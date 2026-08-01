"""Screen-driven application controller (state machine + input handling).

Drives the multi-screen flow — main menu → (Register: name entry → guided
multi-angle scan → saved) or (Login: scan → welcome → success) — by composing
pipeline components, services, and UI renderers. It interprets keyboard/mouse
input and manages the camera between scans. It contains no drawing primitives,
cosine math, or SQL; rendering and persistence are delegated.
"""
from __future__ import annotations

from enum import Enum, auto
from time import perf_counter
from typing import Callable, Optional

import cv2

from face_login.cv.camera import CameraError, Frame
from face_login.cv.embedder import EmbeddingError
from face_login.cv.quality import QualityReason
from face_login.database.database import DatabaseError
from face_login.services.coverage import CoverageTracker
from face_login.services.register import RegisterService

_MAX_NAME = 24
_WELCOME_SECONDS = 2.0
_REASON_HINTS = {
    QualityReason.FACE_TOO_SMALL: "Move closer",
    QualityReason.TOO_BLURRY: "Hold still",
    QualityReason.LOW_DETECTION_SCORE: "Show your face clearly",
    QualityReason.LOW_POSE_CONFIDENCE: "Keep your face straight",
    QualityReason.INVALID_EMBEDDING: "Try again",
}


class State(Enum):
    """Screens of the application flow."""

    MENU = auto()
    NAME_INPUT = auto()
    SCAN_REGISTER = auto()
    SCAN_LOGIN = auto()
    WELCOME = auto()
    MESSAGE = auto()


class ScreenController:
    """Run the screen state machine over injected collaborators."""

    def __init__(self, *, window, screens, overlay, coverage_bar, repository,
                 login_service, camera, detector, aligner, pose, embedder,
                 quality, coverage_config, logger) -> None:
        """Store injected collaborators and initialize the menu state."""
        self._window = window
        self._screens = screens
        self._overlay = overlay
        self._coverage_bar = coverage_bar
        self._repository = repository
        self._login = login_service
        self._camera = camera
        self._detector = detector
        self._aligner = aligner
        self._pose = pose
        self._embedder = embedder
        self._quality = quality
        self._coverage_config = coverage_config
        self._logger = logger
        self._size = (screens.width, screens.height)
        self._register: Optional[RegisterService] = None
        self._tracker = CoverageTracker(coverage_config)
        self._state = State.MENU
        self._running = False
        self._click: Optional[tuple[int, int]] = None
        self._name = ""
        self._active_name = ""
        self._welcome_at = 0.0
        self._frame_no = 0
        self._msg = ("", "", "info")
        self._handlers: dict[State, Callable[[], None]] = {
            State.MENU: self._menu,
            State.NAME_INPUT: self._name_input,
            State.SCAN_REGISTER: self._scan_register,
            State.SCAN_LOGIN: self._scan_login,
            State.WELCOME: self._welcome,
            State.MESSAGE: self._message,
        }

    def run(self) -> None:
        """Register input handling and run the state machine until quit."""
        self._window.set_mouse_callback(self._on_mouse)
        self._running = True
        self._logger.info("Application UI started.")
        while self._running:
            self._handlers[self._state]()

    def _menu(self) -> None:
        frame, buttons = self._screens.menu()
        action = self._resolve(self._window.show(frame), buttons)
        if action == "register":
            self._name = ""
            self._state = State.NAME_INPUT
        elif action == "login" and self._open_camera():
            self._state = State.SCAN_LOGIN
        elif action in ("quit", "back"):
            self._running = False

    def _name_input(self) -> None:
        frame, buttons = self._screens.name_input(self._name)
        key = self._window.show(frame)
        click = self._hit(buttons)
        code = key & 0xFF
        if click == "submit" or code in (10, 13):
            self._submit_name()
        elif click == "back" or code == 27:
            self._state = State.MENU
        elif code in (8, 127):
            self._name = self._name[:-1]
        elif 32 <= code < 127 and len(self._name) < _MAX_NAME:
            self._name += chr(code)

    def _scan_register(self) -> None:
        frame = self._read()
        if frame is None:
            return self._to_menu()
        face = self._first_face(frame)
        tone, text = self._register_guide(frame, face)
        state = self._tracker.state()
        self._overlay.draw(frame.image, detected_face=face)
        self._coverage_bar.draw(frame.image, state)
        display = self._screens.scan_banner(frame.image, text, tone)
        if (self._window.show(display) & 0xFF) == 27:
            return self._to_menu()
        if state.complete:
            self._camera.close()
            self._show_message("Registration Complete", self._active_name, "success")

    def _scan_login(self) -> None:
        frame = self._read()
        if frame is None:
            return self._to_menu()
        face = self._first_face(frame)
        self._frame_no += 1
        if face is None:
            tone, text = "negative", "No face detected"
        else:
            tone, text = "info", "Scanning... hold still"
            if self._frame_no % 3 == 0:  # throttle the heavy embed + match
                result = self._login_attempt(frame, face)
                if result is not None and result.success:
                    self._active_name = result.user_name or "User"
                    self._camera.close()
                    self._welcome_at = perf_counter()
                    self._state = State.WELCOME
                    return
        self._overlay.draw(frame.image, detected_face=face)
        display = self._screens.scan_banner(frame.image, text, tone)
        if (self._window.show(display) & 0xFF) == 27:
            self._to_menu()

    def _welcome(self) -> None:
        frame = self._screens.message(f"Welcome, {self._active_name}!",
                                      "Verifying login…", "success")
        key = self._window.show(frame)
        if key != -1 or perf_counter() - self._welcome_at > _WELCOME_SECONDS:
            self._show_message("Login Successful", self._active_name, "success")

    def _message(self) -> None:
        title, subtitle, tone = self._msg
        key = self._window.show(self._screens.message(title, subtitle, tone))
        clicked = self._take_click() is not None
        if key != -1 or clicked:
            if (key & 0xFF) in (ord("q"), 27):
                self._running = False
            else:
                self._state = State.MENU

    def _register_guide(self, frame: Frame, face):
        if face is None:
            return "negative", "No face detected - look at the camera"
        pose = self._pose.estimate(face)
        state = self._tracker.state()
        if state.bins[self._bin_index(pose.yaw)].captured:
            return "info", self._guide_text(pose.yaw, state)
        aligned = self._aligner.align(frame, face)
        embedding = self._embedder.embed(aligned)
        quality = self._quality.evaluate(frame, face, pose, embedding)
        if not quality.passed:
            return "negative", self._reason_hint(quality.reasons)
        result = self._register.process(pose, quality, embedding, timestamp=frame.timestamp)
        if result.stored:
            done = self._tracker.state().captured_count
            return "positive", f"Captured {result.current_bin:+.0f} deg   ({done}/{state.total_bins})"
        return "info", "Hold still..."

    def _guide_text(self, yaw: float, state) -> str:
        targets = [b.yaw_center for b in state.remaining_bins]
        if not targets:
            return "Almost done..."
        target = min(targets, key=lambda c: abs(c - yaw))
        arrow = "turn right ->" if target > yaw else "<- turn left"
        return f"You {yaw:+.0f} deg   {arrow} to {target:+.0f}   ({state.captured_count}/{state.total_bins})"

    def _bin_index(self, yaw: float) -> int:
        cfg = self._coverage_config
        width = (cfg.yaw_max - cfg.yaw_min) / max(cfg.yaw_bins, 1)
        return min(max(int((yaw - cfg.yaw_min) // width), 0), cfg.yaw_bins - 1)

    def _first_face(self, frame: Frame):
        try:
            faces = self._detector.detect(frame)
        except Exception as exc:  # keep the loop alive on a detection error
            self._logger.warning("Detection error: %s", exc)
            return None
        return faces[0] if faces else None

    def _login_attempt(self, frame: Frame, face):
        try:
            aligned = self._aligner.align(frame, face)
            embedding = self._embedder.embed(aligned)
            return self._login.login(embedding)
        except (EmbeddingError, DatabaseError) as exc:
            self._logger.warning("Login error: %s", exc)
            return None

    def _submit_name(self) -> None:
        name = self._name.strip()
        if not name:
            return
        existing = self._repository.get_user_by_name(name)
        if existing is not None:
            self._repository.delete_user(existing.id)
            self._logger.info("Overwriting existing user '%s'.", name)
        self._tracker = CoverageTracker(self._coverage_config)
        self._register = RegisterService(name, self._repository, self._tracker)
        self._active_name = name
        if self._open_camera():
            self._state = State.SCAN_REGISTER

    @staticmethod
    def _reason_hint(reasons) -> str:
        for reason in reasons:
            if reason in _REASON_HINTS:
                return _REASON_HINTS[reason]
        return "Improve quality"

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click = (x, y)

    def _take_click(self) -> Optional[tuple[int, int]]:
        click, self._click = self._click, None
        return click

    def _hit(self, buttons) -> Optional[str]:
        click = self._take_click()
        if click is None:
            return None
        for button in buttons:
            x1, y1, x2, y2 = button.rect
            if x1 <= click[0] <= x2 and y1 <= click[1] <= y2:
                return button.action
        return None

    def _resolve(self, key: int, buttons) -> Optional[str]:
        action = self._hit(buttons)
        if action is not None:
            return action
        code = key & 0xFF
        char = chr(code).lower() if 32 <= code < 127 else ""
        for button in buttons:
            if button.key and button.key == char:
                return button.action
        return "back" if code == 27 else None

    def _read(self) -> Optional[Frame]:
        try:
            raw = self._camera.read()
        except CameraError as exc:
            self._logger.warning("Camera read failed: %s", exc)
            return None
        image = cv2.flip(cv2.resize(raw.image, self._size), 1)  # mirror: natural selfie view
        return Frame(image=image, timestamp=raw.timestamp, frame_id=raw.frame_id)

    def _open_camera(self) -> bool:
        try:
            self._camera.open()
            return True
        except CameraError as exc:
            self._logger.error("Camera open failed: %s", exc)
            self._show_message("Camera Unavailable", "Check the device", "error")
            return False

    def _to_menu(self) -> None:
        self._camera.close()
        self._state = State.MENU

    def _show_message(self, title: str, subtitle: str, tone: str) -> None:
        self._msg = (title, subtitle, tone)
        self._state = State.MESSAGE
