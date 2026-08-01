"""Application composition root.

Wires every implemented component together into :class:`FaceLoginApplication`
and drives the capture/recognition/display loop. This module contains no
algorithms, drawing, cosine math, or SQL of its own — it only composes the
existing modules, interprets keystrokes, and manages the resource lifecycle.
"""
from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

from face_login.config import AppConfig, load_config
from face_login.cv.alignment import FaceAligner
from face_login.cv.camera import Camera, CameraError
from face_login.cv.detector import FaceDetector
from face_login.cv.embedder import EmbeddingError, FaceEmbedder
from face_login.cv.pose import SolvePnPPoseEstimator
from face_login.cv.quality import QualityGate
from face_login.database.database import Database, DatabaseError
from face_login.database.repository import Repository
from face_login.logging_setup import configure_logging, get_logger
from face_login.services.coverage import CoverageTracker
from face_login.services.login import LoginService
from face_login.services.matcher import Matcher
from face_login.services.register import RegisterService
from face_login.ui.window import ApplicationWindow

_FPS_SMOOTHING = 0.1


class Mode(Enum):
    """The two operating modes of the application."""

    REGISTER = auto()
    LOGIN = auto()


class FaceLoginApplication:
    """Composition root that runs the face registration/login loop.

    ``__init__`` only constructs and wires components (opening the database and
    creating its schema); no camera capture or loop work starts until
    :meth:`run`. Use as a context manager so resources are always released.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Build every component from configuration; start no capture work."""
        self._config = config or load_config()
        self._logger = get_logger(__name__)
        cfg = self._config

        Path(cfg.database.path).parent.mkdir(parents=True, exist_ok=True)
        self._database = Database(cfg.database.path, initialize=True)
        self._repository = Repository(self._database)

        self._camera = Camera.from_config(cfg.camera)
        self._detector = FaceDetector(cfg.detection)
        self._aligner = FaceAligner(cfg.alignment)
        self._pose = SolvePnPPoseEstimator()
        self._embedder = FaceEmbedder(model_name=cfg.detection.model_name)
        self._quality = QualityGate(cfg.quality)
        self._matcher = Matcher.from_config(cfg.recognition)
        self._coverage_tracker = CoverageTracker(cfg.coverage)
        self._login_service = LoginService(self._repository, self._matcher)
        self._register_service: Optional[RegisterService] = None
        self._window = ApplicationWindow(window_title=cfg.app.name)

        self._mode = Mode.LOGIN
        self._running = False
        self._last_tick: Optional[float] = None
        self._fps: Optional[float] = None

    def __enter__(self) -> "FaceLoginApplication":
        """Enter a ``with`` block, returning this application."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Always release resources when leaving a ``with`` block."""
        self.close()

    def run(self) -> None:
        """Open the camera and run the main loop until the user quits."""
        self._camera.open()
        self._running = True
        self._logger.info("Running in %s mode.", self._mode.name)
        while self._running:
            key = self._tick()
            if key is not None:
                self._handle_key(key)

    def _tick(self) -> Optional[int]:
        """Process and display one frame; return the key, or ``None`` if skipped."""
        try:
            frame = self._camera.read()
        except CameraError as exc:
            self._logger.warning("Camera read failed: %s", exc)
            return None
        context = self._new_context()
        try:
            self._run_pipeline(frame, context)
        except (EmbeddingError, DatabaseError, CameraError) as exc:
            self._logger.warning("Pipeline stage failed: %s", exc)
        except Exception as exc:  # keep the loop alive on unexpected frame errors
            self._logger.error("Unexpected pipeline error: %s", exc)
        return self._render(frame, context)

    def _run_pipeline(self, frame: object, context: dict) -> None:
        """Run detection→align→pose→embed→quality→(register|login) for one frame."""
        faces = self._detector.detect(frame)
        if not faces:
            return
        face = faces[0]
        context["detected_face"] = face
        aligned = self._aligner.align(frame, face)
        pose = self._pose.estimate(face)
        embedding = self._embedder.embed(aligned)
        quality = self._quality.evaluate(frame, face, pose, embedding)
        context["pose"] = pose
        context["quality"] = quality
        if self._mode is Mode.REGISTER and self._register_service is not None:
            result = self._register_service.process(
                pose, quality, embedding, timestamp=frame.timestamp
            )
            context["registration"] = result
            if result.completed:
                self._logger.info("Registration complete; switching to login.")
                self._enter_login_mode()
        else:
            context["login"] = self._login_service.login(embedding)

    def _render(self, frame: object, context: dict) -> int:
        """Draw overlays and coverage, display the frame, and return the key."""
        registering = self._mode is Mode.REGISTER and self._register_service is not None
        coverage = self._coverage_tracker.state() if registering else None
        return self._window.show(
            frame.image,
            detected_face=context["detected_face"],
            pose=context["pose"],
            quality=context["quality"],
            login=context["login"],
            registration=context["registration"],
            coverage=coverage,
            fps=self._update_fps(),
        )

    def _handle_key(self, key: int) -> None:
        """Interpret a key: quit, or switch between register and login modes."""
        code = key & 0xFF
        char = chr(code).lower() if 32 <= code < 127 else ""
        if code == 27 or char == "q":  # ESC or Q
            self._running = False
        elif char == "r":
            self._enter_register_mode()
        elif char == "l":
            self._enter_login_mode()

    def _enter_register_mode(self) -> None:
        """Prompt for a name, optionally overwrite an existing user, and start."""
        name = input("Enter user name: ").strip()
        if not name:
            self._logger.warning("Empty name; staying in %s mode.", self._mode.name)
            return
        if not self._prepare_user(name):
            return
        self._coverage_tracker = CoverageTracker(self._config.coverage)
        self._register_service = RegisterService(
            name, self._repository, self._coverage_tracker
        )
        self._mode = Mode.REGISTER
        self._logger.info("Registering user '%s'.", name)

    def _prepare_user(self, name: str) -> bool:
        """Handle a pre-existing user; return ``True`` to proceed with capture.

        If the name is already enrolled, ask whether to overwrite. On "yes" the
        existing user (and, by cascade, all their embeddings) is deleted so
        registration starts fresh; on "no" registration is cancelled.
        """
        existing = self._repository.get_user_by_name(name)
        if existing is None:
            return True
        count = len(self._repository.get_embeddings(existing.id))
        answer = input(
            f"User '{name}' already exists with {count} embedding(s). "
            "Overwrite? [y/N]: "
        ).strip().lower()
        if answer in ("y", "yes"):
            self._repository.delete_user(existing.id)
            self._logger.info("Deleted existing '%s' (%d embeddings).", name, count)
            return True
        self._logger.info("Registration cancelled; kept existing '%s'.", name)
        return False

    def _enter_login_mode(self) -> None:
        """Return to login mode and clear any registration session."""
        self._mode = Mode.LOGIN
        self._register_service = None
        self._logger.info("Login mode.")

    def _update_fps(self) -> Optional[float]:
        """Return a smoothed frames-per-second estimate for the overlay."""
        now = perf_counter()
        if self._last_tick is not None:
            delta = now - self._last_tick
            if delta > 0.0:
                instant = 1.0 / delta
                self._fps = (
                    instant if self._fps is None
                    else self._fps * (1.0 - _FPS_SMOOTHING) + instant * _FPS_SMOOTHING
                )
        self._last_tick = now
        return self._fps

    def close(self) -> None:
        """Release resources in reverse order: camera, window, database."""
        self._safe_close(self._camera.close, "camera")
        self._safe_close(self._window.close, "window")
        self._safe_close(self._database.close, "database")

    def _safe_close(self, closer: Callable[[], None], name: str) -> None:
        """Invoke a resource's close callback, logging any failure."""
        try:
            closer()
        except Exception as exc:  # shutdown must never raise
            self._logger.warning("Error closing %s: %s", name, exc)

    @staticmethod
    def _new_context() -> dict:
        """Return an empty per-frame result context for the renderer."""
        return {
            "detected_face": None,
            "pose": None,
            "quality": None,
            "login": None,
            "registration": None,
        }


def main() -> int:
    """Entry point: configure logging, then run the application to completion."""
    config = load_config()
    configure_logging(
        level=config.logging.level,
        log_format=config.logging.format,
        log_file=config.logging.file,
    )
    logger = get_logger(__name__)
    try:
        with FaceLoginApplication(config) as app:
            app.run()
    except (CameraError, DatabaseError, EmbeddingError) as exc:
        logger.error("Fatal startup error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    return 0
