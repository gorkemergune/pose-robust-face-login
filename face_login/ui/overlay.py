"""Frame overlay rendering: bounding boxes, pose, quality, and status panels.

Single responsibility: draw already-produced pipeline objects onto a BGR frame.
It consumes DTOs (detected face, pose, quality, login, registration, fps) and
renders them; it never creates windows, reads input, accesses the camera, or
touches services. All visual values come from the constructor, so nothing is
hardcoded in the drawing code. The renderer is stateless and thread-safe.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:  # types for annotations only — objects are passed in
    from face_login.cv.detector import DetectedFace
    from face_login.cv.pose import PoseResult
    from face_login.cv.quality import QualityResult
    from face_login.services.login import LoginResult
    from face_login.services.register import RegistrationResult

Color = tuple[int, int, int]


class OverlayRenderer:
    """Stateless OpenCV renderer for pipeline overlays.

    Every geometry, color, and typography value is a constructor argument and is
    stored read-only, so one instance is reusable across frames and threads.
    ``draw`` reads only its arguments and the stored configuration.
    """

    def __init__(
        self,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        line_thickness: int = 2,
        landmark_radius: int = 2,
        padding: int = 8,
        margin: int = 12,
        line_height: int = 20,
        corner_radius: int = 8,
        panel_alpha: float = 0.5,
        text_color: Color = (255, 255, 255),
        pass_color: Color = (0, 200, 0),
        fail_color: Color = (0, 0, 255),
        box_default_color: Color = (180, 180, 180),
        landmark_color: Color = (0, 255, 255),
        panel_info_color: Color = (40, 40, 40),
        panel_success_color: Color = (0, 120, 0),
        panel_fail_color: Color = (0, 0, 140),
    ) -> None:
        """Store all visual parameters (no magic numbers in the drawing code)."""
        self._font = font
        self._font_scale = font_scale
        self._font_thickness = font_thickness
        self._line_thickness = line_thickness
        self._landmark_radius = landmark_radius
        self._padding = padding
        self._margin = margin
        self._line_height = line_height
        self._corner_radius = corner_radius
        self._panel_alpha = panel_alpha
        self._text_color = text_color
        self._pass_color = pass_color
        self._fail_color = fail_color
        self._box_default_color = box_default_color
        self._landmark_color = landmark_color
        self._panel_info_color = panel_info_color
        self._panel_success_color = panel_success_color
        self._panel_fail_color = panel_fail_color

    def draw(
        self,
        frame: np.ndarray,
        *,
        detected_face: Optional["DetectedFace"] = None,
        pose: Optional["PoseResult"] = None,
        quality: Optional["QualityResult"] = None,
        login: Optional["LoginResult"] = None,
        registration: Optional["RegistrationResult"] = None,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """Render the supplied objects onto ``frame`` in place and return it."""
        if detected_face is not None:
            self._draw_face(frame, detected_face, quality)
        self._draw_info_panel(frame, pose, quality)
        self._draw_status_panel(frame, login, registration)
        if fps is not None:
            self._draw_fps(frame, fps)
        return frame

    # -- face --------------------------------------------------------------

    def _draw_face(
        self,
        frame: np.ndarray,
        face: "DetectedFace",
        quality: Optional["QualityResult"],
    ) -> None:
        """Draw the bounding box (colored by quality) and five landmarks."""
        x1, y1, x2, y2 = (int(v) for v in face.bounding_box)
        color = self._box_color(quality)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self._line_thickness)
        for point in face.five_landmarks:
            center = (int(point[0]), int(point[1]))
            cv2.circle(frame, center, self._landmark_radius, self._landmark_color, cv2.FILLED)

    def _box_color(self, quality: Optional["QualityResult"]) -> Color:
        """Green when quality passed, red when failed, neutral when unknown."""
        if quality is None:
            return self._box_default_color
        return self._pass_color if quality.passed else self._fail_color

    # -- pose + quality info panel ----------------------------------------

    def _draw_info_panel(
        self,
        frame: np.ndarray,
        pose: Optional["PoseResult"],
        quality: Optional["QualityResult"],
    ) -> None:
        """Draw the top-left pose/quality text block."""
        lines = self._info_lines(pose, quality)
        if lines:
            self._draw_text_block(
                frame, lines, (self._margin, self._margin), self._panel_info_color
            )

    def _info_lines(
        self, pose: Optional["PoseResult"], quality: Optional["QualityResult"]
    ) -> list[str]:
        """Build the pose and quality text lines from the given objects."""
        lines: list[str] = []
        if pose is not None:
            lines += [
                f"Yaw: {pose.yaw:.1f}",
                f"Pitch: {pose.pitch:.1f}",
                f"Roll: {pose.roll:.1f}",
                f"Conf: {pose.confidence:.2f}",
            ]
        if quality is not None:
            lines.append(f"Quality: {quality.score:.2f}")
            lines.append("PASS" if quality.passed else "FAIL")
            if not quality.passed:
                lines += [f"- {reason.value}" for reason in quality.reasons]
        return lines

    # -- login / registration status panel --------------------------------

    def _draw_status_panel(
        self,
        frame: np.ndarray,
        login: Optional["LoginResult"],
        registration: Optional["RegistrationResult"],
    ) -> None:
        """Draw login and/or registration status, stacked up from the bottom."""
        bottom = frame.shape[0] - self._margin
        if login is not None:
            lines, background = self._login_lines(login)
            bottom = self._draw_block_bottom(frame, lines, background, bottom)
        if registration is not None:
            lines, background = self._registration_lines(registration)
            self._draw_block_bottom(frame, lines, background, bottom)

    def _login_lines(self, login: "LoginResult") -> tuple[list[str], Color]:
        """Build login-result lines and the panel background color."""
        if login.success:
            yaw = f"{login.best_yaw:.0f}" if login.best_yaw is not None else "-"
            lines = [
                f"Logged in: {login.user_name}",
                f"Similarity: {login.similarity:.3f}",
                f"Best pose: {yaw}",
            ]
            return lines, self._panel_success_color
        lines = ["Unknown user", f"Similarity: {login.similarity:.3f}"]
        return lines, self._panel_fail_color

    def _registration_lines(
        self, registration: "RegistrationResult"
    ) -> tuple[list[str], Color]:
        """Build registration-result lines and the panel background color."""
        current = (
            f"{registration.current_bin:.0f}"
            if registration.current_bin is not None else "-"
        )
        lines = [
            f"Registration: {'Complete' if registration.completed else 'In progress'}",
            f"Coverage: {registration.coverage_percentage:.0f}%",
            f"Current bin: {current}",
            f"Remaining bins: {len(registration.remaining_bins)}",
        ]
        background = (
            self._panel_success_color
            if registration.completed else self._panel_info_color
        )
        return lines, background

    # -- fps ---------------------------------------------------------------

    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw the FPS readout in the top-right corner."""
        text = f"FPS: {fps:.1f}"
        (text_w, text_h), _ = cv2.getTextSize(
            text, self._font, self._font_scale, self._font_thickness
        )
        x = frame.shape[1] - self._margin - text_w
        y = self._margin + text_h
        cv2.putText(
            frame, text, (x, y), self._font, self._font_scale,
            self._text_color, self._font_thickness, cv2.LINE_AA,
        )

    # -- primitives --------------------------------------------------------

    def _draw_block_bottom(
        self, frame: np.ndarray, lines: list[str], background: Color, bottom_y: int
    ) -> int:
        """Draw a text block whose bottom sits at ``bottom_y``; return its top."""
        block_height = len(lines) * self._line_height + 2 * self._padding
        top = bottom_y - block_height
        self._draw_text_block(frame, lines, (self._margin, top), background)
        return top - self._padding

    def _draw_text_block(
        self,
        frame: np.ndarray,
        lines: list[str],
        top_left: tuple[int, int],
        background: Optional[Color],
    ) -> None:
        """Draw a (optionally paneled) block of left-aligned text lines."""
        if not lines:
            return
        widths = [
            cv2.getTextSize(line, self._font, self._font_scale, self._font_thickness)[0][0]
            for line in lines
        ]
        block_w = max(widths) + 2 * self._padding
        block_h = len(lines) * self._line_height + 2 * self._padding
        x, y = top_left
        if background is not None:
            self._panel(frame, (x, y), (x + block_w, y + block_h), background)
        for index, line in enumerate(lines):
            baseline = y + self._padding + self._line_height * (index + 1)
            cv2.putText(
                frame, line, (x + self._padding, baseline), self._font,
                self._font_scale, self._text_color, self._font_thickness, cv2.LINE_AA,
            )

    def _panel(
        self, frame: np.ndarray, top_left: tuple[int, int],
        bottom_right: tuple[int, int], color: Color,
    ) -> None:
        """Blend a translucent rounded panel onto the frame ROI."""
        x1, y1 = max(top_left[0], 0), max(top_left[1], 0)
        x2 = min(bottom_right[0], frame.shape[1])
        y2 = min(bottom_right[1], frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        self._rounded_rect(overlay, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), color)
        cv2.addWeighted(overlay, self._panel_alpha, roi, 1.0 - self._panel_alpha, 0.0, dst=roi)

    def _rounded_rect(
        self, image: np.ndarray, top_left: tuple[int, int],
        bottom_right: tuple[int, int], color: Color,
    ) -> None:
        """Fill a rounded rectangle (plain rectangle when radius is 0)."""
        x1, y1 = top_left
        x2, y2 = bottom_right
        radius = max(0, min(self._corner_radius, (x2 - x1) // 2, (y2 - y1) // 2))
        if radius == 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, cv2.FILLED)
            return
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(image, (cx, cy), radius, color, cv2.FILLED)
