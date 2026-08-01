"""Coverage bar widget: visualize occupied pose bins during registration.

Single responsibility: draw a segmented coverage bar and its text summary onto a
BGR frame from an immutable :class:`CoverageState`. One segment is rendered per
yaw bin (green when captured, gray otherwise), with a coverage percentage and a
captured/total count.

This is a pure renderer: no camera, detection, pose, matching, registration, UI
window, ``waitKey``, or global state. Every visual value comes from the
constructor, so nothing is hardcoded in the drawing code.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:  # type for annotations only
    from face_login.services.coverage import CoverageState

Color = tuple[int, int, int]


class CoverageBarRenderer:
    """Stateless OpenCV renderer for a pose-coverage bar.

    All geometry, colors, and typography are supplied to the constructor and
    stored read-only, so a single instance can be reused across frames and
    threads. ``draw`` reads only its arguments and the stored configuration.
    """

    def __init__(
        self,
        margin: int = 20,
        bar_height: int = 28,
        spacing: int = 4,
        captured_color: Color = (0, 200, 0),
        empty_color: Color = (110, 110, 110),
        text_color: Color = (255, 255, 255),
        outline_color: Color = (30, 30, 30),
        outline_thickness: int = 1,
        font_scale: float = 0.6,
        font_thickness: int = 1,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        text_gap: int = 10,
        line_height: int = 22,
    ) -> None:
        """Store all visual parameters (no magic numbers live in the drawing)."""
        self._margin = margin
        self._bar_height = bar_height
        self._spacing = spacing
        self._captured_color = captured_color
        self._empty_color = empty_color
        self._text_color = text_color
        self._outline_color = outline_color
        self._outline_thickness = outline_thickness
        self._font_scale = font_scale
        self._font_thickness = font_thickness
        self._font = font
        self._text_gap = text_gap
        self._line_height = line_height

    def draw(self, frame: np.ndarray, coverage: "CoverageState") -> np.ndarray:
        """Draw the coverage bar and summary onto ``frame`` (in place).

        Args:
            frame: BGR image to draw on; it is modified directly.
            coverage: Coverage snapshot providing the bins and totals.

        Returns:
            The same ``frame`` instance, for convenient chaining.
        """
        width = frame.shape[1]
        self._draw_segments(frame, coverage, width)
        self._draw_summary(frame, coverage)
        return frame

    def _draw_segments(
        self, frame: np.ndarray, coverage: "CoverageState", width: int
    ) -> None:
        """Render one rectangle per yaw bin across the available width."""
        count = coverage.total_bins
        if count <= 0:
            return
        inner_width = width - 2 * self._margin
        total_spacing = self._spacing * (count - 1)
        segment_width = max((inner_width - total_spacing) / count, 1.0)
        top = self._margin
        bottom = self._margin + self._bar_height
        cursor = float(self._margin)
        for pose_bin in coverage.bins:
            left = int(round(cursor))
            right = int(round(cursor + segment_width))
            color = self._captured_color if pose_bin.captured else self._empty_color
            cv2.rectangle(frame, (left, top), (right, bottom), color, cv2.FILLED)
            if self._outline_thickness > 0:
                cv2.rectangle(
                    frame, (left, top), (right, bottom),
                    self._outline_color, self._outline_thickness,
                )
            cursor += segment_width + self._spacing

    def _draw_summary(self, frame: np.ndarray, coverage: "CoverageState") -> None:
        """Render the coverage percentage and captured/total count lines."""
        percentage = int(round(coverage.coverage_percentage))
        first_baseline = self._margin + self._bar_height + self._text_gap + self._line_height
        self._put_text(frame, f"Coverage: {percentage}%", first_baseline)
        second_baseline = first_baseline + self._line_height
        self._put_text(
            frame,
            f"Captured: {coverage.captured_count} / {coverage.total_bins}",
            second_baseline,
        )

    def _put_text(self, frame: np.ndarray, text: str, baseline_y: int) -> None:
        """Draw one left-aligned text line at the given baseline."""
        cv2.putText(
            frame, text, (self._margin, baseline_y), self._font,
            self._font_scale, self._text_color, self._font_thickness, cv2.LINE_AA,
        )
