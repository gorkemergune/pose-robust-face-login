"""Application window (presentation): compose overlays and forward keystrokes.

Single responsibility: display already-processed results in an OpenCV window and
return the raw pressed key. It draws via :class:`OverlayRenderer` and
:class:`CoverageBarRenderer`, owns only the window lifecycle, and forwards
``waitKey`` without interpreting it. No camera, database, recognition,
registration, or login logic lives here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from face_login.ui.coverage_bar import CoverageBarRenderer
from face_login.ui.overlay import OverlayRenderer

if TYPE_CHECKING:  # types for annotations only — objects are passed through
    from face_login.cv.detector import DetectedFace
    from face_login.cv.pose import PoseResult
    from face_login.cv.quality import QualityResult
    from face_login.services.coverage import CoverageState
    from face_login.services.login import LoginResult
    from face_login.services.register import RegistrationResult


class ApplicationWindow:
    """A lazily-created OpenCV window that renders overlays and forwards keys.

    State is limited to the window lifecycle; rendering is delegated to injected
    (or default) renderers. Usable as a context manager so the window is always
    destroyed on exit.
    """

    def __init__(
        self,
        window_title: str = "Pose-Robust Face Login",
        resizable: bool = True,
        fullscreen: bool = False,
        window_flags: Optional[int] = None,
        overlay_renderer: Optional[OverlayRenderer] = None,
        coverage_renderer: Optional[CoverageBarRenderer] = None,
    ) -> None:
        """Configure the window; it is not created until the first :meth:`show`.

        Args:
            window_title: OpenCV window name.
            resizable: Whether the window may be resized (ignored if
                ``window_flags`` is given).
            fullscreen: Whether to display the window fullscreen.
            window_flags: Explicit OpenCV window flags overriding ``resizable``.
            overlay_renderer: Overlay renderer (a default is built if ``None``).
            coverage_renderer: Coverage-bar renderer (default built if ``None``).
        """
        self._title = window_title
        self._fullscreen = fullscreen
        self._flags = (
            window_flags if window_flags is not None
            else (cv2.WINDOW_NORMAL if resizable else cv2.WINDOW_AUTOSIZE)
        )
        self._overlay = overlay_renderer or OverlayRenderer()
        self._coverage = coverage_renderer or CoverageBarRenderer()
        self._created = False

    @property
    def title(self) -> str:
        """The window's title."""
        return self._title

    @property
    def is_open(self) -> bool:
        """Whether the OpenCV window currently exists."""
        return self._created

    def show(
        self,
        frame: np.ndarray,
        *,
        detected_face: Optional["DetectedFace"] = None,
        pose: Optional["PoseResult"] = None,
        quality: Optional["QualityResult"] = None,
        login: Optional["LoginResult"] = None,
        registration: Optional["RegistrationResult"] = None,
        coverage: Optional["CoverageState"] = None,
        fps: Optional[float] = None,
    ) -> int:
        """Render overlays onto ``frame``, display it, and return the pressed key.

        Args:
            frame: BGR image to render and display (modified in place).
            detected_face, pose, quality, login, registration, coverage, fps:
                Already-processed results to visualize (any may be ``None``).

        Returns:
            The raw key code from ``cv2.waitKey(1)`` (``-1`` when no key). The
            caller decides what the key means.
        """
        self._ensure_window()
        self._overlay.draw(
            frame,
            detected_face=detected_face,
            pose=pose,
            quality=quality,
            login=login,
            registration=registration,
            fps=fps,
        )
        if coverage is not None:
            self._coverage.draw(frame, coverage)
        cv2.imshow(self._title, frame)
        return cv2.waitKey(1)

    def close(self) -> None:
        """Destroy the window if it exists. Safe to call multiple times."""
        if self._created:
            cv2.destroyWindow(self._title)
            self._created = False

    def _ensure_window(self) -> None:
        """Create and configure the OpenCV window on first use."""
        if self._created:
            return
        cv2.namedWindow(self._title, self._flags)
        if self._fullscreen:
            cv2.setWindowProperty(
                self._title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        self._created = True

    def __enter__(self) -> "ApplicationWindow":
        """Enter a ``with`` block, returning this window."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Always destroy the window when leaving a ``with`` block."""
        self.close()
