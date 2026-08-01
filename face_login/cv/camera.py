"""Camera capture: webcam initialization and frame acquisition.

Single responsibility: open a capture device, apply the configured resolution
and frame rate, and deliver raw BGR frames to the pipeline. Higher-level
concerns (detection, resize, reconnect, UI) live elsewhere.
"""
from __future__ import annotations

from types import TracebackType
from typing import Optional

import cv2
import numpy as np

from face_login.config import CameraConfig


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a frame cannot be read."""


class Camera:
    """OpenCV-backed webcam capture with configurable resolution and FPS.

    The device is *not* opened on construction; call :meth:`open` explicitly or
    use the instance as a context manager. Resources are always released, via
    :meth:`close`, the ``with`` block exit, or garbage collection, so a camera
    is never left held open.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        """Store capture settings without touching the hardware yet."""
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._capture: Optional[cv2.VideoCapture] = None

    @classmethod
    def from_config(cls, config: CameraConfig) -> "Camera":
        """Build a :class:`Camera` from a :class:`CameraConfig` section."""
        return cls(
            index=config.index,
            width=config.width,
            height=config.height,
            fps=config.target_fps,
        )

    @property
    def index(self) -> int:
        """Return the configured capture-device index."""
        return self._index

    @property
    def is_opened(self) -> bool:
        """Return ``True`` while the underlying capture device is open."""
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> "Camera":
        """Open the capture device and apply resolution/FPS settings.

        Returns:
            This camera instance, to allow ``camera = Camera(...).open()``.

        Raises:
            CameraError: If the device at the configured index cannot be opened.
        """
        if self.is_opened:
            return self
        capture = cv2.VideoCapture(self._index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Unable to open camera at index {self._index}."
            )
        self._capture = capture
        self._apply_settings()
        return self

    def read(self) -> np.ndarray:
        """Grab the next frame as a BGR image.

        Returns:
            The captured frame as an ``H x W x 3`` BGR ``numpy`` array.

        Raises:
            CameraError: If the camera is not open, or a frame cannot be read.
        """
        if self._capture is None or not self._capture.isOpened():
            raise CameraError("Camera is not open; call open() first.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError(
                f"Failed to read a frame from camera index {self._index}."
            )
        return frame

    def close(self) -> None:
        """Release the capture device. Idempotent and safe to call twice."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _apply_settings(self) -> None:
        """Push the requested width, height, and FPS to the capture device."""
        assert self._capture is not None  # guaranteed by the caller (open)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        self._capture.set(cv2.CAP_PROP_FPS, float(self._fps))

    def __enter__(self) -> "Camera":
        """Open the camera when entering a ``with`` block."""
        return self.open()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Release the camera when leaving a ``with`` block."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup if the camera was never closed explicitly."""
        try:
            self.close()
        except Exception:  # pragma: no cover - defensive at interpreter shutdown
            pass
