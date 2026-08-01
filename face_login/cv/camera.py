"""Camera capture: webcam initialization and frame acquisition.

Single responsibility: open a capture device, apply the configured resolution
and frame rate, and deliver timestamped BGR frames to the pipeline. Higher-level
concerns (detection, resize, reconnect, UI) live elsewhere.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Optional

import cv2
import numpy as np

from face_login.config import CameraConfig
from face_login.logging_setup import get_logger

_LOGGER = get_logger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a frame cannot be read."""


@dataclass(frozen=True)
class Frame:
    """An immutable captured frame with acquisition metadata.

    Attributes:
        image: The captured frame as an ``H x W x 3`` BGR ``numpy`` array.
        timestamp: Wall-clock capture time in seconds since the epoch.
        frame_id: Zero-based index of the frame within the current session.
    """

    image: np.ndarray
    timestamp: float
    frame_id: int


class Camera:
    """OpenCV-backed webcam capture with configurable resolution and FPS.

    The device is *not* opened on construction; call :meth:`open` explicitly or
    use the instance as a context manager. Resources are always released, via
    :meth:`close`, the ``with`` block exit, or garbage collection, so a camera
    is never left held open. All device operations are guarded by a lock, so a
    single instance may be shared safely across threads.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: Optional[int] = None,
    ) -> None:
        """Store capture settings without touching the hardware yet.

        Args:
            backend: Optional OpenCV capture backend (e.g. ``cv2.CAP_DSHOW``,
                ``cv2.CAP_AVFOUNDATION``, ``cv2.CAP_V4L2``). ``None`` lets
                OpenCV select a backend automatically.
        """
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._backend = backend
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._actual_width = 0
        self._actual_height = 0
        self._actual_fps = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_config(
        cls, config: CameraConfig, backend: Optional[int] = None
    ) -> "Camera":
        """Build a :class:`Camera` from a :class:`CameraConfig` section."""
        return cls(
            index=config.index,
            width=config.width,
            height=config.height,
            fps=config.target_fps,
            backend=backend,
        )

    @property
    def index(self) -> int:
        """Return the configured capture-device index."""
        return self._index

    @property
    def is_opened(self) -> bool:
        """Return ``True`` while the underlying capture device is open."""
        capture = self._capture  # local ref: avoids a close() race
        return capture is not None and capture.isOpened()

    @property
    def actual_resolution(self) -> tuple[int, int]:
        """Return the ``(width, height)`` the device actually applied."""
        return (self._actual_width, self._actual_height)

    @property
    def actual_fps(self) -> float:
        """Return the frame rate the device actually applied."""
        return self._actual_fps

    def open(self) -> "Camera":
        """Open the capture device and apply resolution/FPS settings.

        Returns:
            This camera instance, to allow ``camera = Camera(...).open()``.

        Raises:
            CameraError: If the device at the configured index cannot be opened.
        """
        with self._lock:
            if self.is_opened:
                return self
            capture = self._create_capture()
            if not capture.isOpened():
                capture.release()
                raise CameraError(
                    f"Unable to open camera at index {self._index}."
                )
            self._capture = capture
            self._frame_count = 0
            self._apply_settings()
            return self

    def read(self) -> Frame:
        """Grab the next frame.

        Returns:
            A :class:`Frame` wrapping the BGR image, capture timestamp, and a
            session-local frame id.

        Raises:
            CameraError: If the camera is not open, or a frame cannot be read.
        """
        with self._lock:
            if self._capture is None or not self._capture.isOpened():
                raise CameraError("Camera is not open; call open() first.")
            ok, image = self._capture.read()
            if not ok or image is None:
                raise CameraError(
                    f"Failed to read a frame from camera index {self._index}."
                )
            frame = Frame(
                image=image, timestamp=time.time(), frame_id=self._frame_count
            )
            self._frame_count += 1
            return frame

    def close(self) -> None:
        """Release the capture device. Idempotent and safe to call twice."""
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def _create_capture(self) -> cv2.VideoCapture:
        """Construct the ``VideoCapture``, honouring the optional backend."""
        if self._backend is None:
            return cv2.VideoCapture(self._index)
        return cv2.VideoCapture(self._index, self._backend)

    def _apply_settings(self) -> None:
        """Apply width/height/FPS, then read back and log the actual values."""
        assert self._capture is not None  # guaranteed by the caller (open)
        capture = self._capture
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        capture.set(cv2.CAP_PROP_FPS, float(self._fps))
        self._actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        _LOGGER.info(
            "Camera %d opened: requested %dx%d@%dfps, actual %dx%d@%.1ffps.",
            self._index, self._width, self._height, self._fps,
            self._actual_width, self._actual_height, self._actual_fps,
        )

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
