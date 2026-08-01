"""Face detection adapter built on InsightFace ``FaceAnalysis`` (buffalo_l).

Single responsibility: locate faces in a :class:`~face_login.cv.camera.Frame`
and return their bounding boxes, detection scores, and five-point landmarks.
The heavy model is lazy-loaded once on first use and reused thereafter. This
module performs no alignment, embedding, pose estimation, UI, or database work.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from face_login.config import DetectionConfig
from face_login.cv.camera import Frame
from face_login.logging_setup import get_logger

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """An immutable single face detection result.

    Attributes:
        bounding_box: ``[x1, y1, x2, y2]`` box in pixel coordinates.
        detection_score: Detector confidence in ``[0, 1]``.
        five_landmarks: ``5 x 2`` array of eye/nose/mouth-corner points.
        image: Optional unaligned crop, when already available (else ``None``).
    """

    bounding_box: np.ndarray
    detection_score: float
    five_landmarks: np.ndarray
    image: np.ndarray | None = None


class FaceDetector:
    """Detect faces in frames using a lazily-loaded buffalo_l model.

    The InsightFace model is built exactly once, on the first :meth:`detect`
    call, under a lock so concurrent first calls still initialise it a single
    time. CPU/CUDA execution is selected automatically from the available
    ONNX Runtime providers.
    """

    def __init__(
        self, config: Optional[DetectionConfig] = None, root: str = "."
    ) -> None:
        """Store detection settings; the model is not loaded until first use.

        Args:
            config: Detection thresholds/model name (defaults to
                :class:`DetectionConfig`).
            root: InsightFace model root; ``buffalo_l`` resolves to
                ``{root}/models/buffalo_l`` (the repository ``models/`` folder).
        """
        self._config = config or DetectionConfig()
        self._root = root
        self._app: Any = None
        self._lock = threading.Lock()

    def detect(self, frame: Frame) -> list[DetectedFace]:
        """Detect all faces in ``frame`` and return them as ``DetectedFace``.

        Args:
            frame: A camera :class:`Frame` whose ``image`` is a BGR array.

        Returns:
            One :class:`DetectedFace` per detected face (possibly empty).
        """
        app = self._ensure_app()
        faces = app.get(frame.image)
        return [self._convert(face) for face in faces]

    def _ensure_app(self) -> Any:
        """Return the loaded model, building it once on first use."""
        if self._app is None:
            with self._lock:
                if self._app is None:
                    self._app = self._build_app()
        return self._app

    def _build_app(self) -> Any:
        """Construct and prepare the detection-only InsightFace model."""
        from insightface.app import FaceAnalysis  # lazy: import on first use

        providers, ctx_id = self._select_runtime()
        app = FaceAnalysis(
            name=self._config.model_name,
            root=self._root,
            allowed_modules=["detection"],
            providers=providers,
        )
        det_size = (self._config.det_size, self._config.det_size)
        app.prepare(
            ctx_id=ctx_id,
            det_size=det_size,
            det_thresh=self._config.det_threshold,
        )
        _LOGGER.info(
            "Face detector loaded: model=%s, providers=%s, ctx_id=%d, det_size=%s.",
            self._config.model_name, providers, ctx_id, det_size,
        )
        return app

    @staticmethod
    def _select_runtime() -> tuple[list[str], int]:
        """Pick CUDA when available, otherwise CPU, from ONNX Runtime."""
        import onnxruntime as ort  # lazy: only needed when the model loads

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
        return ["CPUExecutionProvider"], -1

    @staticmethod
    def _convert(face: Any) -> DetectedFace:
        """Map one InsightFace ``Face`` to an immutable :class:`DetectedFace`."""
        return DetectedFace(
            bounding_box=np.asarray(face.bbox, dtype=np.float32),
            detection_score=float(face.det_score),
            five_landmarks=np.asarray(face.kps, dtype=np.float32),
        )
