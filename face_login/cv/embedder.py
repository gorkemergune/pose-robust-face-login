"""ArcFace embedding generator (512-D) via InsightFace ONNX Runtime.

Single responsibility: turn an aligned face chip
(:class:`~face_login.cv.alignment.AlignedFace`) into an L2-normalized ArcFace
embedding. The buffalo_l recognition model is lazy-loaded once on first use and
reused thereafter, selecting CUDA when available and CPU otherwise. This module
never matches, estimates pose, checks quality, or touches the database.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from face_login.cv.alignment import AlignedFace
from face_login.logging_setup import get_logger

_LOGGER = get_logger(__name__)

# buffalo_l's recognition model file; other files in the pack are non-recognition.
_RECOGNITION_FILE = "w600k_r50.onnx"
_RECOGNITION_HINTS = ("w600k", "glint", "arcface", "r50", "r100")


class EmbeddingError(RuntimeError):
    """Raised when the recognition model cannot be located or loaded."""


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    """An immutable L2-normalized face embedding.

    Attributes:
        embedding: Unit-length ``float32`` ArcFace vector (512-D for buffalo_l).
        norm: L2 norm of the *raw* embedding before normalization.
        model_name: Name of the model pack that produced the embedding.
    """

    embedding: np.ndarray
    norm: float
    model_name: str


class FaceEmbedder:
    """Generate L2-normalized ArcFace embeddings from aligned faces.

    The recognition model is built exactly once, on the first :meth:`embed`
    call, guarded by a lock so concurrent first calls initialise it a single
    time. The embedder is otherwise stateless and safe to share across threads.
    """

    def __init__(self, model_name: str = "buffalo_l", root: str = ".") -> None:
        """Store the model pack name and root; the model is not loaded yet.

        Args:
            model_name: InsightFace model pack (default ``buffalo_l``).
            root: Model root; the pack resolves to ``{root}/models/{model_name}``.
        """
        self._model_name = model_name
        self._root = root
        self._model: Any = None
        self._lock = threading.Lock()

    def embed(self, aligned: AlignedFace) -> FaceEmbedding:
        """Compute the L2-normalized embedding of an aligned face.

        Args:
            aligned: The aligned RGB face chip (112x112 for buffalo_l).

        Returns:
            A :class:`FaceEmbedding` with a unit-length vector and its raw norm.
        """
        model = self._ensure_model()
        # AlignedFace is RGB; ArcFaceONNX.get_feat swaps R/B internally, so it
        # expects BGR. Convert back to BGR to feed the model correct colours.
        bgr = cv2.cvtColor(aligned.image, cv2.COLOR_RGB2BGR)
        raw = np.asarray(model.get_feat(bgr), dtype=np.float32).flatten()
        norm = float(np.linalg.norm(raw))
        embedding = (raw / norm).astype(np.float32) if norm > 0.0 else raw
        return FaceEmbedding(
            embedding=embedding, norm=norm, model_name=self._model_name
        )

    def _ensure_model(self) -> Any:
        """Return the loaded recognition model, building it once on first use."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._build_model()
        return self._model

    def _build_model(self) -> Any:
        """Locate, load, and prepare the buffalo_l recognition model."""
        from insightface.model_zoo import get_model  # lazy import
        from insightface.utils import storage  # lazy import

        model_dir = storage.ensure_available(
            "models", self._model_name, root=self._root
        )
        onnx_path = self._recognition_onnx(model_dir)
        providers, ctx_id = self._select_runtime()
        model = get_model(onnx_path, providers=providers)
        if getattr(model, "taskname", None) != "recognition":
            raise EmbeddingError(
                f"Model at {onnx_path} is not a recognition model."
            )
        model.prepare(ctx_id)
        _LOGGER.info(
            "ArcFace embedder loaded: pack=%s, file=%s, providers=%s, ctx_id=%d.",
            self._model_name, onnx_path, providers, ctx_id,
        )
        return model

    @staticmethod
    def _recognition_onnx(model_dir: str) -> str:
        """Return the path to the recognition ONNX file inside ``model_dir``."""
        import glob
        import os

        preferred = os.path.join(model_dir, _RECOGNITION_FILE)
        if os.path.isfile(preferred):
            return preferred
        for path in sorted(glob.glob(os.path.join(model_dir, "*.onnx"))):
            if any(h in os.path.basename(path).lower() for h in _RECOGNITION_HINTS):
                return path
        raise EmbeddingError(
            f"No recognition ONNX found in {model_dir} for {_RECOGNITION_FILE!r}."
        )

    @staticmethod
    def _select_runtime() -> tuple[list[str], int]:
        """Pick CUDA when available, otherwise CPU, from ONNX Runtime."""
        import onnxruntime as ort  # lazy: only needed when the model loads

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
        return ["CPUExecutionProvider"], -1
