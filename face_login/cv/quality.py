"""Quality gate: decide whether a detected face is good enough to enrol/match.

Single responsibility: evaluate frame/face quality and report a pass/fail
verdict with an interpretable breakdown. This module only evaluates quality; it
performs no matching, coverage, UI, or database work.

Checks: minimum face size, detection confidence, blur (variance of Laplacian),
pose confidence, and embedding-norm validity. Failures are reported as
:class:`QualityReason` values so callers can explain or act on them.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from face_login.config import QualityConfig
from face_login.cv.camera import Frame
from face_login.cv.detector import DetectedFace
from face_login.cv.embedder import FaceEmbedding
from face_login.cv.pose import PoseResult

# Reference raw ArcFace norm used to scale the embedding component into [0, 1].
_EMBEDDING_NORM_REFERENCE = 20.0
_DEFAULT_MIN_POSE_CONFIDENCE = 0.5
_DEFAULT_MIN_EMBEDDING_NORM = 1.0


class QualityReason(Enum):
    """Reason a face failed the quality gate (one per failing check)."""

    FACE_TOO_SMALL = "face_too_small"
    TOO_BLURRY = "too_blurry"
    LOW_DETECTION_SCORE = "low_detection_score"
    LOW_POSE_CONFIDENCE = "low_pose_confidence"
    INVALID_EMBEDDING = "invalid_embedding"


@dataclass(frozen=True, slots=True)
class QualityResult:
    """Immutable outcome of the quality gate.

    Attributes:
        passed: ``True`` when every mandatory check passed (``reasons`` empty).
        score: Aggregate quality indicator in ``[0, 1]`` (mean of components).
        blur_score: Variance of the Laplacian over the face region (higher =
            sharper).
        pose_score: Pose-estimation confidence in ``[0, 1]``.
        embedding_norm: Raw L2 norm of the embedding before normalization.
        reasons: Failing checks as :class:`QualityReason` values (empty on pass).
    """

    passed: bool
    score: float
    blur_score: float
    pose_score: float
    embedding_norm: float
    reasons: list[QualityReason]


class QualityGate:
    """Evaluate face quality from a frame, detection, pose, and embedding.

    Thresholds come from :class:`QualityConfig`; pose-confidence and
    embedding-norm floors are separate constructor arguments because they are
    not part of that config. The gate is stateless and reusable.
    """

    def __init__(
        self,
        config: Optional[QualityConfig] = None,
        min_pose_confidence: float = _DEFAULT_MIN_POSE_CONFIDENCE,
        min_embedding_norm: float = _DEFAULT_MIN_EMBEDDING_NORM,
    ) -> None:
        """Store thresholds; no model or per-call state is held."""
        self._config = config or QualityConfig()
        self._min_pose_confidence = min_pose_confidence
        self._min_embedding_norm = min_embedding_norm

    def evaluate(
        self,
        frame: Frame,
        face: DetectedFace,
        pose: PoseResult,
        embedding: FaceEmbedding,
    ) -> QualityResult:
        """Assess quality and return a verdict with a per-metric breakdown.

        Args:
            frame: Source camera frame (BGR image) for blur measurement.
            face: Detected face providing the bounding box and detection score.
            pose: Pose estimate providing the pose confidence.
            embedding: Face embedding providing the raw norm and vector.

        Returns:
            A :class:`QualityResult` whose ``reasons`` explain any failure.
        """
        face_size = self._face_size(face.bounding_box)
        blur_score = self._blur_score(frame.image, face.bounding_box)
        pose_score = float(pose.confidence)
        embedding_norm = float(embedding.norm)

        reasons = self._collect_reasons(
            face_size, face.detection_score, blur_score, pose_score, embedding
        )
        score = self._aggregate_score(
            face_size, face.detection_score, blur_score, pose_score, embedding_norm
        )
        return QualityResult(
            passed=not reasons,
            score=score,
            blur_score=blur_score,
            pose_score=pose_score,
            embedding_norm=embedding_norm,
            reasons=reasons,
        )

    def _collect_reasons(
        self,
        face_size: float,
        detection_score: float,
        blur_score: float,
        pose_score: float,
        embedding: FaceEmbedding,
    ) -> list[QualityReason]:
        """Return the list of failing checks (empty when all pass)."""
        reasons: list[QualityReason] = []
        if face_size < self._config.min_face_size:
            reasons.append(QualityReason.FACE_TOO_SMALL)
        if detection_score < self._config.min_confidence:
            reasons.append(QualityReason.LOW_DETECTION_SCORE)
        if blur_score < self._config.blur_threshold:
            reasons.append(QualityReason.TOO_BLURRY)
        if pose_score < self._min_pose_confidence:
            reasons.append(QualityReason.LOW_POSE_CONFIDENCE)
        if not self._valid_embedding(embedding):
            reasons.append(QualityReason.INVALID_EMBEDDING)
        return reasons

    def _valid_embedding(self, embedding: FaceEmbedding) -> bool:
        """Embedding is valid when its norm clears the floor and it is finite."""
        return (
            embedding.norm >= self._min_embedding_norm
            and bool(np.all(np.isfinite(embedding.embedding)))
        )

    def _aggregate_score(
        self,
        face_size: float,
        detection_score: float,
        blur_score: float,
        pose_score: float,
        embedding_norm: float,
    ) -> float:
        """Combine per-metric components (each in ``[0, 1]``) into a mean score."""
        min_size = max(self._config.min_face_size, 1)
        blur_threshold = max(self._config.blur_threshold, 1e-6)
        components = [
            _clip01(face_size / (2.0 * min_size)),
            _clip01(detection_score),
            _clip01(blur_score / blur_threshold),
            _clip01(pose_score),
            _clip01(embedding_norm / _EMBEDDING_NORM_REFERENCE),
        ]
        return float(np.mean(components))

    @staticmethod
    def _face_size(bounding_box: np.ndarray) -> float:
        """Return the shorter side of the bounding box, in pixels."""
        x1, y1, x2, y2 = (float(v) for v in bounding_box)
        return min(x2 - x1, y2 - y1)

    @staticmethod
    def _blur_score(image: np.ndarray, bounding_box: np.ndarray) -> float:
        """Variance of the Laplacian over the face region (0.0 if empty)."""
        height, width = image.shape[:2]
        x1 = max(int(bounding_box[0]), 0)
        y1 = max(int(bounding_box[1]), 0)
        x2 = min(int(bounding_box[2]), width)
        y2 = min(int(bounding_box[3]), height)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _clip01(value: float) -> float:
    """Clip a value into the closed unit interval ``[0, 1]``."""
    return float(min(max(value, 0.0), 1.0))
