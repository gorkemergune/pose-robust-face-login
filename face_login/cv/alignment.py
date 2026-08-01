"""Face alignment: landmark-based normalized crop via InsightFace ``norm_crop``.

Single responsibility: warp a :class:`~face_login.cv.detector.DetectedFace` out
of its source :class:`~face_login.cv.camera.Frame` into a normalized,
aligned RGB chip suitable for embedding. This module produces no embeddings,
pose, or quality scores, and performs no database access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from face_login.config import AlignmentConfig
from face_login.cv.camera import Frame
from face_login.cv.detector import DetectedFace


@dataclass(frozen=True, slots=True)
class AlignedFace:
    """An immutable aligned face chip together with its source geometry.

    Attributes:
        image: Aligned face as an ``image_size x image_size x 3`` RGB array
            (112x112 by default). OpenCV captures BGR; this chip is already
            converted, so every module after the aligner works on RGB.
        bounding_box: Source ``[x1, y1, x2, y2]`` box carried over from detection.
        five_landmarks: Source ``5 x 2`` landmark array used for the alignment.
        transform_matrix: Optional ``2 x 3`` affine matrix mapping original-image
            coordinates into the aligned crop (``None`` when not provided).
    """

    image: np.ndarray
    bounding_box: np.ndarray
    five_landmarks: np.ndarray
    transform_matrix: np.ndarray | None = None


class FaceAligner:
    """Align detected faces into normalized RGB chips using ``norm_crop``.

    Alignment is stateless apart from the configured target crop size, so a
    single instance can be reused freely across frames.
    """

    def __init__(self, config: Optional[AlignmentConfig] = None) -> None:
        """Store the target crop size (defaults to :class:`AlignmentConfig`)."""
        self._config = config or AlignmentConfig()

    def align(self, frame: Frame, face: DetectedFace) -> AlignedFace:
        """Warp ``face`` from ``frame`` into a normalized aligned chip.

        Args:
            frame: The source camera frame (BGR image).
            face: The detected face whose five landmarks drive the similarity
                transform.

        Returns:
            An :class:`AlignedFace` whose ``image`` is a normalized RGB crop of
            the configured size (112x112 by default).
        """
        from insightface.utils.face_align import estimate_norm  # lazy import

        size = self._config.image_size
        transform = estimate_norm(face.five_landmarks, image_size=size)
        aligned_bgr = cv2.warpAffine(
            frame.image, transform, (size, size), borderValue=0.0
        )
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        return AlignedFace(
            image=aligned_rgb,
            bounding_box=face.bounding_box,
            five_landmarks=face.five_landmarks,
            transform_matrix=transform,
        )
