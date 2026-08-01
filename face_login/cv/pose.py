"""Head pose estimation from five facial landmarks via ``solvePnP``.

Single responsibility: estimate 3D head orientation (yaw / pitch / roll) for a
detected face and report a confidence. The estimator is stateless and hidden
behind the :class:`PoseEstimator` protocol so it can be swapped for another
implementation without changing callers. No coverage, quality, embedding, or
database logic lives here.

Coordinate system (OpenCV camera convention):
    * X axis points to the right of the image.
    * Y axis points down the image.
    * Z axis points forward, into the scene, away from the camera.

Angle sign conventions (degrees), described in image space to avoid any
left/right ambiguity:
    * yaw   > 0: the nose points toward the RIGHT side of the image
                 (the head turns to the viewer's right); yaw < 0 is left.
    * pitch > 0: the head tips UP (the face looks upward); pitch < 0 is down.
    * roll  > 0: the head tilts CLOCKWISE as seen by the viewer
                 (the top of the head leans toward the image's right).

Magnitudes are approximate: camera intrinsics are inferred from the face
bounding box rather than a calibrated camera, so use the angles as relative
head-orientation estimates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from face_login.cv.detector import DetectedFace

# Sign mapping applied to the raw Euler decomposition so the returned angles
# match the documented conventions above. Validated against synthetic rotations.
_YAW_SIGN = -1.0
_PITCH_SIGN = -1.0
_ROLL_SIGN = 1.0


@dataclass(frozen=True, slots=True)
class PoseResult:
    """Immutable head-pose estimate.

    Attributes:
        yaw: Left/right head rotation in degrees (see module sign convention).
        pitch: Up/down head rotation in degrees.
        roll: In-plane head tilt in degrees.
        confidence: Reprojection-based confidence in ``[0, 1]`` (1 = best fit).
    """

    yaw: float
    pitch: float
    roll: float
    confidence: float


class PoseEstimator(Protocol):
    """Structural interface for any head-pose estimator.

    Any object exposing ``estimate(DetectedFace) -> PoseResult`` satisfies this
    protocol, so alternative estimators (e.g. a learned model) can replace the
    default without touching call sites.
    """

    def estimate(self, face: DetectedFace) -> PoseResult:
        """Estimate head pose for a detected face."""
        ...


class SolvePnPPoseEstimator:
    """Estimate head pose by fitting a canonical 3D face model with ``solvePnP``.

    A fixed 5-point 3D model is aligned to the detected 2D landmarks. Camera
    intrinsics are approximated from the face bounding box (no calibration is
    available), so the reported angles are estimates rather than metric truth.
    The estimator holds no per-call state and is safe to reuse and share.
    """

    # Canonical 3D face points in millimetres, in the InsightFace landmark
    # order: left eye, right eye, nose tip, left mouth corner, right mouth
    # corner. Axes follow the module coordinate system (X right, Y down,
    # Z forward); the nose tip is the frontmost point (Z = 0).
    _MODEL_POINTS = np.array(
        [
            [-35.0, -35.0, 30.0],  # left eye
            [35.0, -35.0, 30.0],   # right eye
            [0.0, 0.0, 0.0],       # nose tip
            [-25.0, 30.0, 25.0],   # left mouth corner
            [25.0, 30.0, 25.0],    # right mouth corner
        ],
        dtype=np.float64,
    )

    def estimate(self, face: DetectedFace) -> PoseResult:
        """Estimate head pose from the five landmarks of ``face``.

        Args:
            face: The detected face providing 2D landmarks and a bounding box.

        Returns:
            A :class:`PoseResult`. On solver failure all angles are ``0`` and
            ``confidence`` is ``0``.
        """
        landmarks = np.asarray(face.five_landmarks, dtype=np.float64)
        camera_matrix = self._camera_matrix(face.bounding_box)
        ok, rvec, tvec = cv2.solvePnP(
            self._MODEL_POINTS, landmarks, camera_matrix, None,
            flags=cv2.SOLVEPNP_SQPNP,  # works with 5 points (ITERATIVE needs 6)
        )
        if not ok:
            return PoseResult(0.0, 0.0, 0.0, 0.0)
        rotation, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = self._rotation_to_euler(rotation)
        confidence = self._confidence(
            landmarks, camera_matrix, rvec, tvec, face.bounding_box
        )
        return PoseResult(yaw=yaw, pitch=pitch, roll=roll, confidence=confidence)

    @staticmethod
    def _camera_matrix(bounding_box: np.ndarray) -> np.ndarray:
        """Approximate a pinhole camera matrix from the face bounding box."""
        x1, y1, x2, y2 = (float(v) for v in bounding_box)
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        focal = 2.0 * max(width, height)  # heuristic; no calibration available
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return np.array(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
        """Decompose a rotation matrix into (yaw, pitch, roll) degrees."""
        sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.atan2(rotation[2, 1], rotation[2, 2])
            yaw = math.atan2(-rotation[2, 0], sy)
            roll = math.atan2(rotation[1, 0], rotation[0, 0])
        else:  # gimbal lock
            pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
            yaw = math.atan2(-rotation[2, 0], sy)
            roll = 0.0
        return (
            _YAW_SIGN * math.degrees(yaw),
            _PITCH_SIGN * math.degrees(pitch),
            _ROLL_SIGN * math.degrees(roll),
        )

    def _confidence(
        self,
        landmarks: np.ndarray,
        camera_matrix: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        bounding_box: np.ndarray,
    ) -> float:
        """Confidence from mean landmark reprojection error, in ``[0, 1]``."""
        projected, _ = cv2.projectPoints(
            self._MODEL_POINTS, rvec, tvec, camera_matrix, None
        )
        error = float(
            np.linalg.norm(projected.reshape(-1, 2) - landmarks, axis=1).mean()
        )
        x1, y1, x2, y2 = (float(v) for v in bounding_box)
        face_size = max(x2 - x1, y2 - y1, 1.0)
        return float(math.exp(-error / face_size))
