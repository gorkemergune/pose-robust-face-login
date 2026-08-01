"""Registration use case (register branch).

Single responsibility: orchestrate multi-pose registration from already-processed
pipeline outputs. Per frame it receives a :class:`PoseResult`,
:class:`QualityResult`, and :class:`FaceEmbedding`, advances the
:class:`CoverageTracker`, and — only when a new pose bin is accepted — persists
the embedding through the :class:`Repository`.

It performs no detection, pose estimation, quality evaluation, matching, or UI;
it merely wires the existing modules together and writes through the repository.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # types for annotations only — dependencies are injected
    from face_login.cv.embedder import FaceEmbedding
    from face_login.cv.pose import PoseResult
    from face_login.cv.quality import QualityResult
    from face_login.database.repository import Repository
    from face_login.services.coverage import CoverageState, CoverageTracker, PoseBin


@dataclass(frozen=True, slots=True)
class RegistrationProgress:
    """An immutable snapshot of overall registration coverage progress."""

    captured_count: int
    total_bins: int
    coverage_percentage: float
    complete: bool
    remaining_bins: list[float]


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    """An immutable outcome of processing a single registration frame.

    Attributes:
        completed: Whether registration coverage is now complete.
        stored: Whether this frame produced a newly stored embedding.
        current_bin: Yaw center of the bin captured this frame, else ``None``.
        coverage_percentage: Captured bins as a percentage of total bins.
        remaining_bins: Yaw centers of the not-yet-captured bins.
    """

    completed: bool
    stored: bool
    current_bin: Optional[float]
    coverage_percentage: float
    remaining_bins: list[float]


class RegisterService:
    """Orchestrate registration for one user across multiple head poses.

    Dependencies are injected: a :class:`Repository` for persistence and a
    :class:`CoverageTracker` for pose-bin bookkeeping. The user row is created
    lazily on the first accepted capture, so an aborted session leaves no empty
    user behind.
    """

    def __init__(
        self,
        name: str,
        repository: "Repository",
        coverage_tracker: "CoverageTracker",
    ) -> None:
        """Store the target user name and the injected collaborators."""
        self._name = name
        self._repository = repository
        self._tracker = coverage_tracker
        self._user_id: Optional[int] = None

    @property
    def user_id(self) -> Optional[int]:
        """Id of the registered user, or ``None`` before the first capture."""
        return self._user_id

    def process(
        self,
        pose: "PoseResult",
        quality: "QualityResult",
        embedding: "FaceEmbedding",
        timestamp: Optional[float] = None,
    ) -> RegistrationResult:
        """Advance registration with one processed frame.

        Failed quality results and already-captured bins are ignored (no store).
        A store happens only when the tracker accepts a new pose bin.

        Args:
            pose: Head-pose estimate whose yaw selects the bin.
            quality: Quality verdict; failed results never store.
            embedding: L2-normalized embedding to persist on acceptance.
            timestamp: Optional caller-supplied capture time for the bin.

        Returns:
            The :class:`RegistrationResult` for this frame.
        """
        if self._tracker.state().complete:  # stop once coverage is complete
            return self._result(self._tracker.state(), stored=False, current_bin=None)

        before = self._tracker.state()
        state = self._tracker.update(pose, quality, timestamp)
        if state.captured_count <= before.captured_count:  # ignored / duplicate
            return self._result(state, stored=False, current_bin=None)

        captured_bin = self._newly_captured_bin(before, state)
        self._store(captured_bin, quality, embedding)
        return self._result(state, stored=True, current_bin=captured_bin.yaw_center)

    def progress(self) -> RegistrationProgress:
        """Return the current overall registration progress snapshot."""
        state = self._tracker.state()
        return RegistrationProgress(
            captured_count=state.captured_count,
            total_bins=state.total_bins,
            coverage_percentage=state.coverage_percentage,
            complete=state.complete,
            remaining_bins=[pose_bin.yaw_center for pose_bin in state.remaining_bins],
        )

    def _store(
        self,
        captured_bin: "PoseBin",
        quality: "QualityResult",
        embedding: "FaceEmbedding",
    ) -> None:
        """Persist the accepted embedding for a bin through the repository."""
        user_id = self._ensure_user()
        self._repository.add_embedding(
            user_id=user_id,
            yaw_center=captured_bin.yaw_center,
            quality_score=quality.score,
            embedding=embedding.embedding,
        )

    def _ensure_user(self) -> int:
        """Return the user id, creating (or reusing by name) on first need."""
        if self._user_id is None:
            existing = self._repository.get_user_by_name(self._name)
            user = existing or self._repository.create_user(self._name)
            self._user_id = user.id
        return self._user_id

    @staticmethod
    def _newly_captured_bin(
        before: "CoverageState", after: "CoverageState"
    ) -> "PoseBin":
        """Return the single bin that became captured between two states."""
        previous = {pose_bin.yaw_center for pose_bin in before.captured_bins}
        for pose_bin in after.captured_bins:
            if pose_bin.yaw_center not in previous:
                return pose_bin
        raise RuntimeError("Expected a newly captured bin but found none.")

    @staticmethod
    def _result(
        state: "CoverageState", stored: bool, current_bin: Optional[float]
    ) -> RegistrationResult:
        """Build a :class:`RegistrationResult` from a coverage state."""
        return RegistrationResult(
            completed=state.complete,
            stored=stored,
            current_bin=current_bin,
            coverage_percentage=state.coverage_percentage,
            remaining_bins=[pose_bin.yaw_center for pose_bin in state.remaining_bins],
        )
