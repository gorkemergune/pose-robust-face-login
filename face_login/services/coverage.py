"""Coverage tracker: yaw-bin occupancy map for multi-pose registration.

Single responsibility: divide the yaw range into bins and track which bins have
been captured from passing frames, exposing an immutable :class:`CoverageState`
snapshot. The tracker is fully deterministic: it reads no clock and uses no
randomness, so an identical sequence of inputs always yields identical output.

It generates no embeddings, estimates no pose, draws no UI, and never touches
the database.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from face_login.config import CoverageConfig
from face_login.cv.pose import PoseResult
from face_login.cv.quality import QualityResult


@dataclass(frozen=True, slots=True)
class PoseBin:
    """An immutable yaw bin and its capture status.

    Attributes:
        yaw_center: Center yaw angle of the bin, in degrees.
        captured: Whether a passing frame has been captured for this bin.
        quality_score: Quality score of the captured frame (0.0 if uncaptured).
        timestamp: Capture time supplied by the caller (``None`` if uncaptured).
    """

    yaw_center: float
    captured: bool = False
    quality_score: float = 0.0
    timestamp: float | None = None


@dataclass(frozen=True, slots=True)
class CoverageState:
    """An immutable snapshot of registration pose coverage.

    Attributes:
        bins: All yaw bins in ascending center order.
        coverage_percentage: Captured bins as a percentage of total bins.
        complete: Whether every bin has been captured.
    """

    bins: tuple[PoseBin, ...]
    coverage_percentage: float
    complete: bool

    @property
    def total_bins(self) -> int:
        """Total number of yaw bins."""
        return len(self.bins)

    @property
    def captured_count(self) -> int:
        """Number of bins that have been captured."""
        return sum(1 for pose_bin in self.bins if pose_bin.captured)

    @property
    def captured_bins(self) -> tuple[PoseBin, ...]:
        """The captured bins, in ascending center order."""
        return tuple(pose_bin for pose_bin in self.bins if pose_bin.captured)

    @property
    def remaining_bins(self) -> tuple[PoseBin, ...]:
        """The not-yet-captured bins, in ascending center order."""
        return tuple(pose_bin for pose_bin in self.bins if not pose_bin.captured)


class CoverageTracker:
    """Track captured yaw bins across passing frames during registration.

    Bins are derived from :class:`CoverageConfig` (range and count). Each bin is
    captured at most once (first passing frame wins), which suppresses
    duplicates. All state transitions depend solely on the inputs, keeping the
    tracker deterministic.
    """

    def __init__(self, config: Optional[CoverageConfig] = None) -> None:
        """Build the empty bin grid from ``config`` (defaults applied)."""
        self._config = config or CoverageConfig()
        span = self._config.yaw_max - self._config.yaw_min
        self._width = span / max(self._config.yaw_bins, 1)
        self._bins: list[PoseBin] = self._initial_bins()

    def update(
        self,
        pose: PoseResult,
        quality: QualityResult,
        timestamp: Optional[float] = None,
    ) -> CoverageState:
        """Record a capture when quality passes, then return the new state.

        Args:
            pose: Head-pose estimate whose yaw selects the bin.
            quality: Quality verdict; failed results are ignored.
            timestamp: Optional caller-supplied capture time (e.g.
                ``Frame.timestamp``). Kept external so the tracker stays
                deterministic.

        Returns:
            The updated :class:`CoverageState` snapshot.
        """
        if quality.passed:
            index = self._bin_index(pose.yaw)
            current = self._bins[index]
            if not current.captured:  # duplicate suppression: first capture wins
                self._bins[index] = PoseBin(
                    yaw_center=current.yaw_center,
                    captured=True,
                    quality_score=quality.score,
                    timestamp=timestamp,
                )
        return self.state()

    def state(self) -> CoverageState:
        """Return the current immutable coverage snapshot."""
        bins = tuple(self._bins)
        total = len(bins)
        captured = sum(1 for pose_bin in bins if pose_bin.captured)
        percentage = (100.0 * captured / total) if total else 0.0
        return CoverageState(
            bins=bins,
            coverage_percentage=percentage,
            complete=total > 0 and captured == total,
        )

    def reset(self) -> None:
        """Clear all captures, returning to an empty bin grid."""
        self._bins = self._initial_bins()

    def _initial_bins(self) -> list[PoseBin]:
        """Create the empty, ascending-order bin grid from the config."""
        return [
            PoseBin(yaw_center=self._center(index))
            for index in range(self._config.yaw_bins)
        ]

    def _center(self, index: int) -> float:
        """Return the center yaw (degrees) of the bin at ``index``."""
        return self._config.yaw_min + (index + 0.5) * self._width

    def _bin_index(self, yaw: float) -> int:
        """Map a yaw angle to a bin index, clamped to the valid range."""
        raw = (yaw - self._config.yaw_min) / self._width
        last = self._config.yaw_bins - 1
        return int(min(max(math.floor(raw), 0), last))
