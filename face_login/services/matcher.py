"""Matcher: cosine-similarity nearest-neighbour search over the gallery.

Single responsibility: compare a probe embedding against a list of stored
embeddings and apply the decision threshold. Everything is passed in as
arguments; the matcher never touches SQLite, the repository, pose, quality, or
recognition. It is stateless (only an immutable threshold) and therefore
thread-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # import types for annotations only — no runtime coupling
    from face_login.config import RecognitionConfig
    from face_login.cv.embedder import FaceEmbedding
    from face_login.database.repository import EmbeddingRecord

_DEFAULT_THRESHOLD = 0.44


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    """An immutable per-embedding similarity result.

    Attributes:
        user_id: Owner of the gallery embedding.
        embedding_id: Identifier of the specific stored embedding.
        yaw_center: Pose-bin yaw of the stored embedding, in degrees.
        similarity: Cosine similarity to the probe, in ``[-1, 1]``.
    """

    user_id: int
    embedding_id: int
    yaw_center: float
    similarity: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    """An immutable matching decision with the full ranked candidate list.

    Attributes:
        matched: Whether the best similarity met the threshold.
        best_user_id: Matched user's id, or ``None`` when not matched.
        similarity: Best (top) similarity score (0.0 for an empty gallery).
        best_yaw: Matched embedding's yaw, or ``None`` when not matched.
        candidates: All candidates, ranked best-first (deterministic order).
    """

    matched: bool
    best_user_id: Optional[int]
    similarity: float
    best_yaw: Optional[float]
    candidates: list[MatchCandidate]


class Matcher:
    """Rank a probe embedding against a gallery by cosine similarity.

    Embeddings are assumed L2-normalized, so cosine similarity reduces to a dot
    product computed with a single vectorized matrix-vector multiply.
    """

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        """Store the immutable acceptance threshold (default 0.44)."""
        self._threshold = threshold

    @classmethod
    def from_config(cls, config: "RecognitionConfig") -> "Matcher":
        """Build a :class:`Matcher` from a :class:`RecognitionConfig`."""
        return cls(threshold=config.threshold)

    @property
    def threshold(self) -> float:
        """The cosine-similarity acceptance threshold."""
        return self._threshold

    def match(
        self, probe: "FaceEmbedding", gallery: list["EmbeddingRecord"]
    ) -> MatchResult:
        """Match ``probe`` against ``gallery`` and return the ranked decision.

        Args:
            probe: The L2-normalized query embedding.
            gallery: Stored L2-normalized embeddings to compare against.

        Returns:
            A :class:`MatchResult` with the best match and the full ranking.
        """
        if not gallery:
            return MatchResult(False, None, 0.0, None, [])
        matrix = np.stack([record.embedding for record in gallery]).astype(
            np.float32
        )
        query = np.asarray(probe.embedding, dtype=np.float32)
        similarities = matrix @ query  # dot product == cosine (unit vectors)
        candidates = self._rank(gallery, similarities)
        best = candidates[0]
        matched = best.similarity >= self._threshold
        return MatchResult(
            matched=matched,
            best_user_id=best.user_id if matched else None,
            similarity=best.similarity,
            best_yaw=best.yaw_center if matched else None,
            candidates=candidates,
        )

    @staticmethod
    def _rank(
        gallery: list["EmbeddingRecord"], similarities: np.ndarray
    ) -> list[MatchCandidate]:
        """Build candidates and sort best-first with deterministic tie-breaks."""
        candidates = [
            MatchCandidate(
                user_id=record.user_id,
                embedding_id=record.id,
                yaw_center=record.yaw_center,
                similarity=float(similarity),
            )
            for record, similarity in zip(gallery, similarities)
        ]
        # Descending similarity; ties broken by user_id then embedding_id (asc)
        # so identical scores always order the same way.
        candidates.sort(
            key=lambda c: (-c.similarity, c.user_id, c.embedding_id)
        )
        return candidates
