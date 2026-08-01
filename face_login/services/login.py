"""Login use case (login branch).

Single responsibility: orchestrate authentication from an already-computed probe
:class:`FaceEmbedding`. It loads the gallery through the :class:`Repository`,
delegates all similarity logic to the :class:`Matcher`, logs the attempt, and
returns a decision.

It performs no detection, pose, embedding, quality, cosine, or UI work. The
service is stateless (collaborators are injected) and therefore thread-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # types for annotations only — dependencies are injected
    from face_login.cv.embedder import FaceEmbedding
    from face_login.database.repository import Repository
    from face_login.services.matcher import Matcher


@dataclass(frozen=True, slots=True)
class LoginResult:
    """An immutable authentication outcome.

    Attributes:
        success: Whether the probe matched an enrolled user.
        user_id: Matched user's id, or ``None`` when not matched.
        user_name: Matched user's name, or ``None`` when not matched.
        similarity: Best similarity score (0.0 for an empty gallery).
        best_yaw: Yaw of the matched embedding, or ``None`` when not matched.
        candidate_count: Number of gallery embeddings compared against.
    """

    success: bool
    user_id: Optional[int]
    user_name: Optional[str]
    similarity: float
    best_yaw: Optional[float]
    candidate_count: int


class LoginService:
    """Authenticate a probe embedding against the stored gallery.

    Dependencies are injected: a :class:`Repository` (the only DB access) and a
    :class:`Matcher` (all similarity logic). The service holds no state.
    """

    def __init__(self, repository: "Repository", matcher: "Matcher") -> None:
        """Store the injected repository and matcher collaborators."""
        self._repository = repository
        self._matcher = matcher

    def login(self, probe: "FaceEmbedding") -> LoginResult:
        """Match ``probe`` against the gallery, log the attempt, and decide.

        Args:
            probe: The L2-normalized query embedding.

        Returns:
            A :class:`LoginResult` describing the authentication outcome.
        """
        gallery = self._repository.get_all_embeddings()
        result = self._matcher.match(probe, gallery)
        candidate_count = len(result.candidates)

        if not result.matched:
            self._repository.log_session(None, result.similarity, False)
            return LoginResult(
                success=False,
                user_id=None,
                user_name=None,
                similarity=result.similarity,
                best_yaw=None,
                candidate_count=candidate_count,
            )

        user = self._repository.get_user(result.best_user_id)
        self._repository.log_session(result.best_user_id, result.similarity, True)
        return LoginResult(
            success=True,
            user_id=result.best_user_id,
            user_name=user.name if user is not None else None,
            similarity=result.similarity,
            best_yaw=result.best_yaw,
            candidate_count=candidate_count,
        )
