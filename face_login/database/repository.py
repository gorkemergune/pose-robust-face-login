"""Repository layer: the only gateway to persistent storage.

Single responsibility: map between SQLite rows and immutable Python dataclasses
for users, embeddings, and sessions. This is the ONLY component that talks to
:class:`~face_login.database.database.Database`; the rest of the application
depends on these typed methods and never issues SQL.

It contains no matching, cosine similarity, quality, coverage, or UI logic, and
delegates all embedding (de)serialization to ``Database.to_blob``/``from_blob``.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from face_login.database.database import Database


@dataclass(frozen=True, slots=True)
class User:
    """An immutable registered user."""

    id: int
    name: str
    created_at: float


@dataclass(frozen=True, slots=True)
class EmbeddingRecord:
    """An immutable stored embedding with its pose/quality metadata."""

    id: int
    user_id: int
    yaw_center: float
    quality_score: float
    embedding: np.ndarray
    created_at: float


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """An immutable authentication-attempt log entry."""

    id: int
    user_id: Optional[int]
    similarity: Optional[float]
    success: bool
    created_at: float


class Repository:
    """Typed data-access layer over :class:`Database`.

    Write operations run inside ``Database.transaction()`` for atomicity; reads
    use the plain query helpers. All SQL is parameterized.
    """

    def __init__(self, database: Database) -> None:
        """Store the database gateway this repository operates through."""
        self._db = database

    # -- users -------------------------------------------------------------

    def create_user(self, name: str) -> User:
        """Insert a new user and return it (raises on duplicate name)."""
        created_at = time.time()
        with self._db.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO users (name, created_at) VALUES (?, ?)",
                (name, created_at),
            )
            user_id = int(cursor.lastrowid)
        return User(id=user_id, name=name, created_at=created_at)

    def get_user(self, user_id: int) -> Optional[User]:
        """Return the user with ``user_id``, or ``None`` if absent."""
        row = self._db.query_one(
            "SELECT id, name, created_at FROM users WHERE id = ?", (user_id,)
        )
        return self._to_user(row) if row is not None else None

    def get_user_by_name(self, name: str) -> Optional[User]:
        """Return the user named ``name``, or ``None`` if absent."""
        row = self._db.query_one(
            "SELECT id, name, created_at FROM users WHERE name = ?", (name,)
        )
        return self._to_user(row) if row is not None else None

    def list_users(self) -> list[User]:
        """Return all users, ordered by id."""
        rows = self._db.query(
            "SELECT id, name, created_at FROM users ORDER BY id"
        )
        return [self._to_user(row) for row in rows]

    def delete_user(self, user_id: int) -> bool:
        """Delete a user (cascading its embeddings); return whether one was removed."""
        with self._db.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM users WHERE id = ?", (user_id,)
            )
            return cursor.rowcount > 0

    # -- embeddings --------------------------------------------------------

    def add_embedding(
        self,
        user_id: int,
        yaw_center: float,
        quality_score: float,
        embedding: np.ndarray,
    ) -> EmbeddingRecord:
        """Store an embedding for a user and return the persisted record."""
        blob = self._db.to_blob(embedding)  # validates + serializes float32
        created_at = time.time()
        with self._db.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO embeddings "
                "(user_id, yaw_center, quality_score, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, yaw_center, quality_score, blob, created_at),
            )
            embedding_id = int(cursor.lastrowid)
        return EmbeddingRecord(
            id=embedding_id,
            user_id=user_id,
            yaw_center=yaw_center,
            quality_score=quality_score,
            embedding=self._db.from_blob(blob),
            created_at=created_at,
        )

    def get_embeddings(self, user_id: int) -> list[EmbeddingRecord]:
        """Return all embeddings for a user, ordered by id."""
        rows = self._db.query(
            "SELECT id, user_id, yaw_center, quality_score, embedding, created_at "
            "FROM embeddings WHERE user_id = ? ORDER BY id",
            (user_id,),
        )
        return [self._to_embedding(row) for row in rows]

    def get_all_embeddings(self) -> list[EmbeddingRecord]:
        """Return every stored embedding, ordered by user then id."""
        rows = self._db.query(
            "SELECT id, user_id, yaw_center, quality_score, embedding, created_at "
            "FROM embeddings ORDER BY user_id, id"
        )
        return [self._to_embedding(row) for row in rows]

    # -- sessions ----------------------------------------------------------

    def log_session(
        self,
        user_id: Optional[int],
        similarity: Optional[float],
        success: bool,
    ) -> SessionRecord:
        """Record an authentication attempt and return the log entry."""
        created_at = time.time()
        with self._db.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO sessions (user_id, similarity, success, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, similarity, int(success), created_at),
            )
            session_id = int(cursor.lastrowid)
        return SessionRecord(
            id=session_id,
            user_id=user_id,
            similarity=similarity,
            success=success,
            created_at=created_at,
        )

    def recent_sessions(self, limit: int) -> list[SessionRecord]:
        """Return the most recent sessions, newest first, up to ``limit``."""
        rows = self._db.query(
            "SELECT id, user_id, similarity, success, created_at "
            "FROM sessions ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [self._to_session(row) for row in rows]

    # -- row mapping -------------------------------------------------------

    @staticmethod
    def _to_user(row: sqlite3.Row) -> User:
        """Map a users row to a :class:`User`."""
        return User(
            id=row["id"], name=row["name"], created_at=row["created_at"]
        )

    def _to_embedding(self, row: sqlite3.Row) -> EmbeddingRecord:
        """Map an embeddings row to an :class:`EmbeddingRecord`."""
        return EmbeddingRecord(
            id=row["id"],
            user_id=row["user_id"],
            yaw_center=row["yaw_center"],
            quality_score=row["quality_score"],
            embedding=self._db.from_blob(row["embedding"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_session(row: sqlite3.Row) -> SessionRecord:
        """Map a sessions row to a :class:`SessionRecord`."""
        return SessionRecord(
            id=row["id"],
            user_id=row["user_id"],
            similarity=row["similarity"],
            success=bool(row["success"]),
            created_at=row["created_at"],
        )
