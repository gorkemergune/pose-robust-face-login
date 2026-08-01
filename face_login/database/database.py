"""SQLite database layer: connection, schema init, transactions, and helpers.

Single responsibility: own the SQLite connection and provide a small, typed,
thread-safe API for the rest of the application. It opens/closes the
connection, enforces foreign keys, WAL journaling, and ``NORMAL`` synchronous
mode, creates the schema idempotently, manages transactions, and offers
parameterized execute/query helpers plus float32 embedding (de)serialization.

No ORM, no repository logic, no matching, no business logic. Embeddings are
stored as raw float32 ``BLOB`` bytes — never Pickle.
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np

from face_login.database.schema import (
    INSERT_METADATA,
    METADATA_DEFAULTS,
    SCHEMA_STATEMENTS,
)
from face_login.logging_setup import get_logger

_LOGGER = get_logger(__name__)

# Expected embedding length (buffalo_l ArcFace is 512-D). Enforced on storage so
# a wrong-sized vector (384/768/1024/...) is rejected at the database boundary.
EMBEDDING_DIMENSION = 512


class DatabaseError(RuntimeError):
    """Raised for database connection or execution failures."""


class Database:
    """Thread-safe SQLite connection manager with a typed helper API.

    A single connection (``check_same_thread=False``) is guarded by a re-entrant
    lock, so the instance may be shared across threads. The connection runs in
    autocommit mode; use :meth:`transaction` for explicit atomic blocks.
    """

    def __init__(self, path: str | Path, *, initialize: bool = True) -> None:
        """Open the database and, by default, create the schema.

        Args:
            path: Filesystem path to the SQLite database file.
            initialize: When ``True`` (default), create the schema idempotently.
        """
        self._path = str(path)
        self._lock = threading.RLock()
        self._connection: Optional[sqlite3.Connection] = None
        self.open()
        if initialize:
            self.initialize()

    # -- lifecycle ---------------------------------------------------------

    @property
    def path(self) -> str:
        """Filesystem path of the database file."""
        return self._path

    @property
    def is_open(self) -> bool:
        """Whether the underlying connection is currently open."""
        return self._connection is not None

    def open(self) -> "Database":
        """Open and configure the connection (idempotent)."""
        with self._lock:
            if self._connection is not None:
                return self
            try:
                connection = sqlite3.connect(
                    self._path, check_same_thread=False, isolation_level=None
                )
                connection.row_factory = sqlite3.Row
                self._connection = connection
                self._configure()
            except sqlite3.Error as exc:
                raise DatabaseError(
                    f"Failed to open database at {self._path}: {exc}"
                ) from exc
            return self

    def close(self) -> None:
        """Close the connection. Idempotent and safe to call twice."""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                finally:
                    self._connection = None

    def initialize(self) -> None:
        """Create the schema idempotently within a single transaction."""
        with self.transaction() as connection:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)
            for key, value in METADATA_DEFAULTS:
                connection.execute(INSERT_METADATA, (key, value))
        _LOGGER.info("Database schema initialized at %s.", self._path)

    # -- query helpers -----------------------------------------------------

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Execute a single parameterized statement and return its cursor."""
        with self._lock:
            connection = self._require_connection()
            try:
                return connection.execute(sql, tuple(params))
            except sqlite3.Error as exc:
                raise DatabaseError(f"Execute failed: {exc}") from exc

    def executemany(
        self, sql: str, seq_of_params: Sequence[Sequence[Any]]
    ) -> sqlite3.Cursor:
        """Execute a parameterized statement over many parameter rows."""
        with self._lock:
            connection = self._require_connection()
            try:
                return connection.executemany(
                    sql, [tuple(params) for params in seq_of_params]
                )
            except sqlite3.Error as exc:
                raise DatabaseError(f"Executemany failed: {exc}") from exc

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Run a parameterized query and return all rows."""
        with self._lock:
            connection = self._require_connection()
            try:
                return connection.execute(sql, tuple(params)).fetchall()
            except sqlite3.Error as exc:
                raise DatabaseError(f"Query failed: {exc}") from exc

    def query_one(
        self, sql: str, params: Sequence[Any] = ()
    ) -> Optional[sqlite3.Row]:
        """Run a parameterized query and return the first row, or ``None``."""
        with self._lock:
            connection = self._require_connection()
            try:
                return connection.execute(sql, tuple(params)).fetchone()
            except sqlite3.Error as exc:
                raise DatabaseError(f"Query failed: {exc}") from exc

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block atomically, committing on success and rolling back on error."""
        with self._lock:
            connection = self._require_connection()
            try:
                connection.execute("BEGIN")
                yield connection
                connection.execute("COMMIT")
            except Exception as exc:
                connection.execute("ROLLBACK")
                if isinstance(exc, sqlite3.Error):
                    raise DatabaseError(f"Transaction failed: {exc}") from exc
                raise

    # -- embedding (de)serialization --------------------------------------

    @staticmethod
    def to_blob(vector: np.ndarray) -> bytes:
        """Serialize a validated float32 vector to raw bytes for BLOB storage.

        Raises:
            DatabaseError: If the vector is not a 1-D float32 array of exactly
                :data:`EMBEDDING_DIMENSION` elements.
        """
        if vector.dtype != np.float32:
            raise DatabaseError(
                f"Embedding must be float32, got {vector.dtype}."
            )
        if vector.ndim != 1:
            raise DatabaseError(
                f"Embedding must be 1-D, got {vector.ndim}-D."
            )
        if vector.shape[0] != EMBEDDING_DIMENSION:
            raise DatabaseError(
                f"Embedding must have {EMBEDDING_DIMENSION} elements, "
                f"got {vector.shape[0]}."
            )
        return np.ascontiguousarray(vector).tobytes()

    @staticmethod
    def from_blob(blob: bytes) -> np.ndarray:
        """Deserialize a float32 BLOB back into a writable array.

        Raises:
            DatabaseError: If the stored bytes do not decode to exactly
                :data:`EMBEDDING_DIMENSION` float32 values.
        """
        vector = np.frombuffer(blob, dtype=np.float32).copy()
        if vector.shape[0] != EMBEDDING_DIMENSION:
            raise DatabaseError(
                f"Stored embedding has {vector.shape[0]} elements, "
                f"expected {EMBEDDING_DIMENSION}."
            )
        return vector

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "Database":
        """Return the open database for use in a ``with`` block."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close the connection when leaving a ``with`` block."""
        self.close()

    # -- internals ---------------------------------------------------------

    def _configure(self) -> None:
        """Apply per-connection PRAGMAs (foreign keys, WAL, synchronous)."""
        assert self._connection is not None  # set by open() before this call
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")

    def _require_connection(self) -> sqlite3.Connection:
        """Return the live connection or raise if the database is closed."""
        if self._connection is None:
            raise DatabaseError("Database connection is not open.")
        return self._connection
