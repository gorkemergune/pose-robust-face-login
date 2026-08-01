"""SQLite schema definitions for the face-login database.

Single responsibility: declare the DDL for the ``users``, ``embeddings``, and
``sessions`` tables plus their supporting indexes. Every statement is
idempotent (``IF NOT EXISTS``) and is executed by the database layer; no
connection, query, or business logic lives here. Embeddings are stored as
float32 ``BLOB`` values — never Pickle.
"""
from __future__ import annotations

CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    UNIQUE NOT NULL,
    created_at  REAL    NOT NULL
)
"""

CREATE_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS embeddings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    yaw_center    REAL    NOT NULL,
    quality_score REAL    NOT NULL,
    embedding     BLOB    NOT NULL,
    created_at    REAL    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
)
"""

CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER,
    similarity  REAL,
    success     INTEGER NOT NULL,
    created_at  REAL    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
)
"""

CREATE_METADATA = """
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""

CREATE_EMBEDDINGS_USER_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_embeddings_user_id "
    "ON embeddings (user_id)"
)

CREATE_SESSIONS_USER_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_sessions_user_id "
    "ON sessions (user_id)"
)

# Ordered so referenced tables exist before their foreign keys/indexes.
SCHEMA_STATEMENTS: tuple[str, ...] = (
    CREATE_USERS,
    CREATE_EMBEDDINGS,
    CREATE_SESSIONS,
    CREATE_METADATA,
    CREATE_EMBEDDINGS_USER_INDEX,
    CREATE_SESSIONS_USER_INDEX,
)

# Idempotent seed for the metadata table (INSERT OR IGNORE never overwrites an
# existing key, so values survive future migrations that update them).
INSERT_METADATA = "INSERT OR IGNORE INTO metadata (key, value) VALUES (?, ?)"

METADATA_DEFAULTS: tuple[tuple[str, str], ...] = (
    ("schema_version", "1"),
    ("embedding_model", "buffalo_l"),
    ("embedding_dimension", "512"),
)
