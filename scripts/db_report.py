"""Ad-hoc database inspection for demos and verification.

Prints the enrolled users, how many embeddings each has (with their pose bins),
and the most recent authentication sessions. Handy for screenshots that prove
registration created data and that overwriting/deleting removed it.

Run from the project root:

    python scripts/db_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

from face_login.config import load_config
from face_login.database.database import Database
from face_login.database.repository import Repository


def main() -> int:
    """Open the configured database and print a human-readable summary."""
    config = load_config()
    with Database(config.database.path) as database:
        repository = Repository(database)
        users = repository.list_users()
        print(f"Database : {config.database.path}")
        print(f"Users    : {len(users)}")
        for user in users:
            embeddings = repository.get_embeddings(user.id)
            yaws = sorted(round(record.yaw_center, 1) for record in embeddings)
            print(
                f"  [{user.id}] {user.name}: "
                f"{len(embeddings)} embeddings; pose bins {yaws}"
            )
        sessions = repository.recent_sessions(10)
        print(f"Sessions : {len(sessions)} most recent")
        for session in sessions:
            similarity = (
                f"{session.similarity:.3f}" if session.similarity is not None else "-"
            )
            print(
                f"  #{session.id} user={session.user_id} "
                f"success={session.success} similarity={similarity}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
