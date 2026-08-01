"""Package entry point enabling ``python -m face_login``.

Delegates to the application composition root in :mod:`face_login.app`. No
business logic lives here.
"""
from __future__ import annotations

from face_login.app import main

if __name__ == "__main__":
    raise SystemExit(main())
