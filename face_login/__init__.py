"""Pose-robust face registration & login system (desktop application).

Top-level package. The codebase follows Clean Architecture: the perception
adapters (:mod:`face_login.cv`) and storage (:mod:`face_login.database`)
sit beneath the service layer (:mod:`face_login.services`), while the UI
(:mod:`face_login.ui`) stays decoupled from the pipeline. See ``CLAUDE.md``
and ``ROADMAP.md`` for the full architecture.
"""

__version__ = "0.1.0"
