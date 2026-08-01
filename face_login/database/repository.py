"""Repository layer: the only gateway to persistent storage.

Single responsibility: read and write users, embeddings, and sessions so the
rest of the application never issues SQL directly. Implementation deferred to
Phase 6.
"""
