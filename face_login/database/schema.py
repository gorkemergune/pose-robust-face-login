"""Database schema definitions for users, embeddings, and sessions tables.

Single responsibility: declare and create the SQLite schema. Embeddings are
stored as numeric blobs/arrays — never Pickle. Implementation deferred to
Phase 6.
"""
