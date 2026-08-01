"""Application composition root.

Bootstraps configuration and logging and, in later phases, will wire the
camera, recognition pipeline, and UI together. At the skeleton stage it only
proves the package is executable end-to-end; no pipeline logic runs here.
"""
from __future__ import annotations

from face_login.config import load_config
from face_login.logging_setup import configure_logging, get_logger


def main() -> int:
    """Bootstrap the application skeleton and return a process exit code."""
    config = load_config()
    configure_logging(
        level=config.logging.level,
        log_format=config.logging.format,
        log_file=config.logging.file,
    )
    logger = get_logger(__name__)
    logger.info("%s skeleton initialized.", config.app.name)
    logger.info("Recognition pipeline components are not implemented yet.")
    return 0
