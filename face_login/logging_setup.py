"""Centralized logging configuration.

Provides a single entry point to configure the standard-library root logger so
every module shares one format, level, and set of handlers. Modules obtain
loggers via :func:`get_logger` and never configure logging themselves.
"""
from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(
    level: str = "INFO",
    log_format: str | None = None,
    log_file: str | None = None,
) -> None:
    """Configure the root logger with a shared format and optional file sink.

    Call once at application start-up. When ``log_file`` is provided its parent
    directory is created and a file handler is added alongside the console.
    ``force=True`` makes repeated calls safely reconfigure logging.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format or DEFAULT_FORMAT,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger sharing the central configuration."""
    return logging.getLogger(name)
