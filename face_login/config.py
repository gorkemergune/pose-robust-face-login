"""Centralized configuration loading.

Parses ``configs/config.yaml`` into strongly-typed, immutable dataclasses so the rest
of the application depends on typed settings rather than raw dictionaries.
Every section is optional: missing keys fall back to the documented defaults,
which mirror the values in ``config.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


@dataclass(frozen=True)
class AppInfo:
    """Human-facing application metadata."""

    name: str = "Pose-Robust Face Login"


@dataclass(frozen=True)
class LoggingConfig:
    """Logging level, format, and optional file sink."""

    level: str = "INFO"
    file: str | None = None
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


@dataclass(frozen=True)
class CameraConfig:
    """Capture device selection and target resolution/frame-rate."""

    index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30


@dataclass(frozen=True)
class DetectionConfig:
    """Face-detection model and thresholds."""

    model_name: str = "buffalo_l"
    det_size: int = 640
    det_threshold: float = 0.5


@dataclass(frozen=True)
class AlignmentConfig:
    """Aligned-crop geometry (square, ``image_size`` pixels per side)."""

    image_size: int = 112


@dataclass(frozen=True)
class PoseConfig:
    """Accepted head-yaw range in degrees."""

    yaw_min: float = -90.0
    yaw_max: float = 90.0


@dataclass(frozen=True)
class QualityConfig:
    """Quality-gate thresholds for frame acceptance."""

    blur_threshold: float = 100.0
    brightness_min: float = 40.0
    brightness_max: float = 220.0
    min_face_size: int = 112
    min_confidence: float = 0.6
    stability_frames: int = 3


@dataclass(frozen=True)
class CoverageConfig:
    """Yaw-bin configuration for registration pose coverage."""

    yaw_bins: int = 18
    yaw_min: float = -90.0
    yaw_max: float = 90.0


@dataclass(frozen=True)
class RecognitionConfig:
    """Embedding dimensionality and cosine-similarity decision threshold."""

    embedding_dim: int = 512
    threshold: float = 0.44


@dataclass(frozen=True)
class DatabaseConfig:
    """Filesystem location of the SQLite database."""

    path: str = "data/face_login.db"


@dataclass(frozen=True)
class AppConfig:
    """Aggregate, immutable application configuration."""

    app: AppInfo = field(default_factory=AppInfo)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Return the mapping for ``key`` from ``raw`` (empty when absent)."""
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{key}' must be a mapping.")
    return value


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load ``configs/config.yaml`` into a typed :class:`AppConfig`.

    A missing file or missing sections yield the in-code defaults. PyYAML is
    imported lazily so that importing this module has no third-party
    dependency; it is required only when an existing file is actually parsed.
    """
    config_path = Path(path)
    raw: dict[str, Any] = {}
    if config_path.exists():
        import yaml  # lazy import: only needed to parse an existing file

        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    return AppConfig(
        app=AppInfo(**_section(raw, "app")),
        logging=LoggingConfig(**_section(raw, "logging")),
        camera=CameraConfig(**_section(raw, "camera")),
        detection=DetectionConfig(**_section(raw, "detection")),
        alignment=AlignmentConfig(**_section(raw, "alignment")),
        pose=PoseConfig(**_section(raw, "pose")),
        quality=QualityConfig(**_section(raw, "quality")),
        coverage=CoverageConfig(**_section(raw, "coverage")),
        recognition=RecognitionConfig(**_section(raw, "recognition")),
        database=DatabaseConfig(**_section(raw, "database")),
    )
