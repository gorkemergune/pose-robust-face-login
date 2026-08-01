"""Application composition root.

Constructs and wires every component from configuration, then delegates the
screen-driven flow to :class:`~face_login.controller.ScreenController`. It opens
the database and creates its schema; no camera capture or loop work happens in
``__init__``. This module contains no recognition algorithms, drawing, cosine
math, or SQL — only composition and lifecycle.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from face_login.config import AppConfig, load_config
from face_login.controller import ScreenController
from face_login.cv.alignment import FaceAligner
from face_login.cv.camera import Camera, CameraError
from face_login.cv.detector import FaceDetector
from face_login.cv.embedder import EmbeddingError, FaceEmbedder
from face_login.cv.pose import SolvePnPPoseEstimator
from face_login.cv.quality import QualityGate
from face_login.database.database import Database, DatabaseError
from face_login.database.repository import Repository
from face_login.logging_setup import configure_logging, get_logger
from face_login.services.login import LoginService
from face_login.services.matcher import Matcher
from face_login.ui.coverage_bar import CoverageBarRenderer
from face_login.ui.overlay import OverlayRenderer
from face_login.ui.screens import ScreenRenderer
from face_login.ui.window import ApplicationWindow


class FaceLoginApplication:
    """Build every component and run the screen-driven application."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Construct and wire all components; start no capture work."""
        self._config = config or load_config()
        self._logger = get_logger(__name__)
        cfg = self._config

        self._prepare_model_dir(cfg.detection.model_name)
        Path(cfg.database.path).parent.mkdir(parents=True, exist_ok=True)
        self._database = Database(cfg.database.path, initialize=True)
        repository = Repository(self._database)

        self._camera = Camera.from_config(cfg.camera)
        overlay = OverlayRenderer()
        coverage_bar = CoverageBarRenderer()
        self._window = ApplicationWindow(
            window_title=cfg.app.name,
            overlay_renderer=overlay,
            coverage_renderer=coverage_bar,
        )
        self._controller = ScreenController(
            window=self._window,
            screens=ScreenRenderer(),
            overlay=overlay,
            coverage_bar=coverage_bar,
            repository=repository,
            login_service=LoginService(repository, Matcher.from_config(cfg.recognition)),
            camera=self._camera,
            detector=FaceDetector(cfg.detection),
            aligner=FaceAligner(cfg.alignment),
            pose=SolvePnPPoseEstimator(),
            embedder=FaceEmbedder(model_name=cfg.detection.model_name),
            quality=QualityGate(cfg.quality),
            coverage_config=cfg.coverage,
            logger=get_logger("face_login.controller"),
        )

    def __enter__(self) -> "FaceLoginApplication":
        """Enter a ``with`` block, returning this application."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Always release resources when leaving a ``with`` block."""
        self.close()

    @staticmethod
    def _prepare_model_dir(model_name: str) -> None:
        """Remove an empty model placeholder so InsightFace can auto-download.

        An existing but empty ``models/<name>/`` directory makes InsightFace
        skip the download and then fail (no models found). If the folder holds
        no ``.onnx`` files it is removed so the pack downloads on first use;
        a folder with real weights is left untouched.
        """
        model_dir = Path("models") / model_name
        if model_dir.exists() and not any(model_dir.glob("*.onnx")):
            shutil.rmtree(model_dir, ignore_errors=True)

    def run(self) -> None:
        """Run the screen-driven application loop until the user quits."""
        self._controller.run()

    def close(self) -> None:
        """Release resources in reverse order: camera, window, database."""
        for closer, name in ((self._camera.close, "camera"),
                             (self._window.close, "window"),
                             (self._database.close, "database")):
            try:
                closer()
            except Exception as exc:  # shutdown must never raise
                self._logger.warning("Error closing %s: %s", name, exc)


def main() -> int:
    """Entry point: configure logging, then run the application to completion."""
    config = load_config()
    configure_logging(level=config.logging.level, log_format=config.logging.format,
                      log_file=config.logging.file)
    logger = get_logger(__name__)
    try:
        with FaceLoginApplication(config) as app:
            app.run()
    except (CameraError, DatabaseError, EmbeddingError) as exc:
        logger.error("Fatal startup error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    return 0
