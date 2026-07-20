"""Centralized logging configuration for AutoLabel Dock."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path


LOG_DIR_ENV = "AUTOLABEL_LOG_DIR"
LOG_FILE_NAME = "autolabel.log"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 3

_configured = False


def default_log_dir() -> Path:
    """Return the default folder used for application log files."""
    override = os.environ.get(LOG_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()

    return Path("logs")


def log_file_path(log_dir: Path | None = None) -> Path:
    """Return the main application log file path."""
    return (log_dir or default_log_dir()) / LOG_FILE_NAME


def setup_logging(log_dir: Path | None = None) -> Path:
    """Configure application-wide logging and return the active log file path.

    The app writes detailed DEBUG logs to ``autolabel.log`` in a log folder.
    By default this folder is ``logs`` under the app start directory. Set ``AUTOLABEL_LOG_DIR``
    to redirect logs to another folder without changing the UI.
    """
    global _configured

    active_log_dir = log_dir or default_log_dir()
    active_log_file = log_file_path(active_log_dir)

    if _configured:
        return active_log_file

    active_log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "[%(asctime)s] [%(levelname)-5s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)


    for handler in root.handlers[:]:
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


    if sys.stderr is not None:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        root.addHandler(console)


    file_handler = logging.handlers.RotatingFileHandler(
        filename=active_log_file,
        mode="a",
        maxBytes=DEFAULT_MAX_BYTES,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
        errors="replace",
        delay=False,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.captureWarnings(True)


    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    _configured = True

    logging.getLogger(__name__).info(
        "Logging initialized, log file: %s",
        active_log_file,
    )
    return active_log_file
