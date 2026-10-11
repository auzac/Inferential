"""Application logging: rotating file in the user log folder plus console.

Privacy rule: log events, file names, shapes and settings, never bulk process data values.
"""

from __future__ import annotations

import logging
import logging.handlers
import platform
import sys
from pathlib import Path

from inferential_studio import APP_NAME, __version__
from inferential_studio.infrastructure.paths import log_dir

LOG_FILE_NAME = "inferential_studio.log"
_FORMAT = "%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s: %(message)s"


def configure_logging(level: str | int = "INFO", *, to_console: bool = True) -> Path:
    """Configure the root logger once and return the log file path."""
    path = log_dir() / LOG_FILE_NAME
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_ims_handler", False):
            root.removeHandler(handler)
            handler.close()

    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    file_handler._ims_handler = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    if to_console and sys.stderr is not None:  # stderr is None in windowed PyInstaller builds
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_FORMAT))
        console._ims_handler = True  # type: ignore[attr-defined]
        root.addHandler(console)

    root.setLevel(level if isinstance(level, int) else level.upper())
    logging.getLogger(__name__).info(
        "%s %s starting (Python %s, %s)",
        APP_NAME,
        __version__,
        platform.python_version(),
        platform.platform(),
    )
    return path


def install_excepthook() -> None:
    """Log uncaught exceptions instead of letting them vanish in a windowed app."""
    previous = sys.excepthook

    def hook(exc_type, exc, tb):  # type: ignore[no-untyped-def]
        if not issubclass(exc_type, KeyboardInterrupt):
            logging.getLogger("inferential_studio.uncaught").critical(
                "Uncaught exception", exc_info=(exc_type, exc, tb)
            )
        previous(exc_type, exc, tb)

    sys.excepthook = hook
