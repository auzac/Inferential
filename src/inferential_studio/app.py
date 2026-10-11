"""Application entry point.

Usage:
    python -m inferential_studio [project.imsproj]
    python -m inferential_studio --smoke-test     # start, build the window, exit 0 (for packaging checks)
"""

from __future__ import annotations

import argparse
import logging
import sys

from inferential_studio import APP_ID, APP_NAME, __version__


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="inferential-studio", description=APP_NAME)
    parser.add_argument("project", nargs="?", help="Project file to open")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Start, create the main window and exit"
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    # Qt adds its own arguments (e.g. -platform); ignore what we do not know.
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = _parse_args(argv)

    # Imports are deferred so --version works quickly and import errors get logged.
    from inferential_studio.infrastructure.logging_config import (
        configure_logging,
        install_excepthook,
    )
    from inferential_studio.infrastructure.settings import load_settings

    settings = load_settings()
    configure_logging(settings.log_level)
    install_excepthook()
    log = logging.getLogger(__name__)

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    # Exact scaling at 125%/150%/175% Windows display settings instead of rounding.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance() or QApplication([sys.argv[0], *argv])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(APP_ID)

    from inferential_studio.ui.main_window import MainWindow
    from inferential_studio.ui.theme import apply_theme

    apply_theme(app)
    window = MainWindow(settings)
    window.show()

    if args.project:
        window.open_project_file(args.project)

    if args.smoke_test:
        log.info("Smoke test: main window created; exiting")
        QTimer.singleShot(0, app.quit)

    code = app.exec()
    log.info("%s exiting with code %s", APP_NAME, code)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
