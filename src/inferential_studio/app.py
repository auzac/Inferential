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
    from PySide6.QtGui import QIcon

    from inferential_studio.infrastructure.paths import resource_path

    icon = resource_path("assets/icons/app.png")
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))
    window = MainWindow(settings)
    window.show()

    if args.project:
        window.open_project_file(args.project)

    if not args.smoke_test:
        QTimer.singleShot(0, window.offer_recovery)  # after the window is on screen
    if args.smoke_test:
        _numeric_self_check()
        log.info("Smoke test: main window created; exiting")
        QTimer.singleShot(0, app.quit)

    code = app.exec()
    log.info("%s exiting with code %s", APP_NAME, code)
    return int(code)


def _numeric_self_check() -> None:
    """Align and train on a small synthetic dataset (proves the bundled numerics load and run)."""
    import numpy as np
    import pandas as pd

    from inferential_studio.core.features import TargetSamples, build_feature_matrix
    from inferential_studio.core.models import train_models
    from inferential_studio.core.schemas import FeatureSpec, ModellingConfig
    from inferential_studio.core.synthetic import SyntheticConfig, generate_distillation_dataset

    ds = generate_distillation_dataset(SyntheticConfig(days=6))
    lab = ds.lab
    n = len(lab)
    target = TargetSamples(
        "y", pd.DatetimeIndex(lab["Sample_Timestamp"]), lab["Product_Composition"].to_numpy(),
        np.zeros(n, bool), np.zeros(n, bool),
    )  # fmt: skip
    specs = [FeatureSpec(c, deadtime_s=1200) for c in ("Tray_15_Temp", "Column_Pressure")]
    run = train_models(build_feature_matrix(target, ds.process, specs), ModellingConfig())
    failed = [m.name for m in run.models if not m.ok] + run.errors
    if failed:
        raise RuntimeError(f"Numeric self-check failed: {failed}")
    logging.getLogger(__name__).info("Smoke test: aligned and trained %d models", len(run.models))


if __name__ == "__main__":
    raise SystemExit(main())
