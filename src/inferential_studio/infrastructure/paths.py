"""Per-user application directories. No Qt dependency so tests can redirect them easily.

Windows:  %LOCALAPPDATA%\\InferentialModelStudio\\{config,logs}
Other:    $XDG_CONFIG_HOME/InferentialModelStudio, $XDG_STATE_HOME/InferentialModelStudio/logs

Set ``INFERENTIAL_STUDIO_HOME`` to override the base folder (used by tests and portable runs).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from inferential_studio import APP_ID

ENV_OVERRIDE = "INFERENTIAL_STUDIO_HOME"


def _base_dirs() -> tuple[Path, Path]:
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        base = Path(override)
        return base / "config", base / "logs"
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return local / APP_ID / "config", local / APP_ID / "logs"
    config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    state = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return config / APP_ID, state / APP_ID / "logs"


def config_dir() -> Path:
    path = _base_dirs()[0]
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_dir() -> Path:
    path = _base_dirs()[1]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_base() -> Path:
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        return Path(override)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / APP_ID
    state = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return state / APP_ID


def recovery_dir() -> Path:
    """Automatic recovery copies of unsaved projects."""
    path = _state_base() / "recovery"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    """Data files extracted from portable projects (safe to delete when the app is closed)."""
    path = _state_base() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resource_path(relative: str) -> Path:
    """A file shipped with the application (bundle folder when frozen, else the repository)."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = Path(__file__).resolve().parents[3]
    return base / relative
