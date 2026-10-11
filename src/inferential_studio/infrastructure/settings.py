"""User settings stored as a small JSON file in the per-user config folder.

JSON rather than the Windows registry keeps settings transparent, easy to reset (delete the
file) and testable without Qt. A corrupt file falls back to defaults and is kept aside.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from inferential_studio.core.schemas import SchemaError, validate_timezone
from inferential_studio.infrastructure.paths import config_dir

log = logging.getLogger(__name__)

SETTINGS_FILE = "settings.json"
MAX_RECENT = 10
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


@dataclass
class AppSettings:
    display_timezone: str = "UTC"
    log_level: str = "INFO"
    recent_projects: list[str] = field(default_factory=list)
    last_directory: str = ""
    author_name: str = ""  # recorded on exclusions and model cards
    window_geometry: str = ""  # Qt saveGeometry() as base64; empty = size to the screen
    help_panel: bool | None = None  # None = automatic (shown on wide screens only)

    def add_recent(self, path: str | os.PathLike[str]) -> None:
        text = str(Path(path))
        self.recent_projects = [text] + [p for p in self.recent_projects if p != text]
        del self.recent_projects[MAX_RECENT:]

    def validated(self) -> AppSettings:
        """Return a copy with any invalid values replaced by defaults."""
        default = AppSettings()
        tz = self.display_timezone
        try:
            validate_timezone(tz)
        except SchemaError:
            tz = default.display_timezone
        level = self.log_level.upper() if isinstance(self.log_level, str) else ""
        return AppSettings(
            display_timezone=tz,
            log_level=level if level in LOG_LEVELS else default.log_level,
            recent_projects=[p for p in self.recent_projects if isinstance(p, str)][:MAX_RECENT],
            last_directory=self.last_directory if isinstance(self.last_directory, str) else "",
            author_name=self.author_name if isinstance(self.author_name, str) else "",
            window_geometry=self.window_geometry if isinstance(self.window_geometry, str) else "",
            help_panel=self.help_panel if isinstance(self.help_panel, bool) else None,
        )


def settings_path() -> Path:
    return config_dir() / SETTINGS_FILE


def load_settings(path: Path | None = None) -> AppSettings:
    path = path or settings_path()
    if not path.exists():
        return AppSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("settings root is not an object")
        known = {f.name for f in fields(AppSettings)}
        return AppSettings(**{k: v for k, v in data.items() if k in known}).validated()
    except (OSError, ValueError, TypeError) as exc:
        backup = path.with_suffix(".corrupt.json")
        log.warning(
            "Settings file unreadable (%s); using defaults and keeping a copy at %s", exc, backup
        )
        with contextlib.suppress(OSError):
            os.replace(path, backup)
        return AppSettings()


def save_settings(settings: AppSettings, path: Path | None = None) -> None:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".settings-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(asdict(settings.validated()), handle, indent=2)
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        log.exception("Could not save settings to %s", path)
