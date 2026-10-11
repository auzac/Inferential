"""Crash recovery: periodic copies of unsaved projects, offered again at the next start.

A recovery copy is an ordinary project file plus a small JSON note with the original file name.
Source paths are rewritten so they still resolve from the recovery folder. Copies are deleted
after a successful save or a clean exit.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from inferential_studio.core.persistence import (
    PROJECT_EXTENSION,
    ProjectFileError,
    rebase_source_paths,
    save_project,
)
from inferential_studio.core.schemas import Project, utc_now
from inferential_studio.infrastructure.paths import recovery_dir

log = logging.getLogger(__name__)


@dataclass
class RecoveryCopy:
    project_file: Path
    note_file: Path
    original_path: Path | None
    name: str
    saved_at: datetime


class RecoveryWriter:
    """Writes this application instance's recovery copy (one per running window)."""

    def __init__(self) -> None:
        self.key = uuid.uuid4().hex[:12]

    @property
    def project_file(self) -> Path:
        return recovery_dir() / f"{self.key}{PROJECT_EXTENSION}"

    @property
    def note_file(self) -> Path:
        return recovery_dir() / f"{self.key}.json"

    def write(self, project: Project, original_path: Path | None) -> bool:
        try:
            copy = rebase_source_paths(project, original_path, self.project_file)
            save_project(copy, self.project_file)
            note = {
                "original_path": str(original_path) if original_path else None,
                "name": project.name,
                "saved_at": utc_now().isoformat(),
            }
            self.note_file.write_text(json.dumps(note), encoding="utf-8")
            return True
        except (ProjectFileError, OSError) as exc:
            log.warning("Could not write recovery copy: %s", exc)
            return False

    def discard(self) -> None:
        for path in (self.project_file, self.note_file):
            path.unlink(missing_ok=True)


def find_recovery_copies() -> list[RecoveryCopy]:
    out = []
    for note in sorted(recovery_dir().glob("*.json")):
        project_file = note.with_suffix(PROJECT_EXTENSION)
        try:
            data = json.loads(note.read_text(encoding="utf-8"))
            saved = datetime.fromisoformat(data["saved_at"])
        except (OSError, ValueError, KeyError, TypeError):
            note.unlink(missing_ok=True)
            continue
        if not project_file.is_file():
            note.unlink(missing_ok=True)
            continue
        original = Path(data["original_path"]) if data.get("original_path") else None
        out.append(RecoveryCopy(project_file, note, original, data.get("name", ""), saved))
    return out


def discard_copy(copy: RecoveryCopy) -> None:
    for path in (copy.project_file, copy.note_file):
        path.unlink(missing_ok=True)
