"""The open project in the GUI: the core Project, its file path and the unsaved-changes flag."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from inferential_studio.core.persistence import load_project, save_project
from inferential_studio.core.schemas import Project


class ProjectSession(QObject):
    changed = Signal()  # project content replaced or edited
    dirty_changed = Signal(bool)
    path_changed = Signal(object)  # Path | None

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._project = Project()
        self._path: Path | None = None
        self._dirty = False

    @property
    def project(self) -> Project:
        return self._project

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def display_name(self) -> str:
        return self._path.name if self._path else self._project.name

    def new(self, project: Project | None = None) -> None:
        self._replace(project or Project(), None)

    def open(self, path: str | Path) -> None:
        """Load synchronously (project files hold configuration only). Raises ProjectFileError."""
        project = load_project(path)
        self._replace(project, Path(path))

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ValueError("No file name given for the project.")
        self._project = save_project(self._project, target)
        self._path = target
        self.path_changed.emit(target)
        self._set_dirty(False)
        return target

    def mark_dirty(self) -> None:
        self._set_dirty(True)
        self.changed.emit()

    def _replace(self, project: Project, path: Path | None) -> None:
        self._project = project
        self._path = path
        self.path_changed.emit(path)
        self.changed.emit()
        self._set_dirty(False)

    def _set_dirty(self, dirty: bool) -> None:
        if dirty != self._dirty:
            self._dirty = dirty
            self.dirty_changed.emit(dirty)
