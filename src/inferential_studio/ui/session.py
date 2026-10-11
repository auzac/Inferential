"""The open project in the GUI: configuration, loaded data, file path and unsaved-changes flag."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from inferential_studio.core.dataset import ProjectData, SourceData
from inferential_studio.core.persistence import load_project, rebase_source_paths, save_project
from inferential_studio.core.schemas import DataSource, Project, SourceKind, VariableMeta


class ProjectSession(QObject):
    changed = Signal()  # project configuration replaced or edited
    data_changed = Signal()  # imported data added, removed or reloaded
    dirty_changed = Signal(bool)
    path_changed = Signal(object)  # Path | None

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._project = Project()
        self._data = ProjectData()
        self._path: Path | None = None
        self._dirty = False
        self.data_version = 0  # increments whenever data changes; used to invalidate caches

    # ------------------------------------------------------------------ state

    @property
    def project(self) -> Project:
        return self._project

    @property
    def data(self) -> ProjectData:
        return self._data

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def display_name(self) -> str:
        return self._path.name if self._path else self._project.name

    # ------------------------------------------------------------------ project lifecycle

    def new(self, project: Project | None = None) -> None:
        self._replace(project or Project(), None)

    def open(self, path: str | Path) -> None:
        """Load the configuration synchronously (small). Data is loaded separately in the background."""
        project = load_project(path)
        self._replace(project, Path(path))

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ValueError("No file name given for the project.")
        project = rebase_source_paths(self._project, self._path, target)
        self._project = save_project(project, target)
        for src in self._project.sources:  # keep loaded data pointing at the rebased source spec
            loaded = self._data.sources.get(src.source_id)
            if loaded is not None:
                loaded.source = replace(loaded.source, path=src.path)
        self._path = target
        self.path_changed.emit(target)
        self._set_dirty(False)
        return target

    def mark_dirty(self) -> None:
        self._set_dirty(True)
        self.changed.emit()

    # ------------------------------------------------------------------ data

    def set_data(self, data: ProjectData) -> None:
        """Install data loaded for the current project (checksums recorded at import kept as is)."""
        self._data = data
        self._bump_data()

    def add_source(self, source_data: SourceData, variables: list[VariableMeta]) -> None:
        """Add a newly imported file and its variables to the project."""
        source = source_data.source  # carries the checksum computed during import
        project = self._project
        project.sources.append(source)
        project.variables.extend(variables)
        try:
            project.validate()
        except Exception:
            project.sources.remove(source)
            for v in variables:
                project.variables.remove(v)
            raise
        self._data.sources[source.source_id] = source_data
        self._data.failures.pop(source.source_id, None)
        self.mark_dirty()
        self._bump_data()

    def accept_changed_source(self, source_data: SourceData) -> None:
        """Replace a source whose file changed, recording the new checksum."""
        sid = source_data.source_id
        self._project.sources = [
            replace(s, sha256=source_data.sha256) if s.source_id == sid else s
            for s in self._project.sources
        ]
        self._data.sources[sid] = source_data
        self._data.failures.pop(sid, None)
        self.mark_dirty()
        self._bump_data()

    def remove_source(self, source_id: str) -> None:
        project = self._project
        removed = {v.name for v in project.variables if v.source_id == source_id}
        project.sources = [s for s in project.sources if s.source_id != source_id]
        project.variables = [v for v in project.variables if v.source_id != source_id]
        project.features = [f for f in project.features if f.variable not in removed]
        project.exclusions = [
            ex for ex in project.exclusions if not ex.signals or not set(ex.signals) & removed
        ]
        self._data.sources.pop(source_id, None)
        self._data.failures.pop(source_id, None)
        self.mark_dirty()
        self._bump_data()

    def new_source_id(self, kind: SourceKind) -> str:
        existing = {s.source_id for s in self._project.sources}
        base = kind.value
        if base not in existing:
            return base
        i = 2
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def source_for(self, source_id: str) -> DataSource | None:
        for s in self._project.sources:
            if s.source_id == source_id:
                return s
        return None

    # ------------------------------------------------------------------ internals

    def _replace(self, project: Project, path: Path | None) -> None:
        self._project = project
        self._path = path
        self._data = ProjectData()
        self.path_changed.emit(path)
        self.changed.emit()
        self._bump_data()
        self._set_dirty(False)

    def _bump_data(self) -> None:
        self.data_version += 1
        self.data_changed.emit()

    def _set_dirty(self, dirty: bool) -> None:
        if dirty != self._dirty:
            self._dirty = dirty
            self.dirty_changed.emit(dirty)
