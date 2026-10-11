"""Undoable edits. Edits replace whole project lists or configurations, so undo/redo is exact."""

from __future__ import annotations

from PySide6.QtGui import QUndoCommand, QUndoStack

from inferential_studio.ui.session import ProjectSession


class ReplaceListsCommand(QUndoCommand):
    def __init__(
        self, session: ProjectSession, text: str, before: dict[str, list], after: dict[str, list]
    ) -> None:
        super().__init__(text)
        self._session = session
        self._before = before
        self._after = after

    def redo(self) -> None:
        self._session.set_lists(**self._after)

    def undo(self) -> None:
        self._session.set_lists(**self._before)


def _copy(value: object) -> object:
    return list(value) if isinstance(value, list) else value


def edit_lists(stack: QUndoStack, session: ProjectSession, text: str, **after: object) -> None:
    """Apply a change to project lists or configurations as a single undo step."""
    project = session.project
    before = {name: _copy(getattr(project, name)) for name in after}
    if all(_copy(after[name]) == before[name] for name in after):
        return  # nothing changed: keep the undo history clean
    stack.push(ReplaceListsCommand(session, text, before, {k: _copy(v) for k, v in after.items()}))
