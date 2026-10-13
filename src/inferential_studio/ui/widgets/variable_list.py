"""Checkable, filterable list of variables (used by several pages)."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)


class _VariableCheckDelegate(QStyledItemDelegate):
    def editorEvent(self, event, model, option, index):  # noqa: N802 - Qt override
        if (
            event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
            and option.rect.contains(event.position().toPoint())
            and index.flags() & Qt.ItemFlag.ItemIsEnabled
            and index.flags() & Qt.ItemFlag.ItemIsUserCheckable
        ):
            state = index.data(Qt.ItemDataRole.CheckStateRole)
            checked = state == Qt.CheckState.Checked or state == Qt.CheckState.Checked.value
            return model.setData(
                index,
                Qt.CheckState.Unchecked.value if checked else Qt.CheckState.Checked.value,
                Qt.ItemDataRole.CheckStateRole,
            )
        return super().editorEvent(event, model, option, index)


class CheckableVariableList(QListWidget):
    """Variable names and their checkboxes share the same click target."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setItemDelegate(_VariableCheckDelegate(self))


class VariableList(QWidget):
    selection_changed = Signal()

    def __init__(self, title: str = "Variables", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter variables…")
        self.search.textChanged.connect(self._filter)
        self.list = CheckableVariableList()
        self.list.itemChanged.connect(lambda _i: self.selection_changed.emit())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(title))
        layout.addWidget(self.search)
        layout.addWidget(self.list, 1)

    def set_variables(
        self, names: list[str], units: dict[str, str | None], checked: set[str]
    ) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for name in names:
            u = units.get(name)
            item = QListWidgetItem(name + (f"  [{u}]" if u else ""))
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if name in checked else Qt.CheckState.Unchecked
            )
            self.list.addItem(item)
        self.list.blockSignals(False)
        self._filter(self.search.text())

    def names(self) -> list[str]:
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]

    def checked(self) -> list[str]:
        return [
            self.list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.list.count())
            if self.list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def set_checked(self, name: str, checked: bool = True) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == name:
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def _filter(self, text: str) -> None:
        text = text.lower()
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(text not in item.text().lower())
