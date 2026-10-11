"""Checkable, filterable list of variables (used by several pages)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout, QWidget


class VariableList(QWidget):
    selection_changed = Signal()

    def __init__(self, title: str = "Variables", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter variables…")
        self.search.textChanged.connect(self._filter)
        self.list = QListWidget()
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
