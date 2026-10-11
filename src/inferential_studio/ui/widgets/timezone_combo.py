"""Editable, searchable combo box of IANA timezone names."""

from __future__ import annotations

from functools import lru_cache
from zoneinfo import available_timezones

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QCompleter, QWidget


@lru_cache(maxsize=1)
def timezone_names() -> list[str]:
    return sorted(available_timezones())


class TimezoneCombo(QComboBox):
    def __init__(self, current: str = "UTC", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        names = timezone_names()
        self.addItems(names)
        completer = QCompleter(names, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.setCompleter(completer)
        self.setCurrentText(current)
        self.setMinimumContentsLength(22)

    def timezone(self) -> str:
        return self.currentText().strip()

    def is_valid(self) -> bool:
        return self.timezone() in set(timezone_names())
