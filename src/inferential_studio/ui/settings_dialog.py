"""Application settings dialog."""

from __future__ import annotations

from zoneinfo import available_timezones

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QWidget,
)

from inferential_studio.core.schemas import SchemaError, validate_timezone
from inferential_studio.infrastructure.settings import LOG_LEVELS, AppSettings


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self._settings = settings

        zones = sorted(available_timezones())
        self.timezone = QComboBox()
        self.timezone.setEditable(True)
        self.timezone.addItems(zones)
        completer = QCompleter(zones, self.timezone)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.timezone.setCompleter(completer)
        self.timezone.setCurrentText(settings.display_timezone)
        self.timezone.setToolTip(
            "Default timezone used to display times in new projects. Data is always stored in UTC."
        )

        self.author = QLineEdit(settings.author_name)
        self.author.setToolTip("Recorded on exclusions and model reports you create.")

        self.log_level = QComboBox()
        self.log_level.addItems(LOG_LEVELS)
        self.log_level.setCurrentText(settings.log_level)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout(self)
        form.addRow("Display timezone:", self.timezone)
        form.addRow("Your name:", self.author)
        form.addRow("Log level:", self.log_level)
        form.addRow(buttons)

    def accept(self) -> None:
        try:
            validate_timezone(self.timezone.currentText().strip())
        except SchemaError as exc:
            QMessageBox.warning(self, "Invalid timezone", str(exc))
            return
        super().accept()

    def result_settings(self) -> AppSettings:
        return AppSettings(
            display_timezone=self.timezone.currentText().strip(),
            log_level=self.log_level.currentText(),
            recent_projects=list(self._settings.recent_projects),
            last_directory=self._settings.last_directory,
            author_name=self.author.text().strip(),
        )
