"""Application settings dialog."""

from __future__ import annotations

from dataclasses import replace
from zoneinfo import available_timezones

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
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
from inferential_studio.infrastructure.settings import LOG_LEVELS, THEME_SETTINGS, AppSettings

THEME_LABELS = {"auto": "Follow Windows", "light": "Light", "dark": "Dark"}


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

        self.theme = QComboBox()
        for value in THEME_SETTINGS:
            self.theme.addItem(THEME_LABELS[value], value)
        self.theme.setCurrentIndex(max(self.theme.findData(settings.theme), 0))
        self.theme.setToolTip("The switch in the title bar changes it too.")

        self.system_title_bar = QCheckBox("Use the system title bar (after a restart)")
        self.system_title_bar.setChecked(settings.system_title_bar)
        self.system_title_bar.setToolTip(
            "Show the standard Windows frame above the application's own title bar, "
            "for example with screen readers or window-management tools."
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout(self)
        form.addRow("Display timezone:", self.timezone)
        form.addRow("Your name:", self.author)
        form.addRow("Log level:", self.log_level)
        form.addRow("Theme:", self.theme)
        form.addRow("", self.system_title_bar)
        form.addRow(buttons)

    def accept(self) -> None:
        try:
            validate_timezone(self.timezone.currentText().strip())
        except SchemaError as exc:
            QMessageBox.warning(self, "Invalid timezone", str(exc))
            return
        super().accept()

    def result_settings(self) -> AppSettings:
        """The settings as edited; everything the dialog does not show is kept."""
        return replace(
            self._settings,
            recent_projects=list(self._settings.recent_projects),
            display_timezone=self.timezone.currentText().strip(),
            log_level=self.log_level.currentText(),
            author_name=self.author.text().strip(),
            theme=self.theme.currentData(),
            system_title_bar=self.system_title_bar.isChecked(),
        )
