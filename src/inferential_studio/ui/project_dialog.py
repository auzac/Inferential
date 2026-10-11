"""Project settings: name, display timezone and notes."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QWidget,
)

from inferential_studio.core.schemas import Project
from inferential_studio.ui.widgets.timezone_combo import TimezoneCombo


class ProjectSettingsDialog(QDialog):
    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project settings")
        self.name = QLineEdit(project.name)
        self.timezone = TimezoneCombo(project.display_timezone)
        self.notes = QPlainTextEdit(project.notes)
        hint = QLabel("Times are always stored in UTC; this only changes how they are shown.")
        hint.setObjectName("placeholderNote")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form = QFormLayout(self)
        form.addRow("Project name:", self.name)
        form.addRow("Display timezone:", self.timezone)
        form.addRow("", hint)
        form.addRow("Notes:", self.notes)
        form.addRow(buttons)
        self.resize(520, 360)

    def accept(self) -> None:
        if not self.name.text().strip():
            QMessageBox.warning(self, "Project name", "Enter a project name.")
            return
        if not self.timezone.is_valid():
            QMessageBox.warning(
                self, "Display timezone", "Choose a valid IANA timezone, e.g. Asia/Singapore."
            )
            return
        super().accept()

    def apply_to(self, project: Project) -> bool:
        """Write the edited values into ``project``. Returns True if anything changed."""
        new = (self.name.text().strip(), self.timezone.timezone(), self.notes.toPlainText())
        old = (project.name, project.display_timezone, project.notes)
        if new == old:
            return False
        project.name, project.display_timezone, project.notes = new
        return True
