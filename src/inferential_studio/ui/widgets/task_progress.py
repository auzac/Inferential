"""Status-bar widget showing the running background task with a Cancel button."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget

from inferential_studio.infrastructure.workers import TaskManager


class TaskProgressWidget(QWidget):
    def __init__(self, manager: TaskManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._current: int | None = None

        self.label = QLabel()
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedWidth(180)
        self.bar.setTextVisible(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip("Request cancellation of the running task")
        self.cancel_button.clicked.connect(self._cancel)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.bar)
        layout.addWidget(self.cancel_button, 0, Qt.AlignmentFlag.AlignVCenter)

        manager.task_started.connect(self._on_started)
        manager.task_progress.connect(self._on_progress)
        manager.task_ended.connect(self._on_ended)
        self.setVisible(False)

    @Slot(int, str)
    def _on_started(self, task_id: int, name: str) -> None:
        self._current = task_id
        self.label.setText(name)
        self.bar.setValue(0)
        self.cancel_button.setEnabled(True)
        self.setVisible(True)

    @Slot(int, str, float, str)
    def _on_progress(self, task_id: int, name: str, fraction: float, message: str) -> None:
        if task_id == self._current:
            self.bar.setValue(int(fraction * 1000))
            self.label.setText(f"{name}: {message}" if message else name)

    @Slot(int, str, str)
    def _on_ended(self, task_id: int, name: str, outcome: str) -> None:
        remaining = self._manager.active
        if remaining:
            self._on_started(remaining[-1].task_id, remaining[-1].name)
        else:
            self._current = None
            self.setVisible(False)

    @Slot()
    def _cancel(self) -> None:
        if self._current is not None:
            self._manager.cancel(self._current)
            self.cancel_button.setEnabled(False)
            self.label.setText(self.label.text() + " (cancelling…)")
