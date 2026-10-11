"""Application style: Qt Fusion with a restrained light palette and a few named styles."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

STYLESHEET = """
#sidebar { background: #f3f5f8; border: none; border-right: 1px solid #d8dde4; }
#sidebar::item { padding: 8px 10px; border-radius: 4px; margin: 1px 6px; }
#sidebar::item:selected { background: #dbe7f6; color: #10335c; }
#sidebar::item:hover:!selected { background: #e8edf3; }
#pageTitle { font-size: 16pt; font-weight: 600; color: #1d2733; }
#pageGoal { color: #4a5563; }
#placeholderNote { color: #6b7685; font-style: italic; }
QGroupBox { font-weight: 600; margin-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QStatusBar QLabel { padding: 0 8px; color: #3a4553; }
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
