"""Consistent light theme for Inferential Model Studio (Qt Fusion)."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

STYLESHEET = """
/* Sidebar: fully specified normal, hover and selected visual states. */
QListWidget#sidebar {
    background-color: #f3f5f8;
    color: #1d2733;
    border: none;
    border-right: 1px solid #d8dde4;
    outline: none;
}

QListWidget#sidebar::item {
    color: #1d2733;
    background-color: transparent;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    margin: 0px 5px;
}

QListWidget#sidebar::item:hover:!selected {
    background-color: #e8edf3;
    color: #1d2733;
}

/* Use the same selected palette with or without keyboard focus. */
QListWidget#sidebar::item:selected,
QListWidget#sidebar::item:selected:active,
QListWidget#sidebar::item:selected:!active {
    background-color: #dbe7f6;
    color: #10335c;
}

QListWidget#sidebar::item:selected:hover {
    background-color: #d0e0f3;
    color: #10335c;
}

QLabel#pageTitle {
    font-size: 16pt;
    font-weight: 600;
    color: #1d2733;
}

QLabel#pageGoal {
    color: #4a5563;
}

QLabel#placeholderNote {
    color: #6b7685;
    font-style: italic;
}

QGroupBox {
    font-weight: 600;
    margin-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}

QStatusBar QLabel {
    padding: 0 8px;
    color: #3a4553;
}
"""


def _light_palette() -> QPalette:
    """Define colors in every widget state, including inactive focus groups.

    Windows dark-mode preferences can influence Fusion's default palette.
    Explicitly setting both active and inactive groups prevents white-on-light
    text and inconsistent selection colors after focus changes.
    """
    palette = QPalette()

    roles = {
        QPalette.ColorRole.Window: "#f3f5f8",
        QPalette.ColorRole.WindowText: "#1d2733",
        QPalette.ColorRole.Base: "#ffffff",
        QPalette.ColorRole.AlternateBase: "#f3f5f8",
        QPalette.ColorRole.Text: "#1d2733",
        QPalette.ColorRole.Button: "#e8edf3",
        QPalette.ColorRole.ButtonText: "#1d2733",
        QPalette.ColorRole.Highlight: "#dbe7f6",
        QPalette.ColorRole.HighlightedText: "#10335c",
        QPalette.ColorRole.ToolTipBase: "#ffffff",
        QPalette.ColorRole.ToolTipText: "#1d2733",
        QPalette.ColorRole.PlaceholderText: "#8a93a0",
        QPalette.ColorRole.Link: "#1a5fb4",
        QPalette.ColorRole.Light: "#ffffff",
        QPalette.ColorRole.Midlight: "#e8edf3",
        QPalette.ColorRole.Mid: "#c4ccd6",
        QPalette.ColorRole.Dark: "#8a93a0",
        QPalette.ColorRole.Shadow: "#555e6b",
    }

    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
        QPalette.ColorGroup.Disabled,
    ):
        for role, color in roles.items():
            palette.setColor(group, role, QColor(color))

    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#8a93a0"))

    return palette


def apply_theme(app: QApplication) -> None:
    """Apply the global theme after QApplication is constructed."""
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    app.setStyleSheet(STYLESHEET)
