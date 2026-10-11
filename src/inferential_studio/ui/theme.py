"""Light, high-contrast industrial theme (Qt Fusion + one stylesheet).

Layout of the window: a dark slate navigation rail and status bar frame a light work area; white
surfaces (tables, charts, group boxes) sit on a cool grey background; one steel-blue accent marks
selection and primary actions. Severity colours (info / warning / error) are reserved for the
message panel. All colours are defined once in ``TOKENS``.

The palette is set for every colour group explicitly so Windows dark-mode preferences cannot
leak into Fusion (white-on-light text, inconsistent selection after focus changes).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPolygonF
from PySide6.QtWidgets import QApplication, QWidget

TOKENS = {
    "bg": "#eceff3",  # work area
    "surface": "#ffffff",  # tables, charts, cards
    "surface_alt": "#f6f8fa",  # alternate rows, headers
    "border": "#d0d7df",
    "border_strong": "#b4bfca",
    "text": "#18222d",
    "text_muted": "#596675",
    "text_faint": "#8a96a3",
    "rail": "#1d2833",  # navigation rail, status bar
    "rail_hover": "#26343f",
    "rail_selected": "#2c3d4c",
    "rail_text": "#c3cdd7",
    "accent": "#2f6fa3",
    "accent_hover": "#285f8c",
    "accent_pressed": "#214f75",
    "accent_soft": "#dce8f3",
    "info": "#2f6fa3",
    "warning": "#b86e00",
    "warning_soft": "#fff3df",
    "error": "#b3261e",
    "error_soft": "#fbe7e5",
    "success": "#2e7d4f",  # "OK" status marks only
}

T = TOKENS
STYLESHEET = f"""
QMainWindow, QDialog {{ background: {T["bg"]}; }}
QToolTip {{ background: {T["surface"]}; color: {T["text"]}; border: 1px solid {T["border_strong"]};
            padding: 4px 6px; }}

/* ---- menu bar: one command row ---------------------------------------------------- */
QMenuBar {{ background: {T["surface"]}; border-bottom: 1px solid {T["border"]}; padding: 1px 4px; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; color: {T["text"]}; }}
QMenuBar::item:selected {{ background: {T["accent_soft"]}; }}
QMenu {{ background: {T["surface"]}; border: 1px solid {T["border_strong"]}; padding: 4px 0; }}
QMenu::item {{ padding: 5px 24px 5px 18px; color: {T["text"]}; }}
QMenu::item:selected {{ background: {T["accent_soft"]}; }}
QMenu::item:disabled {{ color: {T["text_faint"]}; }}
QMenu::separator {{ height: 1px; background: {T["border"]}; margin: 4px 8px; }}
QMenuBar QToolButton {{ background: transparent; border: none; border-radius: 3px; padding: 4px 9px;
                        color: {T["text"]}; }}
QMenuBar QToolButton:hover {{ background: {T["accent_soft"]}; }}
QMenuBar QToolButton:pressed {{ background: #c8dcee; }}
QMenuBar QToolButton:disabled {{ color: {T["text_faint"]}; }}
QMenuBar QFrame#menuBarSeparator {{ background: {T["border"]}; margin: 7px 6px; }}

/* ---- navigation rail --------------------------------------------------------------- */
QWidget#rail {{ background: {T["rail"]}; }}
QLabel#railTitle {{ color: #ffffff; font-weight: 600; font-size: 10.5pt; padding: 14px 16px 0 16px; }}
QLabel#railSubtitle {{ color: {T["text_faint"]}; font-size: 7.5pt; letter-spacing: 1px;
                       padding: 2px 16px 12px 16px; }}
QListWidget#sidebar {{ background: {T["rail"]}; color: {T["rail_text"]}; border: none; outline: none; }}
QListWidget#sidebar::item {{ color: {T["rail_text"]}; background: transparent; border: none;
                             border-left: 3px solid transparent; padding: 4px 12px; margin: 0; }}
QListWidget#sidebar::item:hover:!selected {{ background: {T["rail_hover"]}; color: #ffffff; }}
QListWidget#sidebar::item:selected, QListWidget#sidebar::item:selected:active,
QListWidget#sidebar::item:selected:!active {{ background: {T["rail_selected"]}; color: #ffffff;
                                             border-left: 3px solid {T["accent"]}; }}

/* ---- page header ------------------------------------------------------------------- */
QFrame#pageHeader {{ background: {T["surface"]}; border: none; border-bottom: 1px solid {T["border"]}; }}
QLabel#pageEyebrow {{ color: {T["accent"]}; font-size: 7.5pt; font-weight: 700; letter-spacing: 1px; }}
QLabel#pageTitle {{ font-size: 15pt; font-weight: 600; color: {T["text"]}; }}
QLabel#pageGoal {{ color: {T["text_muted"]}; }}
QLabel#placeholderNote {{ color: {T["text_muted"]}; font-size: 8.5pt; }}
QLabel#sectionTitle {{ color: {T["text_muted"]}; font-weight: 600; }}

/* ---- surfaces ---------------------------------------------------------------------- */
QGroupBox {{ background: {T["surface"]}; border: 1px solid {T["border"]}; border-radius: 4px;
             margin-top: 16px; padding: 12px 8px 8px 8px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 8px;
                    padding: 0 4px; color: {T["text_muted"]}; }}
QTabWidget::pane {{ background: {T["surface"]}; border: 1px solid {T["border"]}; top: -1px; }}
QTabBar::tab {{ background: transparent; color: {T["text_muted"]}; padding: 6px 14px; margin-right: 2px;
                border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {T["text"]}; }}
QTabBar::tab:selected {{ color: {T["text"]}; border-bottom: 2px solid {T["accent"]}; font-weight: 600; }}
QTabBar::tab:disabled {{ color: {T["text_faint"]}; }}
QSplitter::handle {{ background: {T["bg"]}; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QScrollArea {{ border: none; background: transparent; }}

/* ---- tables and lists -------------------------------------------------------------- */
QTableView, QTableWidget, QTreeWidget, QListWidget, QTextBrowser, QPlainTextEdit {{
    background: {T["surface"]}; border: 1px solid {T["border"]}; gridline-color: #e4e8ed;
    selection-background-color: {T["accent_soft"]}; selection-color: {T["text"]};
    alternate-background-color: {T["surface_alt"]}; }}
QHeaderView::section {{ background: {T["surface_alt"]}; color: {T["text_muted"]}; font-weight: 600;
                        border: none; border-right: 1px solid #e4e8ed; border-bottom: 1px solid {T["border"]};
                        padding: 4px 6px; }}
QTableCornerButton::section {{ background: {T["surface_alt"]}; border: none; }}

/* ---- inputs and buttons ------------------------------------------------------------ */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateTimeEdit {{
    background: {T["surface"]}; border: 1px solid {T["border_strong"]}; border-radius: 3px;
    padding: 3px 6px; min-height: 18px; selection-background-color: {T["accent_soft"]};
    selection-color: {T["text"]}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateTimeEdit:focus,
QPlainTextEdit:focus {{ border: 1px solid {T["accent"]}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QDateTimeEdit:disabled {{ background: {T["surface_alt"]}; color: {T["text_faint"]}; }}
QComboBox QAbstractItemView {{ background: {T["surface"]}; border: 1px solid {T["border_strong"]};
                               selection-background-color: {T["accent_soft"]}; selection-color: {T["text"]}; }}
QPushButton {{ background: {T["surface"]}; border: 1px solid {T["border_strong"]}; border-radius: 3px;
               padding: 4px 12px; min-height: 18px; color: {T["text"]}; }}
QPushButton:hover {{ background: #f1f5f9; border-color: {T["accent"]}; }}
QPushButton:pressed {{ background: {T["accent_soft"]}; }}
QPushButton:checked {{ background: {T["accent_soft"]}; border-color: {T["accent"]}; }}
QPushButton:disabled {{ color: {T["text_faint"]}; background: {T["surface_alt"]}; border-color: {T["border"]}; }}
QPushButton[primary="true"] {{ background: {T["accent"]}; color: #ffffff; border: 1px solid {T["accent"]};
                               font-weight: 600; padding: 5px 16px; }}
QPushButton[primary="true"]:hover {{ background: {T["accent_hover"]}; }}
QPushButton[primary="true"]:pressed {{ background: {T["accent_pressed"]}; }}
QPushButton[primary="true"]:disabled {{ background: #9fb7cc; border-color: #9fb7cc; color: #eef3f7; }}
QToolButton {{ border-radius: 3px; padding: 3px 8px; }}
QToolButton:hover {{ background: {T["accent_soft"]}; }}
QCheckBox, QRadioButton {{ spacing: 6px; }}

/* ---- scroll bars ------------------------------------------------------------------- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: #c3ccd5; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: #a5b1bd; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---- docks: help panel and message panel ------------------------------------------- */
QDockWidget::title {{ background: {T["surface_alt"]}; padding: 6px 8px; border-bottom: 1px solid {T["border"]};
                      font-weight: 600; color: {T["text_muted"]}; }}
QDockWidget#propertiesDock QTextBrowser {{ border: none; padding: 6px; }}
QFrame#messagePanel {{ background: {T["surface"]}; border: none; border-top: 2px solid {T["border_strong"]}; }}
QFrame#messagePanel[flash="true"] {{ border-top: 2px solid {T["warning"]}; }}
QLabel#messageTitle {{ color: {T["text_muted"]}; font-weight: 700; font-size: 7.5pt; letter-spacing: 1px; }}
QLabel#badgeError {{ background: {T["error_soft"]}; color: {T["error"]}; border-radius: 8px;
                     padding: 0 8px; font-weight: 600; font-size: 8pt; }}
QLabel#badgeWarning {{ background: {T["warning_soft"]}; color: {T["warning"]}; border-radius: 8px;
                       padding: 0 8px; font-weight: 600; font-size: 8pt; }}
QTreeWidget#messageList {{ border: none; font-size: 8.5pt; }}
QTreeWidget#messageList::item {{ height: 22px; border-bottom: 1px solid #eef1f4; }}
QTreeWidget#messageList::item:selected {{ background: {T["accent_soft"]}; color: {T["text"]}; }}

/* ---- status bar -------------------------------------------------------------------- */
QStatusBar {{ background: {T["rail"]}; color: {T["rail_text"]}; }}
QStatusBar QLabel {{ color: {T["rail_text"]}; padding: 0 10px; }}
QStatusBar::item {{ border: none; }}
QStatusBar QPushButton, QStatusBar QToolButton {{ background: transparent; color: #ffffff;
                                                  border: 1px solid #4a5a69; padding: 1px 8px; }}
QStatusBar QProgressBar {{ background: {T["rail_hover"]}; border: none; color: #ffffff; max-height: 12px;
                           text-align: center; }}
QStatusBar QProgressBar::chunk {{ background: {T["accent"]}; }}
"""


def _palette() -> QPalette:
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: T["bg"],
        QPalette.ColorRole.WindowText: T["text"],
        QPalette.ColorRole.Base: T["surface"],
        QPalette.ColorRole.AlternateBase: T["surface_alt"],
        QPalette.ColorRole.Text: T["text"],
        QPalette.ColorRole.Button: T["surface"],
        QPalette.ColorRole.ButtonText: T["text"],
        QPalette.ColorRole.Highlight: T["accent_soft"],
        QPalette.ColorRole.HighlightedText: T["text"],
        QPalette.ColorRole.ToolTipBase: T["surface"],
        QPalette.ColorRole.ToolTipText: T["text"],
        QPalette.ColorRole.PlaceholderText: T["text_faint"],
        QPalette.ColorRole.Link: T["accent"],
        QPalette.ColorRole.Light: "#ffffff",
        QPalette.ColorRole.Midlight: T["surface_alt"],
        QPalette.ColorRole.Mid: T["border_strong"],
        QPalette.ColorRole.Dark: T["text_faint"],
        QPalette.ColorRole.Shadow: "#4b5866",
    }
    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
        QPalette.ColorGroup.Disabled,
    ):
        for role, colour in roles.items():
            palette.setColor(group, role, QColor(colour))
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(T["text_faint"]))
    return palette


def _arrow_images() -> dict[str, str]:
    """Small arrow PNGs for combo boxes and spin boxes (QSS needs image files)."""
    folder = Path(tempfile.gettempdir()) / "inferential_studio_theme"
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, points in {
        "down": ((1, 3), (9, 3), (5, 8)),
        "up": ((1, 7), (9, 7), (5, 2)),
    }.items():
        for state, colour in (("", T["text_muted"]), ("_disabled", T["text_faint"])):
            path = folder / f"arrow_{name}{state}.png"
            if not path.exists():
                img = QImage(10, 10, QImage.Format.Format_ARGB32)
                img.fill(Qt.GlobalColor.transparent)
                painter = QPainter(img)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(colour))
                painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
                painter.end()
                img.save(str(path))
            out[f"{name}{state}"] = path.as_posix()
    return out


def _arrow_stylesheet() -> str:
    a = _arrow_images()
    return f"""
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 18px;
                        border: none; border-left: 1px solid {T["border"]}; }}
QComboBox::down-arrow {{ image: url({a["down"]}); width: 10px; height: 10px; }}
QComboBox::down-arrow:disabled {{ image: url({a["down_disabled"]}); }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border; width: 16px;
    border: none; border-left: 1px solid {T["border"]}; background: {T["surface_alt"]}; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 3px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 3px; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {T["accent_soft"]}; }}
QAbstractSpinBox::up-arrow {{ image: url({a["up"]}); width: 8px; height: 8px; }}
QAbstractSpinBox::down-arrow {{ image: url({a["down"]}); width: 8px; height: 8px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{ image: url({a["up_disabled"]}); }}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{ image: url({a["down_disabled"]}); }}
QDateTimeEdit::drop-down {{ width: 18px; border: none; border-left: 1px solid {T["border"]}; }}
QDateTimeEdit::down-arrow {{ image: url({a["down"]}); width: 10px; height: 10px; }}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the global theme after QApplication is constructed."""
    app.setStyle("Fusion")
    app.setPalette(_palette())
    font = QFont()
    font.setFamilies(["Segoe UI", "Inter", "Noto Sans", "DejaVu Sans", "Arial"])
    font.setPointSizeF(9.0)
    app.setFont(font)
    app.setStyleSheet(STYLESHEET + _arrow_stylesheet())
    # Charts: dark axis text on white plots (antialiasing stays off: large trends stay fast).
    pg.setConfigOptions(foreground="#3b4754", background="w")


def mark_primary(*buttons: QWidget) -> None:
    """Style buttons as the page's main action (filled accent)."""
    for b in buttons:
        b.setProperty("primary", True)
        b.style().unpolish(b)
        b.style().polish(b)


def theme_colour(token: str) -> QColor:
    """A theme colour as a QColor (e.g. ``theme_colour("error")`` for a table cell)."""
    return QColor(TOKENS[token])


def tinted(text: str, token: str = "error", *, bold: bool = False) -> str:
    """Rich text in a theme colour, for inline validation in dialogs (empty text stays empty)."""
    if not text:
        return ""
    weight = "font-weight:600;" if bold else ""
    return f"<span style='color:{TOKENS[token]};{weight}'>{text}</span>"
