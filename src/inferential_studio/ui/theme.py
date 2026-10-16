"""Two themes, switchable at run time: a warm light theme and Windows 11 dark.

The window is one seamless surface: the title bar, the navigation rail and the status bar share
the ``base`` colour, and the pages sit on a slightly lighter content layer (``bg``) with a
rounded top-left corner, as in Windows 11 applications. Cards, tables, inputs and charts use
``surface``. One accent marks selection and primary actions; severity colours (info / warning /
error) are reserved for messages and validation text.

All colours come from ``TOKENS``, a dict that :func:`apply_theme` updates in place, so code
that reads ``TOKENS[...]`` when it draws always gets the current theme. Widgets that cache
colours (icons, chart headers) implement ``apply_theme()`` and are refreshed by
:func:`refresh_widgets` after a switch.

The palette is set for every colour group explicitly so operating-system colour preferences
cannot leak into Fusion (white-on-light text, inconsistent selection after focus changes).
"""

from __future__ import annotations

import logging
import tempfile
from html import escape
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QPainter,
    QPalette,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QApplication, QWidget

LIGHT = "light"
DARK = "dark"
AUTO = "auto"
THEME_NAMES = {LIGHT: "Light", DARK: "Dark"}

log = logging.getLogger(__name__)

THEMES: dict[str, dict[str, str]] = {
    # Warm off-white paper tones with a terracotta accent.
    LIGHT: {
        "base": "#efede6",  # title bar, navigation rail, status bar
        "bg": "#f8f7f3",  # content layer behind the pages
        "surface": "#fffefc",  # cards, tables, inputs, charts
        "surface_alt": "#f4f2ec",  # table headers, alternate rows
        "border": "#e3dfd4",
        "border_strong": "#cdc7b8",
        "gridline": "#ebe8df",
        "text": "#1f1e1b",
        "text_muted": "#68645a",
        "text_faint": "#9d988c",
        "hover": "#e5e2d8",  # subtle fill on the base layer
        "menu_hover": "#e5e2d8",  # menu row highlight (menus sit on ``surface``)
        "selected": "#e0dccf",
        "accent": "#b85a38",
        "accent_hover": "#ad5434",
        "accent_pressed": "#96482c",
        "accent_soft": "#f4e2d9",  # selection background
        "accent_disabled": "#e2b9a8",
        "on_accent": "#ffffff",
        "info": "#3b6ea5",
        "warning": "#a35a00",
        "warning_soft": "#fbecd5",
        "error": "#b3261e",
        "error_soft": "#f9e3df",
        "success": "#3d7d4a",
        "scroll": "#d3cec1",
        "scroll_hover": "#b9b3a3",
        "chart_bg": "#fffefc",
        "chart_fg": "#5e5a51",
        "close_hover": "#c42b1c",
    },
    # Windows 11 dark mode (Mica base, layered surfaces, light-blue accent).
    DARK: {
        "base": "#202020",
        "bg": "#272727",
        "surface": "#2d2d2d",
        "surface_alt": "#333333",
        "border": "#3b3b3b",
        "border_strong": "#4d4d4d",
        "gridline": "#383838",
        "text": "#f2f2f2",
        "text_muted": "#c5c5c5",
        "text_faint": "#8f8f8f",
        "hover": "#2d2d2d",
        "menu_hover": "#3a3a3a",  # lighter than ``surface``: the highlight must read on menus
        "selected": "#333333",
        "accent": "#4cc2ff",
        "accent_hover": "#47b1e8",
        "accent_pressed": "#42a1d2",
        "accent_soft": "#1f3a4c",
        "accent_disabled": "#3d5a6b",
        "on_accent": "#000000",
        "info": "#60cdff",
        "warning": "#fce100",
        "warning_soft": "#433519",
        "error": "#ff99a4",
        "error_soft": "#442726",
        "success": "#6ccb5f",
        "scroll": "#5a5a5a",
        "scroll_hover": "#787878",
        "chart_bg": "#2b2b2b",
        "chart_fg": "#bdbdbd",
        "close_hover": "#c42b1c",
    },
}

TOKENS: dict[str, str] = dict(THEMES[LIGHT])  # the current theme (updated in place)
_current = LIGHT


def current_theme() -> str:
    return _current


def resolve_theme(setting: str | None) -> str:
    """``light`` / ``dark``; ``auto`` (or anything else) follows the operating system."""
    if setting in THEMES:
        return setting
    hints = QGuiApplication.styleHints() if QGuiApplication.instance() else None
    if hints is not None and hints.colorScheme() == Qt.ColorScheme.Dark:
        return DARK
    return LIGHT


def build_stylesheet(t: dict[str, str]) -> str:
    return f"""
QMainWindow, QDialog {{ background: {t["bg"]}; }}
QMainWindow {{ background: {t["base"]}; }}
QToolTip {{ background: {t["surface"]}; color: {t["text"]}; border: 1px solid {t["border_strong"]};
            padding: 4px 6px; }}
QMenu {{ background: {t["surface"]}; border: 1px solid {t["border_strong"]}; padding: 4px; }}
QMenu::item {{ padding: 6px 28px 6px 10px; color: {t["text"]}; border-radius: 4px; }}
QMenu::item:selected {{ background: {t["menu_hover"]}; }}
QMenu::item:disabled {{ color: {t["text_faint"]}; }}
QMenu::icon {{ padding-left: 8px; }}
QMenu::separator {{ height: 1px; background: {t["border"]}; margin: 4px 6px; }}
QMenu::right-arrow {{ width: 8px; height: 8px; }}

/* ---- window shell: title bar, navigation rail, content layer, status bar -------------- */
QWidget#titleBar {{ background: {t["base"]}; }}
QLabel#windowTitle {{ color: {t["text"]}; }}
QLabel#windowTitleApp {{ color: {t["text_muted"]}; }}
QFrame#titleSeparator {{ background: {t["border_strong"]}; margin: 10px 4px; }}
QToolButton#titleButton {{ background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 3px;
                           min-width: 30px; min-height: 30px; }}
QToolButton#titleButton:hover {{ background: {t["hover"]}; }}
QToolButton#titleButton:pressed {{ background: {t["selected"]}; }}
QToolButton#titleButton:checked {{ background: {t["selected"]}; }}
QToolButton#titleButton::menu-indicator {{ image: none; width: 0; }}
QToolButton#captionButton {{ background: transparent; border: 1px solid transparent; border-radius: 0; padding: 0;
                             min-width: 46px; max-width: 46px; }}
QToolButton#captionButton:hover {{ background: {t["hover"]}; }}
QToolButton#captionButton:pressed {{ background: {t["selected"]}; }}
QToolButton#closeButton {{ background: transparent; border: 1px solid transparent; border-radius: 0; padding: 0;
                           min-width: 46px; max-width: 46px; }}
QToolButton#closeButton:hover {{ background: {t["close_hover"]}; }}
QToolButton#closeButton:pressed {{ background: #b22a1c; }}
QWidget#navRail {{ background: {t["base"]}; }}
QListWidget#sidebar {{ background: transparent; border: none; outline: none; color: {t["text"]}; }}
QToolButton#navToggle {{ background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 0;
                         margin: 0 4px; min-height: 36px; }}
QToolButton#titleButton:focus, QToolButton#captionButton:focus,
QToolButton#closeButton:focus, QToolButton#navToggle:focus {{ border-color: {t["accent"]}; }}
QToolButton#navToggle:hover {{ background: {t["hover"]}; }}
QFrame#contentLayer {{ background: {t["bg"]}; border: 1px solid {t["border"]}; border-right: none;
                       border-bottom: none; border-top-left-radius: 8px; }}
QWidget#helpPane {{ background: {t["bg"]}; border-left: 1px solid {t["border"]}; }}
QLabel#helpPaneTitle {{ color: {t["text_muted"]}; font-weight: 600; }}
QWidget#helpPane QTextBrowser {{ background: transparent; border: none; padding: 2px 4px; }}
QStatusBar {{ background: {t["base"]}; color: {t["text_muted"]}; }}
QStatusBar QLabel {{ color: {t["text_muted"]}; padding: 0 10px; }}
QStatusBar::item {{ border: none; }}
QStatusBar QPushButton, QStatusBar QToolButton {{ background: transparent; color: {t["text"]};
                                                  border: 1px solid {t["border_strong"]}; padding: 1px 8px; }}
QStatusBar QProgressBar {{ background: {t["hover"]}; border: none; color: {t["text"]}; max-height: 12px;
                           text-align: center; border-radius: 3px; }}
QStatusBar QProgressBar::chunk {{ background: {t["accent"]}; border-radius: 3px; }}

/* ---- page header ------------------------------------------------------------------- */
QFrame#pageHeader {{ background: transparent; border: none; border-bottom: 1px solid {t["border"]}; }}
QLabel#pageEyebrow {{ color: {t["accent"]}; font-size: 8.5pt; font-weight: 600; }}
QLabel#pageTitle {{ font-size: 14pt; font-weight: 600; color: {t["text"]}; }}
QLabel#pageGoal {{ color: {t["text_muted"]}; }}
QLabel#pageProgress {{ color: {t["text_faint"]}; font-size: 8.5pt; }}
QLabel#placeholderNote {{ color: {t["text_muted"]}; font-size: 8.5pt; }}
QLabel#sectionTitle {{ color: {t["text_muted"]}; font-weight: 600; }}

/* ---- surfaces ---------------------------------------------------------------------- */
QGroupBox {{ background: {t["surface"]}; border: 1px solid {t["border"]}; border-radius: 6px;
             margin-top: 16px; padding: 12px 8px 8px 8px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 8px;
                    padding: 0 4px; color: {t["text_muted"]}; }}
QTabWidget::pane {{ background: {t["surface"]}; border: 1px solid {t["border"]}; top: -1px;
                    border-radius: 4px; }}
QTabBar::tab {{ background: transparent; color: {t["text_muted"]}; padding: 6px 12px; margin-right: 1px;
                border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {t["text"]}; }}
QTabBar::tab:selected {{ color: {t["text"]}; border-bottom: 2px solid {t["accent"]}; font-weight: 600; }}
QTabBar::tab:disabled {{ color: {t["text_faint"]}; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

/* ---- tables and lists -------------------------------------------------------------- */
QTableView, QTableWidget, QTreeWidget, QListWidget, QTextBrowser, QPlainTextEdit, QTextEdit {{
    background: {t["surface"]}; color: {t["text"]}; border: 1px solid {t["border"]};
    gridline-color: {t["gridline"]}; selection-background-color: {t["accent_soft"]};
    selection-color: {t["text"]}; alternate-background-color: {t["surface_alt"]}; }}
QHeaderView {{ background: {t["surface"]}; }}  /* no band under the last row */
QHeaderView::section {{ background: {t["surface_alt"]}; color: {t["text_muted"]}; font-weight: 600;
                        border: none; border-right: 1px solid {t["gridline"]};
                        border-bottom: 1px solid {t["border"]}; padding: 4px 6px; }}
QTableCornerButton::section {{ background: {t["surface_alt"]}; border: none; }}

/* ---- inputs and buttons ------------------------------------------------------------ */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateTimeEdit {{
    background: {t["surface"]}; color: {t["text"]}; border: 1px solid {t["border_strong"]};
    border-radius: 4px; padding: 3px 6px; min-height: 18px;
    selection-background-color: {t["accent_soft"]}; selection-color: {t["text"]}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateTimeEdit:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {t["accent"]}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QDateTimeEdit:disabled {{ background: {t["surface_alt"]}; color: {t["text_faint"]}; }}
QComboBox QAbstractItemView {{ background: {t["surface"]}; color: {t["text"]};
                               border: 1px solid {t["border_strong"]};
                               selection-background-color: {t["accent_soft"]}; selection-color: {t["text"]}; }}
QPushButton {{ background: {t["surface"]}; border: 1px solid {t["border_strong"]}; border-radius: 4px;
               padding: 4px 12px; min-height: 18px; color: {t["text"]}; }}
QPushButton:hover {{ background: {t["surface_alt"]}; border-color: {t["accent"]}; }}
QPushButton:pressed {{ background: {t["accent_soft"]}; }}
QPushButton:checked {{ background: {t["accent_soft"]}; border-color: {t["accent"]}; }}
QPushButton:disabled {{ color: {t["text_faint"]}; background: {t["surface_alt"]}; border-color: {t["border"]}; }}
QPushButton[primary="true"] {{ background: {t["accent"]}; color: {t["on_accent"]}; border: 1px solid {t["accent"]};
                               font-weight: 600; padding: 5px 16px; }}
QPushButton[primary="true"]:hover {{ background: {t["accent_hover"]}; border-color: {t["accent_hover"]}; }}
QPushButton[primary="true"]:pressed {{ background: {t["accent_pressed"]}; }}
QPushButton[primary="true"]:focus {{ border: 2px solid {t["text"]}; padding: 4px 15px; }}
QPushButton[primary="true"]:disabled {{ background: {t["accent_disabled"]}; border-color: {t["accent_disabled"]};
                                        color: {t["bg"]}; }}
QToolButton {{ border-radius: 4px; padding: 3px 8px; color: {t["text"]}; }}
QToolButton#iconButton {{ padding: 0; min-width: 24px; min-height: 24px; color: {t["text_muted"]}; }}
QToolButton#iconButton:hover {{ color: {t["error"]}; background: {t["error_soft"]}; }}
QToolButton:hover {{ background: {t["hover"]}; }}
QCheckBox, QRadioButton {{ spacing: 6px; color: {t["text"]}; }}
QLabel {{ color: {t["text"]}; }}

/* ---- scroll bars ------------------------------------------------------------------- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {t["scroll"]}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {t["scroll_hover"]}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---- message panel ----------------------------------------------------------------- */
QFrame#messagePanel {{ background: {t["bg"]}; border: none; border-top: 1px solid {t["border"]}; }}
QFrame#messagePanel[flash="true"] {{ border-top: 2px solid {t["warning"]}; }}
QLabel#messageTitle {{ color: {t["text_muted"]}; font-weight: 600; }}
QLabel#badgeError {{ background: {t["error_soft"]}; color: {t["error"]}; border-radius: 8px;
                     padding: 0 8px; font-weight: 600; font-size: 8pt; }}
QLabel#badgeWarning {{ background: {t["warning_soft"]}; color: {t["warning"]}; border-radius: 8px;
                       padding: 0 8px; font-weight: 600; font-size: 8pt; }}
QTreeWidget#messageList {{ border: none; background: transparent; font-size: 8.5pt; }}
QTreeWidget#messageList::item {{ height: 22px; border-bottom: 1px solid {t["gridline"]}; }}
QTreeWidget#messageList::item:selected {{ background: {t["accent_soft"]}; color: {t["text"]}; }}

/* ---- wizard ------------------------------------------------------------------------ */
QWizard {{ background: {t["bg"]}; }}
"""


def _palette(t: dict[str, str]) -> QPalette:
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: t["bg"],
        QPalette.ColorRole.WindowText: t["text"],
        QPalette.ColorRole.Base: t["surface"],
        QPalette.ColorRole.AlternateBase: t["surface_alt"],
        QPalette.ColorRole.Text: t["text"],
        QPalette.ColorRole.Button: t["surface"],
        QPalette.ColorRole.ButtonText: t["text"],
        QPalette.ColorRole.BrightText: t["on_accent"],
        QPalette.ColorRole.Highlight: t["accent_soft"],
        QPalette.ColorRole.HighlightedText: t["text"],
        QPalette.ColorRole.ToolTipBase: t["surface"],
        QPalette.ColorRole.ToolTipText: t["text"],
        QPalette.ColorRole.PlaceholderText: t["text_faint"],
        QPalette.ColorRole.Link: t["accent"],
        QPalette.ColorRole.LinkVisited: t["accent_pressed"],
        QPalette.ColorRole.Light: t["surface"],
        QPalette.ColorRole.Midlight: t["surface_alt"],
        QPalette.ColorRole.Mid: t["border_strong"],
        QPalette.ColorRole.Dark: t["text_faint"],
        QPalette.ColorRole.Shadow: t["base"],
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
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(t["text_faint"]))
    return palette


def _save_image(path: Path, img: QImage) -> None:
    """Write one theme asset; raise OSError on failure (stylesheet() falls back)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not img.save(str(path)) or not path.exists():
        raise OSError(f"cannot write theme asset {path}")


def _arrow_images(name: str, t: dict[str, str]) -> dict[str, str]:
    """Small arrow PNGs for combo boxes, spin boxes and tab scrollers (QSS needs files)."""
    folder = Path(tempfile.gettempdir()) / "inferential_studio_theme" / name
    out = {}
    for arrow, points in {
        "down": ((1, 3), (9, 3), (5, 8)),
        "up": ((1, 7), (9, 7), (5, 2)),
        "left": ((7, 1), (7, 9), (2, 5)),
        "right": ((3, 1), (3, 9), (8, 5)),
    }.items():
        for state, colour in (("", t["text_muted"]), ("_disabled", t["text_faint"])):
            path = folder / f"arrow_{arrow}{state}_{colour.lstrip('#')}.png"
            if not path.exists():
                img = QImage(10, 10, QImage.Format.Format_ARGB32)
                img.fill(Qt.GlobalColor.transparent)
                painter = QPainter(img)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(colour))
                painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
                painter.end()
                _save_image(path, img)
            out[f"{arrow}{state}"] = path.as_posix()
    return out


def _indicator_images(name: str, t: dict[str, str]) -> dict[str, str]:
    """Check-mark and dash PNGs (1x and @2x) for checked check boxes, list items and menus."""
    folder = Path(tempfile.gettempdir()) / "inferential_studio_theme" / name
    out = {}
    colour = t["on_accent"]
    shapes = {
        "check": [(3.2, 7.3), (5.9, 10.0), (10.9, 4.4)],
        "dash": [(3.5, 7.0), (10.5, 7.0)],
    }
    for box in (14, 12):  # check boxes; the smaller boxes of list rows
        for shape, points in shapes.items():
            stem = f"{shape}{box}_{colour.lstrip('#')}"
            for scale, suffix in ((1, ""), (2, "@2x")):
                path = folder / f"{stem}{suffix}.png"
                if not path.exists():
                    k = scale * box / 14
                    img = QImage(box * scale, box * scale, QImage.Format.Format_ARGB32)
                    img.fill(Qt.GlobalColor.transparent)
                    painter = QPainter(img)
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                    pen = QPen(QColor(colour), 1.6 * scale)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                    painter.setPen(pen)
                    painter.drawPolyline(QPolygonF([QPointF(x * k, y * k) for x, y in points]))
                    painter.end()
                    _save_image(path, img)
            out[f"{shape}{box}"] = (folder / f"{stem}.png").as_posix()
    return out


def _indicator_stylesheet(name: str, t: dict[str, str]) -> str:
    i = _indicator_images(name, t)
    box = "QCheckBox::indicator, QGroupBox::indicator"
    row = "QAbstractItemView::indicator"  # smaller: list rows are only ~15 px tall
    return f"""
{box}, {row} {{ width: 14px; height: 14px; border: 1px solid {t["text_muted"]}; border-radius: 3px;
    background: {t["surface"]}; }}
{row} {{ width: 12px; height: 12px; border-radius: 2px; }}
QCheckBox::indicator:hover, {row}:hover {{ border-color: {t["accent"]}; }}
QCheckBox::indicator:checked, QGroupBox::indicator:checked {{
    background: {t["accent"]}; border-color: {t["accent"]}; image: url({i["check14"]}); }}
{row}:checked {{ background: {t["accent"]}; border-color: {t["accent"]}; image: url({i["check12"]}); }}
QCheckBox::indicator:indeterminate {{
    background: {t["accent"]}; border-color: {t["accent"]}; image: url({i["dash14"]}); }}
{row}:indeterminate {{ background: {t["accent"]}; border-color: {t["accent"]}; image: url({i["dash12"]}); }}
QCheckBox::indicator:disabled, {row}:disabled {{
    border-color: {t["text_faint"]}; background: {t["surface_alt"]}; }}
QCheckBox::indicator:checked:disabled, {row}:checked:disabled {{
    background: {t["accent_disabled"]}; border-color: {t["accent_disabled"]}; }}
QRadioButton::indicator {{ width: 14px; height: 14px; border: 1px solid {t["text_muted"]};
    border-radius: 8px; background: {t["surface"]}; }}
QRadioButton::indicator:hover {{ border-color: {t["accent"]}; }}
QRadioButton::indicator:checked {{ border: 1px solid {t["accent"]};
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 {t["on_accent"]}, stop:0.38 {t["on_accent"]}, stop:0.46 {t["accent"]}, stop:1 {t["accent"]}); }}
QRadioButton::indicator:disabled {{ border-color: {t["text_faint"]}; background: {t["surface_alt"]}; }}
QMenu::indicator {{ width: 14px; height: 14px; margin-left: 6px; }}
QMenu::indicator:checked {{ background: {t["accent"]}; border-radius: 3px; image: url({i["check14"]}); }}
"""


def _arrow_stylesheet(name: str, t: dict[str, str]) -> str:
    a = _arrow_images(name, t)
    return f"""
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 18px;
                        border: none; border-left: 1px solid {t["border"]}; }}
QComboBox::down-arrow {{ image: url({a["down"]}); width: 10px; height: 10px; }}
QComboBox::down-arrow:disabled {{ image: url({a["down_disabled"]}); }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border; width: 16px;
    border: none; border-left: 1px solid {t["border"]}; background: {t["surface_alt"]}; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 4px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 4px; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {t["accent_soft"]}; }}
QAbstractSpinBox::up-arrow {{ image: url({a["up"]}); width: 8px; height: 8px; }}
QAbstractSpinBox::down-arrow {{ image: url({a["down"]}); width: 8px; height: 8px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{ image: url({a["up_disabled"]}); }}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{ image: url({a["down_disabled"]}); }}
QDateTimeEdit::drop-down {{ width: 18px; border: none; border-left: 1px solid {t["border"]}; }}
QDateTimeEdit::down-arrow {{ image: url({a["down"]}); width: 10px; height: 10px; }}
QMenu::right-arrow {{ image: url({a["right"]}); }}
QTabBar::scroller {{ width: 40px; }}
QTabBar QToolButton {{ background: {t["surface"]}; border: 1px solid {t["border"]}; border-radius: 0;
                       margin: 3px 0 3px 0; padding: 0; }}
QTabBar QToolButton:hover {{ background: {t["accent_soft"]}; }}
QTabBar QToolButton::left-arrow {{ image: url({a["left"]}); width: 8px; height: 8px; }}
QTabBar QToolButton::right-arrow {{ image: url({a["right"]}); width: 8px; height: 8px; }}
QTabBar QToolButton::left-arrow:disabled {{ image: url({a["left_disabled"]}); }}
QTabBar QToolButton::right-arrow:disabled {{ image: url({a["right_disabled"]}); }}
"""


def apply_theme(app: QApplication, name: str = LIGHT) -> None:
    """Make ``name`` the current theme: tokens, palette, fonts and chart defaults.

    The stylesheet is *not* set on the application: Qt re-polishes every widget of every
    window when the application stylesheet changes (about a second with all pages built).
    The main window carries it instead (:func:`stylesheet`; dialogs it opens inherit it),
    which makes a theme switch about five times faster. Tooltips and the few unparented
    widgets follow the palette.
    """
    global _current
    name = name if name in THEMES else LIGHT
    _current = name
    TOKENS.clear()
    TOKENS.update(THEMES[name])
    if app.style().name().lower() != "fusion":
        app.setStyle("Fusion")
    app.setPalette(_palette(TOKENS))
    if not _font_set(app):
        font = QFont()
        font.setFamilies(
            ["Segoe UI Variable Text", "Segoe UI", "Inter", "Noto Sans", "DejaVu Sans", "Arial"]
        )
        font.setPointSizeF(9.0)
        app.setFont(font)
    if app.styleSheet():
        app.setStyleSheet("")
    # New charts pick these up; existing ones are recoloured by refresh_widgets().
    # (Antialiasing stays off: large trends stay fast.)
    pg.setConfigOptions(foreground=TOKENS["chart_fg"], background=TOKENS["chart_bg"])


def _font_set(app: QApplication) -> bool:
    return "Segoe UI Variable Text" in app.font().families()


def stylesheet() -> str:
    """The current theme's stylesheet, for the main window (see apply_theme)."""
    base = build_stylesheet(TOKENS)
    try:
        return base + _arrow_stylesheet(_current, TOKENS) + _indicator_stylesheet(_current, TOKENS)
    except OSError as exc:  # TEMP unavailable or read-only: Qt draws default arrows/indicators
        log.warning("Theme image assets unavailable (%s); using the unstyled fallback", exc)
        return base


def refresh_widgets(root: QWidget) -> None:
    """After a theme switch: recolour charts and every widget that caches colours.

    Charts: background, axes and grid of every PyQtGraph view under ``root``. Widgets and chart
    items that draw with cached colours (icons, headers, messages) define ``apply_theme()``.
    """
    for widget in [root, *root.findChildren(QWidget)]:
        if isinstance(widget, pg.GraphicsView):
            widget.setBackground(TOKENS["chart_bg"])
            for item in widget.scene().items():
                if isinstance(item, pg.AxisItem):
                    item.setPen(TOKENS["chart_fg"])
                    item.setTextPen(TOKENS["chart_fg"])
                elif hasattr(item, "apply_theme"):
                    item.apply_theme()
        apply = getattr(widget, "apply_theme", None)
        if callable(apply):
            apply()


def chart_background() -> str:
    return TOKENS["chart_bg"]


def chart_foreground() -> str:
    return TOKENS["chart_fg"]


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
    # Escaped: messages quote user input and operators ("Feed_Flow <: value is required").
    return f"<span style='color:{TOKENS[token]};{weight}'>{escape(text)}</span>"
