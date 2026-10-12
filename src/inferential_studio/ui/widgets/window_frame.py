"""Frameless-window support: resize grips on the window edges and Windows 11 corner rounding.

The main window draws its own title bar (``title_bar.py``). Moving and resizing still go
through the operating system (``startSystemMove`` / ``startSystemResize``), so snapping,
multi-monitor moves and DPI changes behave like any other window. Users who prefer the
system's own frame can switch it back on in Settings (it applies after a restart).
"""

from __future__ import annotations

import ctypes
import logging
import sys

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)

GRIP = 6  # px of each edge that resizes the window

_EDGES = {
    "left": (Qt.Edge.LeftEdge, Qt.CursorShape.SizeHorCursor),
    "right": (Qt.Edge.RightEdge, Qt.CursorShape.SizeHorCursor),
    "top": (Qt.Edge.TopEdge, Qt.CursorShape.SizeVerCursor),
    "bottom": (Qt.Edge.BottomEdge, Qt.CursorShape.SizeVerCursor),
    "top_left": (Qt.Edge.TopEdge | Qt.Edge.LeftEdge, Qt.CursorShape.SizeFDiagCursor),
    "bottom_right": (Qt.Edge.BottomEdge | Qt.Edge.RightEdge, Qt.CursorShape.SizeFDiagCursor),
    "top_right": (Qt.Edge.TopEdge | Qt.Edge.RightEdge, Qt.CursorShape.SizeBDiagCursor),
    "bottom_left": (Qt.Edge.BottomEdge | Qt.Edge.LeftEdge, Qt.CursorShape.SizeBDiagCursor),
}


class _Grip(QWidget):
    def __init__(self, edges: Qt.Edge, cursor: Qt.CursorShape, parent: QWidget) -> None:
        super().__init__(parent)
        self.edges = edges
        self.setCursor(cursor)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None and handle.startSystemResize(self.edges):
                event.accept()
                return
        super().mousePressEvent(event)


class EdgeGrips:
    """Eight invisible strips along the edges and corners of a frameless window."""

    def __init__(self, window: QWidget) -> None:
        self._window = window
        self.grips = {
            name: _Grip(edges, cursor, window) for name, (edges, cursor) in _EDGES.items()
        }
        self._enabled = True

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.update_geometry()

    def update_geometry(self) -> None:
        w, h = self._window.width(), self._window.height()
        corner = GRIP * 2
        rects = {
            "left": QRect(0, corner, GRIP, h - 2 * corner),
            "right": QRect(w - GRIP, corner, GRIP, h - 2 * corner),
            "top": QRect(corner, 0, w - 2 * corner, GRIP),
            "bottom": QRect(corner, h - GRIP, w - 2 * corner, GRIP),
            "top_left": QRect(0, 0, corner, corner),
            "top_right": QRect(w - corner, 0, corner, corner),
            "bottom_left": QRect(0, h - corner, corner, corner),
            "bottom_right": QRect(w - corner, h - corner, corner, corner),
        }
        for name, grip in self.grips.items():
            grip.setGeometry(rects[name])
            grip.setVisible(self._enabled)
            if self._enabled:
                grip.raise_()


def round_windows_corners(window: QWidget, border: str) -> bool:
    """Windows 11: rounded corners and a themed 1 px border drawn by the system.

    Returns True when the system draws them (Windows 11); False elsewhere (Windows 10, other
    platforms), where the window draws its own border instead.
    """
    if sys.platform != "win32":
        return False
    try:
        hwnd = int(window.winId())
        dwmapi = ctypes.windll.dwmapi  # type: ignore[attr-defined]
        dwmwa_window_corner_preference, dwmwcp_round = 33, 2
        preference = ctypes.c_int(dwmwcp_round)
        result = dwmapi.DwmSetWindowAttribute(
            hwnd,
            dwmwa_window_corner_preference,
            ctypes.byref(preference),
            ctypes.sizeof(preference),
        )
        if result != 0:
            return False
        set_windows_border(window, border)
        return True
    except (AttributeError, OSError, ValueError) as exc:  # not Windows 11 / no DWM
        log.debug("Window corner rounding unavailable: %s", exc)
        return False


def set_windows_border(window: QWidget, colour: str) -> None:
    """Windows 11: the colour of the system-drawn window border (COLORREF is 0x00BBGGRR)."""
    if sys.platform != "win32":
        return
    try:
        c = QColor(colour)
        colorref = ctypes.c_uint(c.red() | (c.green() << 8) | (c.blue() << 16))
        dwmwa_border_color = 34
        ctypes.windll.dwmapi.DwmSetWindowAttribute(  # type: ignore[attr-defined]
            int(window.winId()), dwmwa_border_color, ctypes.byref(colorref), ctypes.sizeof(colorref)
        )
    except (AttributeError, OSError, ValueError) as exc:
        log.debug("Window border colour unavailable: %s", exc)
