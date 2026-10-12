"""Light/dark theme switch for the title bar: sun, a sliding toggle, moon (checked = dark)."""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPointF, QRectF, QSize, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QAbstractButton, QWidget

from inferential_studio.ui.icons import icon
from inferential_studio.ui.theme import TOKENS

ICON = 16
TRACK_W, TRACK_H = 34, 18
GAP = 6


class ThemeSwitch(QAbstractButton):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("themeSwitch")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._pos = 0.0  # knob position 0 (light) … 1 (dark)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._move_knob)
        self.toggled.connect(self._on_toggled)
        self._update_tooltip()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(ICON + GAP + TRACK_W + GAP + ICON + 8, 30)

    def set_dark(self, dark: bool) -> None:
        """Set the state without animating (e.g. at start-up)."""
        self.blockSignals(True)
        self.setChecked(dark)
        self.blockSignals(False)
        self._anim.stop()
        self._pos = 1.0 if dark else 0.0
        self._update_tooltip()
        self.update()

    def apply_theme(self) -> None:
        self.update()

    # -- internals ----------------------------------------------------------------------

    def _on_toggled(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()
        self._update_tooltip()

    def _move_knob(self, value: float) -> None:
        self._pos = float(value)
        self.update()

    def _update_tooltip(self) -> None:
        self.setToolTip(
            "Switch to the light theme" if self.isChecked() else "Switch to the dark theme"
        )

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h = self.height()
        x = 4
        dark = self.isChecked()
        sun, moon = icon("weather_sunny", "text_muted"), icon("weather_moon", "text_muted")
        y_icon = (h - ICON) // 2
        p.setOpacity(1.0 if not dark else 0.55)
        sun.paint(p, x, y_icon, ICON, ICON)
        p.setOpacity(1.0)
        track = QRectF(x + ICON + GAP, (h - TRACK_H) / 2, TRACK_W, TRACK_H)
        on = QColor(TOKENS["accent"])
        off_border = QColor(TOKENS["text_muted"])
        if self._pos > 0.5:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(on)
        else:
            p.setPen(off_border)
            p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(track.adjusted(0.5, 0.5, -0.5, -0.5), TRACK_H / 2, TRACK_H / 2)
        knob_d = 12 if (self.underMouse() or self.isDown()) else 10
        travel = TRACK_W - 4 - 12
        cx = track.left() + 2 + 6 + travel * self._pos
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(TOKENS["on_accent"]) if self._pos > 0.5 else off_border)
        p.drawEllipse(QPointF(cx, track.center().y()), knob_d / 2, knob_d / 2)
        if self.hasFocus():
            p.setPen(QColor(TOKENS["accent"]))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(track.adjusted(-3, -3, 3, 3), TRACK_H / 2 + 3, TRACK_H / 2 + 3)
        p.setOpacity(1.0 if dark else 0.55)
        moon.paint(p, int(track.right()) + GAP, y_icon, ICON, ICON)
        p.end()

    def enterEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().leaveEvent(event)
        self.update()
