"""Workflow progress: one short segment per stage in the page header.

Done stages are filled with the accent, stages started but not finished with a soft accent,
the rest stay faint; the current stage is drawn taller. Hover names a stage and what it still
needs; a click opens it.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QToolTip, QWidget

from inferential_studio.ui.stages import ATTENTION, DONE, STAGES, TODO
from inferential_studio.ui.theme import TOKENS

SEGMENT = 22
GAP = 4
HEIGHT = 16  # hit area; the bars are thinner
BAR = 4
CURRENT_BAR = 8
COLOUR = {DONE: "accent", ATTENTION: "accent_disabled", TODO: "border_strong"}


class StageProgress(QWidget):
    stage_clicked = Signal(int)

    def __init__(self, current: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current = current
        self._states: list[tuple[str, str]] = [(TODO, "")] * len(STAGES)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(self.sizeHint())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(len(STAGES) * (SEGMENT + GAP) - GAP, HEIGHT)

    def set_states(self, states: list[tuple[str, str]]) -> None:
        if states != self._states:
            self._states = states
            self.update()

    def done_count(self) -> int:
        return sum(state == DONE for state, _ in self._states)

    def _index_at(self, x: float) -> int:
        i = int(x // (SEGMENT + GAP))
        return i if 0 <= i < len(STAGES) else -1

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        for i, (state, _note) in enumerate(self._states):
            h = CURRENT_BAR if i == self._current else BAR
            p.setBrush(QColor(TOKENS[COLOUR[state]]))
            rect = QRectF(i * (SEGMENT + GAP), (HEIGHT - h) / 2, SEGMENT, h)
            p.drawRoundedRect(rect, h / 2, h / 2)
        p.end()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        i = self._index_at(event.position().x())
        if i < 0:
            QToolTip.hideText()
            return
        stage, note = STAGES[i], self._states[i][1]
        QToolTip.showText(
            event.globalPosition().toPoint(),
            f"<b>{stage.number}. {stage.title}</b><br>{note}",
            self,
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        i = self._index_at(event.position().x())
        if i >= 0 and event.button() is Qt.MouseButton.LeftButton:
            self.stage_clicked.emit(i)
