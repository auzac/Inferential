"""Chart overlays for exclusions and flags that stay fast with thousands of intervals.

Instead of one graphics item per excluded interval, :class:`IntervalShade` is a single item per
plot that paints only the intervals inside the visible range and merges those closer than a
pixel, so drawing cost depends on screen width, not on the number of exclusions.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtGui import QBrush, QColor, QPainter

from inferential_studio.core.rules import runs


def mask_spans(mask: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[start, end] x-spans (epoch s) covering True runs, padded by half a sample each side."""
    starts, ends = runs(np.asarray(mask, dtype=bool))
    if starts.size == 0:
        return np.zeros(0), np.zeros(0)
    half = float(np.median(np.diff(x))) / 2 if len(x) > 1 else 30.0
    return x[starts] - half, x[ends] + half


class IntervalShade(pg.GraphicsObject):
    def __init__(self, starts: np.ndarray, ends: np.ndarray, colour: str, alpha: int = 70) -> None:
        super().__init__()
        order = np.argsort(starts)
        self._starts = np.asarray(starts, dtype=float)[order]
        self._ends = np.asarray(ends, dtype=float)[order]
        self._max_end = np.maximum.accumulate(self._ends) if self._ends.size else self._ends
        c = QColor(colour)
        c.setAlpha(alpha)
        self._brush = QBrush(c)
        self.setZValue(-20)
        self._vb = None

    def attach(self, plot: pg.PlotItem) -> None:
        plot.addItem(self, ignoreBounds=True)
        self._vb = plot.vb
        plot.vb.sigRangeChanged.connect(self._range_changed)

    def _range_changed(self, *_args) -> None:
        self.prepareGeometryChange()
        self.update()

    @property
    def count(self) -> int:
        return int(self._starts.size)

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt override
        if self._vb is None:
            return QRectF()
        return self._vb.viewRect()

    def paint(self, painter: QPainter, *_args) -> None:
        if self._vb is None or self._starts.size == 0:
            return
        view = self._vb.viewRect()
        x0, x1 = view.left(), view.right()
        px = self.pixelWidth() or (x1 - x0) / 1000.0
        first = int(np.searchsorted(self._max_end, x0, side="left"))
        last = int(np.searchsorted(self._starts, x1, side="right"))
        painter.setPen(pg.mkPen(None))
        painter.setBrush(self._brush)
        cur_s = cur_e = None
        for s, e in zip(self._starts[first:last], self._ends[first:last], strict=True):
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s - cur_e <= px:  # closer than a pixel: draw as one rectangle
                cur_e = max(cur_e, e)
            else:
                painter.drawRect(QRectF(cur_s, view.top(), max(cur_e - cur_s, px), view.height()))
                cur_s, cur_e = s, e
        if cur_s is not None:
            painter.drawRect(QRectF(cur_s, view.top(), max(cur_e - cur_s, px), view.height()))
