"""Small plotting helpers shared by the modelling pages (PyQtGraph, white background)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from inferential_studio.core.splits import PURGED, TEST, TRAIN, VALIDATION
from inferential_studio.ui.widgets.time_axis import TzDateAxisItem

SPLIT_COLOURS = {
    TRAIN: "#0072B2",
    VALIDATION: "#E69F00",
    TEST: "#D55E00",
    PURGED: "#999999",
    "not eligible": "#CCCCCC",
}
SPLIT_NAMES = {TRAIN: "Training", VALIDATION: "Validation", TEST: "Final test"}
AXIS_WIDTH = 72
POSITIVE = "#0072B2"
NEGATIVE = "#D55E00"


def plot_widget(title: str = "", *, time_axis: str | None = None) -> pg.PlotWidget:
    """White plot with a grid; ``time_axis`` = display timezone for a date axis at the bottom."""
    axis = {"bottom": TzDateAxisItem(time_axis)} if time_axis else None
    w = pg.PlotWidget(background="w", axisItems=axis)
    w.showGrid(x=True, y=True, alpha=0.15)
    for name in ("left", "bottom"):
        w.getAxis(name).enableAutoSIPrefix(False)  # show values as they are, no "(x0.001)"
    w.getAxis("left").setWidth(AXIS_WIDTH)
    if title:
        w.setTitle(title, color="#333333", size="10pt")
    return w


def legend(plot: pg.PlotWidget, corner: str = "top-left") -> pg.LegendItem:
    """Legend with an opaque background; ``corner`` keeps it clear of where the data usually is."""
    offsets = {"top-left": (5, 5), "top-right": (-5, 5), "bottom-right": (-5, -5)}
    return plot.addLegend(
        offset=offsets[corner], labelTextSize="8pt", brush=(255, 255, 255, 230), pen="#cccccc"
    )


def scatter(plot: pg.PlotItem | pg.PlotWidget, x, y, colour: str, name: str | None = None,  # noqa: ANN001
            size: int = 6, symbol: str = "o") -> pg.PlotDataItem:  # fmt: skip
    c = pg.mkColor(colour)
    c.setAlpha(190)
    return plot.plot(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float), pen=None, symbol=symbol,
        symbolSize=size, symbolBrush=pg.mkBrush(c), symbolPen=None, name=name,
    )  # fmt: skip


def bar_chart(
    plot: pg.PlotWidget, labels: list[str], values: np.ndarray, *, horizontal: bool = True
) -> None:
    """Bars coloured by sign; category labels on the axis (horizontal bars read better)."""
    plot.clear()
    values = np.asarray(values, dtype=float)
    pos = np.arange(len(values))
    brushes = [pg.mkBrush(POSITIVE if v >= 0 else NEGATIVE) for v in np.nan_to_num(values)]
    if horizontal:
        item = pg.BarGraphItem(
            x0=0, y=pos, height=0.7, width=np.nan_to_num(values), brushes=brushes
        )
        left = plot.getAxis("left")
        left.setTicks([[(i, n) for i, n in enumerate(labels)]])
        left.setWidth(None)  # size to the longest label instead of the fixed trend width
        left.autoExpandTextSpace = True
        plot.getAxis("bottom").setTicks(None)
        plot.getPlotItem().invertY(True)
    else:
        item = pg.BarGraphItem(x=pos, height=np.nan_to_num(values), width=0.7, brushes=brushes)
        plot.getAxis("bottom").setTicks([[(i, n) for i, n in enumerate(labels)]])
        plot.getAxis("left").setTicks(None)
    plot.addItem(item)
    if horizontal:
        plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#666666")))
    else:
        plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#666666")))
    plot.autoRange()


def note_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("placeholderNote")
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def with_note(widget: QWidget, note: QLabel) -> QWidget:
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget, 1)
    layout.addWidget(note)
    return box
