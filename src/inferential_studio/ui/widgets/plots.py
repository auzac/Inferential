"""Small plotting helpers shared by the modelling pages (PyQtGraph, white background)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from inferential_studio.core.splits import PURGED, TEST, TRAIN, VALIDATION
from inferential_studio.ui.widgets.chart_menu import ChartViewBox, tidy_plot
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
TEXT_MUTED = "#596675"  # theme text_muted (kept literal: widgets do not import the theme)
POSITIVE = "#0072B2"
NEGATIVE = "#D55E00"


EMPTY_TEXT = "No data to show yet."
_DATA_ITEMS = (
    pg.PlotDataItem,
    pg.PlotCurveItem,
    pg.ScatterPlotItem,
    pg.BarGraphItem,
    pg.ImageItem,
    pg.ErrorBarItem,
)


class ChartPlotItem(pg.PlotItem):
    """Plot that shows a short centred message instead of bare axes while it holds no data.

    The empty state is automatic: :meth:`clear` (or a new chart) shows the message and hides
    axes, grid and legend; adding the first item that carries data restores them. Pages set
    the message with :meth:`set_empty_text` to say what is missing ("Train models first.").
    """

    def __init__(self, *args, empty_text: str = EMPTY_TEXT, **kwargs) -> None:  # noqa: ANN002, ANN003
        kwargs.setdefault("viewBox", ChartViewBox())
        super().__init__(*args, **kwargs)
        tidy_plot(self)
        self._grid: tuple = (None, None, None)
        self._mouse = (True, True)
        self._axes_shown: dict[str, bool] = {}
        self._message = pg.TextItem(empty_text, color=TEXT_MUTED, anchor=(0.5, 0.5))
        self._message.setPos(0.5, 0.5)
        self._empty = False
        self._set_empty(True)

    # -- public ---------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return self._empty

    @property
    def empty_text(self) -> str:
        return self._message.toPlainText()

    def set_empty_text(self, text: str) -> None:
        self._message.setText(text)

    # -- PlotItem overrides ---------------------------------------------------

    def showGrid(self, x=None, y=None, alpha=None) -> None:  # noqa: ANN001, N802
        self._grid = (x, y, alpha)
        if not self._empty:
            super().showGrid(x, y, alpha)

    def addLegend(self, *args, **kwargs) -> pg.LegendItem:  # noqa: ANN002, ANN003, N802
        legend = super().addLegend(*args, **kwargs)
        legend.setVisible(not self._empty)
        return legend

    def addItem(self, item, *args, **kwargs) -> None:  # noqa: ANN001, ANN002, ANN003, N802
        super().addItem(item, *args, **kwargs)
        if self._empty and item is not self._message and _has_data(item):
            self._set_empty(False)

    def clear(self) -> None:
        super().clear()  # also removes the message item
        self._empty = False
        self._set_empty(True)

    # -- internals ------------------------------------------------------------

    def _set_empty(self, empty: bool) -> None:
        if empty == self._empty:
            return
        self._empty = empty
        self.vb.empty = empty  # the chart menu stays closed on an empty chart
        if empty:
            self._mouse = tuple(self.vb.state["mouseEnabled"])
            self._axes_shown = {s: self.getAxis(s).isVisible() for s in ("left", "bottom")}
            for side in ("left", "bottom"):
                self.getAxis(side).hide()
            super().showGrid(False, False)
            if self.legend is not None:
                self.legend.setVisible(False)
            super().addItem(self._message, ignoreBounds=True)
            self.vb.setMouseEnabled(False, False)
            self.vb.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
        else:
            self.removeItem(self._message)
            for side, shown in self._axes_shown.items():
                self.getAxis(side).setVisible(shown)
            if any(v is not None for v in self._grid):
                super().showGrid(*self._grid)
            if self.legend is not None:
                self.legend.setVisible(True)
            self.vb.setMouseEnabled(*self._mouse)
            self.enableAutoRange()


def _has_data(item: object) -> bool:
    if not isinstance(item, _DATA_ITEMS):
        return False
    if isinstance(item, (pg.PlotDataItem, pg.PlotCurveItem)):
        return item.xData is not None and len(item.xData) > 0
    if isinstance(item, pg.ScatterPlotItem):
        return len(item.data) > 0
    if isinstance(item, pg.ImageItem):
        return item.image is not None
    return True


def plot_widget(
    title: str = "",
    *,
    time_axis: str | None = None,
    name: str | None = None,
    empty_text: str = EMPTY_TEXT,
) -> pg.PlotWidget:
    """White plot with a grid, the standard right-click menu and an automatic empty state.

    ``time_axis`` = display timezone for a date axis at the bottom; ``name`` = file-name stem
    offered by *Save image…* (defaults to the title); ``empty_text`` = message while empty.
    """
    axis = {"bottom": TzDateAxisItem(time_axis)} if time_axis else None
    w = pg.PlotWidget(background="w", plotItem=ChartPlotItem(axisItems=axis, empty_text=empty_text))
    if name:
        w.setProperty("chartName", name)
    w.showGrid(x=True, y=True, alpha=0.15)
    for side in ("left", "bottom"):
        w.getAxis(side).enableAutoSIPrefix(False)  # show values as they are, no "(x0.001)"
    w.getAxis("left").setWidth(AXIS_WIDTH)
    if title:
        w.setTitle(title, color="#333333", size="10pt")
    return w


def chart_item(plot: pg.PlotWidget) -> ChartPlotItem:
    return plot.getPlotItem()


def show_placeholder(plot: pg.PlotWidget, text: str) -> None:
    """Empty the chart and show ``text`` until data is plotted again."""
    chart_item(plot).set_empty_text(text)
    plot.clear()


def clear_placeholder(plot: pg.PlotWidget) -> None:
    """Empty the chart before plotting new data (the message goes away with the first data)."""
    plot.clear()


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
