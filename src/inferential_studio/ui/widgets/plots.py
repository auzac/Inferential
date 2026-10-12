"""Chart helpers shared by the pages (PyQtGraph): themed charts, empty states, series key.

Data colours are mid-tones that read on both the light and the dark chart background, so a
theme switch only recolours the chart chrome (background, axes, header), never the data.
"""

from __future__ import annotations

from html import escape

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextOption
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from inferential_studio.core.splits import PURGED, TEST, TRAIN, VALIDATION
from inferential_studio.ui.theme import TOKENS
from inferential_studio.ui.widgets.chart_menu import ChartViewBox, tidy_plot
from inferential_studio.ui.widgets.time_axis import TzDateAxisItem

# Colour-blind-safe series colours (Okabe–Ito hues, lightness adjusted so every colour keeps
# contrast on both the light and the dark chart background).
BLUE = "#2F8FD8"
ORANGE = "#E69F00"
GREEN = "#14A37F"
VERMILION = "#E0632B"
PINK = "#C978A9"
SKY = "#5BB7E8"
OLIVE = "#C9B23A"
GREY = "#8C8C8C"  # reference lines, lab values shown for comparison
SERIES = [BLUE, ORANGE, GREEN, VERMILION, PINK, SKY, OLIVE, GREY]
FAINT = "#8C8C8C66"  # de-emphasised points (e.g. not eligible)

SPLIT_COLOURS = {
    TRAIN: BLUE,
    VALIDATION: ORANGE,
    TEST: VERMILION,
    PURGED: "#9A9A9A",
    "not eligible": FAINT,
}
SPLIT_NAMES = {TRAIN: "Training", VALIDATION: "Validation", TEST: "Final test"}
AXIS_WIDTH = 72
POSITIVE = BLUE
NEGATIVE = VERMILION


class ValueAxis(pg.AxisItem):
    """Value axis whose tick labels never overlap.

    PyQtGraph always labels every major tick, so on short charts (stacked trends) labels can
    be drawn on top of each other. Ticks and grid lines are unchanged; a label that would
    overlap one already placed is skipped (major-level labels are placed first, so they win).
    """

    def generateDrawSpecs(self, p):  # noqa: ANN001, ANN201, N802 - PyQtGraph override
        specs = super().generateDrawSpecs(p)
        if not specs:
            return specs
        axis_spec, tick_specs, text_specs = specs
        kept: list = []
        for spec in text_specs:
            if not any(spec[0].intersects(other[0]) for other in kept):
                kept.append(spec)
        return axis_spec, tick_specs, kept


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
    """Plot with an automatic empty state and a key in the header instead of a legend box.

    Empty state: :meth:`clear` (or a new chart) shows a short centred message and hides axes,
    grid and key; adding the first item that carries data restores them. Pages set the message
    with :meth:`set_empty_text` to say what is missing ("Train models first.").

    Key: after :meth:`enable_key` (see :func:`legend`), every named item gets a colour mark and
    its name on the chart's title line, left-aligned, so no legend ever covers the data.
    """

    def __init__(self, *args, empty_text: str = EMPTY_TEXT, **kwargs) -> None:  # noqa: ANN002, ANN003
        kwargs.setdefault("viewBox", ChartViewBox())
        super().__init__(*args, **kwargs)
        tidy_plot(self)
        self._grid: tuple = (None, None, None)
        self._mouse = (True, True)
        self._axes_shown: dict[str, bool] = {}
        self._message = pg.TextItem(empty_text, color=TOKENS["text_muted"], anchor=(0.5, 0.5))
        self._message.setPos(0.5, 0.5)
        option = QTextOption(Qt.AlignmentFlag.AlignHCenter)
        option.setWrapMode(QTextOption.WrapMode.WordWrap)
        self._message.textItem.document().setDefaultTextOption(option)
        self.vb.sigResized.connect(self._fit_message)
        self._title = ""
        self._title_args: dict = {}
        self._key_enabled = False
        self._key: list[tuple[int, str, str, str]] = []  # (id(item), mark, colour, name)
        self._empty = False
        self._set_empty(True)

    # -- public ---------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return self._empty

    @property
    def empty_text(self) -> str:
        return self._message.toPlainText()

    def apply_theme(self) -> None:
        """After a theme switch: message and header colours (data colours stay)."""
        self._message.setColor(TOKENS["text_muted"])
        self._render_header()

    def set_empty_text(self, text: str) -> None:
        self._message.setText(text)
        self._fit_message()

    def enable_key(self) -> None:
        """Show named items as a key on the title line (call once, before plotting)."""
        self._key_enabled = True
        self._render_header()

    @property
    def key_names(self) -> list[str]:
        return [name for _id, _mark, _colour, name in self._key]

    # -- PlotItem overrides ---------------------------------------------------

    def showGrid(self, x=None, y=None, alpha=None) -> None:  # noqa: ANN001, N802
        self._grid = (x, y, alpha)
        if not self._empty:
            super().showGrid(x, y, alpha)

    def showAxis(self, axis: str, show: bool = True) -> None:  # noqa: N802
        # setLabel() shows its axis: while empty, remember the wish and keep it hidden.
        if getattr(self, "_empty", False) and axis in ("left", "bottom"):
            self._axes_shown[axis] = show
            super().showAxis(axis, False)
            return
        super().showAxis(axis, show)

    def setTitle(self, title=None, **args) -> None:  # noqa: ANN001, ANN003, N802
        if not hasattr(self, "_key"):  # PlotItem.__init__ hides the title before we are set up
            super().setTitle(title, **args)
            return
        self._title = title or ""
        self._title_args = args
        self._render_header()

    def addItem(self, item, *args, **kwargs) -> None:  # noqa: ANN001, ANN002, ANN003, N802
        super().addItem(item, *args, **kwargs)
        if item is self._message:
            return
        if self._key_enabled:
            entry = _key_entry(item)
            if entry is not None and entry[2] not in {n for *_x, n in self._key}:
                self._key.append((id(item), *entry))
                self._render_header()
        if self._empty and _has_data(item):
            self._set_empty(False)

    def removeItem(self, item) -> None:  # noqa: ANN001, N802
        super().removeItem(item)
        if any(i == id(item) for i, *_x in self._key):
            self._key = [k for k in self._key if k[0] != id(item)]
            self._render_header()

    def clear(self) -> None:
        was_empty = self._empty
        super().clear()  # also removes the message item
        self._key = []
        if was_empty:  # still empty: keep the saved axis state, put the message back
            super().addItem(self._message, ignoreBounds=True)
            self.vb.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
        else:
            self._set_empty(True)
        self._render_header()

    # -- internals ------------------------------------------------------------

    def _fit_message(self) -> None:
        """Wrap the empty-state message to the plot width (it never runs off the edges)."""
        width = self.vb.width()
        if width > 0:
            self._message.setTextWidth(max(width - 32, 120))

    def _render_header(self) -> None:
        """Title line: the chart title (left-aligned, semibold) followed by the key, if any."""
        key = self._key if self._key_enabled and not self._empty else []
        if not key and not self._title:
            super().setTitle(None)
            return
        parts = []
        if self._title:
            parts.append(
                f"<span style='font-size:9.5pt; font-weight:600'>{escape(self._title)}</span>"
            )
        parts += [
            f"<span style='color:{colour}'>{mark}</span>&nbsp;{escape(name)}"
            for _id, mark, colour, name in key
        ]
        super().setTitle(
            "&nbsp;&nbsp;&nbsp;&nbsp;".join(parts), color=TOKENS["text"], size="8pt", justify="left"
        )

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
            super().addItem(self._message, ignoreBounds=True)
            self.vb.setMouseEnabled(False, False)
            self.vb.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
        else:
            self.removeItem(self._message)
            for side, shown in self._axes_shown.items():
                self.getAxis(side).setVisible(shown)
            if any(v is not None for v in self._grid):
                super().showGrid(*self._grid)
            self.vb.setMouseEnabled(*self._mouse)
            self.enableAutoRange()
        self._render_header()


def _colour_name(spec: object) -> str:
    return pg.mkColor(spec).name() if spec is not None else TOKENS["text"]


def _key_entry(item: object) -> tuple[str, str, str] | None:
    """(mark, colour, name) for a named data item; None for unnamed or decoration items."""
    name_of = getattr(item, "name", None)
    name = name_of() if callable(name_of) else None
    if not name:
        return None
    if isinstance(item, pg.PlotDataItem):
        o = item.opts
        if o.get("pen") is None and o.get("symbol") is not None:
            return "●", _colour_name(pg.mkBrush(o.get("symbolBrush")).color()), name
        if o.get("pen") is None and o.get("fillLevel") is not None:
            return "■", _colour_name(pg.mkBrush(o.get("fillBrush")).color()), name
        pen = pg.mkPen(o.get("pen"))
        dashed = pen.style() in (Qt.PenStyle.DashLine, Qt.PenStyle.DotLine)
        return ("┄" if dashed else "━"), _colour_name(pen.color()), name
    if isinstance(item, pg.ScatterPlotItem):
        return "●", _colour_name(pg.mkBrush(item.opts.get("brush")).color()), name
    if isinstance(item, pg.BarGraphItem):
        brush = item.opts.get("brush") or (item.opts.get("brushes") or [None])[0]
        return "■", _colour_name(pg.mkBrush(brush).color()), name
    return None


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
    """Themed plot with a grid, the standard right-click menu and an automatic empty state.

    ``time_axis`` = display timezone for a date axis at the bottom; ``name`` = file-name stem
    offered by *Save image…* (defaults to the title); ``empty_text`` = message while empty.
    """
    axes: dict[str, pg.AxisItem] = {"left": ValueAxis("left")}
    if time_axis:
        axes["bottom"] = TzDateAxisItem(time_axis)
    plot_item = ChartPlotItem(axisItems=axes, empty_text=empty_text)
    w = pg.PlotWidget(background=TOKENS["chart_bg"], plotItem=plot_item)
    if name:
        w.setProperty("chartName", name)
    w.showGrid(x=True, y=True, alpha=0.15)
    for side in ("left", "bottom"):
        w.getAxis(side).enableAutoSIPrefix(False)  # show values as they are, no "(x0.001)"
    w.getAxis("left").setWidth(AXIS_WIDTH)
    if title:
        w.setTitle(title)
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


def legend(plot: pg.PlotWidget, corner: str = "top-left") -> None:  # noqa: ARG001
    """Name the plotted series in the chart header (a key that never covers the data).

    ``corner`` is accepted for older callers and ignored: the key always sits on the title line.
    """
    chart_item(plot).enable_key()


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
        plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(GREY)))
    else:
        plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(GREY)))
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
