"""Synchronized time-series trends with a timezone-aware axis, crosshair and fast display.

Plots are added and removed one at a time (ticking a variable no longer rebuilds every chart),
and each line series draws from a precomputed min/max envelope while zoomed out, switching to
the raw samples of the visible window when zoomed in. The number of plots is capped so panning
stays fluid on large data.

Navigation is along the time axis only; each chart's y-axis fits the data on screen. The
mouse works like a selection tool instead of a pan tool: dragging highlights a period on every
chart and zooms to it on release, clicking pins the values at that point in time (emitted
through ``point_selected`` / ``point_cleared``), and the hover crosshair keeps emitting
``hover`` for the "At cursor" box below the variable selection. The wheel zooms and
middle-drag pans. **Back** steps out one drag-zoom; **Fit all** shows all data. Right-click
opens the standard chart menu (copy / save the charts as an image; see ``chart_menu``).
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass
from html import escape

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.ui.theme import TOKENS
from inferential_studio.ui.widgets.chart_menu import ChartViewBox, tidy_plot
from inferential_studio.ui.widgets.plots import BLUE, SERIES, ValueAxis
from inferential_studio.ui.widgets.time_axis import TzDateAxisItem, format_time

# Okabe–Ito colour-blind-safe palette (black replaced by dark grey).
PALETTE = SERIES  # shared series colours (readable on both themes)
AXIS_WIDTH = 72
MAX_PLOTS = 12  # charts shown at once; more than this is slow to pan and hard to read
BINS = 2048  # envelope bins per series: the resolution of the zoomed-out view
RAW_SPAN_BINS = 8  # zoomed in closer than this many bins: draw the raw samples instead
POINTS_MAX = 5000  # markers drawn for sparse point series (lab samples)
HIGHLIGHT_COLOUR = (47, 143, 216)  # drag-to-zoom band (BLUE), as the cleaning page's range
MIN_DRAG_PX = 6  # shorter drags are treated as clicks, not zooms
PIN_COLOUR = BLUE  # vertical line marking the pinned point in time
HEADER_HEIGHT = 16  # px of the per-chart name header
MIN_PLOT_HEIGHT = 120  # scroll stacked plots rather than crushing their axes on small screens


@dataclass
class TrendSeries:
    name: str
    x: np.ndarray  # UTC epoch seconds
    y: np.ndarray
    units: str | None = None
    points: bool = False  # draw as markers (sparse lab samples) instead of a line


class _TrendViewBox(ChartViewBox):
    """View box where the left button selects instead of panning.

    Left-drag highlights a period and zooms to it on release (handled by the owning
    :class:`TrendView`), left-click pins the point in time under the mouse and middle-drag
    pans along time. Right-click opens the standard chart menu; right-drag does nothing.
    """

    def __init__(self, view: TrendView) -> None:
        super().__init__()
        self._view = view
        self.setMouseEnabled(x=True, y=False)  # wheel zooms time; y follows the data
        self.setAutoVisible(y=True)

    def mouseDragEvent(self, ev, axis=None) -> None:  # noqa: ANN001 - pyqtgraph signature
        if ev.button() == Qt.MouseButton.LeftButton:
            self._view._band_drag(self, ev)
            ev.accept()
        elif ev.button() == Qt.MouseButton.MiddleButton:
            self._view._pan_drag(self, ev)
            ev.accept()
        else:
            ev.ignore()

    def mouseClickEvent(self, ev) -> None:  # noqa: ANN001 - pyqtgraph signature
        # NOTE: ViewBox.mouseClickEvent takes no axis parameter (0.14).
        if ev.button() == Qt.MouseButton.LeftButton:
            self._view._click_point(self, ev)
            ev.accept()
        else:
            super().mouseClickEvent(ev)  # right-click: the standard chart menu


class TrendView(QWidget):
    """Stacked or unit-grouped plots sharing one time axis."""

    x_range_changed = Signal(float, float)
    rebuilt = Signal()  # plots were added/removed/updated; owners re-add their overlays
    hover = Signal(str, dict)  # (time text, {series name: formatted value}) at the cursor
    point_selected = Signal(str, dict)  # like hover, but pinned by a click until the next click
    point_cleared = Signal()

    def __init__(self, timezone: str = "UTC", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tz = timezone
        self._series: list[TrendSeries] = []
        self.plots: list[pg.PlotItem] = []
        self._vlines: list[pg.InfiniteLine] = []
        self._plot_series: list[list[TrendSeries]] = []
        self._plot_keys: list[str] = []
        self._plot_by_key: dict[str, pg.PlotItem] = {}
        self._key_of_plot: dict[pg.PlotItem, str] = {}
        self._vline_of: dict[pg.PlotItem, pg.InfiniteLine] = {}
        self._sel_line_of: dict[pg.PlotItem, pg.InfiniteLine] = {}
        self._bands: list[tuple[pg.PlotItem, pg.LinearRegionItem]] = []
        self._drag_x0: float | None = None
        self._zoom_history: list[tuple[float, float] | None] = []  # None = fitted to all data
        self._selected_x: float | None = None  # pinned point in time (None while unlocked)
        # group key -> series name -> (line/min item, max item or None, (points, id(x), id(y)))
        self._items: dict[
            str, dict[str, tuple[pg.PlotDataItem, pg.PlotDataItem | None, tuple]]
        ] = {}
        self._colours: dict[str, str] = {}
        self._env_cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._synced_mode: str | None = None
        self._xrange_plot: pg.PlotItem | None = None
        self._xrange_conn = None

        self.mode = QComboBox()
        self.mode.addItem("One plot per signal", "stacked")
        self.mode.addItem("Group by units", "units")
        self.mode.setToolTip("Signals with different units always use separate y-axes.")
        self.mode.currentIndexChanged.connect(self._rebuild)
        self.back_button = QPushButton("Back")
        self.back_button.setToolTip("Return to the time range before the last drag-zoom")
        self.back_button.clicked.connect(self.zoom_back)
        self.back_button.setEnabled(False)
        self.fit_button = QPushButton("Fit all")
        self.fit_button.setToolTip("Show all data")
        self.fit_button.clicked.connect(self.fit_all)
        self.hint = QLabel("Drag to zoom, click to pin, right-click to copy or save")
        self.hint.setToolTip(
            "Drag across a chart to zoom to that period. The wheel zooms, middle-drag pans, a "
            "click pins the values at a time, and right-click copies or saves the charts."
        )
        self.hint.setObjectName("placeholderNote")
        # Clip rather than widen the toolbar when the window is narrow.
        self.hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        bar = QHBoxLayout()
        self.toolbar_layout = bar
        bar.addWidget(self.mode)
        bar.addWidget(self.back_button)
        bar.addWidget(self.fit_button)
        bar.addSpacing(8)
        bar.addWidget(self.hint, 1)  # takes the rest of the row (left-aligned)

        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setProperty("chartName", "trends")
        self.canvas.setBackground(TOKENS["chart_bg"])
        self.chart_scroll = QScrollArea()
        self.chart_scroll.setWidgetResizable(True)
        self.chart_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.chart_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chart_scroll.setWidget(self.canvas)
        self.chart_scroll.viewport().installEventFilter(self)
        self.empty_label = QLabel("Select variables to plot.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.limit_label = QLabel()
        self.limit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.limit_label.setObjectName("placeholderNote")
        self.limit_label.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.chart_scroll, 1)
        layout.addWidget(self.empty_label, 1)
        layout.addWidget(self.limit_label, 1)
        self.canvas.setVisible(False)
        self.chart_scroll.setVisible(False)

        self._mouse_proxy = pg.SignalProxy(
            self.canvas.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse
        )

    # ------------------------------------------------------------------ public API

    def set_timezone(self, timezone: str) -> None:
        self._tz = timezone
        for plot in self.plots:
            axis = plot.getAxis("bottom")
            if isinstance(axis, TzDateAxisItem):
                axis.set_timezone(timezone)

    def set_series(self, series: list[TrendSeries], *, keep_range: bool = False) -> None:
        """Replace the plotted series. ``keep_range`` preserves the current zoom."""
        previous = self.x_range() if keep_range and self._series else None
        self._series = list(series)
        self._sync()
        if previous is not None and self.plots:
            self.set_x_range(*previous)

    def set_x_range(self, lo: float, hi: float) -> None:
        if self.plots:
            self.plots[0].setXRange(lo, hi, padding=0)

    def visible_x_range(self) -> tuple[float, float] | None:
        """The time range on screen (the data extent while auto-ranging)."""
        if not self.plots:
            return None
        if self.plots[0].vb.autoRangeEnabled()[0]:
            xs = [s.x for s in self._series if len(s.x)]
            if not xs:
                return None
            return float(min(x[0] for x in xs)), float(max(x[-1] for x in xs))
        lo, hi = self.plots[0].viewRange()[0]
        return lo, hi

    @property
    def groups(self) -> list[tuple[pg.PlotItem, list[TrendSeries]]]:
        return list(zip(self.plots, self._plot_series, strict=True))

    def fit_all(self) -> None:
        self._zoom_history.clear()
        self.back_button.setEnabled(False)
        for plot in self.plots:
            plot.enableAutoRange()

    def zoom_to(self, lo: float, hi: float) -> None:
        """Zoom the time axis to [lo, hi], remembering the current range for :meth:`zoom_back`."""
        if not self.plots or hi <= lo:
            return
        self._zoom_history.append(self.x_range())
        self.back_button.setEnabled(True)
        self.set_x_range(lo, hi)

    def zoom_back(self) -> None:
        """Undo the last drag-zoom (back to fit-all at the bottom of the history)."""
        if not self._zoom_history:
            return
        previous = self._zoom_history.pop()
        self.back_button.setEnabled(bool(self._zoom_history))
        if previous is None:
            for plot in self.plots:
                plot.enableAutoRange()
        else:
            self.set_x_range(*previous)

    def x_range(self) -> tuple[float, float] | None:
        """Current visible time range, or None while the view is still auto-ranging to fit all data."""
        if not self.plots or self.plots[0].vb.autoRangeEnabled()[0]:
            return None
        lo, hi = self.plots[0].viewRange()[0]
        return lo, hi

    # ------------------------------------------------------------------ construction

    def _groups(self) -> list[list[TrendSeries]]:
        if self.mode.currentData() == "units":
            groups: dict[str, list[TrendSeries]] = {}
            for s in self._series:
                groups.setdefault(s.units or f"(no units) {s.name}", []).append(s)
            return list(groups.values())
        return [[s] for s in self._series]

    @staticmethod
    def _group_key(group: list[TrendSeries]) -> str:
        if len(group) == 1:
            return f"signal:{group[0].name}"
        return f"units:{group[0].units or f'(no units) {group[0].name}'}"

    def _colour_for(self, name: str) -> str:
        if name not in self._colours:
            self._colours[name] = PALETTE[len(self._colours) % len(PALETTE)]
        return self._colours[name]

    def _make_plot(self, units_label: str) -> pg.PlotItem:
        axis = TzDateAxisItem(self._tz)
        plot = pg.PlotItem(
            viewBox=_TrendViewBox(self), axisItems={"bottom": axis, "left": ValueAxis("left")}
        )
        tidy_plot(plot)  # no PyQtGraph menu or "A" button (Fit all is the toolbar's)
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.getAxis("left").setWidth(AXIS_WIDTH)
        # Names sit in a slim header above each chart (never on top of the data); the axis
        # shows only units to keep labels short.
        plot.setLabel("left", units_label)
        vline = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#888888", style=Qt.PenStyle.DashLine)
        )
        vline.setVisible(False)
        plot.addItem(vline, ignoreBounds=True)
        pin = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(PIN_COLOUR, width=2))
        pin.setVisible(self._selected_x is not None)
        if self._selected_x is not None:
            pin.setPos(self._selected_x)
        plot.addItem(pin, ignoreBounds=True)
        # Every pan/zoom re-fits this plot's curves to the visible window (cheap: the
        # data comes from the precomputed envelope or a small raw slice).
        plot.vb.sigRangeChanged.connect(lambda *_args, p=plot: self._update_plot_data(p))
        self._vlines.append(vline)
        self._vline_of[plot] = vline
        self._sel_line_of[plot] = pin
        return plot

    def _rebuild(self) -> None:
        """Recreate every plot from scratch (mode change, first data, key collisions)."""
        self._clear_plots()
        self._synced_mode = self.mode.currentData()
        has_data = bool(self._series)
        self.canvas.setVisible(has_data)
        self.chart_scroll.setVisible(has_data)
        self.empty_label.setVisible(not has_data)
        self.limit_label.setVisible(False)
        if not has_data:
            return
        groups = self._groups()
        if len(groups) > MAX_PLOTS:
            self.limit_label.setText(
                f"Showing {MAX_PLOTS} of {len(groups)} plots — untick variables to see the rest."
            )
            self.limit_label.setVisible(True)
            groups = groups[:MAX_PLOTS]
        for row, group in enumerate(groups):
            key = self._group_key(group)
            plot = self._make_plot(group[0].units or "")
            self._plot_by_key[key] = plot
            self._key_of_plot[plot] = key
            self.plots.append(plot)
            self._plot_keys.append(key)
            self._plot_series.append(group)
            self.canvas.ci.addItem(plot, row=row, col=0)
            self._sync_plot(plot, group)
        self._finish_sync()

    def _sync(self) -> None:
        """Add/remove only the plots that changed; update each plot's curves in place."""
        has_data = bool(self._series)
        self.canvas.setVisible(has_data)
        self.chart_scroll.setVisible(has_data)
        self.empty_label.setVisible(not has_data)
        self.limit_label.setVisible(False)
        if not has_data:
            self._clear_plots()
            return
        mode = self.mode.currentData()
        groups = self._groups()
        if mode != self._synced_mode or len({self._group_key(g) for g in groups}) != len(groups):
            self._rebuild()
            return
        total = len(groups)
        if total > MAX_PLOTS:
            self.limit_label.setText(
                f"Showing {MAX_PLOTS} of {total} plots — untick variables to see the rest."
            )
            self.limit_label.setVisible(True)
            groups = groups[:MAX_PLOTS]
        keys = [self._group_key(g) for g in groups]

        # Drop plots whose group is gone; create plots for new groups.
        for key in list(self._plot_keys):
            if key not in keys:
                self._remove_plot(key)
        for key, group in zip(keys, groups, strict=True):
            if key not in self._plot_by_key:
                plot = self._make_plot(group[0].units or "")
                self._plot_by_key[key] = plot
                self._key_of_plot[plot] = key

        # Rebuild the ordered lists; re-lay out the rows only when the order changed.
        reordered = keys != self._plot_keys
        if reordered:
            for plot in self.plots:
                self.canvas.ci.removeItem(plot)
        self.plots = [self._plot_by_key[k] for k in keys]
        self._plot_series = [g for g in groups]
        self._plot_keys = list(keys)
        self._vlines = [self._vline_of[p] for p in self.plots]
        for row, plot in enumerate(self.plots):
            if reordered:
                self.canvas.ci.addItem(plot, row=row, col=0)
            self._sync_plot(plot, self._plot_series[row])

        names = {s.name for s in self._series}
        self._env_cache = {k: v for k, v in self._env_cache.items() if k[0] in names}
        self._finish_sync()

    def _finish_sync(self) -> None:
        self.canvas.setMinimumHeight(max(1, len(self.plots)) * MIN_PLOT_HEIGHT)
        self._update_axes()
        for plot in self.plots:
            plot.setXLink(self.plots[0] if plot is not self.plots[0] else None)
        if self.plots and self._xrange_plot is not self.plots[0]:
            if self._xrange_conn is not None and self._xrange_plot is not None:
                with contextlib.suppress(TypeError, RuntimeError):
                    self._xrange_plot.sigXRangeChanged.disconnect(self._xrange_conn)
            self._xrange_conn = self.plots[0].sigXRangeChanged.connect(
                lambda _vb, r: self.x_range_changed.emit(float(r[0]), float(r[1]))
            )
            self._xrange_plot = self.plots[0]
        self.rebuilt.emit()

    def _update_axes(self) -> None:
        scrolling = self.canvas.minimumHeight() > self.chart_scroll.viewport().height()
        for i, plot in enumerate(self.plots):
            plot.getAxis("bottom").setStyle(showValues=scrolling or i == len(self.plots) - 1)

    def eventFilter(self, watched, event) -> bool:  # noqa: ANN001, N802 - Qt override
        if watched is self.chart_scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._update_axes()
        return super().eventFilter(watched, event)

    def _clear_plots(self) -> None:
        self._zoom_history.clear()
        self.back_button.setEnabled(False)
        self.canvas.clear()  # removes and deletes all items (plots, curves, lines)
        self.plots, self._vlines, self._plot_series = [], [], []
        self._plot_keys, self._plot_by_key, self._key_of_plot = [], {}, {}
        self._vline_of = {}
        self._sel_line_of = {}
        self._bands = []
        self._items = {}
        self._colours = {}
        self._env_cache = {}
        self._xrange_plot = None
        self._xrange_conn = None

    def _remove_plot(self, key: str) -> None:
        plot = self._plot_by_key.pop(key)
        self._key_of_plot.pop(plot, None)
        idx = self._plot_keys.index(key)
        del self._plot_keys[idx]
        del self._plot_series[idx]
        self.plots.remove(plot)
        self._vlines.remove(self._vline_of.pop(plot))
        self._sel_line_of.pop(plot, None)
        self._bands = [(p, r) for p, r in self._bands if p is not plot]
        for _name, (lo, hi, _sig) in self._items.pop(key, {}).items():
            plot.removeItem(lo)
            if hi is not None:
                plot.removeItem(hi)
        self.canvas.ci.removeItem(plot)
        plot.deleteLater()

    def _sync_plot(self, plot: pg.PlotItem, group: list[TrendSeries]) -> None:
        key = self._key_of_plot[plot]
        items = self._items.setdefault(key, {})
        wanted = {s.name: s for s in group}
        for name in list(items):
            if name not in wanted:
                plot.removeItem(items[name][0])
                if items[name][1] is not None:
                    plot.removeItem(items[name][1])
                del items[name]
        for name, s in wanted.items():
            signature = (s.points, id(s.x), id(s.y))
            pair = items.get(name)
            if pair is not None and pair[2] == signature:
                continue  # unchanged data: keep the items (and any zoom-in state) as they are
            if pair is not None:
                plot.removeItem(pair[0])
                if pair[1] is not None:
                    plot.removeItem(pair[1])
            colour = self._colour_for(name)
            if s.points:
                xp, yp = self._subsample_points(s)
                item = pg.PlotDataItem(
                    xp,
                    yp,
                    pen=None,
                    symbol="o",
                    symbolSize=5,
                    symbolBrush=colour,
                    symbolPen=None,
                    name=s.name,
                )
                plot.addItem(item)
                items[name] = (item, None, signature)
            else:
                xc, ymin, ymax = self._envelope(s)
                lo = pg.PlotDataItem(
                    xc, ymin, pen=pg.mkPen(colour, width=1), connect="finite", name=s.name
                )
                hi = pg.PlotDataItem(xc, ymax, pen=pg.mkPen(colour, width=1), connect="finite")
                plot.addItem(lo)
                plot.addItem(hi)
                items[name] = (lo, hi, signature)
        self._set_header(plot, group)

    def apply_theme(self) -> None:
        """After a theme switch: chart background and the per-chart name headers."""
        self.canvas.setBackground(TOKENS["chart_bg"])
        for plot, group in zip(self.plots, self._plot_series, strict=True):
            self._set_header(plot, group)

    def _set_header(self, plot: pg.PlotItem, group: list[TrendSeries]) -> None:
        """One-line chart header: a colour key and the name of each series, left-aligned."""
        parts = []
        for s in group:  # units are on the y-axis already
            mark = "●" if s.points else "━"
            parts.append(
                f"<span style='color:{self._colour_for(s.name)}'>{mark}</span>&nbsp;{escape(s.name)}"
            )
        plot.setTitle(
            "&nbsp;&nbsp;&nbsp;".join(parts), size="8pt", color=TOKENS["text"], justify="left"
        )
        plot.titleLabel.setMaximumHeight(HEADER_HEIGHT)
        plot.layout.setRowFixedHeight(0, HEADER_HEIGHT)

    @staticmethod
    def _subsample_points(s: TrendSeries) -> tuple[np.ndarray, np.ndarray]:
        n = len(s.x)
        if n <= POINTS_MAX:
            return s.x, s.y
        idx = np.linspace(0, n - 1, POINTS_MAX).round().astype(int)
        return s.x[idx], s.y[idx]

    def _envelope(self, s: TrendSeries) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x bin centres, min per bin, max per bin) — a fixed-size overview of the series."""
        key = (s.name, id(s.x), id(s.y))
        cached = self._env_cache.get(key)
        if cached is not None:
            return cached
        n = len(s.y)
        if n <= BINS:
            result = (
                np.asarray(s.x, dtype=float),
                np.asarray(s.y, dtype=float),
                np.asarray(s.y, dtype=float),
            )
        else:
            k = BINS
            pad = (k - n % k) % k
            xx = np.asarray(s.x, dtype=float)
            yy = np.asarray(s.y, dtype=float)
            if pad:
                xx = np.concatenate([xx, np.full(pad, xx[-1])])
                yy = np.concatenate([yy, np.full(pad, np.nan)])
            x2 = xx.reshape(k, -1)
            y2 = yy.reshape(k, -1)
            with warnings.catch_warnings():
                # All-NaN bins (data gaps) warn on nanmin/nanmax; the NaNs stay to keep the gap.
                warnings.simplefilter("ignore", RuntimeWarning)
                ymin = np.nanmin(y2, axis=1)
                ymax = np.nanmax(y2, axis=1)
            result = ((x2[:, 0] + x2[:, -1]) / 2, ymin, ymax)
        self._env_cache[key] = result
        return result

    def _update_plot_data(self, plot: pg.PlotItem) -> None:
        """Refit the plot's curves for the visible window: raw slice when zoomed in, envelope out."""
        key = self._key_of_plot.get(plot)
        if key is None or plot not in self.plots:
            return
        items = self._items.get(key, {})
        row = self.plots.index(plot)
        if plot.vb.autoRangeEnabled()[0]:
            lo = hi = None
        else:
            lo, hi = plot.viewRange()[0]
        for s in self._plot_series[row]:
            pair = items.get(s.name)
            if pair is None or s.points:
                continue
            lo_item, hi_item, _sig = pair
            if lo is not None and len(s.x) > 1:
                bin_span = (s.x[-1] - s.x[0]) / BINS
                if bin_span > 0 and hi - lo < RAW_SPAN_BINS * bin_span:
                    i0 = int(np.searchsorted(s.x, lo, side="left")) - 1
                    i1 = int(np.searchsorted(s.x, hi, side="right")) + 1
                    i0, i1 = max(i0, 0), min(i1, len(s.x))
                    if i1 - i0 >= 2:
                        lo_item.setData(s.x[i0:i1], s.y[i0:i1])
                        hi_item.setVisible(False)
                        continue
            xc, ymin, ymax = self._envelope(s)
            lo_item.setData(xc, ymin)
            hi_item.setData(xc, ymax)
            hi_item.setVisible(True)

    # ------------------------------------------------------- crosshair & selection

    def _on_mouse(self, event) -> None:  # noqa: ANN001 - pyqtgraph SignalProxy payload
        pos = event[0]
        for plot in self.plots:
            if plot.sceneBoundingRect().contains(pos):
                x = plot.vb.mapSceneToView(pos).x()
                self._show_cursor(x)
                return

    def _values_at(self, x: float) -> tuple[str, dict[str, str]]:
        """(time text, {series name: formatted value}) of the last sample at or before x."""
        values: dict[str, str] = {}
        for s in self._series:
            i = (
                int(np.searchsorted(s.x, x, side="right")) - 1
            )  # last sample at or before the cursor
            if 0 <= i < len(s.y) and np.isfinite(s.y[i]):
                values[s.name] = f"{s.y[i]:.6g}{' ' + s.units if s.units else ''}"
        return format_time(x, self._tz), values

    def _show_cursor(self, x: float) -> None:
        for line in self._vlines:
            line.setPos(x)
            line.setVisible(True)
        self.hover.emit(*self._values_at(x))

    # -- pinned point (left-click) ---------------------------------------------

    def _click_point(self, vb: pg.ViewBox, ev) -> None:  # noqa: ANN001 - pyqtgraph event
        if self._bands:
            self._clear_band()  # clicking anywhere else removes the highlight
            return
        self._toggle_point(vb.mapSceneToView(ev.scenePos()).x())

    def _toggle_point(self, x: float) -> None:
        """Pin the values at x; a second click anywhere releases the pin."""
        if self._selected_x is not None:
            self._clear_point()
            return
        self._selected_x = float(x)
        for plot in self.plots:
            line = self._sel_line_of.get(plot)
            if line is not None:
                line.setPos(x)
                line.setVisible(True)
        self.point_selected.emit(*self._values_at(x))

    def _clear_point(self) -> None:
        if self._selected_x is None:
            return
        self._selected_x = None
        for plot in self.plots:
            line = self._sel_line_of.get(plot)
            if line is not None:
                line.setVisible(False)
        self.point_cleared.emit()

    # -- drag-to-zoom band (left-drag) -----------------------------------------

    def _band_drag(self, vb: pg.ViewBox, ev) -> None:  # noqa: ANN001 - pyqtgraph event
        if ev.isStart():
            self._drag_x0 = vb.mapSceneToView(ev.buttonDownScenePos()).x()
        if self._drag_x0 is None:
            return
        x1 = vb.mapSceneToView(ev.scenePos()).x()
        self._set_band(self._drag_x0, x1)
        if ev.isFinish():
            x0, self._drag_x0 = self._drag_x0, None
            wide = abs(ev.scenePos().x() - ev.buttonDownScenePos().x()) >= MIN_DRAG_PX
            self._finish_band(x0, x1, zoom=wide)

    def _set_band(self, x0: float, x1: float) -> None:
        """Full-height highlight from x0 to x1 on every plot while dragging."""
        if self._selected_x is not None:
            self._clear_point()  # a new drag replaces the pinned point
        lo, hi = sorted((float(x0), float(x1)))
        if not self._bands:
            for plot in self.plots:
                region = pg.LinearRegionItem(
                    values=(lo, hi), brush=pg.mkBrush(*HIGHLIGHT_COLOUR, 40), movable=False
                )
                region.setZValue(5)
                plot.addItem(region, ignoreBounds=True)
                self._bands.append((plot, region))
        else:
            for _plot, region in self._bands:
                region.setRegion((lo, hi))

    def _finish_band(self, x0: float, x1: float, *, zoom: bool = True) -> None:
        """Release of a drag: zoom every chart to the highlighted period."""
        self._clear_band()
        lo, hi = sorted((float(x0), float(x1)))
        if zoom and hi > lo:
            self.zoom_to(lo, hi)

    def _clear_band(self) -> None:
        for plot, region in self._bands:
            if region.scene() is not None:
                plot.removeItem(region)
        self._bands = []

    # -- pan (middle-drag) -----------------------------------------------------

    def _pan_drag(self, vb: pg.ViewBox, ev) -> None:  # noqa: ANN001 - pyqtgraph event
        """Pan along time only; the y-axis keeps fitting the visible data."""
        p0 = vb.mapSceneToView(ev.lastScenePos())
        p1 = vb.mapSceneToView(ev.scenePos())
        dx = p1.x() - p0.x()
        x0, x1 = vb.viewRange()[0]
        vb.setXRange(x0 - dx, x1 - dx, padding=0)
