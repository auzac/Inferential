"""Synchronized time-series trends with a timezone-aware axis, crosshair and fast display.

Plots are added and removed one at a time (ticking a variable no longer rebuilds every chart),
and each line series draws from a precomputed min/max envelope while zoomed out, switching to
the raw samples of the visible window when zoomed in. The number of plots is capped so panning
stays fluid on large data. The crosshair emits the values at the cursor through ``hover``; the
pages show them in the "At cursor" box below the variable selection.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from inferential_studio.ui.widgets.time_axis import TzDateAxisItem, format_time

# Okabe–Ito colour-blind-safe palette (black replaced by dark grey).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#555555"]
AXIS_WIDTH = 72
MAX_PLOTS = 12  # charts shown at once; more than this is slow to pan and hard to read
BINS = 2048  # envelope bins per series: the resolution of the zoomed-out view
RAW_SPAN_BINS = 8  # zoomed in closer than this many bins: draw the raw samples instead
POINTS_MAX = 5000  # markers drawn for sparse point series (lab samples)


@dataclass
class TrendSeries:
    name: str
    x: np.ndarray  # UTC epoch seconds
    y: np.ndarray
    units: str | None = None
    points: bool = False  # draw as markers (sparse lab samples) instead of a line


class TrendView(QWidget):
    """Stacked or unit-grouped plots sharing one time axis."""

    x_range_changed = Signal(float, float)
    rebuilt = Signal()  # plots were added/removed/updated; owners re-add their overlays
    hover = Signal(str, dict)  # (time text, {series name: formatted value}) at the cursor

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
        self.fit_button = QPushButton("Fit all")
        self.fit_button.setToolTip("Reset zoom to show all data (mouse wheel zooms, drag pans)")
        self.fit_button.clicked.connect(self.fit_all)

        bar = QHBoxLayout()
        self.toolbar_layout = bar
        bar.addWidget(self.mode)
        bar.addWidget(self.fit_button)
        bar.addStretch(1)

        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setBackground("w")
        self.empty_label = QLabel("Select variables to plot.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.limit_label = QLabel()
        self.limit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.limit_label.setObjectName("placeholderNote")
        self.limit_label.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.empty_label, 1)
        layout.addWidget(self.limit_label, 1)
        self.canvas.setVisible(False)

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
        for plot in self.plots:
            plot.enableAutoRange()

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
        plot = pg.PlotItem(axisItems={"bottom": axis})
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.getAxis("left").setWidth(AXIS_WIDTH)
        plot.addLegend(offset=(5, 5), labelTextSize="8pt", brush=(255, 255, 255, 210))
        # Names are in the legend; the axis shows only units to keep labels short.
        plot.setLabel("left", units_label)
        vline = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#888888", style=Qt.PenStyle.DashLine)
        )
        vline.setVisible(False)
        plot.addItem(vline, ignoreBounds=True)
        # Every pan/zoom re-fits this plot's curves to the visible window (cheap: the
        # data comes from the precomputed envelope or a small raw slice).
        plot.vb.sigRangeChanged.connect(
            lambda *_args, p=plot: self._update_plot_data(p)
        )
        self._vlines.append(vline)
        self._vline_of[plot] = vline
        return plot

    def _rebuild(self) -> None:
        """Recreate every plot from scratch (mode change, first data, key collisions)."""
        self._clear_plots()
        self._synced_mode = self.mode.currentData()
        has_data = bool(self._series)
        self.canvas.setVisible(has_data)
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
        for i, plot in enumerate(self.plots):
            plot.getAxis("bottom").setStyle(showValues=(i == len(self.plots) - 1))
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

    def _clear_plots(self) -> None:
        self.canvas.clear()  # removes and deletes all items (plots, curves, lines)
        self.plots, self._vlines, self._plot_series = [], [], []
        self._plot_keys, self._plot_by_key, self._key_of_plot = [], {}, {}
        self._vline_of = {}
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

    # ------------------------------------------------------------------ crosshair

    def _on_mouse(self, event) -> None:  # noqa: ANN001 - pyqtgraph SignalProxy payload
        pos = event[0]
        for plot in self.plots:
            if plot.sceneBoundingRect().contains(pos):
                x = plot.vb.mapSceneToView(pos).x()
                self._show_cursor(x)
                return

    def _show_cursor(self, x: float) -> None:
        for line in self._vlines:
            line.setPos(x)
            line.setVisible(True)
        values: dict[str, str] = {}
        for s in self._series:
            i = (
                int(np.searchsorted(s.x, x, side="right")) - 1
            )  # last sample at or before the cursor
            if 0 <= i < len(s.y) and np.isfinite(s.y[i]):
                values[s.name] = f"{s.y[i]:.6g}{' ' + s.units if s.units else ''}"
        self.hover.emit(format_time(x, self._tz), values)
