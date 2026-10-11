"""Synchronized time-series trends with a timezone-aware axis, crosshair readout and fast display.

Large series are drawn with pyqtgraph's peak downsampling and clip-to-view. That affects only
what is drawn; the data held by the caller is never resampled. Lines break at missing values
(``connect="finite"``) instead of being drawn across gaps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from inferential_studio.ui.widgets.time_axis import TzDateAxisItem, format_time

# Okabe–Ito colour-blind-safe palette (black replaced by dark grey).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#555555"]
AXIS_WIDTH = 72


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
    rebuilt = Signal()  # plots were recreated; owners re-add their overlays

    def __init__(self, timezone: str = "UTC", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tz = timezone
        self._series: list[TrendSeries] = []
        self.plots: list[pg.PlotItem] = []
        self._vlines: list[pg.InfiniteLine] = []
        self._plot_series: list[list[TrendSeries]] = []

        self.mode = QComboBox()
        self.mode.addItem("One plot per signal", "stacked")
        self.mode.addItem("Group by units", "units")
        self.mode.setToolTip("Signals with different units always use separate y-axes.")
        self.mode.currentIndexChanged.connect(self._rebuild)
        self.fit_button = QPushButton("Fit all")
        self.fit_button.setToolTip("Reset zoom to show all data (mouse wheel zooms, drag pans)")
        self.fit_button.clicked.connect(self.fit_all)
        self.readout = QLabel(" ")
        self.readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        bar = QHBoxLayout()
        self.toolbar_layout = bar
        bar.addWidget(self.mode)
        bar.addWidget(self.fit_button)
        bar.addWidget(self.readout, 1)

        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setBackground("w")
        self.empty_label = QLabel("Select variables to plot.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.empty_label, 1)
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
        self._rebuild()
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

    def _rebuild(self) -> None:
        self.canvas.clear()
        self.plots, self._vlines, self._plot_series = [], [], []
        has_data = bool(self._series)
        self.canvas.setVisible(has_data)
        self.empty_label.setVisible(not has_data)
        if not has_data:
            return

        colour_of = {s.name: PALETTE[i % len(PALETTE)] for i, s in enumerate(self._series)}
        groups = self._groups()
        for row, group in enumerate(groups):
            axis = TzDateAxisItem(self._tz)
            plot = self.canvas.addPlot(row=row, col=0, axisItems={"bottom": axis})
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis("left").setWidth(AXIS_WIDTH)
            plot.addLegend(offset=(5, 5), labelTextSize="8pt", brush=(255, 255, 255, 210))
            # Names are in the legend; the axis shows only units to keep labels short.
            plot.setLabel("left", group[0].units or "")
            if row < len(groups) - 1:
                axis.setStyle(showValues=False)
            if self.plots:
                plot.setXLink(self.plots[0])
            for s in group:
                colour = colour_of[s.name]
                if s.points:
                    item = pg.PlotDataItem(
                        s.x,
                        s.y,
                        pen=None,
                        symbol="o",
                        symbolSize=5,
                        symbolBrush=colour,
                        symbolPen=None,
                        name=s.name,
                    )
                else:
                    item = pg.PlotDataItem(
                        s.x, s.y, pen=pg.mkPen(colour, width=1), connect="finite", name=s.name
                    )
                plot.addItem(item)
                if not s.points:
                    # Set after adding: pyqtgraph needs the item's ViewBox to evaluate these.
                    item.setDownsampling(auto=True, method="peak")
                    item.setClipToView(True)
            vline = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen("#888888", style=Qt.PenStyle.DashLine)
            )
            vline.setVisible(False)
            plot.addItem(vline, ignoreBounds=True)
            self.plots.append(plot)
            self._vlines.append(vline)
            self._plot_series.append(group)

        self.plots[0].sigXRangeChanged.connect(
            lambda _vb, r: self.x_range_changed.emit(float(r[0]), float(r[1]))
        )
        self.fit_all()
        self.rebuilt.emit()

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
        parts = [format_time(x, self._tz)]
        for s in self._series:
            i = (
                int(np.searchsorted(s.x, x, side="right")) - 1
            )  # last sample at or before the cursor
            if 0 <= i < len(s.y) and np.isfinite(s.y[i]):
                parts.append(f"{s.name} = {s.y[i]:.6g}{' ' + s.units if s.units else ''}")
        self.readout.setText("   ".join(parts))
