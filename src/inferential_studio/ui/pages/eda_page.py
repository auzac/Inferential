"""Stage 2 page: raw exploratory data analysis (trends, statistics, correlations, coverage)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.core.statistics import (
    EdaResult,
    compute_eda,
    correlation_matrix,
    hierarchical_order,
    histogram,
)
from inferential_studio.infrastructure.workers import TaskManager
from inferential_studio.ui.messages import Level
from inferential_studio.ui.pages.stage_page import StagePage
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.stages import Stage
from inferential_studio.ui.widgets.cursor_readout import CursorReadout
from inferential_studio.ui.widgets.data_table import DataTableView
from inferential_studio.ui.widgets.heatmap import CorrelationHeatmap
from inferential_studio.ui.widgets.plots import clear_placeholder, show_placeholder
from inferential_studio.ui.widgets.time_axis import TzDateAxisItem, to_epoch_seconds
from inferential_studio.ui.widgets.trend_view import PALETTE, TrendSeries, TrendView

COVERAGE_FULL = (63, 143, 107)  # muted green for 100 % valid (white = none)
SCATTER_DISPLAY_MAX = 20_000  # points drawn in scatter plots; statistics always use all data
MATRIX_MAX = 6
DEFAULT_SELECTED = 4

SUMMARY_HEADERS = {
    "rows": "Rows",
    "valid": "Valid",
    "valid_pct": "Valid %",
    "missing_pct": "Missing %",
    "invalid_pct": "Text/invalid %",
    "min": "Min",
    "q1": "Q1",
    "median": "Median",
    "q3": "Q3",
    "max": "Max",
    "mean": "Mean",
    "std": "Std dev",
    "variance": "Variance",
    "cv": "CV",
    "skewness": "Skewness",
    "kurtosis": "Excess kurtosis",
}

STATS_NOTE = (
    "Statistics use every row with a valid timestamp (raw data, before cleaning). Std dev and variance "
    "use n−1. CV = std/|mean| is shown only when all values have the same sign. Skewness and excess "
    "kurtosis are bias-corrected sample estimates."
)
CORR_NOTE = (
    "Pairwise correlation on raw, uncleaned data. It shows association, not causation, and can be "
    "dominated by shutdowns or operating-regime changes. The matrix follows the variables checked "
    "on the left. Hover a cell for r and the number of pairs; click it for a scatter plot."
)


def _subsample(n: int, limit: int) -> np.ndarray:
    """Evenly spaced display subset (deterministic) when there are too many points to draw."""
    if n <= limit:
        return np.arange(n)
    return np.linspace(0, n - 1, limit).round().astype(int)


class EdaPage(StagePage):
    def __init__(
        self,
        stage: Stage,
        session: ProjectSession,
        tasks: TaskManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(stage, parent)
        self._session = session
        self._tasks = tasks
        self._frame = pd.DataFrame()
        self._mask = pd.DataFrame()
        self._target: pd.Series | None = None
        self._x: np.ndarray = np.zeros(0)
        self._result: EdaResult | None = None
        self._computed_version = -1
        self._pending = None
        self._units: dict[str, str | None] = {}

        # -- variable list ----------------------------------------------------------
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter variables…")
        self.search.textChanged.connect(self._filter_list)
        self.var_list = QListWidget()
        self.var_list.itemChanged.connect(self._on_selection_changed)
        self.show_target = QCheckBox("Show target (lab samples)")
        self.show_target.setChecked(True)
        self.show_target.toggled.connect(self._update_trends)
        # Coalesce bursts of ticks: redraw once, a short pause after the last change.
        self._selection_timer = QTimer(self)
        self._selection_timer.setSingleShot(True)
        self._selection_timer.setInterval(150)
        self._selection_timer.timeout.connect(self._apply_selection)
        self.cursor = CursorReadout()
        self.check_all = QToolButton()
        self.check_all.setText("All")
        self.check_all.setToolTip("Check every listed variable (respects the filter)")
        self.check_all.clicked.connect(lambda: self._check_listed(True))
        self.check_none = QToolButton()
        self.check_none.setText("None")
        self.check_none.setToolTip("Uncheck every listed variable (respects the filter)")
        self.check_none.clicked.connect(lambda: self._check_listed(False))
        self.count_label = QLabel()
        self.count_label.setObjectName("placeholderNote")
        head = QHBoxLayout()
        head.setSpacing(4)
        title = QLabel("Variables")
        title.setObjectName("sectionTitle")
        head.addWidget(title)
        head.addWidget(self.count_label)
        head.addStretch(1)
        head.addWidget(self.check_all)
        head.addWidget(self.check_none)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addLayout(head)
        ll.addWidget(self.search)
        ll.addWidget(self.var_list, 1)  # the list scrolls when there are many tags
        ll.addWidget(self.show_target)
        ll.addWidget(self.cursor)

        # -- tabs -----------------------------------------------------------------------
        self.tabs = QTabWidget()
        self.trends = TrendView(session.project.display_timezone)
        self.trends.hover.connect(self.cursor.show_hover)
        self.trends.point_selected.connect(self.cursor.show_point)
        self.trends.point_cleared.connect(self.cursor.unlock)
        self.tabs.addTab(self.trends, "Trends")

        self.stats_table = DataTableView()
        self._stats_tab = QWidget()
        sl = QVBoxLayout(self._stats_tab)
        note = QLabel(STATS_NOTE)
        note.setWordWrap(True)
        note.setObjectName("placeholderNote")
        sl.addWidget(self.stats_table, 1)
        sl.addWidget(note)
        self.tabs.addTab(self._stats_tab, "Statistics")

        self.method = QComboBox()
        self.method.addItem("Pearson (linear)", "pearson")
        self.method.addItem("Spearman (rank / monotonic)", "spearman")
        self.absolute = QCheckBox("Absolute |r|")
        self.cluster = QCheckBox("Group similar variables")
        self.cluster.setToolTip("Hierarchical ordering on 1 − |r| (average linkage)")
        self.method.currentIndexChanged.connect(self._update_correlation)
        for w in (self.absolute, self.cluster):
            w.toggled.connect(self._update_correlation)
        self.corr_scope = QLabel()
        self.corr_scope.setObjectName("placeholderNote")
        self.heatmap = CorrelationHeatmap()
        self.heatmap.cell_clicked.connect(self._show_pair)
        self.pair_plot = pg.PlotWidget(background="w")
        self.pair_plot.showGrid(x=True, y=True, alpha=0.15)
        self.pair_note = QLabel()
        self.pair_note.setWordWrap(True)
        self._pair: tuple[str, str] | None = None
        self._pair_placeholder()
        self._corr_tab = QWidget()
        cl = QVBoxLayout(self._corr_tab)
        controls = QHBoxLayout()
        for w in (self.method, self.absolute, self.cluster):
            controls.addWidget(w)
        controls.addStretch(1)
        controls.addWidget(self.corr_scope)
        cl.addLayout(controls)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.heatmap)
        pair_box = QWidget()
        pl = QVBoxLayout(pair_box)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.pair_plot, 1)
        pl.addWidget(self.pair_note)
        split.addWidget(pair_box)
        split.setSizes([650, 350])
        cl.addWidget(split, 1)
        corr_note = QLabel(CORR_NOTE)
        corr_note.setWordWrap(True)
        corr_note.setObjectName("placeholderNote")
        cl.addWidget(corr_note)
        self.tabs.addTab(self._corr_tab, "Correlation")

        self.matrix = pg.GraphicsLayoutWidget()
        self.matrix.setBackground("w")
        self.matrix_note = QLabel()
        self._matrix_tab = QWidget()
        ml = QVBoxLayout(self._matrix_tab)
        ml.addWidget(self.matrix, 1)
        ml.addWidget(self.matrix_note)
        self.tabs.addTab(self._matrix_tab, "Scatter matrix")

        self.hist_var = QComboBox()
        self.hist_var.currentIndexChanged.connect(self._update_histogram)
        self.hist_plot = pg.PlotWidget(background="w")
        self.hist_plot.showGrid(x=True, y=True, alpha=0.15)
        self._hist_tab = QWidget()
        hl = QVBoxLayout(self._hist_tab)
        hl.addWidget(self.hist_var)
        hl.addWidget(self.hist_plot, 1)
        self.tabs.addTab(self._hist_tab, "Distribution")

        self.coverage_canvas = pg.GraphicsLayoutWidget()
        self.coverage_canvas.setBackground("w")
        self._cov_tab = QWidget()
        cvl = QVBoxLayout(self._cov_tab)
        cvl.addWidget(self.coverage_canvas, 1)
        cov_note = QLabel(
            "Top: share of rows with a valid value per variable and time bin (green = complete, white = "
            "none, grey = no rows). Bottom: rows (timestamps) per bin."
        )
        cov_note.setWordWrap(True)
        cov_note.setObjectName("placeholderNote")
        cvl.addWidget(cov_note)
        self.tabs.addTab(self._cov_tab, "Coverage")

        self.xy_table = DataTableView()
        self.xy_note = QLabel()
        self.xy_note.setWordWrap(True)
        self._xy_tab = QWidget()
        xl = QVBoxLayout(self._xy_tab)
        xl.addWidget(self.xy_note)
        xl.addWidget(self.xy_table, 1)
        self.tabs.addTab(self._xy_tab, "X–Y screening")
        # Each tab is refreshed only when it becomes visible (all of them once when
        # the background statistics finish).
        self._tab_updaters = {
            self._stats_tab: self._update_stats,
            self._corr_tab: self._update_correlation,
            self._matrix_tab: self._update_matrix,
            self._hist_tab: self._update_histogram,
            self._cov_tab: self._update_coverage,
            self._xy_tab: self._update_xy,
        }
        self.tabs.currentChanged.connect(lambda _i: self._update_visible_tab())

        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(left)
        main.addWidget(self.tabs)
        main.setSizes([220, 900])
        self.body.addWidget(main, 1)

        session.data_changed.connect(self._on_data_changed)
        session.changed.connect(self._on_project_changed)
        self._on_data_changed()

    # ------------------------------------------------------------------ data

    @Slot()
    def _on_project_changed(self) -> None:
        self.trends.set_timezone(self._session.project.display_timezone)
        if self._result is not None:
            self._update_coverage()

    @Slot()
    def _on_data_changed(self) -> None:
        project, data = self._session.project, self._session.data
        self._frame, self._mask = data.process_frame(project)
        self._target = data.target_series(project)
        self._x = to_epoch_seconds(self._frame.index) if not self._frame.empty else np.zeros(0)
        self._units = {v.name: v.units for v in project.variables}
        self._result = None
        self._populate_list()
        msgs = self._session.messages
        if self._frame.empty:
            msgs.set(
                "initial_eda.data",
                Level.INFO,
                "No process data loaded. Import data on the Data Import page.",
            )
        else:
            n_lab = 0 if self._target is None else int(self._target.notna().sum())
            msgs.set(
                "initial_eda.data", Level.INFO,
                f"{len(self._frame):,} process rows × {self._frame.shape[1]} variables; {n_lab:,} target samples.",
            )  # fmt: skip
        self._update_trends(keep_range=False)  # new data: fit all
        if self.isVisible():
            self.ensure_computed()

    def _populate_list(self) -> None:
        self.var_list.blockSignals(True)
        self.var_list.clear()
        project = self._session.project
        predictors = {v.name for v in project.predictors}
        ordered = sorted(
            self._frame.columns,
            key=lambda n: (n not in predictors, list(self._frame.columns).index(n)),
        )
        for i, name in enumerate(ordered):
            units = self._units.get(name)
            item = QListWidgetItem(f"{name}" + (f"  [{units}]" if units else ""))
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if i < DEFAULT_SELECTED else Qt.CheckState.Unchecked
            )
            if name not in predictors:
                item.setToolTip("Not currently a candidate predictor")
            self.var_list.addItem(item)
        self.var_list.blockSignals(False)
        self._update_count()
        self.hist_var.blockSignals(True)
        self.hist_var.clear()
        self.hist_var.addItems(
            list(self._frame.columns) + ([self._target.name] if self._target is not None else [])
        )
        self.hist_var.blockSignals(False)

    def selected_variables(self) -> list[str]:
        return [
            self.var_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.var_list.count())
            if self.var_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _filter_list(self, text: str) -> None:
        text = text.lower()
        for i in range(self.var_list.count()):
            item = self.var_list.item(i)
            item.setHidden(text not in item.text().lower())

    def showEvent(self, event) -> None:  # noqa: N802, ANN001
        super().showEvent(event)
        self.ensure_computed()

    @Slot()
    def ensure_computed(self) -> None:
        if (
            self._frame.empty
            or self._computed_version == self._session.data_version
            or self._pending is not None
        ):
            return
        version = self._session.data_version
        self._session.messages.set("initial_eda.calc", Level.INFO, "Calculating statistics…")
        self._pending = self._tasks.submit(
            "Calculating statistics",
            compute_eda,
            self._frame,
            self._mask,
            self._target,
            on_result=lambda r, v=version: self._on_computed(r, v),
            on_error=lambda msg, _d: self._on_failed(msg),
            on_cancelled=lambda: self._on_failed("Calculation cancelled."),
        )

    def _on_computed(self, result: EdaResult, version: int) -> None:
        self._pending = None
        if version != self._session.data_version:  # data changed meanwhile: recompute
            self.ensure_computed()
            return
        self._result = result
        self._computed_version = version
        self._session.messages.clear("initial_eda.calc")
        self._update_visible_tab(all_tabs=True)

    def _on_failed(self, message: str) -> None:
        self._pending = None
        self._session.messages.set(
            "initial_eda.calc", Level.WARNING, f"Statistics not calculated: {message}"
        )

    # ------------------------------------------------------------------ views

    @Slot()
    def _on_selection_changed(self, _item=None) -> None:  # noqa: ANN001
        self._selection_timer.start()

    def _apply_selection(self) -> None:
        self._update_count()
        self._update_trends()
        # Only views that are actually on screen follow the ticks; the others are
        # refreshed when their tab is opened (see _update_visible_tab).
        current = self.tabs.currentWidget()
        if current is self._corr_tab:
            self._update_correlation()
        if current is self._matrix_tab:
            self._update_matrix()

    def _check_listed(self, checked: bool) -> None:
        """Check or uncheck every variable the filter leaves visible (one redraw)."""
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.var_list.blockSignals(True)
        for i in range(self.var_list.count()):
            item = self.var_list.item(i)
            if not item.isHidden():
                item.setCheckState(state)
        self.var_list.blockSignals(False)
        self._on_selection_changed()

    def _update_count(self) -> None:
        self.count_label.setText(f"{len(self.selected_variables())} of {self.var_list.count()}")

    def _update_visible_tab(self, all_tabs: bool = False) -> None:
        if self._result is None:
            return
        updaters = (
            set(self._tab_updaters.values())
            if all_tabs
            else [self._tab_updaters.get(self.tabs.currentWidget())]
        )
        for updater in updaters:
            if updater is not None:
                updater()

    def _update_trends(self, keep_range: bool = True) -> None:
        series = []
        for name in self.selected_variables():
            series.append(
                TrendSeries(
                    name, self._x, self._frame[name].to_numpy(dtype=float), self._units.get(name)
                )
            )
        if self.show_target.isChecked() and self._target is not None and not self._target.empty:
            t = self._target.dropna()
            series.append(
                TrendSeries(
                    f"{t.name} (lab)",
                    to_epoch_seconds(t.index),
                    t.to_numpy(dtype=float),
                    self._units.get(str(t.name)),
                    points=True,
                )
            )
        self.trends.set_series(series, keep_range=keep_range)

    def _update_stats(self) -> None:
        summary = self._result.summary.copy()
        summary.insert(0, "units", [self._units.get(n) or "" for n in summary.index])
        self.stats_table.set_frame(summary, {**SUMMARY_HEADERS, "units": "Units"})

    def _update_correlation(self) -> None:
        if self._result is None:
            return
        corr = (
            self._result.pearson
            if self.method.currentData() == "pearson"
            else self._result.spearman
        )
        pairs = self._result.pairs
        keep = [n for n in self.selected_variables() if n in corr.columns]
        self.corr_scope.setText(f"{len(keep)} checked variable{'s' if len(keep) != 1 else ''}")
        if self._pair is not None and not set(self._pair) <= set(keep):
            self._pair_placeholder()  # the pair shown is no longer in the matrix
        if len(keep) < 2:
            self.heatmap.set_placeholder("Check at least two variables on the left.")
            return
        corr, pairs = corr.loc[keep, keep], pairs.loc[keep, keep]
        if self.cluster.isChecked() and len(corr) > 2:
            order = hierarchical_order(corr)
            corr, pairs = corr.loc[order, order], pairs.loc[order, order]
        self.heatmap.set_matrix(corr, pairs, absolute=self.absolute.isChecked())

    def _pair_placeholder(
        self, text: str = "Click a cell of the matrix to see its scatter plot."
    ) -> None:
        self._pair = None
        for side in ("left", "bottom"):
            self.pair_plot.setLabel(side, "")
        show_placeholder(self.pair_plot, text)
        self.pair_note.setText("")

    @Slot(str, str)
    def _show_pair(self, row_var: str, col_var: str) -> None:
        self.pair_plot.clear()
        if row_var not in self._frame or col_var not in self._frame:
            return
        clear_placeholder(self.pair_plot)
        pair = (
            self._frame[[col_var, row_var]].dropna()
            if row_var != col_var
            else self._frame[[col_var]].dropna()
        )
        if row_var == col_var or pair.empty:
            self._pair_placeholder(f"{row_var}: choose a cell off the diagonal.")
            return
        self._pair = (row_var, col_var)
        idx = _subsample(len(pair), SCATTER_DISPLAY_MAX)
        sub = pair.iloc[idx]
        self.pair_plot.plot(
            sub[col_var].to_numpy(), sub[row_var].to_numpy(), pen=None, symbol="o", symbolSize=3,
            symbolBrush=pg.mkBrush(0, 114, 178, 90), symbolPen=None,
        )  # fmt: skip
        self.pair_plot.setLabel("bottom", col_var, units=self._units.get(col_var))
        self.pair_plot.setLabel("left", row_var, units=self._units.get(row_var))
        method = self.method.currentData()
        r = correlation_matrix(pair, method).iat[0, 1]
        shown = (
            f"showing {len(idx):,} of {len(pair):,} points (display only)"
            if len(idx) < len(pair)
            else f"{len(pair):,} points"
        )
        self.pair_note.setText(f"{method.title()} r = {r:.3f}; {shown}.")

    def _update_matrix(self) -> None:
        self.matrix.clear()
        names = [n for n in self.selected_variables() if n in self._frame.columns][:MATRIX_MAX]
        if len(names) < 2:
            self.matrix_note.setText("Check 2–6 variables in the list to draw a scatter matrix.")
            return
        sub = self._frame[names]
        idx = _subsample(len(sub), 5000)
        sub = sub.iloc[idx]
        for i, yname in enumerate(names):
            for j, xname in enumerate(names):
                p = self.matrix.addPlot(row=i, col=j)
                p.hideButtons()
                p.getAxis("left").setStyle(showValues=j == 0)
                p.getAxis("bottom").setStyle(showValues=i == len(names) - 1)
                if j == 0:
                    p.setLabel("left", yname)
                if i == len(names) - 1:
                    p.setLabel("bottom", xname)
                if i == j:
                    counts, edges = histogram(sub[xname], max_bins=30)
                    if counts.size:
                        p.plot(
                            edges,
                            counts,
                            stepMode="center",
                            fillLevel=0,
                            brush=(0, 114, 178, 120),
                            pen=None,
                        )
                else:
                    xy = sub[[xname, yname]].dropna()
                    p.plot(xy[xname].to_numpy(), xy[yname].to_numpy(), pen=None, symbol="o", symbolSize=2,
                           symbolBrush=pg.mkBrush(0, 114, 178, 80), symbolPen=None)  # fmt: skip
        extra = (
            " (first 6 checked variables)" if len(self.selected_variables()) > MATRIX_MAX else ""
        )
        self.matrix_note.setText(f"{len(idx):,} evenly spaced rows drawn{extra}; display only.")

    def _update_histogram(self) -> None:
        self.hist_plot.clear()
        name = self.hist_var.currentText()
        if not name:
            return
        if self._target is not None and name == self._target.name:
            values = self._target
        elif name in self._frame:
            values = self._frame[name]
        else:
            return
        counts, edges = histogram(values)
        if counts.size:
            self.hist_plot.plot(
                edges,
                counts,
                stepMode="center",
                fillLevel=0,
                brush=(0, 114, 178, 140),
                pen=pg.mkPen("#0072B2"),
            )
        self.hist_plot.setLabel("bottom", name, units=self._units.get(name))
        self.hist_plot.setLabel("left", "Count")

    def _update_coverage(self) -> None:
        self.coverage_canvas.clear()
        cov = self._result.coverage
        if cov.valid_fraction.empty:
            return
        frac = cov.valid_fraction.to_numpy().T  # variables x bins
        n_vars, n_bins = frac.shape
        rgb = np.empty((n_vars, n_bins, 3), dtype=np.uint8)
        f = np.nan_to_num(frac, nan=0.0)
        for channel, full in enumerate(COVERAGE_FULL):  # white (no data) → green (complete)
            rgb[..., channel] = (255 - f * (255 - full)).astype(np.uint8)
        rgb[np.isnan(frac)] = (200, 200, 200)
        x0 = to_epoch_seconds(pd.DatetimeIndex([cov.bin_starts[0]]))[0]
        width = cov.bin_width.total_seconds() * n_bins
        tz = self._session.project.display_timezone
        top = self.coverage_canvas.addPlot(row=0, col=0, axisItems={"bottom": TzDateAxisItem(tz)})
        img = pg.ImageItem(rgb.transpose(1, 0, 2), levels=(0, 255))  # true colours, no stretch
        img.setRect(QRectF(x0, 0, width, n_vars))
        top.addItem(img)
        top.invertY(True)
        names = list(cov.valid_fraction.columns)
        top.getAxis("left").setTicks([[(i + 0.5, n) for i, n in enumerate(names)]])
        top.getAxis("bottom").setStyle(showValues=False)  # the time labels sit under the rows chart
        top.hideButtons()
        top.setMenuEnabled(False)
        bottom = self.coverage_canvas.addPlot(
            row=1, col=0, axisItems={"bottom": TzDateAxisItem(tz)}
        )
        bottom.hideButtons()
        bottom.setMenuEnabled(False)
        # Same left-axis width on both charts so their time axes line up.
        width = 60 + 6 * max((len(n) for n in names), default=0)
        for plot in (top, bottom):
            plot.getAxis("left").setWidth(width)
        centres = x0 + (np.arange(n_bins) + 0.5) * cov.bin_width.total_seconds()
        bottom.plot(centres, cov.rows, pen=pg.mkPen(PALETTE[0], width=1.5), stepMode=None)
        bottom.setLabel("left", "Rows / bin")
        bottom.setYRange(0, max(float(cov.rows.max()), 1.0) * 1.1, padding=0)
        bottom.setXLink(top)
        self.coverage_canvas.ci.layout.setRowStretchFactor(0, 3)

    def _update_xy(self) -> None:
        xy = self._result.xy
        if xy is None:
            self.xy_note.setText(
                "No target values available. Import lab data and mark a target to screen X–Y relationships."
            )
            self.xy_table.set_frame(pd.DataFrame())
            return
        self.xy_note.setText(
            f"<b>Preliminary screening.</b> {xy.note} Pairing tolerance: {xy.tolerance}. "
            f"{xy.n_target_samples} target samples. Correlations are recalculated on aligned features in Signal Processing."
        )
        table = xy.table.copy()
        table.insert(0, "units", [self._units.get(n) or "" for n in table.index])
        self.xy_table.set_frame(
            table,
            {
                "pearson": "Pearson r",
                "spearman": "Spearman ρ",
                "n_pairs": "Pairs",
                "units": "Units",
            },
        )
