"""Stage 2 page: raw exploratory data analysis (trends, statistics, correlations, coverage)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTabWidget,
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
from inferential_studio.ui.pages.stage_page import StagePage
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.stages import Stage
from inferential_studio.ui.widgets.data_table import DataTableView
from inferential_studio.ui.widgets.heatmap import CorrelationHeatmap
from inferential_studio.ui.widgets.time_axis import TzDateAxisItem, to_epoch_seconds
from inferential_studio.ui.widgets.trend_view import PALETTE, TrendSeries, TrendView

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
    "dominated by shutdowns or operating-regime changes. Hover a cell for r and the number of pairs; "
    "click it for a scatter plot."
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
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Variables"))
        ll.addWidget(self.search)
        ll.addWidget(self.var_list, 1)
        ll.addWidget(self.show_target)

        # -- tabs -----------------------------------------------------------------------
        self.tabs = QTabWidget()
        self.trends = TrendView(session.project.display_timezone)
        self.tabs.addTab(self.trends, "Trends")

        self.stats_table = DataTableView()
        stats_tab = QWidget()
        sl = QVBoxLayout(stats_tab)
        note = QLabel(STATS_NOTE)
        note.setWordWrap(True)
        note.setObjectName("placeholderNote")
        sl.addWidget(self.stats_table, 1)
        sl.addWidget(note)
        self.tabs.addTab(stats_tab, "Statistics")

        self.method = QComboBox()
        self.method.addItem("Pearson (linear)", "pearson")
        self.method.addItem("Spearman (rank / monotonic)", "spearman")
        self.absolute = QCheckBox("Absolute |r|")
        self.cluster = QCheckBox("Group similar variables")
        self.cluster.setToolTip("Hierarchical ordering on 1 − |r| (average linkage)")
        self.only_selected = QCheckBox("Only checked variables")
        for w in (self.method,):
            w.currentIndexChanged.connect(self._update_correlation)
        for w in (self.absolute, self.cluster, self.only_selected):
            w.toggled.connect(self._update_correlation)
        self.heatmap = CorrelationHeatmap()
        self.heatmap.cell_clicked.connect(self._show_pair)
        self.pair_plot = pg.PlotWidget(background="w")
        self.pair_plot.showGrid(x=True, y=True, alpha=0.15)
        self.pair_note = QLabel("Click a heatmap cell to see the scatter plot.")
        self.pair_note.setWordWrap(True)
        corr_tab = QWidget()
        cl = QVBoxLayout(corr_tab)
        controls = QHBoxLayout()
        for w in (self.method, self.absolute, self.cluster, self.only_selected):
            controls.addWidget(w)
        controls.addStretch(1)
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
        self.tabs.addTab(corr_tab, "Correlation")

        self.matrix = pg.GraphicsLayoutWidget()
        self.matrix.setBackground("w")
        self.matrix_note = QLabel()
        matrix_tab = QWidget()
        ml = QVBoxLayout(matrix_tab)
        ml.addWidget(self.matrix, 1)
        ml.addWidget(self.matrix_note)
        self.tabs.addTab(matrix_tab, "Scatter matrix")

        self.hist_var = QComboBox()
        self.hist_var.currentIndexChanged.connect(self._update_histogram)
        self.hist_plot = pg.PlotWidget(background="w")
        self.hist_plot.showGrid(x=True, y=True, alpha=0.15)
        hist_tab = QWidget()
        hl = QVBoxLayout(hist_tab)
        hl.addWidget(self.hist_var)
        hl.addWidget(self.hist_plot, 1)
        self.tabs.addTab(hist_tab, "Distribution")

        self.coverage_canvas = pg.GraphicsLayoutWidget()
        self.coverage_canvas.setBackground("w")
        cov_tab = QWidget()
        cvl = QVBoxLayout(cov_tab)
        cvl.addWidget(self.coverage_canvas, 1)
        cov_note = QLabel(
            "Top: share of rows with a valid value per variable and time bin (green = complete, white = "
            "none, grey = no rows). Bottom: rows (timestamps) per bin."
        )
        cov_note.setWordWrap(True)
        cov_note.setObjectName("placeholderNote")
        cvl.addWidget(cov_note)
        self.tabs.addTab(cov_tab, "Coverage")

        self.xy_table = DataTableView()
        self.xy_note = QLabel()
        self.xy_note.setWordWrap(True)
        xy_tab = QWidget()
        xl = QVBoxLayout(xy_tab)
        xl.addWidget(self.xy_note)
        xl.addWidget(self.xy_table, 1)
        self.tabs.addTab(xy_tab, "X–Y screening")
        self.tabs.currentChanged.connect(lambda _i: self._update_visible_tab())

        self.status = QLabel()
        self.status.setObjectName("placeholderNote")
        self.recalc = QPushButton("Recalculate")
        self.recalc.clicked.connect(self.ensure_computed)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.recalc)

        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(left)
        main.addWidget(self.tabs)
        main.setSizes([220, 900])
        self.body.addLayout(status_row)
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
        if self._frame.empty:
            self.status.setText("No process data loaded. Import data on the Data Import page.")
        else:
            n_lab = 0 if self._target is None else int(self._target.notna().sum())
            self.status.setText(
                f"{len(self._frame):,} process rows × {self._frame.shape[1]} variables; {n_lab:,} target samples."
            )
        self._update_trends()
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
        self.status.setText(self.status.text().split("  —")[0] + "  — calculating statistics…")
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
        self.status.setText(self.status.text().split("  —")[0])
        self._update_visible_tab(all_tabs=True)

    def _on_failed(self, message: str) -> None:
        self._pending = None
        self.status.setText(self.status.text().split("  —")[0] + f"  — {message}")

    # ------------------------------------------------------------------ views

    @Slot()
    def _on_selection_changed(self, _item=None) -> None:  # noqa: ANN001
        self._update_trends()
        if self.only_selected.isChecked():
            self._update_correlation()
        self._update_matrix()

    def _update_visible_tab(self, all_tabs: bool = False) -> None:
        if self._result is None:
            return
        self._update_stats()
        self._update_correlation()
        self._update_matrix()
        self._update_histogram()
        self._update_coverage()
        self._update_xy()

    def _update_trends(self) -> None:
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
        self.trends.set_series(series)

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
        if self.only_selected.isChecked():
            keep = [n for n in self.selected_variables() if n in corr.columns]
            corr, pairs = corr.loc[keep, keep], pairs.loc[keep, keep]
        if self.cluster.isChecked() and len(corr) > 2:
            order = hierarchical_order(corr)
            corr, pairs = corr.loc[order, order], pairs.loc[order, order]
        self.heatmap.set_matrix(corr, pairs, absolute=self.absolute.isChecked())

    @Slot(str, str)
    def _show_pair(self, row_var: str, col_var: str) -> None:
        self.pair_plot.clear()
        if row_var not in self._frame or col_var not in self._frame:
            return
        pair = (
            self._frame[[col_var, row_var]].dropna()
            if row_var != col_var
            else self._frame[[col_var]].dropna()
        )
        if row_var == col_var or pair.empty:
            self.pair_note.setText(f"{row_var}: choose a cell off the diagonal.")
            return
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
        rgb[..., 0] = (255 - f * (255 - 0)).astype(np.uint8)
        rgb[..., 1] = (255 - f * (255 - 158)).astype(np.uint8)
        rgb[..., 2] = (255 - f * (255 - 115)).astype(np.uint8)
        rgb[np.isnan(frac)] = (200, 200, 200)
        x0 = to_epoch_seconds(pd.DatetimeIndex([cov.bin_starts[0]]))[0]
        width = cov.bin_width.total_seconds() * n_bins
        tz = self._session.project.display_timezone
        top = self.coverage_canvas.addPlot(row=0, col=0, axisItems={"bottom": TzDateAxisItem(tz)})
        img = pg.ImageItem(rgb.transpose(1, 0, 2))
        img.setRect(QRectF(x0, 0, width, n_vars))
        top.addItem(img)
        top.invertY(True)
        names = list(cov.valid_fraction.columns)
        top.getAxis("left").setTicks([[(i + 0.5, n) for i, n in enumerate(names)]])
        bottom = self.coverage_canvas.addPlot(
            row=1, col=0, axisItems={"bottom": TzDateAxisItem(tz)}
        )
        centres = x0 + (np.arange(n_bins) + 0.5) * cov.bin_width.total_seconds()
        bottom.plot(centres, cov.rows, pen=pg.mkPen(PALETTE[0]), stepMode=None)
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
