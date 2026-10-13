"""Stage 5 page: per-predictor deadtime, window, aggregation, filtering and lab alignment."""

from __future__ import annotations

import contextlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.core.features import (
    OK,
    FeatureMatrix,
    LagScan,
    data_span_s,
    feature_specs,
    lag_scan,
    process_signal,
    target_samples,
)
from inferential_studio.core.schemas import (
    Aggregation,
    FeatureSpec,
    FilterKind,
    FilterSpec,
    SchemaError,
    WindowPlacement,
)
from inferential_studio.core.splits import TRAIN, SplitError, chronological_split
from inferential_studio.infrastructure.workers import TaskManager
from inferential_studio.ui.commands import edit_lists
from inferential_studio.ui.messages import Level
from inferential_studio.ui.pages.stage_page import StagePage
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.stages import Stage
from inferential_studio.ui.theme import mark_primary, theme_colour
from inferential_studio.ui.widgets.data_table import DataTableView
from inferential_studio.ui.widgets.flow_layout import flow_row
from inferential_studio.ui.widgets.plots import (
    BLUE,
    FAINT,
    GREY,
    VERMILION,
    legend,
    note_label,
    plot_widget,
    scatter,
)
from inferential_studio.ui.widgets.time_axis import format_time, to_epoch_seconds

NO_PREDICTOR_TEXT = "Choose a target and predictors on Variable Selection to preview them."
NO_LAB_TEXT = "No aligned lab samples to show."

COLUMNS = [
    "Variable",
    "Deadtime",
    "Shift",
    "Window",
    "Placement",
    "Aggregation",
    "Filter",
    "Resample",
    "Min coverage",
    "Max gap",
    "Uses data from",
    "Causal",
    "OK samples",
]
AGG_LABELS = {
    Aggregation.MEAN: "Mean",
    Aggregation.MEDIAN: "Median",
    Aggregation.MIN: "Minimum",
    Aggregation.MAX: "Maximum",
    Aggregation.STD: "Standard deviation",
    Aggregation.LAST: "Last valid value",
}
FILTER_LABELS = {
    FilterKind.NONE: "None",
    FilterKind.ROLLING_MEAN: "Rolling mean (trailing)",
    FilterKind.ROLLING_MEDIAN: "Rolling median (trailing)",
    FilterKind.EWMA: "EWMA (half-life)",
}
COVERAGE_HEADERS = {
    "samples": "Samples with valid Y",
    "ok": "OK",
    "no data": "No data",
    "low coverage": "Low coverage",
    "gap too long": "Gap too long",
    "too few values": "Too few values",
    "overlaps excluded period": "Overlaps exclusion",
    "mean_coverage_pct": "Mean coverage %",
    "causal": "Causal",
}
RAW_COLOUR = "#9a9a9a"
PROCESSED_COLOUR = BLUE
WINDOW_COLOUR = (230, 159, 0, 60)


def _minutes(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return f"{seconds / 60:g} min"


def _item(text: str, data: object = None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
    if data is not None:
        item.setData(Qt.ItemDataRole.UserRole, data)
    return item


def describe_window(spec: FeatureSpec) -> str:
    start, end = data_span_s(spec)

    def rel(s: float) -> str:
        if s == 0:
            return "t"
        return f"t {'−' if s < 0 else '+'} {abs(s) / 60:g} min"

    return f"({rel(start)}, {rel(end)}]"


def _lag_task(target, raw, spec, delays, excluded, cutoff, **kw):  # noqa: ANN001, ANN201
    return lag_scan(
        target, raw, spec, delays, excluded=excluded, cutoff=cutoff, cancel=kw.get("cancel")
    )


class ProcessingPage(StagePage):
    def __init__(
        self,
        stage: Stage,
        session: ProjectSession,
        tasks: TaskManager,
        undo_stack: QUndoStack,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(stage, parent)
        self._session = session
        self._tasks = tasks
        self._undo = undo_stack
        self._sample = 0  # selected lab sample (row of the feature matrix)
        self._preview_stale = True
        self._lag: LagScan | None = None

        # -- settings table -------------------------------------------------------
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._selection_changed)

        # -- editor -----------------------------------------------------------------
        self.deadtime = self._spin(0, 2880, " min", "Process deadtime: how long a change in X takes to "
                                   "show in Y. Moves the window back from the sample time.")  # fmt: skip
        self.shift = self._spin(-2880, 2880, " min", "Extra signed shift added to the deadtime. "
                                "Positive = further into the past; negative may read future data.")  # fmt: skip
        self.window = self._spin(0.1, 10080, " min", "Length of the aggregation window.")
        self.placement = QComboBox()
        self.placement.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.placement.setMinimumContentsLength(14)
        self.placement.addItem("Lookback (ends at t − delay)", WindowPlacement.LOOKBACK.value)
        self.placement.addItem("Centred (non-causal)", WindowPlacement.CENTERED.value)
        self.aggregation = QComboBox()
        for agg, label in AGG_LABELS.items():
            self.aggregation.addItem(label, agg.value)
        self.filter_kind = QComboBox()
        for kind, label in FILTER_LABELS.items():
            self.filter_kind.addItem(label, kind.value)
        self.filter_len = self._spin(0, 1440, " min", "Rolling window or EWMA half-life.")
        self.resample = self._spin(0, 1440, " min", "Average into bins ending at each interval.")
        self.resample.setSpecialValueText("Original sampling")
        self.coverage = self._spin(0, 100, " %", "Minimum share of the window holding valid data.")
        self.max_gap = self._spin(0, 1440, " min", "Longest time without data inside the window.")
        self.max_gap.setSpecialValueText("No limit")
        form = QFormLayout()
        form.addRow("Deadtime", self.deadtime)
        form.addRow("Shift", self.shift)
        form.addRow("Window", self.window)
        form.addRow("Placement", self.placement)
        form.addRow("Aggregation", self.aggregation)
        form.addRow("Filter", self.filter_kind)
        form.addRow("Filter length", self.filter_len)
        form.addRow("Resample", self.resample)
        form.addRow("Min coverage", self.coverage)
        form.addRow("Max gap", self.max_gap)
        self.apply_selected = QPushButton("Apply to selected")
        self.apply_selected.clicked.connect(lambda: self.apply_form(all_rows=False))
        mark_primary(self.apply_selected)
        self.apply_all = QPushButton("Apply to all")
        self.apply_all.clicked.connect(lambda: self.apply_form(all_rows=True))
        self.regime = QCheckBox("Reject windows touching regime exclusions")
        self.regime.setToolTip(
            "Operating-regime exclusions (shutdowns, start-ups and other exclusions of all "
            "signals) make a sample ineligible when any predictor window touches them, even if "
            "coverage would pass."
        )
        self.regime.toggled.connect(self._regime_toggled)
        editor = QGroupBox("Settings for the selected predictors")
        self.editor_box = editor
        el = QVBoxLayout(editor)
        el.addLayout(form)
        brow = QHBoxLayout()
        brow.addWidget(self.apply_selected)
        brow.addWidget(self.apply_all)
        el.addLayout(brow)
        el.addWidget(self.regime)
        el.addStretch(1)

        # -- preview / diagnostics tabs ------------------------------------------------
        self.tabs = QTabWidget()
        tz = session.project.display_timezone
        self.x_plot = plot_widget(time_axis=tz, name="preview_x", empty_text=NO_PREDICTOR_TEXT)
        self.y_plot = plot_widget(time_axis=tz, name="preview_y", empty_text=NO_LAB_TEXT)
        self.y_plot.setXLink(self.x_plot)
        legend(self.x_plot)
        self.sample_spin = QSpinBox()
        self.sample_spin.setPrefix("Lab sample ")
        self.sample_spin.valueChanged.connect(self._sample_changed)
        prev_btn = QPushButton("◀")
        next_btn = QPushButton("▶")
        prev_btn.clicked.connect(lambda: self.sample_spin.setValue(self.sample_spin.value() - 1))
        next_btn.clicked.connect(lambda: self.sample_spin.setValue(self.sample_spin.value() + 1))
        zoom_btn = QPushButton("Zoom to window")
        zoom_btn.clicked.connect(self._zoom_to_window)
        fit_btn = QPushButton("Fit all")
        fit_btn.clicked.connect(
            lambda: (self.x_plot.enableAutoRange(), self.y_plot.enableAutoRange())
        )
        self.sample_note = note_label()
        prev = QWidget()
        pl = QVBoxLayout(prev)
        pl.addWidget(flow_row(prev_btn, self.sample_spin, next_btn, zoom_btn, fit_btn))
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.x_plot)
        split.addWidget(self.y_plot)
        split.setSizes([300, 160])
        pl.addWidget(split, 1)
        pl.addWidget(self.sample_note)
        self.tabs.addTab(prev, "Preview")

        self.lag_plot = plot_widget(
            name="lag_sensitivity",
            empty_text="Run a scan to see the correlation for each tested delay.",
        )
        self.lag_plot.setLabel("bottom", "Effective delay (deadtime + shift)", units="min")
        self.lag_plot.setLabel("left", "Pearson r with Y")
        self.lag_max = self._spin(1, 2880, " min", "Largest delay tested.")
        self.lag_max.setValue(120)
        self.lag_step = self._spin(0.5, 240, " min", "Delay step.")
        self.lag_step.setValue(5)
        self.lag_run = QPushButton("Scan delays")
        self.lag_run.clicked.connect(self.run_lag_scan)
        self.lag_use = QPushButton("Use best delay as deadtime")
        self.lag_use.setEnabled(False)
        self.lag_use.clicked.connect(self._use_best_delay)
        self.lag_note = note_label()
        self.lag_note.setToolTip(LagScan.note)
        lag = QWidget()
        ll = QVBoxLayout(lag)
        span = QWidget()  # "Up to [ ] step [ ]" stays together when the row wraps
        sl = QHBoxLayout(span)
        sl.setContentsMargins(0, 0, 0, 0)
        for w in (QLabel("Up to"), self.lag_max, QLabel("step"), self.lag_step):
            sl.addWidget(w)
        ll.addWidget(flow_row(span, self.lag_run, self.lag_use))
        ll.addWidget(self.lag_plot, 1)
        ll.addWidget(self.lag_note)
        self.tabs.addTab(lag, "Lag sensitivity")

        self.coverage_table = DataTableView()
        cov = QWidget()
        cl = QVBoxLayout(cov)
        cl.addWidget(self.coverage_table, 1)
        cl.addWidget(
            note_label(
                "Coverage = valid values in the window × sampling interval ÷ window length. "
                "Gap = longest time in the window without a valid value (window edges count). "
                "Counts are lab samples with a valid, non-excluded target."
            )
        )
        self.tabs.addTab(cov, "Coverage report")

        self.samples_table = DataTableView()
        self._samples_version: object = None  # feature matrix shown in the table
        self.tabs.addTab(self.samples_table, "Feature matrix")
        self.tabs.currentChanged.connect(lambda _i: self._fill_samples())

        # Table above the preview on the left; the settings form runs the full height on the
        # right, so the preview charts get the vertical space instead of the form.
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(self.table)
        left.addWidget(self.tabs)
        left.setSizes([200, 480])
        left.setChildrenCollapsible(False)
        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(left)
        editor_scroll = QScrollArea()  # the form scrolls on short screens instead of clipping
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setFrameShape(QFrame.Shape.NoFrame)
        editor_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        editor_scroll.setWidget(editor)
        editor_scroll.setMinimumWidth(editor.minimumSizeHint().width() + 12)
        main.addWidget(editor_scroll)
        main.setSizes([820, 340])
        main.setStretchFactor(0, 1)
        main.setChildrenCollapsible(False)
        self.body.addWidget(main, 1)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._update_preview)

        session.changed.connect(self._refresh)
        session.inputs_changed.connect(self._refresh)
        session.features_changed.connect(self._on_features)
        self._refresh()
        self._on_features()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _spin(lo: float, hi: float, suffix: str, tip: str) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(1)
        s.setSuffix(suffix)
        s.setToolTip(tip)
        s.setKeyboardTracking(False)
        return s

    def specs(self) -> list[FeatureSpec]:
        return feature_specs(self._session.project)

    def selected_variables(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows]

    def current_variable(self) -> str | None:
        sel = self.selected_variables()
        if sel:
            return sel[0]
        return (
            self.table.item(0, 0).data(Qt.ItemDataRole.UserRole) if self.table.rowCount() else None
        )

    # ------------------------------------------------------------------ table and form

    @Slot()
    def _refresh(self) -> None:
        project = self._session.project
        self.regime.blockSignals(True)
        self.regime.setChecked(project.alignment.reject_regime_overlap)
        self.regime.blockSignals(False)
        selected = set(self.selected_variables())
        specs = self.specs()
        self.table.blockSignals(True)
        self.table.setRowCount(len(specs))
        for r, spec in enumerate(specs):
            f = spec.filter
            if f.kind is FilterKind.NONE:
                ftext = "None"
            elif f.kind is FilterKind.EWMA:
                ftext = f"EWMA, half-life {_minutes(f.halflife_s)}"
            else:
                ftext = f"{FILTER_LABELS[f.kind].split(' (')[0]}, {_minutes(f.window_s)}"
            cells = [
                spec.variable,
                _minutes(spec.deadtime_s),
                _minutes(spec.shift_s),
                _minutes(spec.window_s),
                "Lookback" if spec.placement is WindowPlacement.LOOKBACK else "Centred",
                AGG_LABELS[spec.aggregation],
                ftext,
                _minutes(spec.resample_s) or "Original",
                f"{spec.min_coverage * 100:g} %",
                _minutes(spec.max_gap_s) or "No limit",
                describe_window(spec),
                "Yes" if spec.is_causal else "NO — reads future data",
                "",
            ]
            for c, text in enumerate(cells):
                item = _item(text, spec.variable if c == 0 else None)
                if c == 11 and not spec.is_causal:
                    item.setForeground(theme_colour("error"))
                self.table.setItem(r, c, item)
            if spec.variable in selected:
                self.table.selectRow(r)
        self.table.blockSignals(False)
        if not self.selected_variables() and specs:
            self.table.selectRow(0)
        self._update_counts()
        self._load_form()
        self._session.messages.set(
            "processing.inputs", Level.INFO,
            "" if specs else "Choose a target and predictors on the Variable Selection page.",
        )  # fmt: skip
        self._preview_timer.start()

    def _update_counts(self) -> None:
        """OK-sample count per predictor (last column) from the current feature matrix."""
        fm = self._session.features
        usable = fm.y_status == "ok" if fm is not None else None
        for r in range(self.table.rowCount()):
            name = self.table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            text = ""
            if fm is not None and name in fm.X.columns:
                text = f"{int((fm.status[name].to_numpy()[usable] == OK).sum()):,}"
            self.table.setItem(r, len(COLUMNS) - 1, _item(text))

    def apply_theme(self) -> None:
        """Re-apply the error colour cached in the table cells (see ui/theme.py)."""
        for r, spec in enumerate(self.specs()[: self.table.rowCount()]):
            item = self.table.item(r, 11)
            if item is not None and not spec.is_causal:
                item.setForeground(theme_colour("error"))

    @Slot()
    def _selection_changed(self) -> None:
        self._load_form()
        self._lag = None
        self.lag_plot.clear()
        self.lag_use.setEnabled(False)
        self._preview_timer.start()

    def _load_form(self) -> None:
        name = self.current_variable()
        spec = next((s for s in self.specs() if s.variable == name), None)
        enabled = spec is not None
        for w in (self.apply_selected, self.apply_all):
            w.setEnabled(enabled)
        if spec is None:
            return
        self.deadtime.setValue(spec.deadtime_s / 60)
        self.shift.setValue(spec.shift_s / 60)
        self.window.setValue(spec.window_s / 60)
        self.placement.setCurrentIndex(self.placement.findData(spec.placement.value))
        self.aggregation.setCurrentIndex(self.aggregation.findData(spec.aggregation.value))
        self.filter_kind.setCurrentIndex(self.filter_kind.findData(spec.filter.kind.value))
        length = (
            spec.filter.halflife_s if spec.filter.kind is FilterKind.EWMA else spec.filter.window_s
        )
        self.filter_len.setValue((length or 0) / 60)
        self.resample.setValue((spec.resample_s or 0) / 60)
        self.coverage.setValue(spec.min_coverage * 100)
        self.max_gap.setValue((spec.max_gap_s or 0) / 60)
        n = len(self.selected_variables())
        self.editor_box.setTitle(f"Settings — {n} predictors" if n > 1 else f"Settings — {name}")

    def form_spec(self, variable: str) -> FeatureSpec:
        kind = FilterKind(self.filter_kind.currentData())
        length = self.filter_len.value() * 60 or None
        if kind is FilterKind.EWMA:
            filt = FilterSpec(kind, halflife_s=length)
        elif kind is FilterKind.NONE:
            filt = FilterSpec()
        else:
            filt = FilterSpec(kind, window_s=length)
        return FeatureSpec(
            variable=variable,
            window_s=self.window.value() * 60,
            deadtime_s=self.deadtime.value() * 60,
            shift_s=self.shift.value() * 60,
            placement=self.placement.currentData(),
            aggregation=self.aggregation.currentData(),
            filter=filt,
            resample_s=self.resample.value() * 60 or None,
            min_coverage=self.coverage.value() / 100,
            max_gap_s=self.max_gap.value() * 60 or None,
        )

    def apply_form(self, *, all_rows: bool) -> bool:
        specs = self.specs()
        targets = {s.variable for s in specs} if all_rows else set(self.selected_variables())
        if not targets:
            return False
        try:
            new = [self.form_spec(s.variable) if s.variable in targets else s for s in specs]
        except SchemaError as exc:
            self._session.messages.error(f"Feature settings not applied: {exc}", "processing")
            return False
        edit_lists(self._undo, self._session, "Edit feature settings", features=new)
        return True

    @Slot(bool)
    def _regime_toggled(self, on: bool) -> None:
        project = self._session.project
        cfg = replace(project.alignment, reject_regime_overlap=on)
        edit_lists(self._undo, self._session, "Alignment option", alignment=cfg)

    # ------------------------------------------------------------------ features

    @Slot()
    def _on_features(self) -> None:
        fm = self._session.features
        if fm is None:
            self.coverage_table.set_frame(pd.DataFrame())
            self.samples_table.set_frame(pd.DataFrame())
            self._preview_timer.start()
            return
        self.coverage_table.set_frame(fm.coverage_report(), COVERAGE_HEADERS)
        self._fill_samples()
        self.sample_spin.blockSignals(True)
        self.sample_spin.setRange(1, max(len(fm.y), 1))
        self.sample_spin.blockSignals(False)
        self._sample = min(self._sample, max(len(fm.y) - 1, 0))
        self._update_counts()
        self._preview_timer.start()

    def _fill_samples(self) -> None:
        """The per-sample table is long: fill it only while its tab is open."""
        fm = self._session.features
        if self.tabs.currentWidget() is not self.samples_table or fm is self._samples_version:
            return
        self._samples_version = fm
        self.samples_table.set_frame(fm.table() if fm is not None else pd.DataFrame())

    @Slot(int)
    def _sample_changed(self, value: int) -> None:
        self._sample = value - 1
        self._preview_timer.start()

    def showEvent(self, event) -> None:  # noqa: N802, ANN001
        super().showEvent(event)
        if self._preview_stale:
            self._preview_timer.start()

    def _update_preview(self) -> None:
        # Drawing a long raw series costs time: only the page on screen redraws.
        self._preview_stale = not self.isVisible()
        if self._preview_stale:
            return
        self.x_plot.clear()
        self.y_plot.clear()
        name = self.current_variable()
        frame, _mask, target = self._session.frames()
        if name is None or name not in frame.columns:
            self.sample_note.setText("")
            return
        spec = next(s for s in self.specs() if s.variable == name)
        state = self._session.cleaning
        excluded = state.excluded[name].to_numpy() if state is not None else None
        raw = frame[name]
        with contextlib.suppress(SchemaError):
            spec = self.form_spec(name) if self.selected_variables() == [name] else spec
        sig = process_signal(raw, spec, excluded)
        units = next((v.units for v in self._session.project.variables if v.name == name), None)
        clean = raw.where(~excluded) if excluded is not None else raw
        x = to_epoch_seconds(clean.index)
        raw_item = self.x_plot.plot(x, clean.to_numpy(dtype=float), pen=pg.mkPen(RAW_COLOUR),
                                    name=f"{name} (cleaned)", connect="finite")  # fmt: skip
        processed = sig.series()
        proc_item = self.x_plot.plot(
            to_epoch_seconds(processed.index), processed.to_numpy(), name="after resample/filter",
            pen=pg.mkPen(PROCESSED_COLOUR, width=1.5),
        )  # fmt: skip
        for item in (raw_item, proc_item):
            item.setDownsampling(auto=True, method="peak")
            item.setClipToView(True)
        self.x_plot.setLabel("left", name, units=units)
        fm = self._session.features
        if fm is None or not len(fm.y):
            self.sample_note.setText("No aligned lab samples to preview.")
            return
        tx = to_epoch_seconds(fm.sample_time)
        ok = fm.eligible
        y = fm.y
        if ok.any():
            scatter(self.y_plot, tx[ok], y[ok], PROCESSED_COLOUR, size=5)
        if (~ok).any():
            scatter(self.y_plot, tx[~ok], np.nan_to_num(y[~ok], nan=np.nanmean(y)), FAINT, size=5)
        self.y_plot.setLabel("left", "Y (lab)")
        k = self._sample
        start, end = spec.window_offsets_s
        t = tx[k]
        region = pg.LinearRegionItem(
            (t + start, t + end), movable=False, brush=pg.mkBrush(*WINDOW_COLOUR)
        )
        self.x_plot.addItem(region, ignoreBounds=True)
        for p in (self.x_plot, self.y_plot):
            p.addItem(
                pg.InfiniteLine(t, angle=90, pen=pg.mkPen(VERMILION, width=1.5)), ignoreBounds=True
            )
        scatter(
            self.y_plot, [t], [y[k] if np.isfinite(y[k]) else np.nanmean(y)], VERMILION, size=11
        )
        value, status = np.nan, ""
        if name in fm.X.columns:
            value = fm.X[name].iat[k]
            status = fm.status[name].iat[k]
            cov = fm.coverage[name].iat[k]
            gap = fm.gap_s[name].iat[k]
            if np.isfinite(value):
                self.x_plot.plot(
                    [t + start, t + end], [value, value], pen=pg.mkPen(VERMILION, width=3)
                )
        tz = self._session.project.display_timezone
        lines = [
            f"Sample {k + 1} at {format_time(t, tz, with_seconds=False)} ({tz})",
            f"Y = {y[k]:.6g}" if np.isfinite(y[k]) else f"Y: {fm.y_status[k]}",
            f"window {describe_window(spec)}",
        ]
        if status:
            lines.append(f"{name} = {value:.6g}" if np.isfinite(value) else f"{name}: {status}")
            lines.append(
                f"coverage {cov * 100:.0f} %, largest gap {gap / 60:.1f} min, status: {status}"
            )
        if fm.sample_id is not None:
            lines.insert(1, f"id {fm.sample_id[k]}")
        self.sample_note.setText(" · ".join(lines))

    def _zoom_to_window(self) -> None:
        fm = self._session.features
        name = self.current_variable()
        if fm is None or name is None or not len(fm.y):
            return
        spec = next(s for s in self.specs() if s.variable == name)
        t = to_epoch_seconds(fm.sample_time)[self._sample]
        start, end = data_span_s(spec)
        margin = max(end - start, 600)
        self.x_plot.setXRange(t + start - margin, t + max(end, 0) + margin, padding=0)

    # ------------------------------------------------------------------ lag sensitivity

    def training_cutoff(self) -> pd.Timestamp | None:
        """Last training sample time under the current split settings (no test data used)."""
        fm: FeatureMatrix | None = self._session.features
        if fm is None or not fm.eligible.any():
            return None
        try:
            split = chronological_split(
                fm.sample_time,
                fm.eligible,
                self._session.project.modelling.split,
                fm.max_history_s(),
            )
        except SplitError:
            return None
        train = fm.sample_time[split.labels == TRAIN]
        return train.max() if len(train) else None

    def run_lag_scan(self) -> None:
        name = self.current_variable()
        if name is None:
            return
        if self._session._cleaning_running or self._session._features_running:
            self._session.messages.warning("Wait for cleaning and feature alignment before scanning delays.", "processing")
            return
        cutoff = self.training_cutoff()
        if cutoff is None:
            self._session.messages.warning("Define a valid training period before scanning delays.", "processing")
            self._lag = None
            self.lag_use.setEnabled(False)
            return
        frame, _mask, _target = self._session.frames()
        try:
            target = target_samples(
                self._session.project, self._session.data, self._session.cleaning
            )
        except ValueError as exc:
            self._session.messages.error(f"Lag scan: {exc}", "processing")
            return
        spec = next(s for s in self.specs() if s.variable == name)
        state = self._session.cleaning
        excluded = state.excluded[name].to_numpy() if state is not None else None
        delays = np.arange(0, self.lag_max.value() * 60 + 1e-9, self.lag_step.value() * 60)
        self.lag_run.setEnabled(False)
        self._tasks.submit(
            f"Scanning delays for {name}",
            _lag_task,
            target,
            frame[name],
            spec,
            delays,
            excluded,
            cutoff,
            on_result=self._lag_done,
            on_error=lambda msg, _d: (
                self._session.messages.error(f"Lag scan failed: {msg}", "processing"),
                self.lag_run.setEnabled(True),
            ),
            on_cancelled=lambda: self.lag_run.setEnabled(True),
        )

    def _lag_done(self, scan: LagScan) -> None:
        self.lag_run.setEnabled(True)
        self._lag = scan
        self.lag_plot.clear()
        minutes = scan.delays_s / 60
        self.lag_plot.plot(minutes, scan.pearson, pen=pg.mkPen(PROCESSED_COLOUR, width=2),
                           symbol="o", symbolSize=4, symbolBrush=PROCESSED_COLOUR)  # fmt: skip
        spec = next((s for s in self.specs() if s.variable == scan.variable), None)
        if spec is not None:
            self.lag_plot.addItem(pg.InfiniteLine(spec.effective_delay_s / 60, angle=90,
                                                  pen=pg.mkPen(GREY, style=Qt.PenStyle.DashLine)))  # fmt: skip
        best = scan.best_delay_s
        cutoff = (
            f"training samples up to {format_time(to_epoch_seconds(pd.DatetimeIndex([scan.cutoff]))[0], self._session.project.display_timezone, with_seconds=False)}"
            if scan.cutoff is not None
            else "all samples (no split could be defined)"
        )
        if best is not None:
            i = int(np.flatnonzero(scan.delays_s == best)[0])
            scatter(self.lag_plot, [best / 60], [scan.pearson[i]], VERMILION, size=12)
            text = f"Largest |r| = {abs(scan.pearson[i]):.3f} at {best / 60:g} min ({cutoff}; n = {scan.n_pairs[i]})."
        else:
            text = f"Not enough samples to correlate ({cutoff})."
        self.lag_note.setText(f"{text} Dashed line: current setting.")
        self._session.messages.info(f"Lag scan for {scan.variable}: {text}", "processing")
        self.lag_use.setEnabled(best is not None and best >= 0 and scan.cutoff is not None)

    def _use_best_delay(self) -> None:
        scan = self._lag
        if scan is None or scan.best_delay_s is None or scan.cutoff is None:
            return
        specs = self.specs()
        new = [
            replace(s, deadtime_s=float(scan.best_delay_s), shift_s=0.0)
            if s.variable == scan.variable
            else s
            for s in specs
        ]
        edit_lists(self._undo, self._session, f"Deadtime for {scan.variable}", features=new)
