"""Stage 6 page: model families, hyperparameter search, scaling and the chronological split."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDateTime, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.core.models import ERROR, check_feasibility
from inferential_studio.core.schemas import (
    MODEL_LABELS,
    CvKind,
    ModelKind,
    ModellingConfig,
    ModelSpec,
    Scaling,
    SchemaError,
    SplitConfig,
    SplitMode,
)
from inferential_studio.core.splits import (
    PURGED,
    TEST,
    TRAIN,
    VALIDATION,
    SplitError,
    chronological_split,
)
from inferential_studio.ui.commands import edit_lists
from inferential_studio.ui.pages.stage_page import StagePage
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.stages import Stage
from inferential_studio.ui.widgets.plots import (
    SPLIT_COLOURS,
    SPLIT_NAMES,
    legend,
    note_label,
    plot_widget,
    scatter,
)
from inferential_studio.ui.widgets.time_axis import to_epoch_seconds

MODEL_NOTES = {
    ModelKind.MLR: "Reference model. Unstable when predictors are strongly correlated.",
    ModelKind.PLS: "Latent components chosen to predict Y; copes with correlated predictors.",
    ModelKind.PCR: "Components that summarise X (not chosen for Y), then regression.",
    ModelKind.RIDGE: "Shrinks coefficients; alpha sets the strength (searched on a log grid).",
}


class _ModelRow:
    """Widgets editing one ModelSpec."""

    def __init__(self, kind: ModelKind) -> None:
        self.kind = kind
        self.enabled = QCheckBox(MODEL_LABELS[kind])
        self.enabled.setToolTip(MODEL_NOTES[kind])
        self.mode = QComboBox()
        self.value: QSpinBox | QDoubleSpinBox | None = None
        self.range_lo: QDoubleSpinBox | QSpinBox | None = None
        self.range_hi: QDoubleSpinBox | None = None
        self.steps: QSpinBox | None = None
        if kind is ModelKind.MLR:
            self.mode.addItem("No hyperparameter", "none")
            self.mode.setEnabled(False)
            return
        self.mode.addItem("Tune by time-aware CV", "tune")
        self.mode.addItem("Fixed value", "fixed")
        if kind in (ModelKind.PLS, ModelKind.PCR):
            self.value = QSpinBox()
            self.value.setRange(1, 50)
            self.value.setPrefix("components: ")
            self.range_lo = QSpinBox()
            self.range_lo.setRange(1, 50)
            self.range_lo.setPrefix("search 1 … ")
        else:
            self.value = QDoubleSpinBox()
            self.value.setDecimals(6)
            self.value.setRange(1e-6, 1e6)
            self.value.setPrefix("alpha: ")
            self.range_lo = QDoubleSpinBox()
            self.range_lo.setDecimals(6)
            self.range_lo.setRange(1e-8, 1e6)
            self.range_lo.setPrefix("from ")
            self.range_hi = QDoubleSpinBox()
            self.range_hi.setDecimals(3)
            self.range_hi.setRange(1e-6, 1e8)
            self.range_hi.setPrefix("to ")
            self.steps = QSpinBox()
            self.steps.setRange(2, 50)
            self.steps.setSuffix(" values")

    def widgets(self) -> list[QWidget]:
        return [
            w
            for w in (self.enabled, self.mode, self.value, self.range_lo, self.range_hi, self.steps)
            if w
        ]

    def load(self, spec: ModelSpec) -> None:
        for w in self.widgets():
            w.blockSignals(True)
        self.enabled.setChecked(spec.enabled)
        if self.kind in (ModelKind.PLS, ModelKind.PCR):
            fixed = spec.n_components is not None
            self.mode.setCurrentIndex(1 if fixed else 0)
            self.value.setValue(spec.n_components or 2)
            self.range_lo.setValue(spec.max_components)
        elif self.kind is ModelKind.RIDGE:
            fixed = spec.alpha is not None
            self.mode.setCurrentIndex(1 if fixed else 0)
            self.value.setValue(spec.alpha or 1.0)
            self.range_lo.setValue(spec.alpha_min)
            self.range_hi.setValue(spec.alpha_max)
            self.steps.setValue(spec.alpha_steps)
        for w in self.widgets():
            w.blockSignals(False)
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        if self.value is None:
            return
        fixed = self.mode.currentData() == "fixed"
        self.value.setVisible(fixed)
        for w in (self.range_lo, self.range_hi, self.steps):
            if w is not None:
                w.setVisible(not fixed)

    def spec(self) -> ModelSpec:
        self._sync_enabled()
        fixed = self.mode.currentData() == "fixed"
        if self.kind is ModelKind.MLR:
            return ModelSpec(self.kind, enabled=self.enabled.isChecked())
        if self.kind in (ModelKind.PLS, ModelKind.PCR):
            return ModelSpec(
                self.kind,
                enabled=self.enabled.isChecked(),
                n_components=self.value.value() if fixed else None,
                max_components=self.range_lo.value(),
            )
        return ModelSpec(
            self.kind,
            enabled=self.enabled.isChecked(),
            alpha=self.value.value() if fixed else None,
            alpha_min=self.range_lo.value(),
            alpha_max=self.range_hi.value(),
            alpha_steps=self.steps.value(),
        )


class ModelSetupPage(StagePage):
    train_requested = Signal()

    def __init__(
        self,
        stage: Stage,
        session: ProjectSession,
        undo_stack: QUndoStack,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(stage, parent)
        self._session = session
        self._undo = undo_stack
        self._loading = False

        # -- models -------------------------------------------------------------------
        self.rows = {kind: _ModelRow(kind) for kind in ModelKind}
        models_box = QGroupBox("Models to compare")
        grid = QGridLayout(models_box)
        for r, row in enumerate(self.rows.values()):
            grid.addWidget(row.enabled, 2 * r, 0)
            grid.addWidget(row.mode, 2 * r, 1)
            params = QHBoxLayout()
            params.setContentsMargins(24, 0, 0, 6)
            for w in row.widgets()[2:]:
                params.addWidget(w)
            params.addStretch(1)
            grid.addLayout(params, 2 * r + 1, 0, 1, 2)
            for w in row.widgets():
                self._watch(w)
            row.mode.currentIndexChanged.connect(lambda _i, rr=row: rr._sync_enabled())
        grid.setColumnStretch(1, 1)

        # -- preprocessing and split --------------------------------------------------
        self.scaling = QComboBox()
        self.scaling.addItem("Standardise (mean 0, SD 1)", Scaling.STANDARD.value)
        self.scaling.addItem("Robust (median, IQR)", Scaling.ROBUST.value)
        self.scaling.addItem("None (original units)", Scaling.NONE.value)
        self._watch(self.scaling)
        self.mode = QComboBox()
        self.mode.addItem("By share of eligible samples", SplitMode.FRACTIONS.value)
        self.mode.addItem("By dates", SplitMode.DATES.value)
        self._watch(self.mode)
        self.train_pct = QDoubleSpinBox()
        self.train_pct.setRange(5, 95)
        self.train_pct.setSuffix(" % training")
        self.val_pct = QDoubleSpinBox()
        self.val_pct.setRange(0, 90)
        self.val_pct.setSuffix(" % validation")
        self.train_end = QDateTimeEdit()
        self.test_start = QDateTimeEdit()
        for w in (self.train_end, self.test_start):
            w.setDisplayFormat("yyyy-MM-dd HH:mm")
            w.setCalendarPopup(True)
        self.auto_purge = QCheckBox("Automatic purge gap")
        self.purge = QDoubleSpinBox()
        self.purge.setRange(0, 100000)
        self.purge.setSuffix(" min")
        self.cv_kind = QComboBox()
        self.cv_kind.addItem("Expanding window (rolling origin)", CvKind.EXPANDING.value)
        self.cv_kind.addItem("Blocked k-fold (contiguous blocks)", CvKind.BLOCKED.value)
        self.cv_folds = QSpinBox()
        self.cv_folds.setRange(2, 20)
        self.one_se = QCheckBox("Prefer the simplest setting within one standard error")
        for w in (self.train_pct, self.val_pct, self.train_end, self.test_start, self.auto_purge,
                  self.purge, self.cv_kind, self.cv_folds, self.one_se):  # fmt: skip
            self._watch(w)
        split_box = QGroupBox("Chronological split and cross-validation")
        form = QFormLayout(split_box)
        self._split_form = form
        form.addRow("Scaling", self.scaling)
        form.addRow("Split", self.mode)
        self.fraction_row = QWidget()
        fr = QHBoxLayout(self.fraction_row)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.addWidget(self.train_pct)
        fr.addWidget(self.val_pct)
        self.test_pct = QLabel()
        fr.addWidget(self.test_pct)
        form.addRow("Shares", self.fraction_row)
        self.dates_row = QWidget()
        dr = QHBoxLayout(self.dates_row)
        dr.setContentsMargins(0, 0, 0, 0)
        dr.addWidget(QLabel("validation from"))
        dr.addWidget(self.train_end)
        dr.addWidget(QLabel("final test from"))
        dr.addWidget(self.test_start)
        form.addRow("Boundaries", self.dates_row)
        prow = QHBoxLayout()
        prow.addWidget(self.auto_purge)
        prow.addWidget(self.purge)
        form.addRow("Purge gap", prow)
        form.addRow("CV (training only)", self.cv_kind)
        form.addRow("CV folds", self.cv_folds)
        form.addRow("", self.one_se)
        self.config_error = note_label()

        self.train_button = QPushButton("Train models")
        self.train_button.setMinimumHeight(34)
        self.train_button.clicked.connect(self.train_requested.emit)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(models_box)
        ll.addWidget(split_box)
        ll.addWidget(self.config_error)
        ll.addWidget(
            note_label(
                "Hyperparameters are tuned only inside the training period. The validation period "
                "ranks the models; the final test period stays untouched until you choose to "
                "reveal it on the Validation page. The purge gap drops samples whose feature "
                "windows would share process data with the next period."
            )
        )
        ll.addWidget(self.train_button)
        ll.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(left)

        # -- split preview ------------------------------------------------------------
        self.split_plot = plot_widget(
            "Lab samples by period", time_axis=session.project.display_timezone
        )
        legend(self.split_plot, "bottom-right")
        self.split_note = note_label()
        self.problems = note_label()
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.split_plot, 1)
        rl.addWidget(self.split_note)
        rl.addWidget(self.problems)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(right)
        split.setSizes([520, 640])
        self.body.addWidget(split, 1)

        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(350)
        self._apply_timer.timeout.connect(self.apply)

        session.changed.connect(self._load)
        session.modelling_changed.connect(self._load)
        session.features_changed.connect(self._update_preview)
        self._load()

    def _watch(self, w: QWidget) -> None:
        for name in ("toggled", "currentIndexChanged", "valueChanged", "dateTimeChanged"):
            sig = getattr(w, name, None)
            if sig is not None:
                sig.connect(self._edited)

    @Slot()
    def _edited(self, *_args: object) -> None:
        if not self._loading:
            self._sync_visibility()
            self._apply_timer.start()

    def _tz(self) -> ZoneInfo:
        return ZoneInfo(self._session.project.display_timezone)

    def _to_qt(self, moment: pd.Timestamp) -> QDateTime:
        local = moment.tz_convert(self._tz())
        return QDateTime(local.year, local.month, local.day, local.hour, local.minute, 0)

    def _from_qt(self, value: QDateTime) -> pd.Timestamp:
        naive = pd.Timestamp(value.toPython())
        return naive.tz_localize(
            self._tz(), ambiguous=True, nonexistent="shift_forward"
        ).tz_convert("UTC")

    def _sync_visibility(self) -> None:
        dates = self.mode.currentData() == SplitMode.DATES.value
        self._split_form.setRowVisible(self.fraction_row, not dates)
        self._split_form.setRowVisible(self.dates_row, dates)
        self.purge.setEnabled(not self.auto_purge.isChecked())
        test = 100 - self.train_pct.value() - self.val_pct.value()
        self.test_pct.setText(f"{test:g} % final test")

    @Slot()
    def _load(self) -> None:
        cfg = self._session.project.modelling
        self._loading = True
        for kind, row in self.rows.items():
            row.load(cfg.spec(kind) or ModelSpec(kind, enabled=False))
        sp = cfg.split
        self.scaling.setCurrentIndex(self.scaling.findData(cfg.scaling.value))
        self.mode.setCurrentIndex(self.mode.findData(sp.mode.value))
        self.train_pct.setValue(sp.train_fraction * 100)
        self.val_pct.setValue(sp.validation_fraction * 100)
        if sp.train_end is not None:
            self.train_end.setDateTime(self._to_qt(pd.Timestamp(sp.train_end)))
        if sp.validation_end is not None:
            self.test_start.setDateTime(self._to_qt(pd.Timestamp(sp.validation_end)))
        self.auto_purge.setChecked(sp.purge_s is None)
        self.purge.setValue((sp.purge_s or 0) / 60)
        self.cv_kind.setCurrentIndex(self.cv_kind.findData(sp.cv_kind.value))
        self.cv_folds.setValue(sp.cv_folds)
        self.one_se.setChecked(cfg.one_standard_error)
        self._loading = False
        self._sync_visibility()
        self._update_preview()

    def form_config(self) -> ModellingConfig:
        dates = self.mode.currentData() == SplitMode.DATES.value
        if dates and self._session.project.modelling.split.mode is SplitMode.FRACTIONS:
            self._seed_dates()
        split = SplitConfig(
            mode=self.mode.currentData(),
            train_fraction=self.train_pct.value() / 100,
            validation_fraction=self.val_pct.value() / 100,
            train_end=self._from_qt(self.train_end.dateTime()).to_pydatetime() if dates else None,
            validation_end=self._from_qt(self.test_start.dateTime()).to_pydatetime()
            if dates
            else None,
            purge_s=None if self.auto_purge.isChecked() else self.purge.value() * 60,
            cv_kind=self.cv_kind.currentData(),
            cv_folds=self.cv_folds.value(),
        )
        return ModellingConfig(
            scaling=self.scaling.currentData(),
            split=split,
            models=tuple(row.spec() for row in self.rows.values()),
            one_standard_error=self.one_se.isChecked(),
        )

    def _seed_dates(self) -> None:
        """Switching to dates: start from the boundaries of the current share-based split."""
        fm = self._session.features
        if fm is None or not fm.eligible.any():
            return
        try:
            res = chronological_split(fm.sample_time, fm.eligible, self._session.project.modelling.split,
                                      fm.max_history_s())  # fmt: skip
        except SplitError:
            return
        self._loading = True
        last = fm.sample_time[fm.eligible].max()
        self.train_end.setDateTime(self._to_qt(res.validation_start or res.test_start or last))
        self.test_start.setDateTime(self._to_qt(res.test_start or last))
        self._loading = False

    @Slot()
    def apply(self) -> bool:
        try:
            cfg = self.form_config()
        except SchemaError as exc:
            self.config_error.setText(f"<span style='color:#B00020'>{exc}</span>")
            return False
        self.config_error.setText("")
        edit_lists(self._undo, self._session, "Model settings", modelling=cfg)
        return True

    @Slot()
    def _update_preview(self) -> None:
        self.split_plot.clear()
        fm = self._session.features
        cfg = self._session.project.modelling
        if fm is None:
            self.split_note.setText(self._session.features_error or "Aligning features…")
            self.problems.setText("")
            self.train_button.setEnabled(False)
            return
        try:
            res = chronological_split(fm.sample_time, fm.eligible, cfg.split, fm.max_history_s())
        except SplitError as exc:
            self.split_note.setText(str(exc))
            self.train_button.setEnabled(False)
            return
        x = to_epoch_seconds(fm.sample_time)
        y = np.where(np.isfinite(fm.y), fm.y, np.nan)
        for label in (TRAIN, VALIDATION, TEST, PURGED):
            m = res.labels == label
            if m.any():
                scatter(self.split_plot, x[m], y[m], SPLIT_COLOURS[label],
                        SPLIT_NAMES.get(label, "Purge gap"), size=6)  # fmt: skip
        for boundary in (res.validation_start, res.test_start):
            if boundary is not None:
                pos = to_epoch_seconds(pd.DatetimeIndex([boundary]))[0]
                self.split_plot.addItem(pg.InfiniteLine(pos, angle=90, pen=pg.mkPen("#555555", style=Qt.PenStyle.DashLine)))  # fmt: skip
        self.split_plot.setLabel("left", f"{fm.target}")
        counts = res.counts()
        self.split_note.setText(
            f"Training {counts[TRAIN]}, validation {counts[VALIDATION]}, final test {counts[TEST]} "
            f"samples; {counts[PURGED]} in purge gaps of {res.purge_s / 60:g} min; "
            f"{counts['not eligible']} not eligible. Longest feature history: "
            f"{fm.max_history_s() / 60:g} min."
        )
        if self.auto_purge.isChecked():
            self._loading = True
            self.purge.setValue(res.purge_s / 60)
            self._loading = False
        tr = res.labels == TRAIN
        problems = check_feasibility(fm.X[tr].reset_index(drop=True), fm.y[tr], res)
        lines = [("✖ " if p.severity == ERROR else "⚠ ") + p.message for p in problems]
        if fm.non_causal:
            lines.append("⚠ Non-causal features: " + ", ".join(fm.non_causal))
        if not any(m.enabled for m in cfg.models):
            lines.append("✖ Select at least one model.")
        self.problems.setText("<br>".join(lines) or "Ready to train.")
        blocked = any(p.severity == ERROR for p in problems) or not any(
            m.enabled for m in cfg.models
        )
        self.train_button.setEnabled(not blocked)
