"""Stage 4 page: post-cleaning EDA and variable selection (target, predictors, VIF)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.core.diagnostics import (
    CORRELATION_WARNING,
    VIF_WARNING,
    AlignedDiagnostics,
    aligned_diagnostics,
    high_correlation_pairs,
    raw_vs_cleaned,
)
from inferential_studio.core.features import OK, Y_OK
from inferential_studio.core.schemas import SourceKind
from inferential_studio.core.statistics import preliminary_xy_correlation
from inferential_studio.infrastructure.workers import TaskManager
from inferential_studio.ui.commands import edit_lists
from inferential_studio.ui.messages import Level
from inferential_studio.ui.pages.stage_page import StagePage
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.stages import Stage
from inferential_studio.ui.widgets.data_table import DataTableView
from inferential_studio.ui.widgets.heatmap import CorrelationHeatmap
from inferential_studio.ui.widgets.plots import note_label
from inferential_studio.ui.widgets.variable_list import VariableList

RANKING_HEADERS = {
    "units": "Units",
    "pearson": "Pearson r (aligned)",
    "spearman": "Spearman ρ (aligned)",
    "n": "Samples",
    "vif": "VIF",
    "vif_note": "VIF note",
    "screen_raw": "Screening r (raw)",
    "screen_cleaned": "Screening r (cleaned)",
}
RAW_CLEAN_HEADERS = {
    "valid_raw": "Valid (raw)",
    "valid_cleaned": "Valid (cleaned)",
    "removed_pct": "Removed %",
    "mean_raw": "Mean (raw)",
    "mean_cleaned": "Mean (cleaned)",
    "std_raw": "Std (raw)",
    "std_cleaned": "Std (cleaned)",
    "min_raw": "Min (raw)",
    "min_cleaned": "Min (cleaned)",
    "max_raw": "Max (raw)",
    "max_cleaned": "Max (cleaned)",
    "r_y_raw": "Screening r with Y (raw)",
    "r_y_cleaned": "Screening r with Y (cleaned)",
}


def _raw_vs_cleaned_task(
    raw: pd.DataFrame, cleaned: pd.DataFrame, target: pd.Series | None, **_kw: object
) -> pd.DataFrame:
    table = raw_vs_cleaned(raw, cleaned)
    if target is not None and target.notna().any():
        table["r_y_raw"] = preliminary_xy_correlation(raw, target).table["pearson"]
        table["r_y_cleaned"] = preliminary_xy_correlation(cleaned, target).table["pearson"]
    return table


class VariablesPage(StagePage):
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
        self._diag: AlignedDiagnostics | None = None
        self._raw_clean_key: object = None
        self._raw_clean_pending = False
        self._raw_clean: pd.DataFrame | None = None

        # -- left: target and predictors ----------------------------------------
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(self._target_chosen)
        self.predictors = VariableList("Candidate predictors (X)")
        self.predictors.selection_changed.connect(lambda: self._apply_timer.start())
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(400)
        self._apply_timer.timeout.connect(self._apply_predictors)
        all_btn = QPushButton("All")
        none_btn = QPushButton("None")
        all_btn.clicked.connect(lambda: self._check_all(True))
        none_btn.clicked.connect(lambda: self._check_all(False))
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Target (Y)"))
        ll.addWidget(self.target_combo)
        ll.addWidget(self.predictors, 1)
        row = QHBoxLayout()
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch(1)
        ll.addLayout(row)

        # -- right: diagnostics tabs ---------------------------------------------
        self.tabs = QTabWidget()
        self.ranking = DataTableView()
        self.ranking_note = note_label(
            "Ranked by |Pearson r| with the target on aligned, model-eligible samples. VIF = 1 / (1 − R²) "
            "of each predictor regressed on the others. Thresholds raise warnings in the message panel; "
            "nothing is removed automatically."
        )
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.5, 0.999)
        self.threshold.setSingleStep(0.01)
        self.threshold.setDecimals(3)
        self.threshold.setValue(CORRELATION_WARNING)
        self.threshold.setPrefix("Warn when |r| > ")
        self.threshold.valueChanged.connect(self._update_views)
        tab = QWidget()
        tl = QVBoxLayout(tab)
        tl.addWidget(self.ranking_note)
        tl.addWidget(self.ranking, 1)
        trow = QHBoxLayout()
        trow.addWidget(self.threshold)
        trow.addStretch(1)
        tl.addLayout(trow)
        self.tabs.addTab(tab, "Ranking && VIF")

        self.method = QComboBox()
        self.method.addItem("Pearson (linear)", "pearson")
        self.method.addItem("Spearman (rank)", "spearman")
        self.method.currentIndexChanged.connect(self._update_views)
        self.heatmap = CorrelationHeatmap()
        tab = QWidget()
        hl = QVBoxLayout(tab)
        hl.addWidget(self.method)
        hl.addWidget(self.heatmap, 1)
        hl.addWidget(
            note_label(
                "Correlation of the aligned features (and the target, first row) over model-eligible "
                "lab samples. Pairwise correlation cannot reveal every collinearity: check VIF too."
            )
        )
        self.tabs.addTab(tab, "Correlation (aligned)")

        self.impact = DataTableView()
        tab = QWidget()
        il = QVBoxLayout(tab)
        il.addWidget(self.impact, 1)
        il.addWidget(
            note_label(
                "How each predictor affects the number of model-eligible samples: a sample is "
                "eligible only when every selected predictor passes its coverage checks."
            )
        )
        self.tabs.addTab(tab, "Sample impact")

        self.raw_clean = DataTableView()
        self.raw_clean_note = note_label(
            "Process statistics before and after the exclusions of the Cleaning stage. Screening r "
            "pairs each lab sample with the last process value at or before it (no deadtime)."
        )
        tab = QWidget()
        rl = QVBoxLayout(tab)
        rl.addWidget(self.raw_clean, 1)
        rl.addWidget(self.raw_clean_note)
        self._raw_clean_tab = tab
        self.tabs.addTab(tab, "Raw vs cleaned")

        split = QSplitter(Qt.Orientation.Horizontal)
        left.setMinimumWidth(240)  # target name and variable list stay readable
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setSizes([260, 900])
        split.setStretchFactor(1, 1)  # extra width goes to the tables, not the list
        self.body.addWidget(split, 1)

        session.changed.connect(self._refresh_selection)
        session.data_changed.connect(self._refresh_selection)
        session.features_changed.connect(self._on_features)
        session.cleaning_changed.connect(self._invalidate_raw_clean)
        self._refresh_selection()
        self._on_features()

    # ------------------------------------------------------------------ selection

    def _units(self) -> dict[str, str | None]:
        return {v.name: v.units for v in self._session.project.variables}

    @Slot()
    def _refresh_selection(self) -> None:
        project = self._session.project
        kinds = {s.source_id: s.kind for s in project.sources}
        lab = [v.name for v in project.variables if kinds.get(v.source_id) is SourceKind.LAB]
        process = [
            v.name for v in project.variables if kinds.get(v.source_id) is SourceKind.PROCESS
        ]
        target = project.target.name if project.target else None
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItem("(choose a target)", None)
        for name in lab + process:
            src = "lab" if name in lab else "process"
            self.target_combo.addItem(f"{name}  ({src})", name)
        index = self.target_combo.findData(target)
        self.target_combo.setCurrentIndex(max(index, 0))
        self.target_combo.blockSignals(False)
        candidates = [n for n in process if n != target]
        checked = {v.name for v in project.predictors}
        if candidates != self.predictors.names() or set(self.predictors.checked()) != checked:
            self.predictors.set_variables(candidates, self._units(), checked)
        self._update_selection_note()

    def _update_selection_note(self) -> None:
        project = self._session.project
        n = len(project.predictors)
        target = project.target.name if project.target else "no target"
        self._session.messages.set(
            "variables.selection", Level.INFO, f"Target: {target}; {n} predictor(s) selected."
        )

    def _check_all(self, checked: bool) -> None:
        self.predictors.list.blockSignals(True)
        for name in self.predictors.names():
            self.predictors.set_checked(name, checked)
        self.predictors.list.blockSignals(False)
        self._apply_timer.start()

    @Slot()
    def _target_chosen(self) -> None:
        target = self.target_combo.currentData()
        predictors = [n for n in self.predictors.checked() if n != target]
        lists = self._session.model_inputs(target, predictors)
        edit_lists(self._undo, self._session, "Choose target", **lists)

    @Slot()
    def _apply_predictors(self) -> None:
        project = self._session.project
        target = project.target.name if project.target else None
        lists = self._session.model_inputs(target, self.predictors.checked())
        edit_lists(self._undo, self._session, "Choose predictors", **lists)

    # ------------------------------------------------------------------ diagnostics

    @Slot()
    def _on_features(self) -> None:
        fm = self._session.features
        self._diag = None
        if fm is not None:
            _times, X, y = fm.eligible_data()
            if len(y) >= 3:
                self._diag = aligned_diagnostics(X, y, threshold=self.threshold.value())
        self._update_selection_note()
        self._update_views()

    @Slot()
    def _update_views(self) -> None:
        diag, fm = self._diag, self._session.features
        units = self._units()
        if diag is None or fm is None:
            self.ranking.set_frame(pd.DataFrame())
            self.impact.set_frame(pd.DataFrame())
            self.heatmap.set_matrix(pd.DataFrame())
            self._session.messages.set(
                "variables.collinearity", Level.INFO,
                "Diagnostics appear once at least 3 samples are eligible." if fm is not None else "",
            )  # fmt: skip
            return
        table = diag.xy.copy()
        table.insert(0, "units", [units.get(n) or "" for n in table.index])
        table["vif"] = diag.vif["vif"]
        table["vif_note"] = diag.vif["note"]
        if self._raw_clean is not None and "r_y_raw" in self._raw_clean:
            table["screen_raw"] = self._raw_clean["r_y_raw"].reindex(table.index)
            table["screen_cleaned"] = self._raw_clean["r_y_cleaned"].reindex(table.index)
        self.ranking.set_frame(table, RANKING_HEADERS)
        pairs = high_correlation_pairs(diag.pearson, self.threshold.value())
        notes = []
        high_vif = diag.vif.index[diag.vif["vif"] > VIF_WARNING].tolist()
        if high_vif:
            notes.append(f"VIF > {VIF_WARNING:g}: {', '.join(high_vif)}.")
        if not pairs.empty:
            listed = "; ".join(
                f"{a} / {b} (r = {r:.2f})"
                for a, b, r in pairs.head(8).itertuples(index=False, name=None)
            )
            notes.append(f"Highly correlated pairs: {listed}.")
        if notes:
            notes.append("Consider PLS, PCR or Ridge, or keep one variable of each group.")
        self._session.messages.set(
            "variables.collinearity", Level.WARNING if notes else Level.INFO,
            " ".join(notes) or "No collinearity warnings at these thresholds.",
        )  # fmt: skip

        method = self.method.currentData()
        _t, X, y = fm.eligible_data()
        full = pd.concat([pd.Series(y, name=f"{fm.target} (Y)"), X], axis=1)
        corr = full.corr(method=method, min_periods=3)
        counts = pd.DataFrame(len(full), index=corr.index, columns=corr.columns)
        self.heatmap.set_matrix(corr, counts)
        self.impact.set_frame(self._impact_table(), {"eligible_without": "Eligible without it",
                                                     "gain": "Gain", "fails": "Samples failing"})  # fmt: skip

    def _impact_table(self) -> pd.DataFrame:
        fm = self._session.features
        status = fm.status.to_numpy() == OK
        y_ok = fm.y_status == Y_OK
        rows = []
        for j, name in enumerate(fm.features):
            others = np.delete(status, j, axis=1).all(axis=1) if status.shape[1] > 1 else True
            without = int((y_ok & others).sum())
            rows.append(
                {
                    "variable": name,
                    "fails": int((y_ok & ~status[:, j]).sum()),
                    "eligible_without": without,
                    "gain": without - fm.n_eligible,
                }
            )
        return pd.DataFrame(rows).set_index("variable").sort_values("gain", ascending=False)

    # ------------------------------------------------------------------ raw vs cleaned

    @Slot()
    def _invalidate_raw_clean(self) -> None:
        self._raw_clean_key = None
        if self.isVisible():
            self._ensure_raw_clean()

    def showEvent(self, event) -> None:  # noqa: N802, ANN001
        super().showEvent(event)
        self._ensure_raw_clean()

    def _ensure_raw_clean(self) -> None:
        state = self._session.cleaning
        if state is None or self._raw_clean_pending or not self.isVisible():
            return
        key = (self._session.data_version, id(state))
        if key == self._raw_clean_key:
            return
        frame, _mask, target = self._session.frames()
        self._raw_clean_pending = True
        self._tasks.submit(
            "Comparing raw and cleaned data",
            _raw_vs_cleaned_task,
            frame,
            state.cleaned(frame),
            target,
            on_result=lambda t, k=key: self._raw_clean_done(t, k),
            on_error=lambda msg, _d: self._raw_clean_done(None, None, msg),
            on_cancelled=lambda: setattr(self, "_raw_clean_pending", False),
        )

    def _raw_clean_done(self, table: pd.DataFrame | None, key: object, error: str = "") -> None:
        self._raw_clean_pending = False
        if table is None:
            self._session.messages.set(
                "variables.rawclean", Level.ERROR, f"Raw-vs-cleaned comparison failed: {error}"
            )
            return
        self._raw_clean, self._raw_clean_key = table, key
        self._session.messages.clear("variables.rawclean")
        state = self._session.cleaning
        if state is None or key != (self._session.data_version, id(state)):
            self._ensure_raw_clean()  # cleaning changed meanwhile
            return
        units = self._units()
        shown = table.copy()
        shown.insert(0, "units", [units.get(n) or "" for n in shown.index])
        self.raw_clean.set_frame(shown, {**RAW_CLEAN_HEADERS, "units": "Units"})
        self._update_views()
