"""Stage 8 page: model comparison, split metrics, residual diagnostics and applicability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.core.models import METRIC_LABELS, ModelResult, autocorrelation
from inferential_studio.core.splits import SPLITS, TEST, TRAIN, VALIDATION
from inferential_studio.core.statistics import histogram
from inferential_studio.ui.pages.stage_page import StagePage
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.stages import Stage
from inferential_studio.ui.widgets.data_table import DataTableView
from inferential_studio.ui.widgets.plots import (
    SPLIT_COLOURS,
    SPLIT_NAMES,
    legend,
    note_label,
    plot_widget,
    scatter,
)
from inferential_studio.ui.widgets.time_axis import to_epoch_seconds

COMPARISON_HEADERS = {
    "status": "Status",
    "cv_rmse": "CV RMSE (train)",
    **{
        f"{s}_{k}": f"{SPLIT_NAMES[s]} {label}"
        for s in SPLITS
        for k, label in (("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE"), ("bias", "bias"))
    },  # fmt: skip
}
REVEAL_TEXT = (
    "The final test period is meant to be looked at once, after you have chosen a model using the "
    "training and validation results. If you change models or settings after seeing the test "
    "results, those results become exploratory and a new untouched period is needed for an honest "
    "estimate.\n\nShow the final test results now?"
)

NO_RUN_TEXT = "Train models on Model Setup to compare them here."


class ValidationPage(StagePage):
    def __init__(
        self, stage: Stage, session: ProjectSession, parent: QWidget | None = None
    ) -> None:
        super().__init__(stage, parent)
        self._session = session

        self.comparison = DataTableView()
        self.comparison.fit_to_rows(max_rows=6)  # one row per model, no empty band
        self.comparison_note = note_label()
        self.reveal_button = QPushButton("Show final test results…")
        self.reveal_button.clicked.connect(self.ask_reveal)
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._show_model)
        self.metrics = DataTableView(sortable=False)
        self.metrics.fit_to_rows(max_rows=3)  # training, validation and, once revealed, final test

        # Top: every model side by side. Bottom: one model's metrics and diagnostic charts.
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        heading = QLabel("Model comparison (same split for every model)")
        heading.setObjectName("sectionTitle")
        row.addWidget(heading)
        row.addStretch(1)
        row.addWidget(self.reveal_button)
        tl.addLayout(row)
        tl.addWidget(self.comparison)
        tl.addWidget(self.comparison_note)
        tl.addStretch(1)

        self.tabs = QTabWidget()
        tz = session.project.display_timezone
        self.avp = plot_widget("Actual vs predicted", empty_text=NO_RUN_TEXT)
        legend(self.avp, "bottom-right")
        self.trend = plot_widget(
            "Lab samples and predictions", time_axis=tz, empty_text=NO_RUN_TEXT
        )
        legend(self.trend)
        self.res_time = plot_widget(
            "Residual (actual − predicted) over time", time_axis=tz, empty_text=NO_RUN_TEXT
        )
        self.res_pred = plot_widget("Residual vs predicted", empty_text=NO_RUN_TEXT)
        self.res_x_var = QComboBox()
        self.res_x_var.currentIndexChanged.connect(self._update_residual_x)
        self.res_x = plot_widget(name="residuals_vs_x", empty_text=NO_RUN_TEXT)
        self.res_hist = plot_widget("Residual distribution", empty_text=NO_RUN_TEXT)
        legend(self.res_hist, "top-right")
        self.res_acf = plot_widget(
            "Residual autocorrelation (consecutive lab samples)", empty_text=NO_RUN_TEXT
        )
        legend(self.res_acf, "top-right")
        self.applicability = DataTableView()
        self.app_note = note_label()
        self.tabs.addTab(self.avp, "Actual vs predicted")
        self.tabs.addTab(self.trend, "Trend")
        self.tabs.addTab(self.res_time, "Residuals vs time")
        self.tabs.addTab(self.res_pred, "Residuals vs predicted")
        xbox = QWidget()
        xl = QVBoxLayout(xbox)
        xl.addWidget(self.res_x_var)
        xl.addWidget(self.res_x, 1)
        self.tabs.addTab(xbox, "Residuals vs X")
        self.tabs.addTab(self.res_hist, "Residual histogram")
        self.tabs.addTab(self.res_acf, "Residual ACF")
        abox = QWidget()
        al = QVBoxLayout(abox)
        al.addWidget(self.applicability, 1)
        al.addWidget(self.app_note)
        self.tabs.addTab(abox, "Applicability")
        self.plot_note = note_label(
            "Training = fitted samples; Validation = used to compare models; Final test = untouched "
            "until revealed."
        )

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        mrow = QHBoxLayout()
        diag = QLabel("Diagnostics for")
        diag.setObjectName("sectionTitle")
        mrow.addWidget(diag)
        mrow.addWidget(self.model_combo, 1)
        bl.addLayout(mrow)
        bl.addWidget(self.metrics)
        bl.addWidget(self.tabs, 1)
        bl.addWidget(self.plot_note)
        # The comparison keeps its natural height (its table fits its rows); the diagnostics
        # below take the rest of the page.
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(bottom)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        self.body.addWidget(split, 1)

        session.training_changed.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------ state

    def splits_shown(self) -> tuple[str, ...]:
        return SPLITS if self._session.test_revealed else (TRAIN, VALIDATION)

    def current(self) -> ModelResult | None:
        run = self._session.training
        name = self.model_combo.currentData()
        if run is None or name is None:
            return None
        return next((m for m in run.models if m.name == name), None)

    def ask_reveal(self) -> None:
        answer = QMessageBox.question(self, "Final test results", REVEAL_TEXT)
        if answer == QMessageBox.StandardButton.Yes:
            self._session.reveal_test()

    @Slot()
    def refresh(self) -> None:
        run = self._session.training
        revealed = self._session.test_revealed
        self.reveal_button.setEnabled(
            bool(run is not None and not revealed and run.split.mask(TEST).any())
        )
        previous = self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if run is None:
            self.comparison.set_frame(pd.DataFrame())
            self.comparison_note.setText("")
            self.model_combo.blockSignals(False)
            self._show_model()
            return
        table = run.comparison(include_test=revealed)
        self.comparison.set_frame(table, COMPARISON_HEADERS)
        for name in table.index:
            m = next(m for m in run.models if m.name == name)
            if m.ok:
                self.model_combo.addItem(name, name)
        index = self.model_combo.findData(previous)
        self.model_combo.setCurrentIndex(max(index, 0))
        self.model_combo.blockSignals(False)
        ranked = "validation RMSE" if run.ranking_split() == VALIDATION else "cross-validation RMSE"
        units = next(
            (v.units for v in self._session.project.variables if v.name == run.target), None
        )
        self.comparison_note.setText(
            f"Ranked by {ranked}. Errors in {units or 'target units'}; bias = mean(predicted − actual)."
            + ("" if revealed else " Final-test columns stay hidden until revealed.")
        )
        self.comparison_note.setToolTip(
            "Prefer a model that is accurate and also simple, stable and physically sensible over the "
            "highest training R². Fit metrics alone do not show online readiness."
        )
        self.res_x_var.blockSignals(True)
        self.res_x_var.clear()
        self.res_x_var.addItems(run.features)
        self.res_x_var.blockSignals(False)
        self._show_model()

    @Slot()
    def _show_model(self) -> None:
        m = self.current()
        for p in (
            self.avp,
            self.trend,
            self.res_time,
            self.res_pred,
            self.res_x,
            self.res_hist,
            self.res_acf,
        ):
            p.clear()
        if m is None:
            self.metrics.set_frame(pd.DataFrame())
            self.applicability.set_frame(pd.DataFrame())
            return
        shown = self.splits_shown()
        metrics = m.metrics.loc[list(shown)].rename(index=SPLIT_NAMES)
        self.metrics.set_frame(metrics, METRIC_LABELS)
        pred = m.predictions[m.predictions["split"].isin(shown)]
        self._actual_vs_predicted(pred)
        self._trend(pred)
        self._residuals(pred)
        self._update_residual_x()
        self._applicability()

    def _by_split(self, pred: pd.DataFrame):  # noqa: ANN202
        for split in SPLITS:
            sub = pred[pred["split"] == split]
            if len(sub):
                yield split, sub

    def _actual_vs_predicted(self, pred: pd.DataFrame) -> None:
        for split, sub in self._by_split(pred):
            scatter(
                self.avp, sub["actual"], sub["predicted"], SPLIT_COLOURS[split], SPLIT_NAMES[split]
            )
        lo = float(np.nanmin(pred[["actual", "predicted"]].to_numpy()))
        hi = float(np.nanmax(pred[["actual", "predicted"]].to_numpy()))
        self.avp.plot(
            [lo, hi], [lo, hi], pen=pg.mkPen("#555555", style=Qt.PenStyle.DashLine), name="identity"
        )
        self.avp.setLabel("bottom", "Actual (lab)")
        self.avp.setLabel("left", "Predicted")

    def _trend(self, pred: pd.DataFrame) -> None:
        x = to_epoch_seconds(pd.DatetimeIndex(pred["sample_time"]))
        order = np.argsort(x)
        self.trend.plot(x[order], pred["predicted"].to_numpy()[order], pen=pg.mkPen("#555555", width=1),
                        name="Predicted")  # fmt: skip
        for split, sub in self._by_split(pred):
            sx = to_epoch_seconds(pd.DatetimeIndex(sub["sample_time"]))
            scatter(
                self.trend, sx, sub["actual"], SPLIT_COLOURS[split], f"Lab — {SPLIT_NAMES[split]}"
            )

    def _residuals(self, pred: pd.DataFrame) -> None:
        for split, sub in self._by_split(pred):
            sx = to_epoch_seconds(pd.DatetimeIndex(sub["sample_time"]))
            scatter(self.res_time, sx, sub["residual"], SPLIT_COLOURS[split], size=5)
            scatter(self.res_pred, sub["predicted"], sub["residual"], SPLIT_COLOURS[split], size=5)
            counts, edges = histogram(sub["residual"], max_bins=40)
            if counts.size:
                self.res_hist.plot(edges, counts, stepMode="center", fillLevel=0, name=SPLIT_NAMES[split],
                                   brush=pg.mkBrush(pg.mkColor(SPLIT_COLOURS[split]).lighter(130)),
                                   pen=pg.mkPen(SPLIT_COLOURS[split]))  # fmt: skip
            acf = autocorrelation(sub.sort_values("sample_time")["residual"].to_numpy(), 15)
            lags = np.arange(acf.size)
            offset = {TRAIN: -0.25, VALIDATION: 0.0, TEST: 0.25}[split]
            self.res_acf.addItem(pg.BarGraphItem(x=lags + offset, height=acf, width=0.24,
                                                 brush=SPLIT_COLOURS[split], name=SPLIT_NAMES[split]))  # fmt: skip
        for p in (self.res_time, self.res_pred):
            p.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#555555")))
        self.res_pred.setLabel("bottom", "Predicted")
        self.res_hist.setLabel("bottom", "Residual (actual − predicted)")
        self.res_acf.setLabel("bottom", "Lag (lab samples)")
        n = int((pred["split"] == TRAIN).sum())
        if n > 3:
            band = 1.96 / np.sqrt(n)
            for v in (band, -band):
                self.res_acf.addItem(pg.InfiniteLine(v, angle=0, pen=pg.mkPen("#999999", style=Qt.PenStyle.DashLine)))  # fmt: skip

    @Slot()
    def _update_residual_x(self) -> None:
        self.res_x.clear()
        m = self.current()
        run = self._session.training
        name = self.res_x_var.currentText()
        if m is None or run is None or name not in run.features:
            return
        pred = m.predictions[m.predictions["split"].isin(self.splits_shown())].copy()
        pred["x"] = run.feature_values[name].to_numpy()[pred["row"].to_numpy()]
        for split, sub in self._by_split(pred):
            scatter(self.res_x, sub["x"], sub["residual"], SPLIT_COLOURS[split], size=5)
        self.res_x.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#555555")))
        units = next((v.units for v in self._session.project.variables if v.name == name), None)
        self.res_x.setLabel("bottom", f"{name} (aligned feature)", units=units)
        self.res_x.setLabel("left", "Residual")

    def _applicability(self) -> None:
        run = self._session.training
        units = {v.name: v.units for v in self._session.project.variables}
        table = run.train_ranges.copy()
        table.insert(0, "units", [units.get(n) or "" for n in table.index])
        labels = run.split.labels
        headers = {"units": "Units", "min": "Training min", "max": "Training max",
                   "mean": "Training mean", "std": "Training SD"}  # fmt: skip
        for split in self.splits_shown():
            if split == TRAIN:
                continue
            m = labels == split
            col = f"outside_{split}"
            table[col] = run.outside_range[m].sum().to_numpy() if m.any() else 0
            headers[col] = f"{SPLIT_NAMES[split]} samples outside range"
        self.applicability.set_frame(table, headers)
        val = labels == VALIDATION
        any_out = int(run.outside_range[val].any(axis=1).sum()) if val.any() else 0
        self.app_note.setText(
            f"Linear models extrapolate silently outside the training range. {any_out} validation "
            "sample(s) have at least one predictor outside its training range. Keep these limits "
            "with the model; predictions outside them need engineering judgement."
        )
