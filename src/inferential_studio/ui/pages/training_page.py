"""Stage 7 page: fitted models and their interpretation (coefficients, CV, loadings, scores, VIP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.core.models import ModelResult, PcrDetails, PlsDetails
from inferential_studio.core.schemas import ModelKind
from inferential_studio.core.splits import SPLITS, TEST
from inferential_studio.ui.messages import Level
from inferential_studio.ui.pages.stage_page import StagePage
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.stages import Stage
from inferential_studio.ui.theme import mark_primary
from inferential_studio.ui.widgets.data_table import DataTableView
from inferential_studio.ui.widgets.plots import (
    SPLIT_COLOURS,
    SPLIT_NAMES,
    bar_chart,
    legend,
    note_label,
    plot_widget,
    scatter,
)

COEF_NOTE = (
    "Coefficient: change in Y (its units) per unit change of X, holding the other predictors "
    "fixed, after all preprocessing. Standardised: change in Y, in training standard deviations, "
    "per one training standard deviation of X — comparable across predictors. Coefficients show "
    "how the model uses each input, not physical cause and effect. No confidence intervals are "
    "shown: lab samples are autocorrelated and predictors correlated, so textbook standard errors "
    "would overstate certainty."
)
PLS_NOTE = (
    "X weights (W) define how each component is formed from the (deflated) predictors; X loadings "
    "(P) describe how the predictors are reconstructed from the component scores. They are "
    "different objects and generally differ. VIP (Wold) summarises each predictor's contribution "
    "to explaining Y; VIP > 1 is a common rule of thumb for above-average influence, not a "
    "selection rule or a proof of causality."
)
PCR_NOTE = (
    "PCA loadings are the eigenvectors of the scaled predictors. Components are ordered by the X "
    "variance they explain, which says nothing about how well they predict Y."
)
VAR_NOTE = {
    ModelKind.PLS: "Blue: share of scaled-X variation captured by each PLS component. Orange: share "
    "of training-Y variation explained by each component. Predictive quality is judged on the "
    "Validation page, not here.",
    ModelKind.PCR: "Share of scaled-X variance captured by each principal component (PCA). This is "
    "not the share of Y explained.",
}


class TrainingPage(StagePage):
    train_requested = Signal()

    def __init__(
        self, stage: Stage, session: ProjectSession, parent: QWidget | None = None
    ) -> None:
        super().__init__(stage, parent)
        self._session = session

        self.train_button = QPushButton("Train models")
        self.train_button.clicked.connect(self.train_requested.emit)
        mark_primary(self.train_button)
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._show_model)
        top = QHBoxLayout()
        top.addWidget(QLabel("Model"))
        top.addWidget(self.model_combo, 1)
        top.addWidget(self.train_button)

        self.tabs = QTabWidget()
        # Coefficients
        self.coef_plot = plot_widget("Standardised coefficients")
        self.coef_table = DataTableView()
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.coef_plot)
        split.addWidget(self.coef_table)
        split.setSizes([500, 500])
        tab = QWidget()
        tl = QVBoxLayout(tab)
        tl.addWidget(split, 1)
        tl.addWidget(note_label(COEF_NOTE))
        self.tabs.addTab(tab, "Coefficients")
        # CV curve
        self.cv_plot = plot_widget("Cross-validation error (training period only)")
        self.cv_plot.setLabel("left", "Mean RMSE over folds")
        self.cv_note = note_label()
        tab = QWidget()
        cl = QVBoxLayout(tab)
        cl.addWidget(self.cv_plot, 1)
        cl.addWidget(self.cv_note)
        self.tabs.addTab(tab, "Cross-validation")
        # Loadings / weights / VIP
        self.load_kind = QComboBox()
        self.load_component = QComboBox()
        self.load_kind.currentIndexChanged.connect(self._update_loadings)
        self.load_component.currentIndexChanged.connect(self._update_loadings)
        self.load_plot = plot_widget(name="loadings")
        self.load_note = note_label()
        tab = QWidget()
        ll = QVBoxLayout(tab)
        row = QHBoxLayout()
        row.addWidget(self.load_kind)
        row.addWidget(self.load_component)
        row.addStretch(1)
        ll.addLayout(row)
        ll.addWidget(self.load_plot, 1)
        ll.addWidget(self.load_note)
        self._load_tab = tab
        self.tabs.addTab(tab, "Loadings && VIP")
        # Scores
        self.score_x = QComboBox()
        self.score_y = QComboBox()
        self.score_x.currentIndexChanged.connect(self._update_scores)
        self.score_y.currentIndexChanged.connect(self._update_scores)
        self.score_plot = plot_widget(name="scores")
        legend(self.score_plot)
        self.score_note = note_label()
        tab = QWidget()
        sl = QVBoxLayout(tab)
        row = QHBoxLayout()
        row.addWidget(QLabel("x:"))
        row.addWidget(self.score_x)
        row.addWidget(QLabel("y:"))
        row.addWidget(self.score_y)
        row.addStretch(1)
        sl.addLayout(row)
        sl.addWidget(self.score_plot, 1)
        sl.addWidget(self.score_note)
        self._score_tab = tab
        self.tabs.addTab(tab, "Scores")
        # Explained variance
        self.var_plot = plot_widget("Explained variance per component")
        self.var_plot.setLabel("left", "Share (%)")
        legend(self.var_plot)
        self.var_note = note_label()
        tab = QWidget()
        vl = QVBoxLayout(tab)
        vl.addWidget(self.var_plot, 1)
        vl.addWidget(self.var_note)
        self._var_tab = tab
        self.tabs.addTab(tab, "Explained variance")

        self.body.addLayout(top)
        self.body.addWidget(self.tabs, 1)

        session.training_changed.connect(self.refresh)
        session.features_changed.connect(self._update_button)
        self.refresh()

    # ------------------------------------------------------------------ state

    def current(self) -> ModelResult | None:
        run = self._session.training
        i = self.model_combo.currentIndex()
        if run is None or i < 0 or i >= len(run.models):
            return None
        return run.models[i]

    @Slot()
    def _update_button(self) -> None:
        self.train_button.setEnabled(self._session.features is not None)

    @Slot()
    def refresh(self) -> None:
        run = self._session.training
        self._update_button()
        previous = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if run is not None:
            for m in run.models:
                self.model_combo.addItem(m.name if m.ok else f"{m.spec.label} — not trained")
        index = self.model_combo.findText(previous)
        self.model_combo.setCurrentIndex(max(index, 0))
        self.model_combo.blockSignals(False)
        self._show_model()

    @Slot()
    def _show_model(self) -> None:
        m = self.current()
        for plot in (self.coef_plot, self.cv_plot, self.load_plot, self.score_plot, self.var_plot):
            plot.clear()
        self.coef_table.set_frame(pd.DataFrame())
        msgs = self._session.messages
        msgs.clear("training.model")
        if m is None:
            self.cv_note.setText("")
            return
        if not m.ok:
            self.cv_note.setText(m.error)
            msgs.set(
                "training.model", Level.WARNING, f"{m.spec.label} could not be trained: {m.error}"
            )
            return
        self._update_coefficients(m)
        self._update_cv(m)
        latent = isinstance(m.details, (PlsDetails, PcrDetails))
        for tab in (self._load_tab, self._score_tab, self._var_tab):
            self.tabs.setTabEnabled(self.tabs.indexOf(tab), latent)
        if latent:
            self._fill_latent_controls(m)
        if m.warnings:
            msgs.set("training.model", Level.WARNING, f"{m.name}: " + " ".join(m.warnings),
                     also=("validation",))  # fmt: skip

    def _units(self) -> dict[str, str | None]:
        return {v.name: v.units for v in self._session.project.variables}

    def _update_coefficients(self, m: ModelResult) -> None:
        units = self._units()
        run = self._session.training
        table = pd.DataFrame(
            {
                "units": [units.get(n) or "" for n in m.coefficients.index],
                "coefficient": m.coefficients,
                "standardized": m.standardized,
            }
        )
        if isinstance(m.details, PlsDetails):
            table["vip"] = m.details.vip
        order = table["standardized"].abs().sort_values(ascending=False).index
        table = table.loc[order]
        intercept = pd.DataFrame(
            {
                "units": [units.get(run.target) or ""],
                "coefficient": [m.intercept],
                "standardized": [np.nan],
            },
            index=["(intercept)"],
        )
        self.coef_table.set_frame(
            pd.concat([intercept, table]),
            {"units": "X units", "coefficient": f"Coefficient ({units.get(run.target) or 'Y units'} per X unit)",
             "standardized": "Standardised", "vip": "VIP"},
        )  # fmt: skip
        bar_chart(self.coef_plot, list(table.index), table["standardized"].to_numpy())

    def _update_cv(self, m: ModelResult) -> None:
        cv = m.cv
        self.cv_plot.setVisible(cv is not None)
        if cv is None:
            text = (
                f"Mean CV RMSE: {m.cv_rmse:.4g}. This model has no tuned hyperparameter."
                if np.isfinite(m.cv_rmse)
                else "No cross-validation result (fixed hyperparameters or too few samples)."
            )
            self.cv_note.setText(text)
            return
        x = np.log10(cv.values) if cv.parameter == "alpha" else cv.values
        self.cv_plot.setLabel("bottom", "log10(alpha)" if cv.parameter == "alpha" else "Components")
        self.cv_plot.plot(x, cv.mean_rmse, pen=pg.mkPen("#0072B2", width=2), symbol="o", symbolSize=5,
                          symbolBrush="#0072B2")  # fmt: skip
        err = pg.ErrorBarItem(x=x, y=cv.mean_rmse, height=2 * cv.se_rmse, pen=pg.mkPen("#7FB2D9"))
        self.cv_plot.addItem(err)
        pick = np.log10(cv.chosen) if cv.parameter == "alpha" else cv.chosen
        best = np.log10(cv.best) if cv.parameter == "alpha" else cv.best
        i_pick = int(np.flatnonzero(np.isclose(cv.values, cv.chosen))[0])
        i_best = int(np.flatnonzero(np.isclose(cv.values, cv.best))[0])
        scatter(self.cv_plot, [best], [cv.mean_rmse[i_best]], "#999999", size=12, symbol="s")
        scatter(self.cv_plot, [pick], [cv.mean_rmse[i_pick]], "#D55E00", size=12)
        self.cv_note.setText(
            f"Chosen {cv.parameter}: {cv.chosen:g} ({cv.rule}); lowest mean CV RMSE at "
            f"{cv.best:g}. Error bars: ± 1 standard error across folds. Folds use only the training "
            "period, past-to-future where applicable, with the purge gap."
        )

    # ------------------------------------------------------------------ latent models

    def _fill_latent_controls(self, m: ModelResult) -> None:
        d = m.details
        comps = list(d.scores.columns[:-1])
        for combo in (self.load_component, self.score_x, self.score_y):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(comps)
            combo.blockSignals(False)
        self.score_y.setCurrentIndex(1 if len(comps) > 1 else 0)
        self.load_kind.blockSignals(True)
        self.load_kind.clear()
        if isinstance(d, PlsDetails):
            self.load_kind.addItem("VIP (all components)", "vip")
            self.load_kind.addItem("X weights (W)", "weights")
            self.load_kind.addItem("X loadings (P)", "loadings")
        else:
            self.load_kind.addItem("PCA loadings", "loadings")
        self.load_kind.blockSignals(False)
        self._update_loadings()
        self._update_scores()
        self._update_variance(m)

    @Slot()
    def _update_loadings(self) -> None:
        m = self.current()
        if m is None or m.details is None:
            return
        d = m.details
        kind = self.load_kind.currentData()
        comp = self.load_component.currentText()
        self.load_component.setEnabled(kind != "vip")
        if kind == "vip":
            vip = d.vip.sort_values(ascending=False)
            bar_chart(self.load_plot, list(vip.index), vip.to_numpy())
            self.load_plot.addItem(pg.InfiniteLine(1.0, angle=90, pen=pg.mkPen("#555555", style=Qt.PenStyle.DashLine)))  # fmt: skip
            self.load_plot.setTitle("Variable importance in projection (VIP)")
        else:
            frame = (
                d.x_weights
                if kind == "weights"
                else (d.x_loadings if isinstance(d, PlsDetails) else d.loadings)
            )
            if comp not in frame.columns:
                return
            values = frame[comp]
            bar_chart(self.load_plot, list(values.index), values.to_numpy())
            self.load_plot.setTitle(f"{self.load_kind.currentText()} — {comp}")
        self.load_note.setText(PLS_NOTE if isinstance(d, PlsDetails) else PCR_NOTE)

    @Slot()
    def _update_scores(self) -> None:
        m = self.current()
        self.score_plot.clear()
        if m is None or m.details is None:
            return
        scores = m.details.scores
        cx, cy = self.score_x.currentText(), self.score_y.currentText()
        if cx not in scores or cy not in scores:
            return
        shown = [s for s in SPLITS if s != TEST or self._session.test_revealed]
        for split in shown:
            sub = scores[scores["split"] == split]
            if len(sub):
                scatter(
                    self.score_plot,
                    sub[cx],
                    sub[cy],
                    SPLIT_COLOURS[split],
                    SPLIT_NAMES[split],
                    size=6,
                )
        self.score_plot.setLabel("bottom", cx)
        self.score_plot.setLabel("left", cy)
        n_comp = len(scores.columns) - 1
        note = "Scores: position of each lab sample in the component space. Clusters or trends over " \
               "time can reveal operating regimes."  # fmt: skip
        if n_comp < 2:
            note = (
                "⚠ Only one component: both axes show the same score (a 2-D scores plot needs two components). "
                + note
            )
        if not self._session.test_revealed:
            note += " Final-test samples are hidden until revealed on the Validation page."
        self.score_note.setText(note)

    def _update_variance(self, m: ModelResult) -> None:
        d = m.details
        self.var_plot.clear()
        n = len(d.x_variance_explained)
        pos = np.arange(1, n + 1)
        if isinstance(d, PlsDetails):
            self.var_plot.addItem(pg.BarGraphItem(x=pos - 0.18, height=100 * d.x_variance_explained, width=0.35,
                                                  brush="#0072B2", name="X variance"))  # fmt: skip
            self.var_plot.addItem(pg.BarGraphItem(x=pos + 0.18, height=100 * d.y_variance_explained, width=0.35,
                                                  brush="#E69F00", name="Y variance (training)"))  # fmt: skip
            note = VAR_NOTE[ModelKind.PLS]
        else:
            self.var_plot.addItem(pg.BarGraphItem(x=pos, height=100 * d.x_variance_explained, width=0.6,
                                                  brush="#0072B2", name="X variance"))  # fmt: skip
            note = VAR_NOTE[ModelKind.PCR]
        self.var_plot.plot(pos, 100 * np.cumsum(d.x_variance_explained), pen=pg.mkPen("#555555", width=1.5),
                           symbol="o", symbolSize=4, name="Cumulative X variance")  # fmt: skip
        self.var_plot.getAxis("bottom").setTicks([[(int(i), f"{int(i)}") for i in pos]])
        self.var_plot.setLabel("bottom", "Component")
        self.var_note.setText(note)
