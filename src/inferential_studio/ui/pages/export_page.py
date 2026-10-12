"""Stage 9 page: model card, prediction exports, model file and applying a saved model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QGuiApplication, QUndoStack
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.core.artifacts import (
    EQUATION_CAVEAT,
    MODEL_EXTENSION,
    PRED_OK,
    ArtifactError,
    ModelArtifact,
    Prediction,
    build_artifact,
    check_compatibility,
    equation_text,
    load_artifact,
    predict,
    save_artifact,
)
from inferential_studio.core.exclusions import category_table
from inferential_studio.core.exports import (
    ExportError,
    batch_prediction_table,
    feature_matrix_table,
    predictions_table,
    write_table,
)
from inferential_studio.core.report import build_report
from inferential_studio.core.schemas import ReportInfo
from inferential_studio.infrastructure.settings import AppSettings
from inferential_studio.infrastructure.workers import TaskManager
from inferential_studio.ui.commands import edit_lists
from inferential_studio.ui.messages import Level
from inferential_studio.ui.pages.stage_page import StagePage
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.stages import Stage
from inferential_studio.ui.theme import mark_primary
from inferential_studio.ui.widgets.flow_layout import flow_row
from inferential_studio.ui.widgets.plots import (
    BLUE,
    GREY,
    VERMILION,
    clear_placeholder,
    legend,
    note_label,
    plot_widget,
    scatter,
    show_placeholder,
)
from inferential_studio.ui.widgets.time_axis import to_epoch_seconds

APPLY_PLACEHOLDER = "Load a model file, then apply it to the data in this project."
TABLE_FILTER = "Excel workbook (*.xlsx);;CSV, comma separated (*.csv)"


def _predict_task(artifact, process, times, excluded, regime, units, **kw):  # noqa: ANN001, ANN201
    return predict(artifact, process, times, excluded=excluded, regime=regime, units=units, **kw)


class ExportPage(StagePage):
    def __init__(
        self,
        stage: Stage,
        session: ProjectSession,
        tasks: TaskManager,
        undo_stack: QUndoStack,
        settings: AppSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(stage, parent)
        self._session = session
        self._tasks = tasks
        self._undo = undo_stack
        self._settings = settings
        self._loaded: ModelArtifact | None = None
        self._loaded_path: Path | None = None
        self._prediction: Prediction | None = None
        self.last_written: Path | None = None  # for tests and the status bar

        # -- the trained model -----------------------------------------------------------
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._update_equation)
        self.include_test = QCheckBox("Include final-test results")
        self.title = QLineEdit()
        self.version = QLineEdit()
        self.owner = QLineEdit()
        self.purpose = QPlainTextEdit()
        self.purpose.setPlaceholderText(
            "What the model estimates, for whom, and how it will be used."
        )
        self.purpose.setMaximumHeight(70)
        self.limitations = QPlainTextEdit()
        self.limitations.setPlaceholderText(
            "Known limitations, one per line (added to the automatic ones)."
        )
        self.limitations.setMaximumHeight(70)
        for w in (self.title, self.version, self.owner):
            w.editingFinished.connect(self._store_info)
        card = QGroupBox("Model card")
        form = QFormLayout(card)
        form.addRow("Model", self.model_combo)
        form.addRow("", self.include_test)
        form.addRow("Title", self.title)
        form.addRow("Model version", self.version)
        form.addRow("Owner", self.owner)
        form.addRow("Purpose", self.purpose)
        form.addRow("Limitations", self.limitations)

        self.btn_report = QPushButton("Save model card (HTML)…")
        self.btn_report.clicked.connect(lambda: self.save_report())
        mark_primary(self.btn_report)
        self.btn_predictions = QPushButton("Export predictions…")
        self.btn_predictions.clicked.connect(lambda: self.export_predictions())
        self.btn_features = QPushButton("Export feature matrix…")
        self.btn_features.clicked.connect(lambda: self.export_features())
        self.btn_model = QPushButton("Save model file (.imsmodel)…")
        self.btn_model.clicked.connect(lambda: self.save_model())
        self.btn_equation = QPushButton("Copy equation")
        self.btn_equation.clicked.connect(self._copy_equation)
        buttons = QGroupBox("Export")
        bl = QVBoxLayout(buttons)
        for b in (
            self.btn_report,
            self.btn_predictions,
            self.btn_features,
            self.btn_model,
            self.btn_equation,
        ):
            bl.addWidget(b)
        bl.addWidget(
            note_label(
                "The model file is plain JSON (no executable code): feature definitions, "
                "coefficients, training ranges, metrics and provenance. It can be opened safely "
                "and applied below. Final-test rows are exported only when included."
            )
        )
        self.equation = QPlainTextEdit()
        self.equation.setReadOnly(True)
        self.equation.setMaximumHeight(90)
        eq_box = QGroupBox("Equation on aligned features")
        el = QVBoxLayout(eq_box)
        el.addWidget(self.equation)
        el.addWidget(note_label(EQUATION_CAVEAT))

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(card)
        ll.addWidget(buttons)
        ll.addWidget(eq_box)
        ll.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(left)
        scroll.setMinimumWidth(330)

        # -- apply a saved model ---------------------------------------------------------
        self.btn_load = QPushButton("Load model file…")
        self.btn_load.clicked.connect(lambda: self.load_model_file())
        self.loaded_note = note_label("Apply a saved model to the data loaded in this project.")
        self.when = QComboBox()
        self.when.addItem("At lab sample times", "lab")
        self.when.addItem("At regular intervals", "grid")
        self.when.setItemData(
            0, "Predict where lab results exist and compare with them", Qt.ItemDataRole.ToolTipRole
        )
        self.when.setItemData(1, "Predict every N minutes over the loaded period",
                              Qt.ItemDataRole.ToolTipRole)  # fmt: skip
        self.when.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.when.setMinimumContentsLength(24)  # long item texts elide instead of widening the page
        self.every = QDoubleSpinBox()
        self.every.setRange(1, 1440)
        self.every.setValue(60)
        self.every.setSuffix(" min")
        self.every.setEnabled(False)
        self.when.currentIndexChanged.connect(
            lambda _i: self.every.setEnabled(self.when.currentData() == "grid")
        )
        self.use_exclusions = QCheckBox("Use this project's exclusions")
        self.use_exclusions.setChecked(True)
        self.btn_apply = QPushButton("Apply model")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(lambda: self.apply_model())
        self.btn_export_applied = QPushButton("Export results…")
        self.btn_export_applied.setEnabled(False)
        self.btn_export_applied.clicked.connect(lambda: self.export_applied())
        self.apply_plot = plot_widget(
            time_axis=session.project.display_timezone, name="batch_prediction"
        )
        legend(self.apply_plot, "top-right")
        show_placeholder(self.apply_plot, APPLY_PLACEHOLDER)
        self.apply_note = note_label()
        apply_box = QGroupBox("Apply a saved model (batch prediction)")
        al = QVBoxLayout(apply_box)
        al.addWidget(flow_row(self.btn_load, self.btn_apply, self.btn_export_applied))
        al.addWidget(flow_row(self.when, self.every, self.use_exclusions))
        al.addWidget(self.loaded_note)
        al.addWidget(self.apply_plot, 1)
        al.addWidget(self.apply_note)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(apply_box)
        split.setSizes([460, 640])
        split.setChildrenCollapsible(False)
        self.body.addWidget(split, 1)

        session.training_changed.connect(self.refresh)
        session.changed.connect(self._load_info)
        self._load_info()
        self.refresh()

    # ------------------------------------------------------------------ state

    def _run_ok(self) -> bool:
        s = self._session
        return (
            s.training is not None and not s.training_stale and any(m.ok for m in s.training.models)
        )

    @Slot()
    def refresh(self) -> None:
        s = self._session
        run = s.training
        ok = self._run_ok()
        for b in (
            self.btn_report,
            self.btn_predictions,
            self.btn_model,
            self.btn_equation,
            self.model_combo,
        ):
            b.setEnabled(ok)
        self.btn_features.setEnabled(s.features is not None)
        self.include_test.setEnabled(ok and s.test_revealed)
        self.include_test.setChecked(ok and s.test_revealed)
        self.include_test.setToolTip(
            "" if s.test_revealed else "Reveal the final test on the Validation page first."
        )
        previous = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if run is not None:
            for name in run.comparison().index:  # ranked: best first
                m = next(x for x in run.models if x.name == name)
                if m.ok:
                    self.model_combo.addItem(name)
        i = self.model_combo.findText(previous)
        self.model_combo.setCurrentIndex(max(i, 0))
        self.model_combo.blockSignals(False)
        self._update_equation()
        self._update_apply_state()

    @Slot()
    def _load_info(self) -> None:
        info = self._session.project.report
        for w, text in ((self.title, info.title), (self.version, info.model_version),
                        (self.owner, info.owner or self._settings.author_name)):  # fmt: skip
            if not w.hasFocus():
                w.setText(text)
        if not self.purpose.hasFocus() and self.purpose.toPlainText() != info.purpose:
            self.purpose.setPlainText(info.purpose)
        if not self.limitations.hasFocus() and self.limitations.toPlainText() != info.limitations:
            self.limitations.setPlainText(info.limitations)

    def report_info(self) -> ReportInfo:
        return ReportInfo(
            title=self.title.text().strip(),
            model_version=self.version.text().strip() or "1.0",
            owner=self.owner.text().strip(),
            purpose=self.purpose.toPlainText().strip(),
            limitations=self.limitations.toPlainText().strip(),
        )

    @Slot()
    def _store_info(self) -> None:
        info = self.report_info()
        if info != self._session.project.report:
            edit_lists(self._undo, self._session, "Model card details", report=info)

    def artifact(self) -> ModelArtifact:
        s = self._session
        self._store_info()
        info = self.report_info()
        return build_artifact(
            s.project, s.training, self.model_combo.currentText(), owner=info.owner,
            purpose=info.purpose, model_version=info.model_version,
            include_test=self.include_test.isChecked(),
            project_path=str(s.path) if s.path else None,
        )  # fmt: skip

    @Slot()
    def _update_equation(self) -> None:
        if not self._run_ok() or not self.model_combo.currentText():
            self.equation.setPlainText("")
            return
        m = next(
            x for x in self._session.training.models if x.name == self.model_combo.currentText()
        )
        self.equation.setPlainText(
            equation_text(self._session.training.target, m.intercept, dict(m.coefficients))
        )

    def _copy_equation(self) -> None:
        QGuiApplication.clipboard().setText(self.equation.toPlainText() + "\n\n" + EQUATION_CAVEAT)

    def _ask_path(self, title: str, default: str, filters: str) -> Path | None:
        start = str(Path(self._settings.last_directory or Path.home()) / default)
        path, _ = QFileDialog.getSaveFileName(self, title, start, filters)
        return Path(path) if path else None

    def _done(self, path: Path) -> None:
        self.last_written = path
        self._settings.last_directory = str(path.parent)
        self._session.messages.info(f"Saved {path}", "export")

    def _fail(self, title: str, message: str) -> None:
        window = self.window()
        if hasattr(window, "show_error"):
            window.show_error(title, message)

    # ------------------------------------------------------------------ exports

    def save_report(self, path: Path | None = None) -> Path | None:
        if not self._run_ok():
            return None
        name = self._safe_name()
        path = path or self._ask_path("Save model card", f"{name} model card.html", "HTML (*.html)")
        if path is None:
            return None
        if path.suffix.lower() not in (".html", ".htm"):
            path = path.with_suffix(".html")
        try:
            art = self.artifact()
            m = next(x for x in self._session.training.models if x.name == art.name)
            state = self._session.cleaning
            html = build_report(
                art, self._session.project, predictions=m.predictions,
                cleaning_summary=category_table(state) if state is not None else None,
                title=self.report_info().title, limitations=self.report_info().limitations,
            )  # fmt: skip
            path.write_text(html, encoding="utf-8")
        except (ArtifactError, OSError) as exc:
            self._fail("Cannot save the model card", str(exc))
            return None
        self._done(path)
        return path

    def export_predictions(self, path: Path | None = None) -> Path | None:
        if not self._run_ok():
            return None
        path = path or self._ask_path(
            "Export predictions", f"{self._safe_name()} predictions.xlsx", TABLE_FILTER
        )
        if path is None:
            return None
        s = self._session
        try:
            table = predictions_table(
                s.training, self.model_combo.currentText(), s.features,
                timezone=s.project.display_timezone, include_test=self.include_test.isChecked(),
            )  # fmt: skip
            notes = [f"Model: {self.model_combo.currentText()}", "Residual = actual - predicted."]
            if not self.include_test.isChecked():
                notes.append("Final-test rows are not included (not revealed / not selected).")
            write_table(table, path, sheet="Predictions", notes=notes)
        except ExportError as exc:
            self._fail("Cannot export predictions", str(exc))
            return None
        self._done(path)
        return path

    def export_features(self, path: Path | None = None) -> Path | None:
        fm = self._session.features
        if fm is None:
            return None
        path = path or self._ask_path("Export feature matrix", "feature matrix.xlsx", TABLE_FILTER)
        if path is None:
            return None
        try:
            write_table(feature_matrix_table(fm, self._session.project.display_timezone), path,
                        sheet="Features", notes=[f.replace("\n", " ") for f in fm.notes])  # fmt: skip
        except ExportError as exc:
            self._fail("Cannot export the feature matrix", str(exc))
            return None
        self._done(path)
        return path

    def save_model(self, path: Path | None = None) -> Path | None:
        if not self._run_ok():
            return None
        path = path or self._ask_path("Save model file", f"{self._safe_name()}{MODEL_EXTENSION}",
                                      f"Inferential model (*{MODEL_EXTENSION})")  # fmt: skip
        if path is None:
            return None
        if path.suffix.lower() != MODEL_EXTENSION:
            path = path.with_suffix(MODEL_EXTENSION)
        try:
            save_artifact(self.artifact(), path)
        except ArtifactError as exc:
            self._fail("Cannot save the model file", str(exc))
            return None
        self._done(path)
        return path

    def _safe_name(self) -> str:
        base = self.report_info().title or f"{self._session.project.name}"
        return "".join(c if c.isalnum() or c in " -_." else "_" for c in base).strip() or "model"

    # ------------------------------------------------------------------ apply a saved model

    def load_model_file(self, path: Path | None = None) -> bool:
        if path is None:
            name, _ = QFileDialog.getOpenFileName(self, "Load model file", self._settings.last_directory,
                                                  f"Inferential model (*{MODEL_EXTENSION})")  # fmt: skip
            if not name:
                return False
            path = Path(name)
        try:
            self._loaded = load_artifact(path)
        except ArtifactError as exc:
            self._fail("Cannot load the model file", str(exc))
            return False
        self._loaded_path = path
        self._prediction = None
        self.btn_export_applied.setEnabled(False)
        show_placeholder(self.apply_plot, "Model loaded. Click Apply model to predict.")
        self._update_apply_state()
        return True

    def _update_apply_state(self) -> None:
        art = self._loaded
        if art is None:
            self.btn_apply.setEnabled(False)
            return
        frame, _mask, _target = self._session.frames()
        units = {v.name: v.units for v in self._session.project.variables}
        problems = check_compatibility(art, list(frame.columns), units)
        d = art.data
        text = (
            f"<b>{self._loaded_path.name}</b>: {art.name} for {art.target}, version "
            f"{d.get('model_version', '')}, created {str(d.get('created_at', ''))[:16]} UTC; "
            f"predictors: {', '.join(art.coefficients)}."
        )
        self._session.messages.set(
            "export.model", Level.ERROR,
            f"{self._loaded_path.name} cannot be applied to this project: " + " ".join(problems)
            if problems else "",
        )  # fmt: skip
        if problems:
            text += "<br>Not compatible with this project — see Messages."
        elif frame.empty:
            text += "<br>Import process data to apply it."
        self.loaded_note.setText(text)
        self.btn_apply.setEnabled(not problems and not frame.empty)

    def prediction_times(self) -> pd.DatetimeIndex:
        frame, _mask, target = self._session.frames()
        if self.when.currentData() == "lab" and target is not None and len(target):
            return pd.DatetimeIndex(target.index).unique()
        step = pd.Timedelta(minutes=self.every.value())
        start = frame.index.min().ceil(step)
        return pd.date_range(start, frame.index.max(), freq=step)

    def apply_model(self) -> None:
        art = self._loaded
        if art is None:
            return
        frame, _mask, _target = self._session.frames()
        state = self._session.cleaning
        use = self.use_exclusions.isChecked() and state is not None
        units = {v.name: v.units for v in self._session.project.variables}
        self.btn_apply.setEnabled(False)
        self._tasks.submit(
            f"Applying {art.name}",
            _predict_task,
            art,
            frame,
            self.prediction_times(),
            state.excluded if use else None,
            state.regime if use else None,
            units,
            on_result=self._applied,
            on_error=lambda msg, _d: (
                self._fail("Cannot apply the model", msg),
                self._update_apply_state(),
            ),
            on_cancelled=self._update_apply_state,
        )

    def _applied(self, pred: Prediction) -> None:
        self._prediction = pred
        self._update_apply_state()
        self.btn_export_applied.setEnabled(True)
        clear_placeholder(self.apply_plot)
        t = pred.table
        x = to_epoch_seconds(pd.DatetimeIndex(t["time"]))
        ok = (t["status"] == PRED_OK).to_numpy()
        self.apply_plot.plot(x, np.where(ok, t["prediction"], np.nan), pen=pg.mkPen(BLUE, width=1.5),
                             connect="finite", name="Prediction")  # fmt: skip
        out = t["n_outside_training_range"].to_numpy() > 0
        if out.any():
            scatter(
                self.apply_plot,
                x[out],
                t["prediction"].to_numpy()[out],
                VERMILION,
                "Outside training range",
                size=6,
            )
        _f, _m, target = self._session.frames()
        lines = [f"{pred.n_ok:,} of {len(t):,} times predicted."] + pred.warnings
        if (
            target is not None
            and self._loaded is not None
            and str(target.name) == self._loaded.target
        ):
            lab = target.dropna()
            scatter(
                self.apply_plot,
                to_epoch_seconds(lab.index),
                lab.to_numpy(),
                GREY,
                "Lab",
                size=5,
            )
            if self.when.currentData() == "lab":
                merged = t.set_index("time")["prediction"].reindex(lab.index)
                err = (merged - lab).dropna()
                if len(err):
                    lines.append(
                        f"Against the lab ({len(err)} samples): RMSE {np.sqrt(np.mean(err**2)):.4g}, "
                        f"bias (predicted − actual) {err.mean():.4g}."
                    )
        self.apply_plot.setLabel("left", self._loaded.target if self._loaded else "")
        self.apply_note.setText(" ".join(lines[:1] + lines[-1:]) if len(lines) > 1 else lines[0])
        level = Level.WARNING if pred.warnings else Level.INFO
        self._session.messages.post(
            level, f"Applied {self._loaded.name}: " + " ".join(lines), "export"
        )

    def export_applied(self, path: Path | None = None) -> Path | None:
        if self._prediction is None:
            return None
        path = path or self._ask_path(
            "Export batch predictions", "batch predictions.xlsx", TABLE_FILTER
        )
        if path is None:
            return None
        try:
            table = batch_prediction_table(
                self._prediction.table, self._session.project.display_timezone
            )
            write_table(table, path, sheet="Predictions", notes=self._prediction.warnings)
        except ExportError as exc:
            self._fail("Cannot export", str(exc))
            return None
        self._done(path)
        return path

    def showEvent(self, event) -> None:  # noqa: N802, ANN001
        super().showEvent(event)
        self._update_apply_state()
