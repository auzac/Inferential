"""The open project in the GUI: configuration, loaded data, file path and unsaved-changes flag."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from PySide6.QtCore import QObject, Signal

from inferential_studio.core.artifacts import compare_results, results_summary
from inferential_studio.core.dataset import ProjectData, SourceData
from inferential_studio.core.exclusions import CleaningState, compute_cleaning
from inferential_studio.core.features import AlignmentError, FeatureMatrix, build_project_features
from inferential_studio.core.models import ModelError, TrainingRun, evaluate_final_test
from inferential_studio.core.persistence import (
    PORTABLE,
    REFERENCE,
    extract_embedded,
    load_project,
    load_results,
    read_manifest,
    rebase_source_paths,
    resolve_source_path,
    save_project,
)
from inferential_studio.core.quality import DetectorResult
from inferential_studio.core.schemas import (
    DataSource,
    FeatureSpec,
    Project,
    SourceKind,
    VariableMeta,
    VariableRole,
)
from inferential_studio.ui.messages import Level, MessageCenter

if TYPE_CHECKING:
    from inferential_studio.infrastructure.workers import TaskManager

log = logging.getLogger(__name__)


def _compute_cleaning_task(
    project: Project,
    frame: pd.DataFrame,
    text_mask: pd.DataFrame | None,
    detector_results: dict[tuple[str, str], DetectorResult] | None,
    target: pd.Series | None,
    **_task_kwargs: object,
) -> CleaningState:
    """TaskManager entry point: compute_cleaning itself ignores progress/cancel."""
    return compute_cleaning(project, frame, text_mask, detector_results, target)


def _build_features_task(
    project: Project,
    data: ProjectData,
    frame: pd.DataFrame,
    cleaning: CleaningState | None,
    **task_kwargs: object,
) -> FeatureMatrix:
    return build_project_features(project, data, frame, cleaning, **task_kwargs)


class ProjectSession(QObject):
    changed = Signal()  # project configuration replaced or edited
    data_changed = Signal()  # imported data added, removed or reloaded
    dirty_changed = Signal(bool)
    path_changed = Signal(object)  # Path | None
    cleaning_changed = Signal()  # exclusion masks recomputed
    detectors_changed = Signal()  # detector configuration edited: results must be recomputed
    inputs_changed = Signal()  # predictors, feature settings or alignment options edited
    modelling_changed = Signal()  # model / split configuration edited
    features_changed = Signal()  # aligned feature matrix recomputed (or failed: see features_error)
    FEATURE_STAGES = ("variables", "processing", "model_setup")
    training_changed = Signal()  # a training run finished or was invalidated
    progress_changed = Signal()  # a stage was visited or an export written

    # Project attributes that undoable edits replace wholesale.
    CLEANING_LISTS = ("exclusions", "rules", "detectors", "flag_decisions")
    INPUT_LISTS = ("variables", "features")
    CONFIGS = ("alignment", "modelling", "report")

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._project = Project()
        self.messages = MessageCenter(self)  # every info / warning / error shown in the footer
        self._data = ProjectData()
        self._path: Path | None = None
        self._dirty = False
        self.data_version = 0  # increments whenever data changes; used to invalidate caches
        self._frame_cache: tuple[int, pd.DataFrame, pd.DataFrame, pd.Series | None] | None = None
        self.detector_results: dict[tuple[str, str], DetectorResult] = {}
        self.detector_results_version = -1  # data version the results were computed for
        self.cleaning: CleaningState | None = None
        self._tasks: TaskManager | None = None  # attached by the main window
        self._cleaning_running = False
        self._cleaning_serial = 0  # bumped on every recompute request; stale results are dropped
        self.features: FeatureMatrix | None = None
        self.features_error: str | None = None
        self._features_running = False
        self._features_serial = 0
        self.training: TrainingRun | None = None
        self.training_stale = False  # inputs changed since the run: results kept but marked
        self.test_revealed = False  # the user chose to look at final-test results of this run
        self.storage = REFERENCE  # or PORTABLE: data files embedded in the project file
        self.embedded_paths: dict[str, Path] = {}  # portable: extracted data per source id
        self.saved_results: dict | None = None  # training summary stored in the project file
        self.reproduction: str | None = None  # outcome of re-training a reopened project
        self.cleaning_changed.connect(self.recompute_features)
        self.features_changed.connect(self._report_features)
        self.training_changed.connect(self._report_training)

    # ------------------------------------------------------------------ state

    @property
    def project(self) -> Project:
        return self._project

    @property
    def data(self) -> ProjectData:
        return self._data

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def display_name(self) -> str:
        return self._path.name if self._path else self._project.name

    # ------------------------------------------------------------------ project lifecycle

    def new(self, project: Project | None = None) -> None:
        self._replace(project or Project(), None)

    def open(self, path: str | Path, *, cache_dir: Path | None = None) -> None:
        """Load the configuration synchronously (small). Data is loaded separately in the background.

        Portable projects have their embedded data files extracted (verified) to ``cache_dir``.
        """
        path = Path(path)
        project = load_project(path)
        manifest = read_manifest(path)
        embedded: dict[str, Path] = {}
        if manifest.get("storage") == PORTABLE:
            if cache_dir is None:
                raise ValueError("A cache folder is needed to open a portable project.")
            key = hashlib.sha256(
                f"{path.resolve()}|{manifest.get('saved_at')}".encode()
            ).hexdigest()
            embedded = extract_embedded(path, cache_dir / key[:16])
        saved_results = load_results(path)
        self._replace(project, path)
        self.storage = manifest.get("storage", REFERENCE)
        self.embedded_paths = embedded
        self.saved_results = saved_results

    def open_recovered(self, copy_path: Path, original_path: Path | None) -> None:
        """Open a crash-recovery copy as unsaved changes to its original file."""
        project = load_project(copy_path)
        self._replace(project, copy_path)
        self._path = original_path
        self.path_changed.emit(original_path)
        self._set_dirty(True)

    def source_file(self, source: DataSource) -> Path:
        """Where a source's data is read from (extracted copy for portable projects)."""
        return self.embedded_paths.get(source.source_id) or resolve_source_path(source, self._path)

    def save(self, path: str | Path | None = None, *, portable: bool | None = None) -> Path:
        """Save (reference or portable). ``portable`` None keeps the current storage mode."""
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ValueError("No file name given for the project.")
        portable = self.storage == PORTABLE if portable is None else portable
        embed = (
            {s.source_id: self.source_file(s) for s in self._project.sources} if portable else None
        )
        results = None
        if self.training is not None and not self.training_stale:
            results = results_summary(self.training, test_revealed=self.test_revealed)
        project = rebase_source_paths(self._project, self._path, target)
        if portable and self.embedded_paths:
            # Keep the original file names; the data travels inside the project.
            project = replace(project, sources=[
                replace(s, path=old.path) for s, old in zip(project.sources, self._project.sources, strict=True)
            ])  # fmt: skip
        self._project = save_project(project, target, embed=embed, results=results)
        for src in self._project.sources:  # keep loaded data pointing at the rebased source spec
            loaded = self._data.sources.get(src.source_id)
            if loaded is not None:
                loaded.source = replace(loaded.source, path=src.path)
        self._path = target
        self.storage = PORTABLE if portable else REFERENCE
        if results is not None:
            self.saved_results = results
        self.path_changed.emit(target)
        self._set_dirty(False)
        return target

    def mark_progress(self, key: str) -> None:
        """Record a visited stage (or ``"export"`` once something is exported). Saved with the
        project but does not mark it unsaved: moving between pages is not an edit."""
        if key not in self._project.progress:
            self._project.progress.append(key)
            self.progress_changed.emit()

    def mark_dirty(self) -> None:
        self._set_dirty(True)
        self.changed.emit()

    # ------------------------------------------------------------------ cleaning

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
        """(process values, text mask, target) for the current data, cached per data version."""
        cache = self._frame_cache
        if cache is None or cache[0] != self.data_version:
            frame, mask = self._data.process_frame(self._project)
            target = self._data.target_series(self._project)
            cache = (self.data_version, frame, mask, target)
            self._frame_cache = cache
        return cache[1], cache[2], cache[3]

    def set_lists(self, **lists: object) -> None:
        """Replace whole project attributes (used by undoable edits)."""
        allowed = set(self.CLEANING_LISTS) | set(self.INPUT_LISTS) | set(self.CONFIGS)
        unknown = set(lists) - allowed
        if unknown:
            raise ValueError(f"{sorted(unknown)} cannot be edited this way.")
        project = self._project
        detectors_before = list(project.detectors)
        target_before = project.target.name if project.target else None
        values = {k: (list(v) if isinstance(v, (list, tuple)) and k != "modelling" else v)
                  for k, v in lists.items()}  # fmt: skip
        replace(project, **values)  # validates; raises before any change
        for name, value in values.items():
            setattr(project, name, value)
        self.mark_dirty()
        if "variables" in lists:
            target_after = project.target.name if project.target else None
            if target_after != target_before:
                self._bump_data()  # a new target: target masks, detectors, statistics change
                return
        if "detectors" in lists and lists["detectors"] != detectors_before:
            self.detectors_changed.emit()
        if set(lists) & set(self.CLEANING_LISTS):
            self.recompute_cleaning()
        if set(lists) & {"variables", "features", "alignment"}:
            self.inputs_changed.emit()
            self.recompute_features()
        if "modelling" in lists:
            self.modelling_changed.emit()
            self._invalidate_training()

    def model_inputs(self, target: str | None, predictors: list[str]) -> dict[str, list]:
        """Variables and features lists for a new target / predictor choice (not applied)."""
        project = self._project
        chosen = set(predictors) - ({target} if target else set())
        variables = []
        for v in project.variables:
            role = (
                VariableRole.TARGET
                if v.name == target
                else VariableRole.PREDICTOR
                if v.name in chosen
                else VariableRole.UNUSED
            )
            variables.append(replace(v, role=role))
        kept = {f.variable: f for f in project.features}
        features = [kept.get(name, FeatureSpec(name)) for name in
                    (v.name for v in variables if v.role is VariableRole.PREDICTOR)]  # fmt: skip
        return {"variables": variables, "features": features}

    def set_detector_results(
        self, results: dict[tuple[str, str], DetectorResult], version: int
    ) -> None:
        if version != self.data_version:
            return  # stale: computed for data that has since changed
        self.detector_results = results
        self.detector_results_version = version
        self.recompute_cleaning()

    def attach_tasks(self, tasks: TaskManager) -> None:
        """Run cleaning recomputation in the background (set once by the main window)."""
        self._tasks = tasks

    def recompute_cleaning(self) -> None:
        """Recompute the exclusion masks. In the app this runs in the background and is
        coalesced: a burst of edits produces one computation with the latest state."""
        self._cleaning_serial += 1
        frame, _mask, _target = self.frames()
        if frame.empty:
            self.cleaning = None
            self.cleaning_changed.emit()
            return
        if self._tasks is None:
            self._recompute_now()
        else:
            self._start_cleaning_task()

    def _recompute_now(self) -> None:
        frame, mask, target = self.frames()
        results = (
            self.detector_results if self.detector_results_version == self.data_version else {}
        )
        self.cleaning = compute_cleaning(self._project, frame, mask, results, target)
        self.cleaning_changed.emit()

    def _start_cleaning_task(self) -> None:
        if self._cleaning_running or self._tasks is None:
            return  # the running task notices the newer serial and reruns
        self._cleaning_running = True
        serial = self._cleaning_serial
        frame, mask, target = self.frames()
        results = (
            self.detector_results if self.detector_results_version == self.data_version else {}
        )
        self._tasks.submit(
            "Recalculating exclusions",
            _compute_cleaning_task,
            self._project,
            frame,
            mask,
            results,
            target,
            on_result=lambda state, s=serial: self._on_cleaning_done(state, s),
            on_error=lambda msg, _details, s=serial: self._on_cleaning_failed(msg, s),
            on_cancelled=lambda s=serial: self._on_cleaning_cancelled(s),
        )

    def _on_cleaning_done(self, state: CleaningState, serial: int) -> None:
        self._cleaning_running = False
        if serial != self._cleaning_serial:
            self._start_cleaning_task()  # edits arrived while computing: run again
            return
        self.cleaning = state
        self.cleaning_changed.emit()

    def _on_cleaning_failed(self, message: str, serial: int) -> None:
        self._cleaning_running = False
        log.error("Exclusion recomputation failed: %s", message)
        if serial != self._cleaning_serial:
            self._start_cleaning_task()
            return
        self.cleaning_changed.emit()  # pages keep showing the previous state

    def _on_cleaning_cancelled(self, serial: int) -> None:
        self._cleaning_running = False
        # No rerun: cancellation happens when the app is shutting down.

    # ------------------------------------------------------------------ aligned features

    def recompute_features(self) -> None:
        """Rebuild the aligned feature matrix (background, coalesced like cleaning)."""
        self._features_serial += 1
        self._invalidate_training()
        project = self._project
        frame, _mask, _target = self.frames()
        if frame.empty or project.target is None or not project.predictors:
            self.features = None
            self.features_error = (
                "Import process and lab data first."
                if frame.empty
                else "Choose a target variable."
                if project.target is None
                else "Choose at least one predictor."
            )
            self.features_changed.emit()
            return
        if self.cleaning is None:
            if not self._cleaning_running:  # cleaning failed: do not align without exclusions
                self.features = None
                self.features_error = "Exclusions could not be computed; see the application log."
                self.features_changed.emit()
            return  # otherwise cleaning is being recomputed; features follow when it is done
        if self._tasks is None:
            self._features_now()
        else:
            self._start_features_task()

    def _features_now(self) -> None:
        frame, _mask, _target = self.frames()
        try:
            self.features = build_project_features(self._project, self._data, frame, self.cleaning)
            self.features_error = None
        except AlignmentError as exc:
            self.features, self.features_error = None, str(exc)
        self.features_changed.emit()

    def _start_features_task(self) -> None:
        if self._features_running or self._tasks is None:
            return
        self._features_running = True
        serial = self._features_serial
        frame, _mask, _target = self.frames()
        self._tasks.submit(
            "Aligning features",
            _build_features_task,
            replace(  # snapshot: later edits do not race the worker
                self._project,
                variables=list(self._project.variables),
                features=list(self._project.features),
            ),
            self._data,
            frame,
            self.cleaning,
            on_result=lambda fm, s=serial: self._on_features_done(fm, None, s),
            on_error=lambda msg, _d, s=serial: self._on_features_done(None, msg, s),
            on_cancelled=lambda: setattr(self, "_features_running", False),
        )

    def _on_features_done(self, fm: FeatureMatrix | None, error: str | None, serial: int) -> None:
        self._features_running = False
        if serial != self._features_serial:
            self._start_features_task()
            return
        self.features, self.features_error = fm, error
        self.features_changed.emit()

    def _report_features(self) -> None:
        """One message about the aligned feature matrix, shown on stages 4–6."""
        fm = self.features
        if fm is None:
            level = Level.WARNING if self.features_error else Level.INFO
            text = self.features_error or "Aligning features…"
        else:
            s = fm.summary()
            level = Level.INFO if s["eligible"] else Level.WARNING
            text = (
                f"{s['eligible']:,} of {s['lab_samples']:,} lab samples are model-eligible "
                f"(target missing {s['target_missing']}, target excluded {s['target_excluded']}, "
                f"failed feature checks {s['feature_failures']}"
                + (
                    f", flagged target values awaiting review {s['target_pending_review']}"
                    if s["target_pending_review"]
                    else ""
                )  # fmt: skip
                + ")."
            )
        self.messages.set("variables.features", level, text, also=self.FEATURE_STAGES[1:])
        non_causal = fm.non_causal if fm is not None else []
        self.messages.set(
            "processing.noncausal",
            Level.WARNING,
            "Non-causal settings read process data recorded after the lab sample time: "
            + ", ".join(non_causal)
            + ". Use them for retrospective analysis only."
            if non_causal
            else "",
            also=("variables", "model_setup", "training", "validation"),
        )

    def _report_training(self) -> None:
        """Messages about the training run, shown on stages 7–9."""
        m, run = self.messages, self.training
        later = ("validation", "export")
        if run is None:
            m.set("training.state", Level.INFO,
                  "No models trained yet. Configure them on Model Setup and press Train.", also=later)  # fmt: skip
        elif self.training_stale:
            m.set("training.state", Level.WARNING,
                  "Settings or data changed since the models were trained: the results shown are out "
                  "of date. Train again (exports are disabled until then).", also=later)  # fmt: skip
        else:
            counts = run.split.counts()
            m.set(
                "training.state", Level.INFO,
                f"Trained on {counts['train']} samples ({run.n_effective:.0f} effectively independent); "
                f"{len(run.features)} predictors; {run.config.scaling.value} scaling; "
                f"{run.config.split.cv_folds}-fold {run.config.split.cv_kind.value} CV in the training period.",
                also=later,
            )  # fmt: skip
        errors = run.errors if run is not None else []
        m.set("training.errors", Level.ERROR, " ".join(errors), also=later)
        warnings = [p.message for p in run.problems if p.severity != "error"] if run else []
        m.set("training.warnings", Level.WARNING, " ".join(warnings), also=later)
        failed = [f"{x.spec.label}: {x.error}" for x in (run.models if run else []) if not x.ok]
        m.set("training.failed", Level.WARNING, "Not trained — " + "; ".join(failed) if failed else "",
              also=later)  # fmt: skip
        repro = self.reproduction
        m.set("training.reproduction", Level.INFO if repro and repro.startswith("✓") else Level.WARNING,
              repro or "", also=later)  # fmt: skip
        m.set("validation.revealed", Level.INFO,
              "Final test results are shown. Model or setting changes made after looking at them make "
              "the test exploratory." if run is not None and self.test_revealed else "",
              also=("export",))  # fmt: skip

    # ------------------------------------------------------------------ training

    def set_training(self, run: TrainingRun, *, reproduce: bool = False) -> None:
        """Install a training run. ``reproduce``: it re-runs the saved results of a reopened project."""
        self.training = run
        self.training_stale = False
        self.test_revealed = False
        self.reproduction = None
        saved = self.saved_results
        if reproduce and saved is not None:
            self.test_revealed = bool(saved.get("test_revealed"))
            selected = saved.get("selected_model")
            if selected and any(m.name == selected and m.ok for m in run.models):
                run.selected_model_name = selected
            if self.test_revealed and run.search_subsets and run.selected_model_name:
                evaluate_final_test(run, run.selected_model_name)
            diffs = compare_results(saved, run)
            when = str(saved.get("saved_at", ""))[:16].replace("T", " ")
            self.reproduction = (
                f"✓ Results saved on {when} UTC were reproduced."
                if not diffs
                else f"⚠ Results differ from those saved on {when} UTC: " + " ".join(diffs[:4])
            )
        self.training_changed.emit()

    def select_model(self, model_name: str) -> None:
        """Remember the candidate chosen for diagnostics, final-test reveal and export."""
        run = self.training
        if run is None or not any(m.name == model_name and m.ok for m in run.models):
            return
        if run.selected_model_name != model_name:
            run.selected_model_name = model_name
            self.mark_dirty()
            self.training_changed.emit()

    def reveal_test(self) -> None:
        if self.training is not None and self.training.search_subsets:
            selected = self.training.selected_model_name
            if selected is None:
                raise ModelError("Choose a candidate before revealing the final test results.")
            evaluate_final_test(self.training, selected)
        self.test_revealed = True
        self.training_changed.emit()

    def _invalidate_training(self) -> None:
        if self.training is not None and not self.training_stale:
            self.training_stale = True
            self.training_changed.emit()

    # ------------------------------------------------------------------ data

    def set_data(self, data: ProjectData) -> None:
        """Install data loaded for the current project (checksums recorded at import kept as is)."""
        self._data = data
        self._bump_data()

    def add_source(self, source_data: SourceData, variables: list[VariableMeta]) -> None:
        """Add a newly imported file and its variables to the project."""
        source = source_data.source  # carries the checksum computed during import
        project = self._project
        project.sources.append(source)
        project.variables.extend(variables)
        try:
            project.validate()
        except Exception:
            project.sources.remove(source)
            for v in variables:
                project.variables.remove(v)
            raise
        self._data.sources[source.source_id] = source_data
        self._data.failures.pop(source.source_id, None)
        self.mark_dirty()
        self._bump_data()

    def accept_changed_source(self, source_data: SourceData) -> None:
        """Replace a source whose file changed, recording the new checksum."""
        sid = source_data.source_id
        self._project.sources = [
            replace(s, sha256=source_data.sha256) if s.source_id == sid else s
            for s in self._project.sources
        ]
        self._data.sources[sid] = source_data
        self._data.failures.pop(sid, None)
        self.mark_dirty()
        self._bump_data()

    def remove_source(self, source_id: str) -> None:
        project = self._project
        removed = {v.name for v in project.variables if v.source_id == source_id}
        project.sources = [s for s in project.sources if s.source_id != source_id]
        project.variables = [v for v in project.variables if v.source_id != source_id]
        project.features = [f for f in project.features if f.variable not in removed]
        project.exclusions = [
            ex for ex in project.exclusions if not ex.signals or not set(ex.signals) & removed
        ]
        project.rules = [
            r for r in project.rules if not (set(r.signals) | r.condition.variables()) & removed
        ]
        kept_detectors = []
        for det in project.detectors:
            variables = tuple(v for v in det.variables if v not in removed)
            if variables:
                kept_detectors.append(replace(det, variables=variables))
        project.detectors = kept_detectors
        detector_ids = {d.detector_id for d in kept_detectors}
        project.flag_decisions = [
            f
            for f in project.flag_decisions
            if f.variable not in removed and f.detector_id in detector_ids
        ]
        self._data.sources.pop(source_id, None)
        self._data.failures.pop(source_id, None)
        self.mark_dirty()
        self._bump_data()

    def new_source_id(self, kind: SourceKind) -> str:
        existing = {s.source_id for s in self._project.sources}
        base = kind.value
        if base not in existing:
            return base
        i = 2
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def source_for(self, source_id: str) -> DataSource | None:
        for s in self._project.sources:
            if s.source_id == source_id:
                return s
        return None

    # ------------------------------------------------------------------ internals

    def _replace(self, project: Project, path: Path | None) -> None:
        self._project = project
        self.storage = REFERENCE
        self.embedded_paths = {}
        self.saved_results = None
        self.reproduction = None
        self.training = None
        self.training_changed.emit()
        self.training_stale = False
        self.test_revealed = False
        self._path = path
        self._data = ProjectData()
        self.path_changed.emit(path)
        self.changed.emit()
        self._bump_data()
        self._set_dirty(False)

    def _bump_data(self) -> None:
        self.data_version += 1
        self.detector_results = {}
        self._cleaning_serial += 1  # discard any in-flight cleaning computation
        self.cleaning = None  # the old masks do not match the new data
        self._features_serial += 1  # and neither do the aligned features
        self.features = None
        self._invalidate_training()
        self.data_changed.emit()
        self.recompute_cleaning()

    def _set_dirty(self, dirty: bool) -> None:
        if dirty != self._dirty:
            self._dirty = dirty
            self.dirty_changed.emit(dirty)
