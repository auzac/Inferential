"""Versioned, code-free model artifacts (``.imsmodel``) and batch prediction.

Every model the application trains (MLR, PLS, PCR, Ridge with any scaling) is an affine
function of its *features*: ``ŷ = b₀ + Σ bⱼ fⱼ``, where each feature ``fⱼ`` is computed from one
process tag by the documented alignment steps (cleaning, resample, trailing filter, window
aggregation; see :mod:`inferential_studio.core.features`). An artifact therefore stores plain
JSON — the feature definitions, the intercept and coefficients, the training ranges, metrics and
provenance — and never a pickled Python object. It is safe to open from any source and does
not depend on the scikit-learn version.

File layout (ZIP): ``manifest.json`` (format id, artifact version, app version) and
``model.json``. Loading checks entry names, sizes and the format, exactly like project files.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inferential_studio import __version__
from inferential_studio.core.features import (
    OK,
    TargetSamples,
    build_feature_matrix,
    data_span_s,
)
from inferential_studio.core.models import PcrDetails, PlsDetails, TrainingRun
from inferential_studio.core.schemas import (
    AlignmentConfig,
    FeatureSpec,
    Project,
    SchemaError,
    from_jsonable,
    to_jsonable,
    utc_now,
)
from inferential_studio.core.splits import SPLITS, TEST, TRAIN, VALIDATION
from inferential_studio.core.tasks import CancelToken, ProgressCallback

MODEL_EXTENSION = ".imsmodel"
MODEL_FORMAT_ID = "inferential-model-studio-model"
ARTIFACT_VERSION = 1
MAX_MODEL_JSON_BYTES = 32 * 1024 * 1024

# Prediction status codes.
PRED_OK = "ok"
PRED_INPUT = "input not usable"


class ArtifactError(Exception):
    """A model file cannot be written, read or applied. The message is shown to the user."""


# --------------------------------------------------------------------------- description


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:g} min"


def describe_feature(spec: FeatureSpec) -> str:
    """Plain-language definition of one feature, e.g. 'mean of X over (t − 50 min, t − 20 min]'."""
    start, end = spec.window_offsets_s

    def rel(s: float) -> str:
        return "t" if s == 0 else f"t {'−' if s < 0 else '+'} {abs(s) / 60:g} min"

    parts = [f"{spec.aggregation.value} of {spec.variable} over ({rel(start)}, {rel(end)}]"]
    if spec.resample_s:
        parts.append(f"after averaging into {_minutes(spec.resample_s)} bins")
    f = spec.filter
    if f.kind.value == "ewma":
        parts.append(f"after a trailing EWMA (half-life {_minutes(f.halflife_s)})")
    elif f.kind.value != "none":
        parts.append(f"after a trailing {f.kind.value.replace('_', ' ')} ({_minutes(f.window_s)})")
    parts.append(f"needs ≥ {spec.min_coverage * 100:g} % coverage")
    if spec.max_gap_s is not None:
        parts.append(f"no gap > {_minutes(spec.max_gap_s)}")
    if not spec.is_causal:
        parts.append("NON-CAUSAL: reads data after t")
    return ", ".join(parts)


def equation_text(target: str, intercept: float, coefficients: dict[str, float]) -> str:
    """The model as an equation on its features (f_name = the aligned feature of that tag)."""
    terms = [f"{intercept:.10g}"]
    for name, b in coefficients.items():
        sign = "−" if b < 0 else "+"
        terms.append(f"{sign} {abs(b):.10g} × f[{name}]")
    return f"{target} = " + " ".join(terms)


EQUATION_CAVEAT = (
    "This equation is exact only for features computed exactly as defined (cleaned inputs, "
    "resampling, trailing filters, deadtime/shift and window aggregation at each sample time, "
    "with the same coverage rules). Applying it to raw tag values, or without the time "
    "alignment, gives different and unvalidated results. Scaling is already folded into the "
    "coefficients."
)


# --------------------------------------------------------------------------- artifact


@dataclass
class ModelArtifact:
    """Everything needed to document and apply one trained model (JSON-serialisable)."""

    data: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.data["model"]["name"]

    @property
    def target(self) -> str:
        return self.data["target"]["name"]

    @property
    def intercept(self) -> float:
        return float(self.data["model"]["intercept"])

    @property
    def coefficients(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.data["model"]["coefficients"].items()}

    @property
    def specs(self) -> list[FeatureSpec]:
        return [from_jsonable(FeatureSpec, p["feature"]) for p in self.data["predictors"]]

    @property
    def alignment(self) -> AlignmentConfig:
        return from_jsonable(AlignmentConfig, self.data["alignment"])

    @property
    def train_ranges(self) -> pd.DataFrame:
        rows = {p["name"]: p["training_range"] for p in self.data["predictors"]}
        return pd.DataFrame.from_dict(rows, orient="index")

    @property
    def units(self) -> dict[str, str | None]:
        return {p["name"]: p.get("units") for p in self.data["predictors"]}

    def equation(self) -> str:
        return equation_text(self.target, self.intercept, self.coefficients)


def build_artifact(
    project: Project,
    run: TrainingRun,
    model_name: str,
    *,
    owner: str = "",
    purpose: str = "",
    model_version: str = "1",
    include_test: bool = False,
    project_path: str | None = None,
) -> ModelArtifact:
    """Describe one model of a training run. Final-test metrics only when ``include_test``."""
    m = next((x for x in run.models if x.name == model_name), None)
    if m is None or not m.ok:
        raise ArtifactError(f"'{model_name}' is not a trained model of this run.")
    variables = {v.name: v for v in project.variables}
    target = variables.get(run.target)
    labels = run.split.labels
    times = run.sample_time

    def period(split: str) -> dict[str, Any]:
        mask = labels == split
        if not mask.any():
            return {"samples": 0}
        t = times[mask]
        return {
            "samples": int(mask.sum()),
            "start": t.min().isoformat(),
            "end": t.max().isoformat(),
        }

    shown = [s for s in SPLITS if s != TEST or include_test]
    metrics = {
        s: {k: (None if pd.isna(v) else float(v)) for k, v in m.metrics.loc[s].items()}
        for s in shown
    }
    predictors = []
    for name in run.features:
        meta = variables.get(name)
        spec = run.feature_specs.get(name) or FeatureSpec(name)
        rng = run.train_ranges.loc[name]
        predictors.append(
            {
                "name": name,
                "units": meta.units if meta else None,
                "source_id": meta.source_id if meta else None,
                "source_column": meta.source_column if meta else None,
                "feature": to_jsonable(spec),
                "description": describe_feature(spec),
                "causal": spec.is_causal,
                "data_span_s": list(data_span_s(spec)),
                "training_range": {k: float(rng[k]) for k in ("min", "max", "mean", "std")},
                "coefficient": float(m.coefficients[name]),
                "standardized": None
                if pd.isna(m.standardized[name])
                else float(m.standardized[name]),
            }
        )
    details: dict[str, Any] = {}
    if isinstance(m.details, PlsDetails):
        d = m.details
        details = {
            "type": "pls",
            "vip": {k: float(v) for k, v in d.vip.items()},
            "x_weights": {c: d.x_weights[c].to_dict() for c in d.x_weights.columns},
            "x_loadings": {c: d.x_loadings[c].to_dict() for c in d.x_loadings.columns},
            "y_loadings": [float(v) for v in d.y_loadings],
            "x_variance_explained": [float(v) for v in d.x_variance_explained],
            "y_variance_explained": [float(v) for v in d.y_variance_explained],
        }
    elif isinstance(m.details, PcrDetails):
        d = m.details
        details = {
            "type": "pcr",
            "loadings": {c: d.loadings[c].to_dict() for c in d.loadings.columns},
            "x_variance_explained": [float(v) for v in d.x_variance_explained],
            "component_coefficients": [float(v) for v in d.component_coefficients],
        }
    cv = None
    if m.cv is not None:
        cv = {
            "parameter": m.cv.parameter,
            "values": [float(v) for v in m.cv.values],
            "mean_rmse": [float(v) for v in m.cv.mean_rmse],
            "se_rmse": [float(v) for v in m.cv.se_rmse],
            "chosen": m.cv.chosen,
            "rule": m.cv.rule,
        }
    sources = [
        {"source_id": s.source_id, "kind": s.kind.value, "path": s.path, "sha256": s.sha256}
        for s in project.sources
    ]
    data = {
        "format": MODEL_FORMAT_ID,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": utc_now().isoformat(),
        "app_version": __version__,
        "libraries": dict(run.versions),
        "model_version": model_version,
        "owner": owner,
        "purpose": purpose,
        "project": {"name": project.name, "file": project_path, "notes": project.notes},
        "display_timezone": project.display_timezone,
        "target": {
            "name": run.target,
            "units": target.units if target else None,
            "source_id": target.source_id if target else None,
            "source_column": target.source_column if target else None,
        },
        "predictors": predictors,
        "alignment": to_jsonable(run.alignment),
        "model": {
            "name": m.name,
            "kind": m.spec.kind.value,
            "label": m.spec.label,
            "hyperparameters": dict(m.hyperparameters),
            "scaling": run.config.scaling.value,
            "intercept": float(m.intercept),
            "coefficients": {k: float(v) for k, v in m.coefficients.items()},
            "cv_rmse": None if pd.isna(m.cv_rmse) else float(m.cv_rmse),
            "cv": cv,
            "details": details,
            "warnings": list(m.warnings),
        },
        "split": {
            "config": to_jsonable(run.config.split),
            "purge_s": run.split.purge_s,
            "periods": {s: period(s) for s in (TRAIN, VALIDATION, TEST)},
            "test_revealed": include_test,
        },
        "metrics": metrics,
        "effective_training_samples": float(run.n_effective),
        "sources": sources,
        "equation": equation_text(run.target, m.intercept, dict(m.coefficients)),
        "equation_caveat": EQUATION_CAVEAT,
        "notes": list(run.feature_notes),
    }
    return ModelArtifact(data)


# --------------------------------------------------------------------------- files


def save_artifact(artifact: ModelArtifact, path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    if target.suffix.lower() != MODEL_EXTENSION:
        raise ArtifactError(f"Model files must use the '{MODEL_EXTENSION}' extension.")
    manifest = {
        "format": MODEL_FORMAT_ID,
        "artifact_version": ARTIFACT_VERSION,
        "app_version": __version__,
    }
    payload = json.dumps(artifact.data, indent=2, ensure_ascii=False, allow_nan=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".imsmodel-", suffix=".tmp", dir=target.parent)
    try:
        with (
            os.fdopen(fd, "wb") as handle,
            zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as zf,
        ):
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("model.json", payload)
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except (OSError, ValueError) as exc:
        Path(tmp).unlink(missing_ok=True)
        raise ArtifactError(f"Could not save the model to '{target}': {exc}") from exc
    return target


def load_artifact(path: str | os.PathLike[str]) -> ModelArtifact:
    """Read a model file. Only JSON is read; nothing in the file is executed."""
    source = Path(path)
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ArtifactError(f"'{source.name}' is not a model file.")
    try:
        with zipfile.ZipFile(source) as zf:
            names = {i.filename: i for i in zf.infolist()}
            if set(names) != {"manifest.json", "model.json"}:
                raise ArtifactError(f"'{source.name}' has unexpected contents: {sorted(names)}.")
            parts = {}
            for name, info in names.items():
                if info.file_size > MAX_MODEL_JSON_BYTES:
                    raise ArtifactError(f"'{name}' in '{source.name}' is unreasonably large.")
                with zf.open(info) as handle:
                    raw = handle.read(MAX_MODEL_JSON_BYTES + 1)
                if len(raw) > MAX_MODEL_JSON_BYTES:
                    raise ArtifactError(f"'{name}' in '{source.name}' is unreasonably large.")
                parts[name] = json.loads(raw.decode("utf-8"))
    except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"'{source.name}' is damaged: {exc}") from exc
    manifest, data = parts["manifest.json"], parts["model.json"]
    if not isinstance(manifest, dict) or manifest.get("format") != MODEL_FORMAT_ID:
        raise ArtifactError(f"'{source.name}' is not an Inferential Model Studio model.")
    version = manifest.get("artifact_version")
    if not isinstance(version, int) or version > ARTIFACT_VERSION:
        raise ArtifactError(
            f"'{source.name}' was written by a newer version (model format {version}); "
            "please upgrade the application."
        )
    artifact = ModelArtifact(data)
    try:  # validate the parts prediction relies on
        _ = artifact.specs, artifact.alignment, artifact.coefficients, artifact.intercept
        names = [p["name"] for p in data["predictors"]]
        if names != list(artifact.coefficients):
            raise ArtifactError("Predictor list and coefficients do not match.")
        if not all(np.isfinite(list(artifact.coefficients.values()))):
            raise ArtifactError("The model contains non-finite coefficients.")
    except (KeyError, TypeError, SchemaError) as exc:
        raise ArtifactError(f"'{source.name}' is incomplete or invalid: {exc}") from exc
    return artifact


# --------------------------------------------------------------------------- prediction


@dataclass
class Prediction:
    table: pd.DataFrame  # time, prediction, status, n_outside, outside, then feature values
    warnings: list[str]

    @property
    def n_ok(self) -> int:
        return int((self.table["status"] == PRED_OK).sum())


def check_compatibility(
    artifact: ModelArtifact,
    available: list[str],
    units: dict[str, str | None] | None = None,
) -> list[str]:
    """Problems that prevent applying the model to this data (empty list = compatible)."""
    problems = []
    missing = [p for p in artifact.coefficients if p not in available]
    if missing:
        problems.append(f"Required tag(s) missing from the data: {', '.join(missing)}.")
    if units:
        for name, expected in artifact.units.items():
            got = units.get(name)
            if expected and got and expected.strip() != got.strip():
                problems.append(
                    f"{name}: the model expects units '{expected}', the data has '{got}'."
                )
    if artifact.target in artifact.coefficients:
        problems.append("The target is listed as a predictor.")
    return problems


def predict(
    artifact: ModelArtifact,
    process: pd.DataFrame,
    times: pd.DatetimeIndex,
    *,
    excluded: pd.DataFrame | None = None,
    regime: np.ndarray | None = None,
    units: dict[str, str | None] | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> Prediction:
    """Apply a saved model at ``times`` (UTC) from process data on its own timestamps.

    Features are rebuilt with the stored definitions and the same coverage rules; a time whose
    inputs fail them gets no prediction (status explains why) instead of a silent guess.
    Inputs outside the training range are counted per row (extrapolation warning).
    """
    problems = check_compatibility(artifact, list(process.columns), units)
    if problems:
        raise ArtifactError(" ".join(problems))
    times = pd.DatetimeIndex(times)
    if times.tz is None:
        raise ArtifactError("Prediction times must be timezone-aware.")
    times = times.sort_values()
    n = len(times)
    dummy = TargetSamples("__y__", times, np.zeros(n), np.zeros(n, bool), np.zeros(n, bool))
    fm = build_feature_matrix(
        dummy, process, artifact.specs, excluded=excluded, regime=regime,
        alignment=artifact.alignment, progress=progress, cancel=cancel,
    )  # fmt: skip
    coef = artifact.coefficients
    names = list(coef)
    X = fm.X[names].to_numpy(dtype=float)
    ok = (fm.status[names] == OK).all(axis=1).to_numpy()
    pred = np.full(n, np.nan)
    pred[ok] = artifact.intercept + X[ok] @ np.array([coef[c] for c in names])
    ranges = artifact.train_ranges.loc[names]
    with np.errstate(invalid="ignore"):
        outside = (ranges["min"].to_numpy() > X) | (ranges["max"].to_numpy() < X)
    outside &= ok[:, None]
    status = np.where(ok, PRED_OK, PRED_INPUT).astype(object)
    reasons = []
    st = fm.status[names].to_numpy()
    for i in range(n):
        if ok[i]:
            reasons.append("")
        else:
            reasons.append(
                "; ".join(f"{names[j]}: {st[i, j]}" for j in range(len(names)) if st[i, j] != OK)
            )
    table = pd.DataFrame(
        {
            "time": times,
            "prediction": pred,
            "status": status,
            "reason": reasons,
            "n_outside_training_range": outside.sum(axis=1),
            "outside_training_range": [
                ", ".join(names[j] for j in np.flatnonzero(row)) for row in outside
            ],
        }
    )
    for j, name in enumerate(names):
        table[f"f[{name}]"] = X[:, j]
    warnings = []
    if not ok.all():
        warnings.append(
            f"{int((~ok).sum())} of {n} times have no prediction: inputs failed the coverage rules."
        )
    if outside.any():
        warnings.append(
            f"{int(outside.any(axis=1).sum())} prediction(s) use inputs outside the training "
            "range (extrapolation)."
        )
    warnings.extend(fm.notes)
    return Prediction(table, warnings)
