"""Time alignment: one row of predictor features per laboratory sample.

Processing order for each predictor X (documented in docs/methodology.md):

1. **Clean** — start from the normalized process values; excluded cells (manual, rule and
   accepted-flag exclusions) and text/blank cells are removed. Nothing is interpolated.
2. **Resample** (optional, interval R) — valid values are averaged in bins ``(kR − R, kR]``
   labelled at the bin's *end*, so a resampled value never uses data recorded after its
   label time. Bins without valid data stay empty.
3. **Filter** (optional, always trailing) — rolling mean / median over ``(τ − F, τ]`` or a
   time-based EWMA, evaluated only at valid points from valid points.
4. **Window aggregation** at each lab sample time ``t`` over ``(t + start, t + end]``
   (see :class:`~inferential_studio.core.schemas.FeatureSpec`): mean, median, min, max,
   std (n − 1, needs two values) or last valid value.

Quality checks for each window:

* coverage = (valid values in the window × sampling interval) / window length, capped at 1,
  where the sampling interval is the variable's median spacing (or R when resampled);
* largest gap = longest time inside the window without a valid value, the window edges
  counting as boundaries (regular sampling gives one sampling interval);
* optional: the window must not overlap an operating-regime exclusion.

Target rows are the lab samples at their **sample (collection) times**, one row per sample.
Y is never forward-filled; replicate samples at the same time stay separate rows and are kept
together by every split. A row is *model-eligible* only when its Y is valid and not excluded
and every predictor feature passed its checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from inferential_studio.core.rules import median_interval_ns, ns
from inferential_studio.core.schemas import (
    Aggregation,
    AlignmentConfig,
    FeatureSpec,
    FilterKind,
    FilterSpec,
    Project,
)
from inferential_studio.core.tasks import CancelToken, ProgressCallback, check, report

NS = 1_000_000_000
EWMA_MEMORY_HALFLIVES = 5.0  # EWMA weight after 5 half-lives is ~3 %: counted as its history

# Feature status codes (per sample x predictor).
OK = "ok"
NO_DATA = "no data"
LOW_COVERAGE = "low coverage"
GAP = "gap too long"
TOO_FEW = "too few values"
REGIME = "overlaps excluded period"
STATUS_ORDER = (OK, NO_DATA, LOW_COVERAGE, GAP, TOO_FEW, REGIME)

# Target status codes (per sample).
Y_OK = "ok"
Y_MISSING = "target missing"
Y_EXCLUDED = "target excluded"


class AlignmentError(ValueError):
    """Alignment cannot run (no target, no predictors …). The message is shown to the user."""


# --------------------------------------------------------------------------- single signals


@dataclass
class ProcessedSignal:
    """One predictor after cleaning, resampling and filtering (valid points only)."""

    name: str
    times_ns: np.ndarray  # int64, sorted, valid points only
    values: np.ndarray  # float, finite
    interval_s: float  # sampling interval used for coverage

    def series(self) -> pd.Series:
        index = pd.DatetimeIndex(pd.to_datetime(self.times_ns, utc=True), name="time")
        return pd.Series(self.values, index=index, name=self.name)


def _resample(times: np.ndarray, values: np.ndarray, step_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of valid values per bin ``(kR − R, kR]``, labelled at ``kR`` (causal)."""
    if times.size == 0:
        return times, values
    labels = -((-times) // step_ns) * step_ns  # ceil to the bin end
    uniq, start = np.unique(labels, return_index=True)
    sums = np.add.reduceat(values, start)
    counts = np.diff(np.append(start, values.size))
    return uniq, sums / counts


def _filter(times: np.ndarray, values: np.ndarray, spec: FilterSpec) -> np.ndarray:
    if spec.kind is FilterKind.NONE or values.size == 0:
        return values
    index = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    s = pd.Series(values, index=index)
    if spec.kind is FilterKind.ROLLING_MEAN:
        out = s.rolling(pd.Timedelta(seconds=spec.window_s), min_periods=1).mean()
    elif spec.kind is FilterKind.ROLLING_MEDIAN:
        out = s.rolling(pd.Timedelta(seconds=spec.window_s), min_periods=1).median()
    else:
        out = s.ewm(halflife=pd.Timedelta(seconds=spec.halflife_s), times=index).mean()
    return out.to_numpy(dtype=float)


def process_signal(
    raw: pd.Series, spec: FeatureSpec, excluded: pd.Series | np.ndarray | None = None
) -> ProcessedSignal:
    """Clean, resample and filter one predictor (steps 1–3 of the module docstring)."""
    values = raw.to_numpy(dtype=float)
    times = ns(raw.index)
    present = ~np.isnan(values)
    interval_ns = median_interval_ns(raw.index[present]) if present.sum() > 1 else 0
    keep = present.copy()
    if excluded is not None:
        keep &= ~np.asarray(excluded, dtype=bool)
    times, values = times[keep], values[keep]
    if spec.resample_s is not None:
        step = int(round(spec.resample_s * NS))
        times, values = _resample(times, values, step)
        interval_ns = step
    values = _filter(times, values, spec.filter)
    interval_s = interval_ns / NS if interval_ns else 60.0
    return ProcessedSignal(str(raw.name), times, values, interval_s)


def history_s(spec: FeatureSpec) -> float:
    """How far before the window start a feature still draws on data (filter + resample)."""
    extra = 0.0
    f = spec.filter
    if f.kind in (FilterKind.ROLLING_MEAN, FilterKind.ROLLING_MEDIAN):
        extra += float(f.window_s)
    elif f.kind is FilterKind.EWMA:
        extra += EWMA_MEMORY_HALFLIVES * float(f.halflife_s)
    if spec.resample_s is not None:
        extra += float(spec.resample_s)
    return extra


def data_span_s(spec: FeatureSpec) -> tuple[float, float]:
    """``(earliest, latest)`` offsets in seconds of process data a feature can depend on."""
    start, end = spec.window_offsets_s
    return start - history_s(spec), end


# --------------------------------------------------------------------------- window aggregation


@dataclass
class WindowResult:
    value: np.ndarray
    coverage: np.ndarray
    gap_s: np.ndarray
    n_values: np.ndarray
    status: np.ndarray  # object array of status codes


def _grouped_aggregate(
    x: np.ndarray, start: np.ndarray, count: np.ndarray, how: Aggregation
) -> np.ndarray:
    """Median or SD of many windows at once: windows with the same number of values are
    gathered into one matrix (regular sampling gives only a few distinct counts)."""
    out = np.empty(start.size)
    chunk_cells = 4_000_000  # bound the temporary matrix (~32 MB)
    for n in np.unique(count):
        rows = np.flatnonzero(count == n)
        step = max(1, chunk_cells // int(n))
        for c in range(0, rows.size, step):
            r = rows[c : c + step]
            block = x[start[r, None] + np.arange(n)]
            if how is Aggregation.MEDIAN:
                out[r] = np.median(block, axis=1)
            else:
                out[r] = block.std(axis=1, ddof=1) if n > 1 else np.nan
    return out


def _range_reduce(ufunc: np.ufunc, a: np.ndarray, s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """``ufunc.reduce(a[s:e])`` for many non-empty ranges at once (``s < e`` required)."""
    if s.size == 0:
        return np.zeros(0, dtype=a.dtype)
    padded = np.append(a, a[:1])  # reduceat needs every index < len
    idx = np.empty(2 * s.size, dtype=np.intp)
    idx[0::2], idx[1::2] = s, e
    return ufunc.reduceat(padded, idx)[0::2]


def aggregate_windows(
    signal: ProcessedSignal,
    sample_ns: np.ndarray,
    spec: FeatureSpec,
    regime_ns: np.ndarray | None = None,
) -> WindowResult:
    """Feature value and quality for every sample time (steps 4 and the checks)."""
    a, b = spec.window_offsets_s
    lo = sample_ns + int(round(a * NS))
    hi = sample_ns + int(round(b * NS))
    tv, xv = signal.times_ns, signal.values
    i0 = np.searchsorted(tv, lo, side="right")  # left-open
    i1 = np.searchsorted(tv, hi, side="right")  # right-closed
    n = i1 - i0
    m = sample_ns.size
    value = np.full(m, np.nan)
    gap = np.full(m, float(spec.window_s))
    coverage = np.minimum(1.0, n * signal.interval_s / spec.window_s)
    has = np.flatnonzero(n > 0)
    s, e = i0[has], i1[has]
    how = spec.aggregation
    if has.size:
        if how is Aggregation.MEAN:
            value[has] = _range_reduce(np.add, xv, s, e) / n[has]
        elif how is Aggregation.MIN:
            value[has] = _range_reduce(np.minimum, xv, s, e)
        elif how is Aggregation.MAX:
            value[has] = _range_reduce(np.maximum, xv, s, e)
        elif how is Aggregation.LAST:
            value[has] = xv[e - 1]
        else:
            value[has] = _grouped_aggregate(xv, s, e - s, how)
        edge = np.maximum(tv[s] - lo[has], hi[has] - tv[e - 1])
        inner = np.zeros(has.size, dtype=np.int64)
        multi = (e - s) > 1
        if multi.any():
            inner[multi] = _range_reduce(np.maximum, np.diff(tv), s[multi], e[multi] - 1)
        gap[has] = np.maximum(edge, inner) / NS
    status = np.full(m, OK, dtype=object)
    too_few = (spec.aggregation is Aggregation.STD) & (n < 2)
    status[too_few] = TOO_FEW
    if spec.max_gap_s is not None:
        status[gap > spec.max_gap_s] = GAP
    status[coverage < spec.min_coverage] = LOW_COVERAGE
    status[n == 0] = NO_DATA
    if regime_ns is not None and regime_ns.size:
        r0 = np.searchsorted(regime_ns, lo, side="right")
        r1 = np.searchsorted(regime_ns, hi, side="right")
        status[(r1 > r0) & (status == OK)] = REGIME
    value[status != OK] = np.nan
    return WindowResult(value, coverage, gap, n, status)


# --------------------------------------------------------------------------- the feature matrix


@dataclass
class TargetSamples:
    """The lab target at its sample times, with result times and ids when known."""

    name: str
    times: pd.DatetimeIndex  # sample (collection) times, UTC, sorted, duplicates allowed
    values: np.ndarray
    excluded: np.ndarray  # bool
    pending: np.ndarray  # bool: flagged, awaiting review (still eligible)
    result_times: pd.DatetimeIndex | None = None
    sample_ids: np.ndarray | None = None


@dataclass
class FeatureMatrix:
    """Deterministic, model-ready view: one row per lab sample (eligible or not)."""

    target: str
    sample_time: pd.DatetimeIndex
    y: np.ndarray
    X: pd.DataFrame  # one column per predictor, NaN where the feature failed its checks
    coverage: pd.DataFrame
    gap_s: pd.DataFrame
    n_values: pd.DataFrame
    status: pd.DataFrame  # per predictor status code (see STATUS_ORDER)
    y_status: np.ndarray  # object: Y_OK / Y_MISSING / Y_EXCLUDED
    y_pending: np.ndarray  # bool
    eligible: np.ndarray  # bool
    specs: dict[str, FeatureSpec]
    result_time: pd.DatetimeIndex | None = None
    sample_id: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)

    @property
    def features(self) -> list[str]:
        return list(self.X.columns)

    @property
    def n_eligible(self) -> int:
        return int(self.eligible.sum())

    @property
    def non_causal(self) -> list[str]:
        return [name for name, spec in self.specs.items() if not spec.is_causal]

    def max_history_s(self) -> float:
        """Longest stretch of process data any one feature row depends on (for purge gaps)."""
        spans = [data_span_s(s) for s in self.specs.values()]
        return max((end - start for start, end in spans), default=0.0)

    def reasons(self) -> list[str]:
        """Why each row is not eligible ('' when it is)."""
        out = []
        cols = self.features
        status = self.status.to_numpy()
        for i in range(len(self.y)):
            if self.y_status[i] != Y_OK:
                out.append(self.y_status[i])
                continue
            bad = [f"{cols[j]}: {status[i, j]}" for j in range(len(cols)) if status[i, j] != OK]
            out.append("; ".join(bad))
        return out

    def eligible_data(self) -> tuple[pd.DatetimeIndex, pd.DataFrame, np.ndarray]:
        rows = self.eligible
        return self.sample_time[rows], self.X.loc[rows].reset_index(drop=True), self.y[rows]

    def table(self) -> pd.DataFrame:
        """Per-sample table for display/export: time, ids, Y, features, eligibility, reason."""
        out = pd.DataFrame({"sample_time": self.sample_time})
        if self.sample_id is not None:
            out["sample_id"] = self.sample_id
        if self.result_time is not None:
            out["result_time"] = self.result_time
        out[self.target] = self.y
        for c in self.features:
            out[c] = self.X[c].to_numpy()
        out["eligible"] = self.eligible
        out["reason"] = self.reasons()
        return out

    def coverage_report(self) -> pd.DataFrame:
        """Per predictor: how many samples with a usable target each check removed."""
        usable_y = self.y_status == Y_OK
        rows = []
        for name in self.features:
            st = self.status[name].to_numpy()[usable_y]
            cov = self.coverage[name].to_numpy()[usable_y]
            spec = self.specs[name]
            row = {"variable": name, "samples": int(usable_y.sum()), "ok": int((st == OK).sum())}
            for code in STATUS_ORDER[1:]:
                row[code] = int((st == code).sum())
            row["mean_coverage_pct"] = 100.0 * float(np.mean(cov)) if cov.size else np.nan
            row["causal"] = spec.is_causal
            rows.append(row)
        return pd.DataFrame(rows).set_index("variable") if rows else pd.DataFrame()

    def summary(self) -> dict[str, int]:
        return {
            "lab_samples": len(self.y),
            "target_missing": int((self.y_status == Y_MISSING).sum()),
            "target_excluded": int((self.y_status == Y_EXCLUDED).sum()),
            "target_pending_review": int((self.y_pending & (self.y_status == Y_OK)).sum()),
            "feature_failures": int(((self.y_status == Y_OK) & ~self.eligible).sum()),
            "eligible": self.n_eligible,
        }


def build_feature_matrix(
    target: TargetSamples,
    process: pd.DataFrame,
    specs: list[FeatureSpec],
    *,
    excluded: pd.DataFrame | None = None,
    regime: np.ndarray | None = None,
    alignment: AlignmentConfig | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> FeatureMatrix:
    """Align every predictor to the target's sample times.

    ``process``: process values on their own timestamps (not cleaned). ``excluded``: bool mask
    of the same shape (final exclusions). ``regime``: bool per process row, rows excluded for
    every variable (shutdowns etc.).
    """
    alignment = alignment or AlignmentConfig()
    if not specs:
        raise AlignmentError("Choose at least one predictor before aligning features.")
    missing = [s.variable for s in specs if s.variable not in process.columns]
    if missing:
        raise AlignmentError(
            f"No process data loaded for: {', '.join(missing)}. Import it or deselect it."
        )
    order = np.argsort(ns(target.times), kind="stable")
    times = target.times[order]
    sample_ns = ns(times)
    y = np.asarray(target.values, dtype=float)[order]
    y_status = np.full(y.size, Y_OK, dtype=object)
    y_status[np.isnan(y)] = Y_MISSING
    y_status[np.asarray(target.excluded, dtype=bool)[order] & (y_status == Y_OK)] = Y_EXCLUDED
    pending = np.asarray(target.pending, dtype=bool)[order]

    regime_ns = None
    if alignment.reject_regime_overlap and regime is not None and np.any(regime):
        regime_ns = ns(process.index)[np.asarray(regime, dtype=bool)]

    cols: dict[str, dict[str, np.ndarray]] = {
        k: {} for k in ("value", "coverage", "gap", "n", "status")
    }
    for i, spec in enumerate(specs):
        check(cancel)
        report(progress, i / len(specs), f"Aligning {spec.variable}")
        mask = excluded[spec.variable].to_numpy() if excluded is not None else None
        signal = process_signal(process[spec.variable], spec, mask)
        res = aggregate_windows(signal, sample_ns, spec, regime_ns)
        cols["value"][spec.variable] = res.value
        cols["coverage"][spec.variable] = res.coverage
        cols["gap"][spec.variable] = res.gap_s
        cols["n"][spec.variable] = res.n_values
        cols["status"][spec.variable] = res.status
    report(progress, 1.0, "Done")
    X = pd.DataFrame(cols["value"])
    status = pd.DataFrame(cols["status"])
    eligible = (y_status == Y_OK) & (status == OK).all(axis=1).to_numpy()
    notes = []
    non_causal = [s.variable for s in specs if not s.is_causal]
    if non_causal:
        notes.append(
            "Non-causal (retrospective) features read data recorded after the sample time: "
            + ", ".join(non_causal)
            + ". Such a model cannot run online as configured."
        )
    return FeatureMatrix(
        target=target.name,
        sample_time=times,
        y=y,
        X=X,
        coverage=pd.DataFrame(cols["coverage"]),
        gap_s=pd.DataFrame(cols["gap"]),
        n_values=pd.DataFrame(cols["n"]),
        status=status,
        y_status=y_status,
        y_pending=pending,
        eligible=eligible,
        specs={s.variable: s for s in specs},
        result_time=target.result_times[order] if target.result_times is not None else None,
        sample_id=target.sample_ids[order] if target.sample_ids is not None else None,
        notes=notes,
        alignment=alignment,
    )


# --------------------------------------------------------------------------- project helpers


def target_samples(project: Project, data, cleaning=None) -> TargetSamples:  # noqa: ANN001
    """The project's target from loaded data (``data``: ProjectData; ``cleaning``: CleaningState)."""
    meta = project.target
    if meta is None:
        raise AlignmentError("Choose a target variable on the Variable Selection page.")
    if meta.source_id not in data.sources:
        raise AlignmentError(f"The target's source '{meta.source_id}' is not loaded.")
    src = data.sources[meta.source_id]
    norm = src.normalized
    if meta.source_column not in norm.values.columns:
        raise AlignmentError(f"Column '{meta.source_column}' is missing from the target source.")
    values = norm.values[meta.source_column]
    excluded = np.zeros(len(values), dtype=bool)
    pending = np.zeros(len(values), dtype=bool)
    if cleaning is not None and cleaning.target is not None:
        tm = cleaning.target
        if len(tm.excluded) == len(values):
            excluded = tm.excluded.to_numpy(dtype=bool)
            pending = tm.pending.to_numpy(dtype=bool)
    sample_ids = norm.sample_id.astype(str).to_numpy() if norm.sample_id is not None else None
    result = pd.DatetimeIndex(norm.result_time) if norm.result_time is not None else None
    return TargetSamples(
        name=meta.name,
        times=pd.DatetimeIndex(values.index),
        values=values.to_numpy(dtype=float),
        excluded=excluded,
        pending=pending,
        result_times=result,
        sample_ids=sample_ids,
    )


def feature_specs(project: Project) -> list[FeatureSpec]:
    """A spec for every predictor, in predictor order; defaults where none is configured."""
    configured = {f.variable: f for f in project.features}
    return [configured.get(v.name, FeatureSpec(v.name)) for v in project.predictors]


def build_project_features(
    project: Project,
    data,  # noqa: ANN001 - ProjectData
    process: pd.DataFrame,
    cleaning=None,  # noqa: ANN001 - CleaningState | None
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> FeatureMatrix:
    """Feature matrix for the project's current target, predictors and cleaning state."""
    target = target_samples(project, data, cleaning)
    excluded = cleaning.excluded if cleaning is not None else None
    regime = cleaning.regime if cleaning is not None else None
    return build_feature_matrix(
        target,
        process,
        feature_specs(project),
        excluded=excluded,
        regime=regime,
        alignment=project.alignment,
        progress=progress,
        cancel=cancel,
    )


# --------------------------------------------------------------------------- lag sensitivity


@dataclass
class LagScan:
    variable: str
    delays_s: np.ndarray
    pearson: np.ndarray
    n_pairs: np.ndarray
    cutoff: pd.Timestamp | None  # only samples at or before this time were used
    note: str = (
        "Correlation of the aligned feature with the target for each tested delay, using only "
        "samples in the training period. Exploratory: autocorrelation and common trends also "
        "produce correlation, so prefer a delay that is physically plausible."
    )

    @property
    def best_delay_s(self) -> float | None:
        r = np.abs(self.pearson)
        if np.all(np.isnan(r)):
            return None
        return float(self.delays_s[int(np.nanargmax(r))])


def lag_scan(
    target: TargetSamples,
    raw: pd.Series,
    spec: FeatureSpec,
    delays_s: np.ndarray,
    *,
    excluded: np.ndarray | None = None,
    cutoff: pd.Timestamp | None = None,
    min_pairs: int = 10,
    cancel: CancelToken | None = None,
) -> LagScan:
    """Correlation versus effective delay (deadtime + shift) for one predictor.

    Only samples with valid, non-excluded Y at or before ``cutoff`` (the training period) are
    used, so the held-out test period never influences the choice of delay.
    """
    signal = process_signal(raw, spec, excluded)
    times = pd.DatetimeIndex(target.times)
    y = np.asarray(target.values, dtype=float)
    use = ~np.isnan(y) & ~np.asarray(target.excluded, dtype=bool)
    if cutoff is not None:
        use &= np.asarray(times <= cutoff)
    sample_ns, y = ns(times)[use], y[use]
    delays = np.asarray(delays_s, dtype=float)
    r = np.full(delays.size, np.nan)
    counts = np.zeros(delays.size, dtype=int)
    for i, d in enumerate(delays):
        check(cancel)
        trial = replace(spec, deadtime_s=max(float(d), 0.0), shift_s=min(float(d), 0.0))
        res = aggregate_windows(signal, sample_ns, trial)
        ok = ~np.isnan(res.value)
        counts[i] = int(ok.sum())
        if counts[i] >= min_pairs and np.ptp(res.value[ok]) > 0 and np.ptp(y[ok]) > 0:
            r[i] = float(np.corrcoef(res.value[ok], y[ok])[0, 1])
    return LagScan(str(raw.name), delays, r, counts, cutoff)
