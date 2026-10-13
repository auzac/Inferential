"""Automatic data-quality detectors. They *flag* values for review; they never delete data.

Every detector works on one variable's own timestamps (missing values are skipped, never
filled) and returns a boolean flag array plus contiguous **events** that the user reviews.

Definitions (plain-language versions are shown in the UI):

* ``zscore``: |x − mean| > k·std.
* ``mad`` (robust z-score): |x − median| > k·1.4826·MAD, where MAD = median(|x − median|).
  Insensitive to the outliers it is looking for. Constant data (MAD = 0) flags nothing.
* ``iqr``: x < Q1 − k·IQR or x > Q3 + k·IQR.
  For these three the statistics can be fitted on a chosen period only (``fit_start``/``fit_end``)
  so that, later, held-out test data does not influence the thresholds.
* ``hampel`` (Hampel identifier, Pearson 2002): for each value, the centred time window
  [t − W/2, t + W/2] gives a local median m and MAD = median(|xⱼ − m|) over the window; flag
  when |x − m| > max(k·1.4826·MAD, min_deviation). At least three values are needed. Centring
  uses values on both sides; that is acceptable for reviewing history and the detector is
  never part of a model's feature pipeline.
* ``rate_of_change``: |Δx / Δt| between consecutive valid samples above a limit (units per
  minute). Steps across a gap longer than 2× the typical interval are not compared. Both edges
  of a spike are flagged; review decides.
* ``flatline``: values that stay within a band of width ``tolerance`` (max − min of the run)
  for at least the given duration (stuck transmitter). A gap ends the plateau. A slow drift
  made of small steps is *not* a plateau.
* ``limits``: outside physical/engineering limits (detector values, else the variable's limits).
* ``invalid_text``: cells that held status text such as "Bad" in the source file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from inferential_studio.core.rules import fill_runs, median_interval_ns, ns, split_runs_at_gaps
from inferential_studio.core.schemas import DetectorConfig, DetectorKind, VariableMeta
from inferential_studio.core.tasks import CancelToken, ProgressCallback, check, report

MAD_SCALE = 1.4826  # makes the MAD a consistent estimator of σ for normal data


@dataclass
class FlagEvent:
    detector_id: str
    variable: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_points: int
    score: float  # detector-specific size of the anomaly (σ, units/min, minutes, excess …)

    @property
    def key(self) -> tuple[str, str, pd.Timestamp, pd.Timestamp]:
        return (self.detector_id, self.variable, self.start, self.end)


@dataclass
class DetectorResult:
    detector: DetectorConfig
    variable: str
    index: pd.DatetimeIndex  # timestamps the flags refer to
    flags: np.ndarray  # bool, aligned with index
    events: list[FlagEvent] = field(default_factory=list)
    note: str = ""


class DetectorError(ValueError):
    """A detector cannot run as configured. The message is shown to the user."""


# --------------------------------------------------------------------------- single detectors


def _fit_values(series: pd.Series, det: DetectorConfig) -> np.ndarray:
    s = series
    if det.fit_start is not None:
        s = s[s.index >= pd.Timestamp(det.fit_start)]
    if det.fit_end is not None:
        s = s[s.index <= pd.Timestamp(det.fit_end)]
    return s.dropna().to_numpy(dtype=float)


def detect(
    det: DetectorConfig,
    series: pd.Series,
    *,
    text_mask: pd.Series | None = None,
    meta: VariableMeta | None = None,
) -> DetectorResult:
    """Run one detector on one variable. ``series`` has a UTC index and may contain NaN."""
    x = series.to_numpy(dtype=float)
    valid = ~np.isnan(x)
    flags = np.zeros(len(x), dtype=bool)
    score = np.zeros(len(x))
    note = ""
    k = det.kind
    name = str(series.name)

    if k in (DetectorKind.ZSCORE, DetectorKind.MAD, DetectorKind.IQR):
        ref = _fit_values(series, det)
        if ref.size < 3:
            return DetectorResult(
                det, name, series.index, flags, [], "Too few values to fit the statistics."
            )
        if k is DetectorKind.ZSCORE:
            centre, scale = ref.mean(), ref.std(ddof=1)
        elif k is DetectorKind.MAD:
            centre = np.median(ref)
            scale = MAD_SCALE * np.median(np.abs(ref - centre))
        else:
            q1, q3 = np.percentile(ref, [25, 75])
            iqr = q3 - q1
            if iqr == 0:
                return DetectorResult(
                    det,
                    name,
                    series.index,
                    flags,
                    [],
                    "IQR is zero (mostly constant data); nothing flagged.",
                )
            lo, hi = q1 - det.threshold * iqr, q3 + det.threshold * iqr
            with np.errstate(invalid="ignore"):
                flags = valid & ((x < lo) | (x > hi))
            score = np.where(x < lo, (lo - x) / iqr, np.where(x > hi, (x - hi) / iqr, 0.0))
        if k is not DetectorKind.IQR:
            if not scale or not np.isfinite(scale):
                return DetectorResult(
                    det,
                    name,
                    series.index,
                    flags,
                    [],
                    "Spread is zero (constant data); nothing flagged.",
                )
            z = np.abs(x - centre) / scale
            flags = valid & (z > det.threshold)
            score = np.nan_to_num(z)

    elif k is DetectorKind.HAMPEL:
        pos = np.flatnonzero(valid)
        med, mad = _window_median_mad(ns(series.index)[pos], x[pos], float(det.window_s))
        dev = np.abs(x[pos] - med)
        threshold = np.maximum(det.threshold * MAD_SCALE * mad, det.min_deviation)
        with np.errstate(invalid="ignore", divide="ignore"):
            hit = (dev > threshold) & (dev > 0)  # NaN (too few neighbours) compares False
            z = np.where(mad > 0, dev / (MAD_SCALE * mad), 0.0)
        flags[pos] = hit
        score[pos] = np.nan_to_num(z)

    elif k is DetectorKind.RATE_OF_CHANGE:
        vals = series.dropna()
        if len(vals) > 1:
            t = ns(vals.index)
            dt_min = np.diff(t) / 60e9
            step = median_interval_ns(vals.index)
            ok_step = np.diff(t) <= 2 * step if step else np.ones(len(t) - 1, dtype=bool)
            rate = np.abs(np.diff(vals.to_numpy())) / np.where(dt_min > 0, dt_min, np.nan)
            hit = ok_step & (rate > det.rate_per_min)
            pos = np.flatnonzero(valid)[1:][hit]  # flag the sample arriving after the jump
            flags[pos] = True
            score[pos] = rate[hit]

    elif k is DetectorKind.FLATLINE:
        vals = series.dropna()
        if len(vals) > 1:
            t = ns(vals.index)
            v = vals.to_numpy()
            step = median_interval_ns(vals.index)
            plateau = _plateaus(t, v, det.tolerance, 2 * step if step else 0)
            first = np.full(plateau[-1] + 1, np.iinfo(np.int64).max)
            last = np.zeros(plateau[-1] + 1, dtype=np.int64)
            np.minimum.at(first, plateau, t)
            np.maximum.at(last, plateau, t)
            duration_ns = last - first + step
            long = duration_ns >= det.window_s * 1e9
            hit = long[plateau]
            pos = np.flatnonzero(valid)[hit]
            flags[pos] = True
            score[pos] = (duration_ns / 60e9)[plateau][hit]

    elif k is DetectorKind.LIMITS:
        lo = det.low if det.low is not None else (meta.eng_low if meta else None)
        hi = det.high if det.high is not None else (meta.eng_high if meta else None)
        if lo is None and hi is None:
            raise DetectorError(
                f"No limits for '{name}': enter them in the detector or set engineering limits for the variable."
            )
        with np.errstate(invalid="ignore"):
            below = valid & (x < lo) if lo is not None else np.zeros(len(x), dtype=bool)
            above = valid & (x > hi) if hi is not None else np.zeros(len(x), dtype=bool)
        flags = below | above
        score = np.where(below, (lo or 0) - x, np.where(above, x - (hi or 0), 0.0))

    elif k is DetectorKind.INVALID_TEXT:
        flags = (
            np.zeros(len(x), dtype=bool) if text_mask is None else text_mask.to_numpy(dtype=bool)
        )
        score = flags.astype(float)

    events = _events(det, name, series.index, flags, score)
    return DetectorResult(det, name, series.index, flags, events, note)


_CHUNK_CELLS = 4_000_000  # bounds the temporary window matrix (~32 MB)


def _window_median_mad(
    t: np.ndarray, v: np.ndarray, window_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Median and MAD of every centred window ``[tᵢ − W/2, tᵢ + W/2]`` (valid points only).

    The MAD is taken about *that window's* median (Hampel's definition), not about each
    neighbour's own local median. Windows with fewer than three values give NaN. Windows with
    the same number of values are evaluated together as one matrix (regular sampling gives
    only a few distinct counts).
    """
    n = v.size
    med = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    if n == 0:
        return med, mad
    half = int(round(window_s * 1e9 / 2))
    lo = np.searchsorted(t, t - half, side="left")
    count = np.searchsorted(t, t + half, side="right") - lo
    for c in np.unique(count):
        if c < 3:
            continue
        rows = np.flatnonzero(count == c)
        step = max(1, _CHUNK_CELLS // int(c))
        for s in range(0, rows.size, step):
            r = rows[s : s + step]
            block = v[lo[r, None] + np.arange(c)]
            m = np.median(block, axis=1)
            med[r] = m
            mad[r] = np.median(np.abs(block - m[:, None]), axis=1)
    return med, mad


def _plateaus(t: np.ndarray, v: np.ndarray, tolerance: float, max_gap_ns: int) -> np.ndarray:
    """Plateau id per value: a plateau is a run whose values all lie within a band of width
    ``tolerance`` (max − min ≤ tolerance). A time gap longer than ``max_gap_ns`` ends it.

    Comparing with the whole run (not only the previous value) keeps a slow, steady drift made
    of small steps from counting as a stuck value.
    """
    n = v.size
    gap = np.diff(t) > max_gap_ns if max_gap_ns else np.zeros(max(n - 1, 0), dtype=bool)
    if tolerance <= 0:  # exact repeats: a vectorised special case of the same rule
        moved = (np.diff(v) != 0) | gap
        return np.concatenate([[0], np.cumsum(moved)])
    ids = np.zeros(n, dtype=np.int64)
    pid, lo, hi = 0, v[0], v[0]
    for i in range(1, n):
        lo_i, hi_i = min(lo, v[i]), max(hi, v[i])
        if gap[i - 1] or hi_i - lo_i > tolerance:
            pid += 1
            lo = hi = v[i]
        else:
            lo, hi = lo_i, hi_i
        ids[i] = pid
    return ids


def _events(
    det: DetectorConfig,
    variable: str,
    index: pd.DatetimeIndex,
    flags: np.ndarray,
    score: np.ndarray,
) -> list[FlagEvent]:
    step = median_interval_ns(index)
    starts, ends = split_runs_at_gaps(flags, index, 2 * step)
    out = []
    for s, e in zip(starts, ends, strict=True):
        out.append(
            FlagEvent(
                det.detector_id,
                variable,
                index[s],
                index[e],
                int(flags[s : e + 1].sum()),
                float(np.max(score[s : e + 1])) if e >= s else 0.0,
            )
        )
    return out


def event_mask(result: DetectorResult, events: list[FlagEvent]) -> np.ndarray:
    """Rows of ``result.index`` covered by the given events (flagged rows only)."""
    if not events:
        return np.zeros(len(result.index), dtype=bool)
    i8 = ns(result.index)
    starts = np.searchsorted(i8, np.array([e.start.value for e in events]), side="left")
    ends = np.searchsorted(i8, np.array([e.end.value for e in events]), side="right") - 1
    return fill_runs(len(i8), starts, ends) & result.flags


# --------------------------------------------------------------------------- all detectors


def run_detectors(
    detectors: list[DetectorConfig],
    series_of: dict[str, pd.Series],
    text_masks: dict[str, pd.Series],
    metas: dict[str, VariableMeta],
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> dict[tuple[str, str], DetectorResult]:
    """Run every enabled detector on each of its variables that has data."""
    jobs = [(d, v) for d in detectors if d.enabled for v in d.variables if v in series_of]
    out: dict[tuple[str, str], DetectorResult] = {}
    for i, (det, var) in enumerate(jobs):
        check(cancel)
        report(progress, i / max(len(jobs), 1), f"{det.kind.value} on {var}")
        out[(det.detector_id, var)] = detect(
            det, series_of[var], text_mask=text_masks.get(var), meta=metas.get(var)
        )
    report(progress, 1.0, "Done")
    return out
