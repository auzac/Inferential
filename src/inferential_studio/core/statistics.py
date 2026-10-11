"""Descriptive statistics and correlations for exploratory data analysis.

Definitions (also shown in the UI):

* Percentages are of all rows with a valid timestamp in the normalized layer.
* ``invalid_pct``: cells that held non-numeric text (e.g. "Bad"); ``missing_pct``: empty cells.
* Standard deviation and variance use ``ddof=1``. Skewness and excess kurtosis are the
  bias-corrected sample estimators (scipy, ``bias=False``).
* Coefficient of variation = std / |mean|, reported **only** when every valid value has the same
  sign (all > 0 or all < 0). Otherwise it is meaningless (e.g. temperatures around 0 °C) and is
  left empty.
* Correlations use pairwise-complete observations and require at least ``min_periods`` pairs.
* Preliminary X–Y correlation pairs each lab sample with the **last process value at or before**
  the sample time (no deadtime, no averaging, never a later value). It is a screening aid only;
  model-ready correlations come after feature alignment (Milestone 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from inferential_studio.core.tasks import CancelToken, ProgressCallback, check, report

SUMMARY_COLUMNS = [
    "rows",
    "valid",
    "valid_pct",
    "missing_pct",
    "invalid_pct",
    "min",
    "q1",
    "median",
    "q3",
    "max",
    "mean",
    "std",
    "variance",
    "cv",
    "skewness",
    "kurtosis",
]


def summary_statistics(
    frame: pd.DataFrame,
    text_mask: pd.DataFrame | None = None,
    *,
    cancel: CancelToken | None = None,
) -> pd.DataFrame:
    """One row per column of ``frame`` with the statistics listed in SUMMARY_COLUMNS."""
    rows = []
    n = len(frame)
    for col in frame.columns:
        check(cancel)
        x = frame[col].to_numpy(dtype=float)
        valid = x[~np.isnan(x)]
        n_text = int(text_mask[col].sum()) if text_mask is not None and col in text_mask else 0
        n_valid = valid.size
        row = {
            "rows": n,
            "valid": n_valid,
            "valid_pct": 100.0 * n_valid / n if n else np.nan,
            "missing_pct": 100.0 * (n - n_valid - n_text) / n if n else np.nan,
            "invalid_pct": 100.0 * n_text / n if n else np.nan,
        }
        if n_valid:
            q1, med, q3 = np.percentile(valid, [25, 50, 75])
            mean = float(valid.mean())
            std = float(valid.std(ddof=1)) if n_valid > 1 else np.nan
            same_sign = valid.min() > 0 or valid.max() < 0
            constant = n_valid > 1 and np.ptp(valid) == 0
            skew, kurt = _sample_skew_kurtosis(valid, mean)
            row.update(
                min=float(valid.min()),
                q1=q1,
                median=med,
                q3=q3,
                max=float(valid.max()),
                mean=mean,
                std=std,
                variance=std**2 if n_valid > 1 else np.nan,
                cv=std / abs(mean) if same_sign and n_valid > 1 else np.nan,
                skewness=skew if not constant else np.nan,
                kurtosis=kurt if not constant else np.nan,
            )
        rows.append(row)
    return pd.DataFrame(
        rows, index=pd.Index(frame.columns, name="variable"), columns=SUMMARY_COLUMNS
    )


def _sample_skew_kurtosis(x: np.ndarray, mean: float) -> tuple[float, float]:
    """Bias-corrected sample skewness (G1) and excess kurtosis (G2), as scipy with ``bias=False``.

    Computed directly from central moments: about 10x faster than scipy on long series.
    """
    n = x.size
    d = x - mean
    d2 = d * d
    m2 = d2.mean()
    if m2 == 0:
        return np.nan, np.nan
    m3 = (d2 * d).mean()
    m4 = (d2 * d2).mean()
    g1 = m3 / m2**1.5
    g2 = m4 / m2**2 - 3.0
    skew = np.sqrt(n * (n - 1)) / (n - 2) * g1 if n > 2 else np.nan
    kurt = ((n + 1) * g2 + 6.0) * (n - 1) / ((n - 2) * (n - 3)) if n > 3 else np.nan
    return float(skew), float(kurt)


def correlation_matrix(
    frame: pd.DataFrame, method: str = "pearson", *, min_periods: int = 30
) -> pd.DataFrame:
    """Pairwise-complete Pearson or Spearman correlation matrix."""
    if method not in ("pearson", "spearman"):
        raise ValueError("Correlation method must be 'pearson' or 'spearman'.")
    if method == "pearson":
        return frame.corr(method="pearson", min_periods=min_periods)
    return _spearman(frame, min_periods)


def _spearman(frame: pd.DataFrame, min_periods: int) -> pd.DataFrame:
    """Exact pairwise-complete Spearman correlation (same result as pandas, much faster).

    Columns are grouped by their missing-value pattern. Within a group every pair shares the
    same complete rows, so the columns are ranked once and Pearson on the ranks is Spearman.
    For two different groups the common rows are found once and those columns re-ranked on them.
    """
    values = frame.to_numpy(dtype=float)
    valid = ~np.isnan(values)
    k = values.shape[1]
    groups: dict[bytes, list[int]] = {}
    for c in range(k):
        groups.setdefault(np.packbits(valid[:, c]).tobytes(), []).append(c)
    members = list(groups.values())
    out = np.full((k, k), np.nan)

    def fill(rows: np.ndarray, cols: list[int]) -> None:
        if rows.sum() < min_periods:
            return
        ranked = stats.rankdata(values[rows][:, cols], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            block = np.corrcoef(ranked, rowvar=False) if len(cols) > 1 else np.ones((1, 1))
        constant = np.ptp(ranked, axis=0) == 0
        block[constant, :] = np.nan
        block[:, constant] = np.nan
        for a, ca in enumerate(cols):
            for b, cb in enumerate(cols):
                if np.isnan(out[ca, cb]) or ca == cb:
                    out[ca, cb] = block[a, b]

    for g in members:
        fill(valid[:, g[0]], g)
    for gi in range(len(members)):
        for gj in range(gi + 1, len(members)):
            a, b = members[gi], members[gj]
            common = valid[:, a[0]] & valid[:, b[0]]
            if common.sum() < min_periods:
                continue
            ranked = stats.rankdata(values[common][:, a + b], axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                block = np.corrcoef(ranked, rowvar=False)
            constant = np.ptp(ranked, axis=0) == 0
            for x, ca in enumerate(a):
                for y, cb in enumerate(b):
                    r = np.nan if constant[x] or constant[len(a) + y] else block[x, len(a) + y]
                    out[ca, cb] = out[cb, ca] = r
    for c in range(k):
        if valid[:, c].sum() >= min_periods and not np.isnan(out[c, c]):
            out[c, c] = 1.0
    return pd.DataFrame(out, index=frame.columns, columns=frame.columns)


def pair_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Number of rows where both columns are valid, for every pair."""
    valid = frame.notna().to_numpy(dtype=np.int32)
    counts = valid.T @ valid
    return pd.DataFrame(counts, index=frame.columns, columns=frame.columns)


def hierarchical_order(corr: pd.DataFrame) -> list[str]:
    """Order variables so strongly correlated ones sit together (average linkage on 1 − |r|)."""
    names = list(corr.columns)
    if len(names) < 3:
        return names
    dist = 1.0 - corr.abs().to_numpy(copy=True)
    dist = np.where(np.isnan(dist), 1.0, dist)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    tree = optimal_leaf_ordering(linkage(condensed, method="average"), condensed)
    return [names[i] for i in leaves_list(tree)]


@dataclass
class XYScreening:
    table: pd.DataFrame  # index = predictor; columns pearson, spearman, n_pairs
    tolerance: pd.Timedelta
    n_target_samples: int
    note: str = (
        "Preliminary: each lab sample is paired with the last process value at or before its "
        "sample time (no deadtime or averaging). Use for screening only."
    )


def preliminary_xy_correlation(
    process: pd.DataFrame,
    target: pd.Series,
    *,
    tolerance: pd.Timedelta | None = None,
    min_pairs: int = 10,
) -> XYScreening:
    """Screen X–Y association at lab sample times using only past-or-present process values."""
    target = target.dropna()
    if tolerance is None:
        spacing = process.index.to_series().diff().median() if len(process) > 1 else pd.Timedelta(0)
        tolerance = (
            max(2 * spacing, pd.Timedelta(minutes=1))
            if pd.notna(spacing)
            else pd.Timedelta(minutes=1)
        )
    left = pd.DataFrame({"time": target.index, "__y__": target.to_numpy()}).sort_values("time")
    right = process.sort_index().reset_index(names="time")
    left["time"] = left["time"].astype(right["time"].dtype)
    merged = pd.merge_asof(
        left, right, on="time", direction="backward", tolerance=tolerance, allow_exact_matches=True
    )
    out = []
    for col in process.columns:
        pair = merged[["__y__", col]].dropna()
        n_pairs = len(pair)
        if n_pairs >= min_pairs and pair[col].nunique() > 1 and pair["__y__"].nunique() > 1:
            pearson = float(pair["__y__"].corr(pair[col]))
            spearman = float(pair["__y__"].corr(pair[col], method="spearman"))
        else:
            pearson = spearman = np.nan
        out.append({"pearson": pearson, "spearman": spearman, "n_pairs": n_pairs})
    table = pd.DataFrame(out, index=pd.Index(process.columns, name="variable"))
    table = table.reindex(
        table["pearson"].abs().sort_values(ascending=False, na_position="last").index
    )
    return XYScreening(table=table, tolerance=tolerance, n_target_samples=len(target))


def histogram(values: pd.Series, max_bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Counts and bin edges using numpy's 'auto' rule, capped at ``max_bins``."""
    x = values.dropna().to_numpy(dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    if np.ptp(x) == 0:
        return np.array([x.size]), np.array([x[0] - 0.5, x[0] + 0.5])
    edges = np.histogram_bin_edges(x, bins="auto")
    if len(edges) - 1 > max_bins:
        edges = np.histogram_bin_edges(x, bins=max_bins)
    counts, edges = np.histogram(x, bins=edges)
    return counts, edges


@dataclass
class Coverage:
    bin_starts: pd.DatetimeIndex  # UTC start of each time bin
    bin_width: pd.Timedelta
    rows: np.ndarray  # rows (timestamps) present per bin — timestamp density
    valid_fraction: pd.DataFrame  # per bin (rows) x variable: share of rows with a valid value


def coverage(frame: pd.DataFrame, n_bins: int = 120) -> Coverage:
    """Data availability over time, for the missing-data view."""
    if frame.empty:
        return Coverage(
            pd.DatetimeIndex([], tz="UTC"), pd.Timedelta(0), np.zeros(0), pd.DataFrame()
        )
    start, end = frame.index.min(), frame.index.max()
    width = max((end - start) / n_bins, pd.Timedelta(seconds=1))
    width = pd.Timedelta(seconds=np.ceil(width.total_seconds()))
    bins = ((frame.index - start) // width).to_numpy()
    nb = int(bins.max()) + 1
    rows = np.bincount(bins, minlength=nb)
    valid = frame.notna().groupby(bins).sum().reindex(range(nb), fill_value=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = valid.to_numpy() / rows[:, None]
    frac = np.where(rows[:, None] > 0, frac, np.nan)
    starts = pd.DatetimeIndex([start + i * width for i in range(nb)])
    return Coverage(starts, width, rows, pd.DataFrame(frac, index=starts, columns=frame.columns))


@dataclass
class EdaResult:
    summary: pd.DataFrame
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    pairs: pd.DataFrame
    coverage: Coverage
    xy: XYScreening | None


def compute_eda(
    process: pd.DataFrame,
    text_mask: pd.DataFrame | None = None,
    target: pd.Series | None = None,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> EdaResult:
    """Everything the initial-EDA page shows, computed in one background call."""
    report(progress, 0.05, "Summary statistics")
    summary = summary_statistics(process, text_mask, cancel=cancel)
    check(cancel)
    report(progress, 0.3, "Pearson correlation")
    pearson = correlation_matrix(process, "pearson")
    check(cancel)
    report(progress, 0.5, "Spearman correlation")
    spearman = correlation_matrix(process, "spearman")
    check(cancel)
    report(progress, 0.8, "Coverage")
    pairs = pair_counts(process)
    cov = coverage(process)
    xy = None
    if target is not None and target.notna().any():
        report(progress, 0.9, "Preliminary X–Y screening")
        xy = preliminary_xy_correlation(process, target)
    report(progress, 1.0, "Done")
    return EdaResult(summary, pearson, spearman, pairs, cov, xy)
