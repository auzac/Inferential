"""Variable diagnostics: raw-vs-cleaned comparison, aligned correlations, VIF.

Definitions:

* **VIF** of predictor j = 1 / (1 − R²ⱼ), where R²ⱼ is from an ordinary least-squares regression
  (with intercept) of xⱼ on all other predictors, using complete rows only. A constant
  predictor has no VIF (shown empty); an exact linear combination has VIF = ∞. VIF depends on
  the whole predictor set and on the rows used; it diagnoses multicollinearity, it does not
  say which variable to drop.
* **Aligned correlations** use model-eligible rows of the feature matrix (one per lab sample),
  not the raw process grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from inferential_studio.core.statistics import summary_statistics

VIF_WARNING = 10.0
CORRELATION_WARNING = 0.9


def vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF per column (complete rows). Columns: vif, r2, note."""
    data = X.dropna()
    names = list(X.columns)
    n, k = data.shape
    out = pd.DataFrame(index=pd.Index(names, name="variable"), columns=["vif", "r2", "note"])
    out["vif"] = np.nan
    out["r2"] = np.nan
    out["note"] = ""
    if k == 0:
        return out
    if k == 1:
        out.loc[names[0], ["vif", "r2"]] = [1.0, 0.0]
        out.loc[names[0], "note"] = "single predictor"
        return out
    values = data.to_numpy(dtype=float)
    if n <= k:
        out["note"] = f"needs more than {k} complete rows (has {n})"
        return out
    centred = values - values.mean(axis=0)
    scale = centred.std(axis=0)
    constant = scale == 0
    for j, name in enumerate(names):
        if constant[j]:
            out.loc[name, "note"] = "constant"
            continue
        others = [i for i in range(k) if i != j and not constant[i]]
        y = centred[:, j] / scale[j]
        A = centred[:, others] / scale[others]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        r2 = 1.0 - float(resid @ resid) / float(y @ y)
        r2 = min(max(r2, 0.0), 1.0)
        out.loc[name, "r2"] = r2
        if r2 > 1.0 - 1e-10:
            out.loc[name, "vif"] = np.inf
            out.loc[name, "note"] = "exact linear combination of others"
        else:
            out.loc[name, "vif"] = 1.0 / (1.0 - r2)
            if out.loc[name, "vif"] > VIF_WARNING:
                out.loc[name, "note"] = f"VIF > {VIF_WARNING:g}"
    out["vif"] = out["vif"].astype(float)
    out["r2"] = out["r2"].astype(float)
    return out


def high_correlation_pairs(
    corr: pd.DataFrame, threshold: float = CORRELATION_WARNING
) -> pd.DataFrame:
    """Predictor pairs with |r| above ``threshold`` (a warning, not a deletion rule)."""
    names = list(corr.columns)
    rows = []
    values = corr.to_numpy()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = values[i, j]
            if np.isfinite(r) and abs(r) > threshold:
                rows.append({"variable_a": names[i], "variable_b": names[j], "r": float(r)})
    out = pd.DataFrame(rows, columns=["variable_a", "variable_b", "r"])
    return out.reindex(out["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


@dataclass
class AlignedDiagnostics:
    n_rows: int  # eligible rows used
    xy: pd.DataFrame  # per predictor: pearson, spearman, n
    pearson: pd.DataFrame  # X–X
    spearman: pd.DataFrame
    vif: pd.DataFrame
    high_pairs: pd.DataFrame


def aligned_diagnostics(
    X: pd.DataFrame, y: np.ndarray, *, threshold: float = CORRELATION_WARNING
) -> AlignedDiagnostics:
    """Correlations and VIF on aligned, model-eligible feature rows."""
    y = np.asarray(y, dtype=float)
    rows = []
    for name in X.columns:
        x = X[name].to_numpy(dtype=float)
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() >= 3 and np.ptp(x[ok]) > 0 and np.ptp(y[ok]) > 0:
            p = float(np.corrcoef(x[ok], y[ok])[0, 1])
            s = float(stats.spearmanr(x[ok], y[ok]).statistic)
        else:
            p = s = np.nan
        rows.append({"variable": name, "pearson": p, "spearman": s, "n": int(ok.sum())})
    xy = pd.DataFrame(rows, columns=["variable", "pearson", "spearman", "n"]).set_index("variable")
    xy = xy.reindex(xy["pearson"].abs().sort_values(ascending=False, na_position="last").index)
    min_periods = 3
    pearson = X.corr(method="pearson", min_periods=min_periods)
    spearman = X.corr(method="spearman", min_periods=min_periods)
    return AlignedDiagnostics(
        n_rows=len(X),
        xy=xy,
        pearson=pearson,
        spearman=spearman,
        vif=vif(X),
        high_pairs=high_correlation_pairs(pearson, threshold),
    )


RAW_CLEAN_COLUMNS = ("valid", "mean", "std", "min", "max")


def raw_vs_cleaned(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side summary statistics before and after cleaning (column pairs raw/cleaned)."""
    a = summary_statistics(raw)
    b = summary_statistics(cleaned)
    out = pd.DataFrame(index=a.index)
    for col in RAW_CLEAN_COLUMNS:
        out[f"{col}_raw"] = a[col]
        out[f"{col}_cleaned"] = b[col]
    out["removed_pct"] = np.where(
        a["valid"] > 0, 100.0 * (a["valid"] - b["valid"]) / a["valid"].clip(lower=1), np.nan
    )
    return out
