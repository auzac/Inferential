"""Linear inferential models: MLR, PLS, PCR and Ridge with leakage-free preprocessing.

Every model is a scikit-learn pipeline ``scaler → estimator`` fitted on **training samples
only**; the same fitted pipeline then predicts validation and test samples. Hyperparameters
(PLS/PCR components, Ridge alpha) are chosen by time-aware cross-validation inside the
training split (:mod:`inferential_studio.core.splits`). Nothing learned touches the final test.

Interpretation objects (see docs/methodology.md for definitions):

* coefficients in original units: the fitted pipeline is affine, so ``ŷ = b0 + Σ bⱼ xⱼ``
  is recovered exactly from predictions at the origin and unit vectors;
* standardised coefficients: ``bⱼ · sd(xⱼ) / sd(y)`` with training standard deviations —
  change in y (in standard deviations) per one standard deviation of xⱼ;
* PLS: X weights W, X loadings P, Y loadings q and X scores T are different objects and are
  labelled as such; VIP uses W (Wold's definition);
* PCR: PCA loadings (eigenvectors), scores and the share of X variance per component, which is
  *not* the share of Y explained.
"""

from __future__ import annotations

import platform
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from inferential_studio.core.features import FeatureMatrix
from inferential_studio.core.schemas import ModelKind, ModellingConfig, ModelSpec, Scaling
from inferential_studio.core.splits import (
    PURGED,
    SPLITS,
    TEST,
    TRAIN,
    VALIDATION,
    SplitError,
    SplitResult,
    chronological_split,
    cv_folds,
)
from inferential_studio.core.tasks import CancelToken, ProgressCallback, check, report

ERROR = "error"
WARNING = "warning"
MIN_TRAIN_SAMPLES = 10

METRIC_COLUMNS = ["n", "r2", "adj_r2", "rmse", "mae", "bias", "max_abs_error", "rmse_pct_sd"]
METRIC_LABELS = {
    "n": "Samples",
    "r2": "R²",
    "adj_r2": "Adjusted R²",
    "rmse": "RMSE",
    "mae": "MAE",
    "bias": "Bias (mean of predicted − actual)",
    "max_abs_error": "Max |error|",
    "rmse_pct_sd": "RMSE as % of training SD of Y",
}


class ModelError(ValueError):
    """A model cannot be trained as configured. The message is shown to the user."""


@dataclass
class Problem:
    severity: str  # ERROR or WARNING
    message: str


# --------------------------------------------------------------------------- pipelines


def make_pipeline(
    kind: ModelKind,
    scaling: Scaling,
    *,
    n_components: int | None = None,
    alpha: float | None = None,
) -> Pipeline:
    steps: list[tuple[str, object]] = []
    if scaling is Scaling.STANDARD:
        steps.append(("scale", StandardScaler()))
    elif scaling is Scaling.ROBUST:
        steps.append(("scale", RobustScaler()))
    else:
        steps.append(("scale", "passthrough"))
    if kind is ModelKind.MLR:
        steps.append(("regress", LinearRegression()))
    elif kind is ModelKind.PLS:
        steps.append(("pls", PLSRegression(n_components=n_components or 1, scale=False)))
    elif kind is ModelKind.PCR:
        steps.append(("pca", PCA(n_components=n_components or 1, svd_solver="full")))
        steps.append(("regress", LinearRegression()))
    else:
        steps.append(("regress", Ridge(alpha=1.0 if alpha is None else alpha)))
    return Pipeline(steps)


def _predict(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    return np.asarray(pipe.predict(X), dtype=float).reshape(-1)


def affine_coefficients(pipe: Pipeline, n_features: int) -> tuple[float, np.ndarray]:
    """Intercept and per-feature coefficients in original units of the fitted affine pipeline."""
    basis = np.vstack([np.zeros(n_features), np.eye(n_features)])
    pred = _predict(pipe, basis)
    return float(pred[0]), pred[1:] - pred[0]


# --------------------------------------------------------------------------- metrics


def regression_metrics(
    y: np.ndarray, yhat: np.ndarray, *, n_params: int | None = None, sd_train: float | None = None
) -> dict[str, float]:
    """Metrics for one split. R² uses that split's own mean; bias = mean(predicted − actual)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    n = int(y.size)
    out = dict.fromkeys(METRIC_COLUMNS, np.nan)
    out["n"] = n
    if n == 0:
        return out
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err**2)))
    out.update(
        rmse=rmse,
        mae=float(np.mean(np.abs(err))),
        bias=float(np.mean(err)),
        max_abs_error=float(np.max(np.abs(err))),
    )
    sst = float(np.sum((y - y.mean()) ** 2))
    if n >= 2 and sst > 0:
        out["r2"] = 1.0 - float(np.sum(err**2)) / sst
        if n_params is not None and n - n_params - 1 > 0:
            out["adj_r2"] = 1.0 - (1.0 - out["r2"]) * (n - 1) / (n - n_params - 1)
    if sd_train:
        out["rmse_pct_sd"] = 100.0 * rmse / sd_train
    return out


def effective_sample_size(y: np.ndarray) -> float:
    """n·(1 − ρ₁)/(1 + ρ₁) from the lag-1 autocorrelation of consecutive samples (ρ₁ ≥ 0 only)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3 or np.ptp(y) == 0:
        return float(n)
    d = y - y.mean()
    rho = float(np.sum(d[1:] * d[:-1]) / np.sum(d * d))
    rho = min(max(rho, 0.0), 0.99)
    return n * (1 - rho) / (1 + rho)


def autocorrelation(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Sample ACF at lags 0..max_lag (biased estimator, as usual for residual diagnostics)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    lags = min(max_lag, max(x.size - 1, 0))
    if denom == 0:
        return np.full(lags + 1, np.nan)
    return np.array([np.sum(x[k:] * x[: x.size - k]) / denom for k in range(lags + 1)])


# --------------------------------------------------------------------------- interpretation


@dataclass
class PlsDetails:
    x_weights: pd.DataFrame  # W: features x components
    x_loadings: pd.DataFrame  # P
    y_loadings: np.ndarray  # q, one per component
    scores: pd.DataFrame  # T for every eligible sample (columns t1..tA) + "split"
    vip: pd.Series
    x_variance_explained: np.ndarray  # share of scaled-X sum of squares per component (train)
    y_variance_explained: np.ndarray  # share of centred-Y sum of squares per component (train)


@dataclass
class PcrDetails:
    loadings: pd.DataFrame  # PCA eigenvectors: features x components
    scores: pd.DataFrame  # every eligible sample + "split"
    x_variance_explained: np.ndarray  # PCA explained variance ratio (X only)
    component_coefficients: np.ndarray  # regression of y on the scores


def vip_scores(W: np.ndarray, T: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Variable importance in projection (Wold): uses X weights W, scores T, Y loadings q."""
    p, a = W.shape
    ss = (q.reshape(-1) ** 2) * np.sum(T**2, axis=0)  # Y sum of squares explained per component
    wnorm = W / np.linalg.norm(W, axis=0, keepdims=True)
    total = float(ss.sum())
    if total == 0:
        return np.full(p, np.nan)
    return np.sqrt(p * (wnorm**2 @ ss) / total)


def _pls_details(
    pipe: Pipeline, X_all: np.ndarray, labels: np.ndarray, X_tr: np.ndarray, y_tr: np.ndarray,
    names: list[str],
) -> PlsDetails:  # fmt: skip
    pls: PLSRegression = pipe.named_steps["pls"]
    pre = pipe[:-1]
    Z_tr = np.asarray(pre.transform(X_tr), dtype=float)
    T_tr = pls.transform(Z_tr)
    a = T_tr.shape[1]
    cols = [f"Component {i + 1}" for i in range(a)]
    W, P = pls.x_weights_, pls.x_loadings_
    q = pls.y_loadings_.reshape(-1)
    z0 = Z_tr - Z_tr.mean(axis=0)
    total_x = float(np.sum(z0**2))
    x_var = np.array(
        [float(np.sum(np.outer(T_tr[:, i], P[:, i]) ** 2)) / total_x if total_x else np.nan
         for i in range(a)]
    )  # fmt: skip
    y0 = y_tr - y_tr.mean()
    total_y = float(np.sum(y0**2))
    ss = (q**2) * np.sum(T_tr**2, axis=0)
    y_var = ss / total_y if total_y else np.full(a, np.nan)
    scores = pd.DataFrame(
        pls.transform(np.asarray(pre.transform(X_all), dtype=float)), columns=cols
    )
    scores["split"] = labels
    return PlsDetails(
        x_weights=pd.DataFrame(W, index=names, columns=cols),
        x_loadings=pd.DataFrame(P, index=names, columns=cols),
        y_loadings=q,
        scores=scores,
        vip=pd.Series(vip_scores(W, T_tr, q), index=names, name="VIP"),
        x_variance_explained=x_var,
        y_variance_explained=y_var,
    )


def _pcr_details(
    pipe: Pipeline, X_all: np.ndarray, labels: np.ndarray, names: list[str]
) -> PcrDetails:
    pca: PCA = pipe.named_steps["pca"]
    reg: LinearRegression = pipe.named_steps["regress"]
    pre = pipe[:-2]
    cols = [f"PC{i + 1}" for i in range(pca.n_components_)]
    scores = pd.DataFrame(
        pca.transform(np.asarray(pre.transform(X_all), dtype=float)), columns=cols
    )
    scores["split"] = labels
    return PcrDetails(
        loadings=pd.DataFrame(pca.components_.T, index=names, columns=cols),
        scores=scores,
        x_variance_explained=np.asarray(pca.explained_variance_ratio_, dtype=float),
        component_coefficients=np.asarray(reg.coef_, dtype=float).reshape(-1),
    )


# --------------------------------------------------------------------------- tuning


@dataclass
class CvCurve:
    parameter: str  # "components" or "alpha"
    values: np.ndarray
    mean_rmse: np.ndarray
    se_rmse: np.ndarray
    best: float
    chosen: float
    rule: str  # how "chosen" was picked


def _cv_rmse(
    kind: ModelKind,
    scaling: Scaling,
    X: np.ndarray,
    y: np.ndarray,
    folds,
    **params: float,  # noqa: ANN001
) -> np.ndarray:
    out = []
    for tr, va in folds:
        pipe = make_pipeline(kind, scaling, **params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X[tr], y[tr])
        err = _predict(pipe, X[va]) - y[va]
        out.append(float(np.sqrt(np.mean(err**2))))
    return np.asarray(out)


def _tune(
    spec: ModelSpec, scaling: Scaling, X: np.ndarray, y: np.ndarray, folds, one_se: bool,  # noqa: ANN001
    cancel: CancelToken | None,
) -> CvCurve:  # fmt: skip
    min_train = min(len(tr) for tr, _ in folds)
    if spec.kind is ModelKind.RIDGE:
        values = np.logspace(np.log10(spec.alpha_min), np.log10(spec.alpha_max), spec.alpha_steps)
        param = "alpha"
    else:
        limit = min(spec.max_components, X.shape[1], min_train - 1)
        if limit < 1:
            raise ModelError("Cross-validation folds are too small to fit even one component.")
        values = np.arange(1, limit + 1)
        param = "components"
    means, ses = [], []
    for v in values:
        check(cancel)
        kw = {"alpha": float(v)} if param == "alpha" else {"n_components": int(v)}
        rmse = _cv_rmse(spec.kind, scaling, X, y, folds, **kw)
        means.append(rmse.mean())
        ses.append(rmse.std(ddof=1) / np.sqrt(rmse.size) if rmse.size > 1 else 0.0)
    means, ses = np.asarray(means), np.asarray(ses)
    ib = int(np.argmin(means))
    chosen, rule = values[ib], "lowest mean CV RMSE"
    if one_se:
        limit_rmse = means[ib] + ses[ib]
        ok = np.flatnonzero(means <= limit_rmse)
        # Simplest model: fewest components, or the strongest regularisation (largest alpha).
        pick = ok.min() if param == "components" else ok.max()
        chosen, rule = values[pick], "simplest within one standard error of the best"
    return CvCurve(param, values.astype(float), means, ses, float(values[ib]), float(chosen), rule)


# --------------------------------------------------------------------------- results


@dataclass
class ModelResult:
    spec: ModelSpec
    name: str
    error: str | None = None
    pipeline: Pipeline | None = None
    hyperparameters: dict[str, float] = field(default_factory=dict)
    cv: CvCurve | None = None
    cv_rmse: float = np.nan  # mean CV RMSE at the chosen setting
    intercept: float = np.nan
    coefficients: pd.Series | None = None  # original units of X and y
    standardized: pd.Series | None = None
    predictions: pd.DataFrame | None = None  # row, sample_time, split, actual, predicted, residual
    metrics: pd.DataFrame | None = None  # index = split
    details: PlsDetails | PcrDetails | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def n_params(self) -> int | None:
        if self.coefficients is None:
            return None
        if self.spec.kind in (ModelKind.PLS, ModelKind.PCR):
            return int(self.hyperparameters.get("components", 0))
        return len(self.coefficients)

    def metric(self, split: str, name: str) -> float:
        if self.metrics is None or split not in self.metrics.index:
            return np.nan
        return float(self.metrics.loc[split, name])


@dataclass
class TrainingRun:
    target: str
    features: list[str]
    config: ModellingConfig
    split: SplitResult
    sample_time: pd.DatetimeIndex  # every feature-matrix row
    feature_values: pd.DataFrame  # the aligned features of every row (same order)
    models: list[ModelResult]
    problems: list[Problem]
    train_ranges: pd.DataFrame  # per feature: min, max, mean, std (training samples)
    outside_range: pd.DataFrame  # per row: number of features outside the training range
    n_effective: float
    versions: dict[str, str]
    feature_notes: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [p.message for p in self.problems if p.severity == ERROR]

    def ranking_split(self) -> str:
        return VALIDATION if self.split.mask(VALIDATION).any() else "cv"

    def comparison(self, include_test: bool = False) -> pd.DataFrame:
        """One row per trained model, ranked by validation RMSE (or CV RMSE without validation)."""
        rows = []
        for m in self.models:
            row = {"model": m.name, "status": "ok" if m.ok else m.error, "cv_rmse": m.cv_rmse}
            splits = (TRAIN, VALIDATION, TEST) if include_test else (TRAIN, VALIDATION)
            for s in splits:
                for k in ("r2", "rmse", "mae", "bias"):
                    row[f"{s}_{k}"] = m.metric(s, k) if m.ok else np.nan
            rows.append(row)
        table = pd.DataFrame(rows).set_index("model")
        key = "validation_rmse" if self.ranking_split() == VALIDATION else "cv_rmse"
        return table.sort_values(key, na_position="last")


def _versions() -> dict[str, str]:
    from inferential_studio import __version__

    return {
        "app": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }


def check_feasibility(
    X_train: pd.DataFrame, y_train: np.ndarray, split: SplitResult
) -> list[Problem]:
    """Situations that make training impossible (errors) or results fragile (warnings)."""
    problems: list[Problem] = []
    n, p = X_train.shape
    if n < MIN_TRAIN_SAMPLES:
        problems.append(
            Problem(ERROR, f"Only {n} training samples; at least {MIN_TRAIN_SAMPLES} are needed.")
        )
        return problems
    if np.ptp(y_train) == 0:
        problems.append(Problem(ERROR, "The target is constant in the training period."))
    constant = [c for c in X_train.columns if np.ptp(X_train[c].to_numpy()) == 0]
    if constant:
        problems.append(
            Problem(
                ERROR, f"Constant in the training period (deselect them): {', '.join(constant)}."
            )
        )
    if n < 5 * p:
        problems.append(
            Problem(
                WARNING, f"{n} training samples for {p} predictors (fewer than 5 per predictor)."
            )
        )
    n_eff = effective_sample_size(y_train)
    if n_eff < 0.5 * n:
        problems.append(
            Problem(
                WARNING,
                f"Consecutive lab samples are strongly autocorrelated: about {n_eff:.0f} effectively "
                f"independent samples out of {n}. Fit statistics will look better than they are.",
            )
        )
    problems.extend(Problem(WARNING, note) for note in split.notes)
    return problems


def train_models(
    fm: FeatureMatrix,
    config: ModellingConfig,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> TrainingRun:
    """Split, tune, fit and evaluate every enabled model on the same split definition."""
    names = fm.features
    split = chronological_split(fm.sample_time, fm.eligible, config.split, fm.max_history_s())
    labels = split.labels
    use = np.isin(labels, SPLITS)
    X_all = fm.X.to_numpy(dtype=float)
    tr = labels == TRAIN
    X_tr, y_tr = X_all[tr], fm.y[tr]
    X_tr_df = pd.DataFrame(X_tr, columns=names)
    problems = check_feasibility(X_tr_df, y_tr, split)
    if fm.non_causal:
        problems.append(
            Problem(WARNING, "Non-causal features (read future data): " + ", ".join(fm.non_causal))
        )
    stats = pd.DataFrame(
        {"min": X_tr_df.min(), "max": X_tr_df.max(), "mean": X_tr_df.mean(), "std": X_tr_df.std()}
    )
    with np.errstate(invalid="ignore"):
        outside = (X_all < stats["min"].to_numpy()) | (X_all > stats["max"].to_numpy())
    outside_df = pd.DataFrame(outside & use[:, None], columns=names)
    run = TrainingRun(
        target=fm.target,
        features=names,
        config=config,
        split=split,
        sample_time=fm.sample_time,
        feature_values=fm.X.copy(),
        models=[],
        problems=problems,
        train_ranges=stats,
        outside_range=outside_df,
        n_effective=effective_sample_size(y_tr) if y_tr.size else 0.0,
        versions=_versions(),
        feature_notes=list(fm.notes),
    )
    if run.errors:
        return run

    folds = None
    fold_error = None
    try:
        folds = cv_folds(
            fm.sample_time[tr], config.split.cv_kind, config.split.cv_folds, split.purge_s
        )
    except SplitError as exc:
        fold_error = str(exc)
        problems.append(Problem(WARNING, f"Cross-validation unavailable: {exc}"))

    specs = [m for m in config.models if m.enabled]
    sd_y = float(np.std(y_tr, ddof=1)) if y_tr.size > 1 else np.nan
    for i, spec in enumerate(specs):
        check(cancel)
        report(progress, i / max(len(specs), 1), f"Training {spec.label}")
        run.models.append(
            _train_one(spec, config, X_all, fm.y, labels, X_tr, y_tr, names, folds, fold_error,
                       sd_y, fm.sample_time, cancel)
        )  # fmt: skip
    report(progress, 1.0, "Done")
    return run


def _train_one(
    spec: ModelSpec, config: ModellingConfig, X_all: np.ndarray, y_all: np.ndarray,
    labels: np.ndarray, X_tr: np.ndarray, y_tr: np.ndarray, names: list[str], folds,  # noqa: ANN001
    fold_error: str | None, sd_y: float, times: pd.DatetimeIndex, cancel: CancelToken | None,
) -> ModelResult:  # fmt: skip
    name = spec.label
    result = ModelResult(spec, name)
    n, p = X_tr.shape
    kind = spec.kind
    try:
        params: dict[str, float] = {}
        if kind is ModelKind.MLR:
            if n <= p + 1:
                raise ModelError(
                    f"MLR needs more training samples ({n}) than predictors + 1 ({p + 1})."
                )
            rank = np.linalg.matrix_rank(X_tr - X_tr.mean(axis=0))
            if rank < p:
                result.warnings.append(
                    "Predictors are linearly dependent in the training data; MLR coefficients are "
                    "not unique (minimum-norm solution shown). Prefer PLS, PCR or Ridge."
                )
        elif kind in (ModelKind.PLS, ModelKind.PCR):
            cap = min(p, n - 1)
            if spec.n_components is not None:
                if spec.n_components > cap:
                    raise ModelError(
                        f"{spec.n_components} components requested; at most {cap} are possible "
                        f"with {p} predictors and {n} training samples."
                    )
                params["components"] = spec.n_components
            else:
                if folds is None:
                    raise ModelError(f"Cannot tune the number of components: {fold_error}")
                result.cv = _tune(
                    spec, config.scaling, X_tr, y_tr, folds, config.one_standard_error, cancel
                )
                params["components"] = int(result.cv.chosen)
        else:
            if spec.alpha is not None:
                params["alpha"] = spec.alpha
            else:
                if folds is None:
                    raise ModelError(f"Cannot tune alpha: {fold_error}")
                result.cv = _tune(
                    spec, config.scaling, X_tr, y_tr, folds, config.one_standard_error, cancel
                )
                params["alpha"] = result.cv.chosen
        kw = (
            {"n_components": int(params["components"])}
            if "components" in params
            else {"alpha": params["alpha"]}
            if "alpha" in params
            else {}
        )
        if folds is not None:
            result.cv_rmse = float(_cv_rmse(kind, config.scaling, X_tr, y_tr, folds, **kw).mean())
        pipe = make_pipeline(kind, config.scaling, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_tr, y_tr)
    except ModelError as exc:
        result.error = str(exc)
        return result
    except (ValueError, np.linalg.LinAlgError) as exc:  # numerical failure inside sklearn
        result.error = f"Fitting failed: {exc}"
        return result

    result.pipeline = pipe
    result.hyperparameters = params
    if "components" in params:
        result.name = f"{name} — {int(params['components'])} component(s)"
    elif "alpha" in params:
        result.name = f"{name} — alpha {params['alpha']:.3g}"
    intercept, coef = affine_coefficients(pipe, p)
    result.intercept = intercept
    result.coefficients = pd.Series(coef, index=names, name="coefficient")
    sd_x = X_tr.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        result.standardized = pd.Series(coef * sd_x / sd_y, index=names, name="standardized")

    use = np.isin(labels, SPLITS)
    pred = np.full(len(y_all), np.nan)
    pred[use] = _predict(pipe, X_all[use])
    result.predictions = pd.DataFrame(
        {
            "row": np.flatnonzero(use),  # row of the feature matrix
            "sample_time": times[use],
            "split": labels[use],
            "actual": y_all[use],
            "predicted": pred[use],
            "residual": y_all[use] - pred[use],
        }
    ).reset_index(drop=True)
    rows = {}
    for s in SPLITS:
        m = labels[use] == s
        rows[s] = regression_metrics(
            y_all[use][m], pred[use][m], n_params=result.n_params if s == TRAIN else None,
            sd_train=sd_y,
        )  # fmt: skip
    result.metrics = pd.DataFrame.from_dict(rows, orient="index")[METRIC_COLUMNS]
    if kind is ModelKind.PLS:
        result.details = _pls_details(pipe, X_all[use], labels[use], X_tr, y_tr, names)
        if params["components"] < 2:
            result.warnings.append(
                "Only one latent component: a two-dimensional scores plot is not available."
            )
    elif kind is ModelKind.PCR:
        result.details = _pcr_details(pipe, X_all[use], labels[use], names)
    return result


__all__ = [
    "ERROR",
    "METRIC_LABELS",
    "PURGED",
    "WARNING",
    "CvCurve",
    "ModelError",
    "ModelResult",
    "PcrDetails",
    "PlsDetails",
    "Problem",
    "TrainingRun",
    "affine_coefficients",
    "autocorrelation",
    "check_feasibility",
    "effective_sample_size",
    "make_pipeline",
    "regression_metrics",
    "train_models",
    "vip_scores",
]
