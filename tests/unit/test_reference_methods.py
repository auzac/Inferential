"""Reference checks: every statistical method against an independent, definition-level version.

Each reference below is written from the textbook definition (plain loops or closed forms), not
from the application's code, so a change that drifts from the standard method fails here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from inferential_studio.core import diagnostics, features, quality, statistics
from inferential_studio.core.models import (
    _tune,
    affine_coefficients,
    autocorrelation,
    effective_sample_size,
    make_pipeline,
    regression_metrics,
    vip_scores,
)
from inferential_studio.core.rules import ns
from inferential_studio.core.schemas import (
    Aggregation,
    CvKind,
    DetectorConfig,
    DetectorKind,
    FeatureSpec,
    FilterKind,
    FilterSpec,
    ModelKind,
    ModelSpec,
    Scaling,
    SplitConfig,
    WindowPlacement,
)
from inferential_studio.core.splits import TEST, TRAIN, VALIDATION, chronological_split, cv_folds

T0 = 1_767_225_600 * 10**9  # 2026-01-01 UTC in ns


# --------------------------------------------------------------------------- models


@pytest.fixture(scope="module")
def collinear():
    """Plant-like data: two latent drivers, six correlated predictors in mixed units."""
    rng = np.random.default_rng(7)
    n, p = 190, 6
    latent = rng.normal(size=(n, 2))
    X = latent @ rng.normal(size=(2, p)) + 0.3 * rng.normal(size=(n, p))
    X = X * [1, 10, 100, 0.01, 5, 2] + [0, 50, 300, 1, -2, 7]
    y = 2 + latent @ [1.0, -0.5] + 0.2 * rng.normal(size=n)
    Xn, yn, Xt = X[:150], y[:150], X[150:]
    mu, sd = Xn.mean(0), Xn.std(0)  # StandardScaler: population SD
    return Xn, yn, Xt, (Xn - mu) / sd, (Xt - mu) / sd, sd


def test_mlr_is_ordinary_least_squares(collinear):
    Xn, yn, _Xt, *_ = collinear
    A = np.column_stack([np.ones(len(Xn)), Xn])
    beta = np.linalg.lstsq(A, yn, rcond=None)[0]
    for scaling in Scaling:  # scaling never changes an OLS fit
        b0, b = affine_coefficients(make_pipeline(ModelKind.MLR, scaling).fit(Xn, yn), Xn.shape[1])
        np.testing.assert_allclose(np.r_[b0, b], beta, rtol=1e-8)
    m = regression_metrics(yn, A @ beta, n_params=Xn.shape[1])
    sse, sst = np.sum((yn - A @ beta) ** 2), np.sum((yn - yn.mean()) ** 2)
    n, p = Xn.shape
    assert m["r2"] == pytest.approx(1 - sse / sst)
    assert m["adj_r2"] == pytest.approx(1 - (sse / (n - p - 1)) / (sst / (n - 1)))
    assert m["rmse"] == pytest.approx(np.sqrt(sse / n))  # not Minitab's S = √(SSE/(n−p−1))


@pytest.mark.parametrize("components", [1, 2, 3, 6])
def test_pls_matches_wold_nipals_pls1(collinear, components):
    Xn, yn, Xt, Z, Zt, sd = collinear
    E, f = Z.copy(), yn - yn.mean()
    W, P, Q, T = [], [], [], []
    for _ in range(components):
        w = E.T @ f
        w /= np.linalg.norm(w)
        t = E @ w
        p_a, q_a = E.T @ t / (t @ t), f @ t / (t @ t)
        E -= np.outer(t, p_a)
        f -= q_a * t
        W.append(w), P.append(p_a), Q.append(q_a), T.append(t)
    W, P, T, Q = np.array(W).T, np.array(P).T, np.array(T).T, np.array(Q)
    B = W @ np.linalg.solve(P.T @ W, Q)  # regression vector for standardised X
    pipe = make_pipeline(ModelKind.PLS, Scaling.STANDARD, n_components=components).fit(Xn, yn)
    np.testing.assert_allclose(pipe.predict(Xt).ravel(), yn.mean() + Zt @ B, rtol=1e-9)
    np.testing.assert_allclose(affine_coefficients(pipe, Xn.shape[1])[1], B / sd, rtol=1e-9)
    ss = Q**2 * np.sum(T**2, axis=0)  # Y variance explained per component
    vip = np.sqrt(W.shape[0] * ((W / np.linalg.norm(W, axis=0)) ** 2 @ ss) / ss.sum())
    pls = pipe.named_steps["pls"]
    got = vip_scores(pls.x_weights_, pls.transform(Z), pls.y_loadings_.ravel())
    np.testing.assert_allclose(got, vip, rtol=1e-9)
    assert np.mean(got**2) == pytest.approx(1.0)  # defining property of VIP


@pytest.mark.parametrize("components", [1, 3, 6])
def test_pcr_is_principal_components_then_ols(collinear, components):
    Xn, yn, Xt, Z, Zt, _sd = collinear
    _u, s, vt = np.linalg.svd(Z, full_matrices=False)
    scores = Z @ vt[:components].T
    g = np.linalg.lstsq(np.column_stack([np.ones(len(scores)), scores]), yn, rcond=None)[0]
    pipe = make_pipeline(ModelKind.PCR, Scaling.STANDARD, n_components=components).fit(Xn, yn)
    np.testing.assert_allclose(
        pipe.predict(Xt).ravel(), g[0] + Zt @ vt[:components].T @ g[1:], rtol=1e-9
    )
    np.testing.assert_allclose(
        pipe.named_steps["pca"].explained_variance_ratio_, (s**2 / np.sum(s**2))[:components]
    )


@pytest.mark.parametrize("alpha", [0.01, 1.0, 100.0])
def test_ridge_is_the_closed_form_with_unpenalised_intercept(collinear, alpha):
    Xn, yn, Xt, Z, Zt, sd = collinear
    b = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (yn - yn.mean()))
    pipe = make_pipeline(ModelKind.RIDGE, Scaling.STANDARD, alpha=alpha).fit(Xn, yn)
    np.testing.assert_allclose(affine_coefficients(pipe, Xn.shape[1])[1], b / sd, rtol=1e-9)
    np.testing.assert_allclose(pipe.predict(Xt).ravel(), yn.mean() + Zt @ b, rtol=1e-9)


def test_one_standard_error_rule(collinear):
    Xn, yn, *_ = collinear
    times = pd.date_range("2026-01-01", periods=len(yn), freq="2h", tz="UTC")
    folds = cv_folds(times, CvKind.EXPANDING, 5, 0)
    cv = _tune(
        ModelSpec(ModelKind.PLS, max_components=6), Scaling.STANDARD, Xn, yn, folds, True, None
    )
    best = int(np.argmin(cv.mean_rmse))
    within = np.flatnonzero(cv.mean_rmse <= cv.mean_rmse[best] + cv.se_rmse[best])
    assert cv.best == cv.values[best] and cv.chosen == cv.values[within.min()]


def test_effective_sample_size_and_acf_on_ar1():
    rng = np.random.default_rng(1)
    e = np.zeros(4000)
    for i in range(1, e.size):
        e[i] = 0.8 * e[i - 1] + rng.normal()
    assert effective_sample_size(e) == pytest.approx(4000 * 0.2 / 1.8, rel=0.1)
    np.testing.assert_allclose(autocorrelation(e, 3)[1:], [0.8, 0.64, 0.512], atol=0.05)


def test_splits_and_folds_never_share_process_data():
    """Brute force over every pair: no two samples in different periods/folds are closer than
    the data span (they would share process values); replicates stay together."""
    rng = np.random.default_rng(2)
    regular = pd.date_range("2026-01-01", periods=300, freq="3h", tz="UTC")
    replicates = pd.date_range("2026-01-05", periods=20, freq="9h", tz="UTC")
    t = pd.DatetimeIndex(sorted([*regular, *replicates]))
    span = 4 * 3600
    eligible = rng.random(len(t)) > 0.1
    split = chronological_split(t, eligible, SplitConfig(), span)
    sec = ns(t) / 1e9
    lab = split.labels

    def close_pairs(a, b, times=sec):
        return int(np.sum(np.abs(times[a][:, None] - times[b][None, :]) < span))

    assert close_pairs(lab == TRAIN, lab == VALIDATION) == 0
    assert close_pairs(lab == VALIDATION, lab == TEST) == 0
    assert (pd.Series(lab[eligible]).groupby(t[eligible]).nunique() == 1).all()
    train_t = t[lab == TRAIN]
    tsec = ns(train_t) / 1e9
    for kind in CvKind:
        for tr, va in cv_folds(train_t, kind, 5, span):
            assert close_pairs(tr, va, tsec) == 0
            assert not set(tsec[tr]) & set(tsec[va])
            if kind is CvKind.EXPANDING:
                assert tsec[tr].max() < tsec[va].min()  # only the past predicts the future


# --------------------------------------------------------------------------- EDA statistics


def test_descriptive_statistics_and_correlations():
    rng = np.random.default_rng(3)
    n = 400
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame({"a": rng.normal(5, 2, n), "b": rng.gamma(2, 2, n)}, index=idx)
    df.loc[df.sample(frac=0.1, random_state=1).index, "a"] = np.nan
    df.loc[df.sample(frac=0.2, random_state=2).index, "b"] = np.nan
    df["c"] = df["a"] * 0.5 + rng.normal(size=n)
    s = statistics.summary_statistics(df)
    for col in df:
        v = df[col].dropna().to_numpy()
        assert s.loc[col, "std"] == pytest.approx(v.std(ddof=1))
        assert s.loc[col, "skewness"] == pytest.approx(stats.skew(v, bias=False))
        assert s.loc[col, "kurtosis"] == pytest.approx(stats.kurtosis(v, bias=False))
    for method in ("pearson", "spearman"):
        np.testing.assert_allclose(
            statistics.correlation_matrix(df, method),
            df.corr(method=method, min_periods=30),
            equal_nan=True,
        )
    X = df.dropna()
    inverse_corr = np.linalg.inv(np.corrcoef(X.to_numpy(), rowvar=False))
    np.testing.assert_allclose(diagnostics.vif(X)["vif"], np.diag(inverse_corr))


# --------------------------------------------------------------------------- cleaning


def _textbook_hampel(t_ns, x, window_s, k):
    half = window_s * 1e9 / 2
    out = np.zeros(x.size, dtype=bool)
    for i in np.flatnonzero(~np.isnan(x)):
        w = x[(t_ns >= t_ns[i] - half) & (t_ns <= t_ns[i] + half)]
        w = w[~np.isnan(w)]
        if w.size >= 3:
            m = np.median(w)
            dev = abs(x[i] - m)
            out[i] = dev > k * 1.4826 * np.median(np.abs(w - m)) and dev > 0
    return out


def test_hampel_identifier_matches_textbook_definition():
    rng = np.random.default_rng(11)
    n = 1500
    t = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    x = np.sin(np.arange(n) / 100) * 3 + rng.normal(0, 0.3, n)
    x[rng.choice(n, 20, replace=False)] += rng.choice([-1, 1], 20) * rng.uniform(2, 6, 20)
    x[800:] += 4  # a genuine step
    x[rng.choice(n, 40, replace=False)] = np.nan
    keep = np.sort(rng.choice(n, n - 60, replace=False))  # irregular sampling
    t, x = t[keep], x[keep]
    det = DetectorConfig(DetectorKind.HAMPEL, ("v",), threshold=3.0, window_s=1800)
    got = quality.detect(det, pd.Series(x, index=t, name="v")).flags
    np.testing.assert_array_equal(got, _textbook_hampel(ns(t), x, 1800, 3.0))


def test_flatline_is_a_band_not_a_chain_of_small_steps():
    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(0, 1, 60), np.linspace(0, 3, 61), rng.normal(0, 1, 30), np.full(40, 2.0)]
    x[151:191] += rng.uniform(-0.02, 0.02, 40)  # stuck, with tiny quantisation noise
    s = pd.Series(x, index=pd.date_range("2026-01-01", periods=x.size, freq="1min", tz="UTC"))
    det = DetectorConfig(DetectorKind.FLATLINE, ("v",), window_s=1800, tolerance=0.06)
    flags = quality.detect(det, s.rename("v")).flags
    assert not flags[60:121].any()  # steady ramp in steps of 0.05 < tolerance: moving, not stuck
    assert flags[151:191].all()  # within a 0.04 band for 40 minutes: stuck


# --------------------------------------------------------------------------- feature alignment


def _brute_window(tt, xv, excl, samples, spec, regime_ns):
    keep = ~np.isnan(xv) & ~excl
    T, V = tt[keep], xv[keep]
    d = np.diff(tt[~np.isnan(xv)])
    interval = np.median(d[d > 0]) / 1e9
    a, b = spec.window_offsets_s
    agg = {
        Aggregation.MEAN: np.mean,
        Aggregation.MEDIAN: np.median,
        Aggregation.MIN: np.min,
        Aggregation.MAX: np.max,
        Aggregation.LAST: lambda z: z[-1],
        Aggregation.STD: lambda z: np.std(z, ddof=1),
    }[spec.aggregation]
    out = []
    for s in samples:
        lo, hi = s + int(round(a * 1e9)), s + int(round(b * 1e9))
        inside = (lo < T) & (hi >= T)
        w = V[inside]
        cov = min(1.0, w.size * interval / spec.window_s)
        gap = np.max(np.diff(np.r_[lo, T[inside], hi])) / 1e9 if w.size else spec.window_s
        bad = (
            w.size < (2 if spec.aggregation is Aggregation.STD else 1)
            or cov < spec.min_coverage
            or gap > spec.max_gap_s
            or np.any((regime_ns > lo) & (regime_ns <= hi))
        )
        out.append((np.nan if bad else agg(w), cov, gap))
    return np.array(out)


@pytest.mark.parametrize("aggregation", list(Aggregation))
@pytest.mark.parametrize("placement", list(WindowPlacement))
def test_window_features_match_brute_force(aggregation, placement):
    rng = np.random.default_rng(9)
    n = 3000
    tt = np.sort(rng.choice(np.arange(0, 15 * 86400, 60), n, replace=False)) * 10**9 + T0
    xv = rng.normal(size=n)
    xv[rng.random(n) < 0.05] = np.nan
    excl = rng.random(n) < 0.05
    regime = rng.random(n) < 0.01
    samples = np.sort(rng.choice(np.arange(3600, 14 * 86400, 600), 100, replace=False)) * 10**9 + T0
    raw = pd.Series(xv, index=pd.DatetimeIndex(pd.to_datetime(tt, utc=True)), name="x")
    spec = FeatureSpec(
        "x", window_s=1800, deadtime_s=600, placement=placement, aggregation=aggregation,
        min_coverage=0.6, max_gap_s=600,
    )  # fmt: skip
    res = features.aggregate_windows(
        features.process_signal(raw, spec, excl), samples, spec, tt[regime]
    )
    ref = _brute_window(tt, xv, excl, samples, spec, tt[regime])
    np.testing.assert_allclose(res.value, ref[:, 0], equal_nan=True)
    np.testing.assert_allclose(res.coverage, ref[:, 1])
    np.testing.assert_allclose(res.gap_s, ref[:, 2])


def test_trailing_filters_and_resampling_match_definitions():
    rng = np.random.default_rng(10)
    n = 1500
    tt = np.sort(rng.choice(np.arange(0, 5 * 86400, 60), n, replace=False)) * 10**9 + T0
    raw = pd.Series(rng.normal(size=n), index=pd.DatetimeIndex(pd.to_datetime(tt, utc=True)))
    raw.name = "x"
    T, V = tt, raw.to_numpy()

    def run(filter_spec=None, resample=None):
        spec = FeatureSpec("x", filter=filter_spec or FilterSpec(), resample_s=resample)
        return features.process_signal(raw, spec)

    past = [(T[i] - 900e9 < T) & (T[i] >= T) for i in range(n)]
    got = run(FilterSpec(FilterKind.ROLLING_MEAN, window_s=900)).values
    np.testing.assert_allclose(got, [V[m].mean() for m in past])
    got = run(FilterSpec(FilterKind.ROLLING_MEDIAN, window_s=900)).values
    np.testing.assert_allclose(got, [np.median(V[m]) for m in past])
    weights = [0.5 ** ((T[i] - T[: i + 1]) / 600e9) for i in range(n)]
    got = run(FilterSpec(FilterKind.EWMA, halflife_s=600)).values
    np.testing.assert_allclose(got, [w @ V[: i + 1] / w.sum() for i, w in enumerate(weights)])
    sig = run(resample=300)
    label = -((-T) // 300_000_000_000) * 300_000_000_000  # bin end: (kR − R, kR]
    np.testing.assert_array_equal(sig.times_ns, np.unique(label))
    np.testing.assert_allclose(sig.values, [V[label == k].mean() for k in np.unique(label)])


def test_tuning_warns_when_the_best_value_is_at_the_edge_of_the_grid(collinear):
    from inferential_studio.core.features import FeatureMatrix
    from inferential_studio.core.models import train_models
    from inferential_studio.core.schemas import ModellingConfig

    Xn, yn, *_ = collinear
    n, p = Xn.shape
    names = [f"x{i}" for i in range(p)]
    X = pd.DataFrame(Xn, columns=names)
    fm = FeatureMatrix(
        "y", pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC"), yn, X, X * 0 + 1,
        X * 0, X * 0 + 30, pd.DataFrame("ok", index=X.index, columns=names),
        np.full(n, "ok", dtype=object), np.zeros(n, bool), np.ones(n, bool),
        {c: FeatureSpec(c) for c in names},
    )  # fmt: skip
    narrow = (
        ModelSpec(ModelKind.RIDGE, alpha_min=100.0, alpha_max=1000.0, alpha_steps=3),
        ModelSpec(ModelKind.PLS, max_components=1),
    )
    run = train_models(fm, ModellingConfig(models=narrow))
    ridge, pls = run.models
    assert ridge.cv.best == 100.0 and any("edge of the alpha range" in w for w in ridge.warnings)
    assert any("maximum of 1 components" in w for w in pls.warnings)
