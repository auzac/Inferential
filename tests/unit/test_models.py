from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inferential_studio.core.diagnostics import aligned_diagnostics, high_correlation_pairs, vif
from inferential_studio.core.features import OK, FeatureMatrix
from inferential_studio.core.models import (
    ERROR,
    affine_coefficients,
    autocorrelation,
    effective_sample_size,
    make_pipeline,
    regression_metrics,
    train_models,
    vip_scores,
)
from inferential_studio.core.schemas import (
    CvKind,
    FeatureSpec,
    ModelKind,
    ModellingConfig,
    ModelSpec,
    Scaling,
    SchemaError,
    SplitConfig,
    SplitMode,
)
from inferential_studio.core.splits import (
    NOT_ELIGIBLE,
    PURGED,
    TEST,
    TRAIN,
    VALIDATION,
    SplitError,
    chronological_split,
    cv_folds,
)

T0 = pd.Timestamp("2026-01-01", tz="UTC")


def make_fm(
    X: np.ndarray, y: np.ndarray, hours: float = 2.0, window_s: float = 1800
) -> FeatureMatrix:
    n, p = X.shape
    names = [f"x{i + 1}" for i in range(p)]
    times = pd.DatetimeIndex([T0 + pd.Timedelta(hours=hours * i) for i in range(n)])
    frame = pd.DataFrame(X, columns=names)
    ok = pd.DataFrame(OK, index=frame.index, columns=names)
    ones = pd.DataFrame(1.0, index=frame.index, columns=names)
    return FeatureMatrix(
        target="y", sample_time=times, y=np.asarray(y, float), X=frame, coverage=ones,
        gap_s=ones * 60, n_values=ones * 30, status=ok,
        y_status=np.full(n, "ok", dtype=object), y_pending=np.zeros(n, bool),
        eligible=np.ones(n, bool), specs={c: FeatureSpec(c, window_s=window_s) for c in names},
    )  # fmt: skip


def linear_data(n=200, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)) * [1.0, 5.0, 0.2] + [10, 100, 3]
    y = 1.5 + 2.0 * X[:, 0] - 0.3 * X[:, 1] + 4.0 * X[:, 2] + rng.normal(0, noise, n)
    return X, y


# --------------------------------------------------------------------------- splits


def test_fraction_split_is_chronological_and_purged():
    times = pd.DatetimeIndex([T0 + pd.Timedelta(hours=i) for i in range(100)])
    eligible = np.ones(100, bool)
    eligible[5] = False
    res = chronological_split(times, eligible, SplitConfig(), history_s=2 * 3600)
    labels = res.labels
    assert labels[5] == NOT_ELIGIBLE
    order = {TRAIN: 0, VALIDATION: 1, TEST: 2}
    ranks = [order[x] for x in labels if x in order]
    assert ranks == sorted(ranks)  # train, then validation, then test
    # 2-h history: the 2 samples within 2 h before each boundary (strictly) are purged... only
    # samples with t > first_next − 2 h, i.e. one hour and zero hours apart → 1 sample each.
    assert (labels == PURGED).sum() == 2
    first_val = res.validation_start
    last_train = times[labels == TRAIN].max()
    assert first_val - last_train >= pd.Timedelta(hours=2)
    assert (labels == TEST).sum() == 20  # the final test set is never trimmed


def test_date_split_and_replicates_stay_together():
    times = pd.DatetimeIndex([T0 + pd.Timedelta(hours=i // 2) for i in range(40)])  # pairs
    cfg = SplitConfig(
        mode=SplitMode.DATES,
        train_end=(T0 + pd.Timedelta(hours=10)).to_pydatetime(),
        validation_end=(T0 + pd.Timedelta(hours=15)).to_pydatetime(),
        purge_s=0,
    )
    res = chronological_split(times, np.ones(40, bool), cfg)
    for t in np.unique(times):
        assert len(set(res.labels[times == t])) == 1
    assert res.validation_start == T0 + pd.Timedelta(hours=10)
    assert res.test_start == T0 + pd.Timedelta(hours=15)


def test_split_config_validation():
    with pytest.raises(SchemaError):
        SplitConfig(train_fraction=0.8, validation_fraction=0.2)
    with pytest.raises(SchemaError):
        SplitConfig(mode=SplitMode.DATES)
    with pytest.raises(SchemaError):
        SplitConfig(cv_folds=1)
    with pytest.raises(SchemaError):
        ModelSpec(ModelKind.PLS, n_components=0)


def test_expanding_cv_only_uses_the_past_and_purges():
    times = pd.DatetimeIndex([T0 + pd.Timedelta(hours=i) for i in range(60)])
    folds = cv_folds(times, CvKind.EXPANDING, 5, purge_s=3 * 3600)
    assert len(folds) == 5
    for tr, va in folds:
        assert times[tr].max() <= times[va].min() - pd.Timedelta(hours=3)
    blocked = cv_folds(times, CvKind.BLOCKED, 4, purge_s=3600)
    covered = np.concatenate([va for _, va in blocked])
    assert sorted(covered) == list(range(60))  # every training sample validated once
    for tr, va in blocked:
        gap = np.abs(times[tr].to_numpy()[:, None] - times[va].to_numpy()[None, :]).min()
        assert gap >= pd.Timedelta(hours=1)
    with pytest.raises(SplitError, match="needs at least"):
        cv_folds(times[:5], CvKind.EXPANDING, 5, 0)


# --------------------------------------------------------------------------- metrics / helpers


def test_regression_metrics_definitions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.array([1.5, 2.0, 2.5, 4.0])
    m = regression_metrics(y, yhat, n_params=1, sd_train=2.0)
    assert m["rmse"] == pytest.approx(np.sqrt(0.125))
    assert m["mae"] == pytest.approx(0.25)
    assert m["bias"] == pytest.approx(0.0)
    assert m["max_abs_error"] == pytest.approx(0.5)
    assert m["r2"] == pytest.approx(1 - 0.5 / 5.0)
    assert m["adj_r2"] == pytest.approx(1 - (1 - m["r2"]) * 3 / 2)
    assert m["rmse_pct_sd"] == pytest.approx(100 * m["rmse"] / 2.0)
    const = regression_metrics(np.ones(5), np.ones(5) * 1.1)
    assert np.isnan(const["r2"]) and const["bias"] == pytest.approx(0.1)  # R² undefined


def test_autocorrelation_and_effective_n():
    rng = np.random.default_rng(0)
    white = rng.normal(size=2000)
    assert abs(autocorrelation(white, 3)[1]) < 0.06
    assert effective_sample_size(white) == pytest.approx(2000, rel=0.15)
    walk = np.cumsum(white)
    assert effective_sample_size(walk) < 50


def test_vif_and_correlation_warnings():
    rng = np.random.default_rng(1)
    a = rng.normal(size=300)
    b = rng.normal(size=300)
    X = pd.DataFrame({"a": a, "b": b, "c": a + 0.05 * rng.normal(size=300), "k": 1.0})
    out = vif(X)
    assert out.loc["b", "vif"] == pytest.approx(1.0, abs=0.1)
    assert out.loc["a", "vif"] > 100 and out.loc["c", "vif"] > 100
    assert out.loc["k", "note"] == "constant" and np.isnan(out.loc["k", "vif"])
    exact = vif(pd.DataFrame({"a": a, "b": b, "s": a + b}))
    assert np.isinf(exact.loc["s", "vif"])
    pairs = high_correlation_pairs(X[["a", "b", "c"]].corr(), 0.9)
    assert pairs[["variable_a", "variable_b"]].values.tolist() == [["a", "c"]]
    diag = aligned_diagnostics(X[["a", "b"]], 2 * a + rng.normal(0, 0.1, 300))
    assert diag.xy.index[0] == "a" and diag.xy.loc["a", "pearson"] > 0.99


# --------------------------------------------------------------------------- models


@pytest.mark.parametrize("scaling", list(Scaling))
def test_mlr_recovers_known_coefficients(scaling):
    X, y = linear_data()
    pipe = make_pipeline(ModelKind.MLR, scaling).fit(X, y)
    b0, b = affine_coefficients(pipe, 3)
    np.testing.assert_allclose(b, [2.0, -0.3, 4.0], atol=0.03)
    assert b0 == pytest.approx(1.5, abs=0.5)


def test_full_rank_pls_pcr_match_mlr_and_small_ridge():
    X, y = linear_data(noise=0.2)
    ref = affine_coefficients(make_pipeline(ModelKind.MLR, Scaling.STANDARD).fit(X, y), 3)[1]
    for kind, kw in (
        (ModelKind.PLS, {"n_components": 3}),
        (ModelKind.PCR, {"n_components": 3}),
        (ModelKind.RIDGE, {"alpha": 1e-8}),
    ):
        b = affine_coefficients(make_pipeline(kind, Scaling.STANDARD, **kw).fit(X, y), 3)[1]
        np.testing.assert_allclose(b, ref, rtol=1e-5, atol=1e-8)


def test_vip_property():
    rng = np.random.default_rng(2)
    W = rng.normal(size=(5, 3))
    T = rng.normal(size=(50, 3))
    q = rng.normal(size=3)
    vip = vip_scores(W, T, q)
    assert np.sum(vip**2) == pytest.approx(5.0)  # Σ VIP² = number of predictors


def test_train_models_end_to_end():
    X, y = linear_data(n=240, noise=0.1)
    fm = make_fm(X, y)
    run = train_models(fm, ModellingConfig())
    assert not run.errors
    labels = run.split.labels
    assert set(labels) <= {TRAIN, VALIDATION, TEST, PURGED}
    assert len(run.models) == 4 and all(m.ok for m in run.models)
    for m in run.models:
        # Every model is evaluated on exactly the same validation and test samples.
        assert m.metrics.loc[TEST, "n"] == (labels == TEST).sum()
        assert m.metrics.loc[VALIDATION, "n"] == (labels == VALIDATION).sum()
        assert m.metric(TEST, "r2") > 0.95
    mlr = run.models[0]
    np.testing.assert_allclose(mlr.coefficients.to_numpy(), [2.0, -0.3, 4.0], rtol=0.03)
    pls = next(m for m in run.models if m.spec.kind is ModelKind.PLS)
    assert pls.cv is not None and pls.details is not None
    assert len(pls.details.vip) == 3
    assert set(pls.details.scores["split"]) == {TRAIN, VALIDATION, TEST}
    pcr = next(m for m in run.models if m.spec.kind is ModelKind.PCR)
    assert pcr.details.x_variance_explained.sum() <= 1.0 + 1e-12
    table = run.comparison()
    assert "test_rmse" not in table.columns  # hidden until the user asks for the final test
    assert "test_rmse" in run.comparison(include_test=True).columns


def test_test_split_cannot_influence_training():
    X, y = linear_data(n=200)
    a = train_models(make_fm(X, y), ModellingConfig())
    test = a.split.mask(TEST)
    X2, y2 = X.copy(), y.copy()
    X2[test] *= 50.0  # wildly different final-test inputs and targets
    y2[test] = -1000.0
    b = train_models(make_fm(X2, y2), ModellingConfig())
    for ma, mb in zip(a.models, b.models, strict=True):
        assert ma.hyperparameters == mb.hyperparameters
        np.testing.assert_allclose(ma.coefficients, mb.coefficients)
        assert ma.cv_rmse == pytest.approx(mb.cv_rmse)
        pd.testing.assert_series_equal(ma.metrics.loc[TRAIN], mb.metrics.loc[TRAIN])
    pd.testing.assert_frame_equal(a.train_ranges, b.train_ranges)


def test_invalid_situations_reported():
    X, y = linear_data(n=12)
    run = train_models(make_fm(X, y), ModellingConfig())
    assert any("training samples" in e for e in run.errors)  # 12 × 0.6 < 10
    X, y = linear_data(n=100)
    run = train_models(make_fm(X, np.ones(100)), ModellingConfig())
    assert any("target is constant" in e for e in run.errors)
    Xc = X.copy()
    Xc[:, 1] = 7.0
    run = train_models(make_fm(Xc, y), ModellingConfig())
    assert any("Constant" in e and "x2" in e for e in run.errors)
    # Too many components requested.
    cfg = ModellingConfig(models=(ModelSpec(ModelKind.PLS, n_components=5),))
    run = train_models(make_fm(X, y), cfg)
    assert not run.models[0].ok and "at most 3" in run.models[0].error
    # Impossible CV folds: tuned models report it, MLR still trains.
    cfg = ModellingConfig(split=SplitConfig(cv_folds=20))
    run = train_models(make_fm(X[:40], y[:40]), cfg)
    kinds = {m.spec.kind: m for m in run.models}
    assert kinds[ModelKind.MLR].ok
    assert not kinds[ModelKind.PLS].ok and "components" in kinds[ModelKind.PLS].error
    assert any(p.severity != ERROR and "Cross-validation" in p.message for p in run.problems)


def test_mlr_with_more_predictors_than_samples():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 25))
    y = X[:, 0] + rng.normal(0, 0.1, 30)
    cfg = ModellingConfig(models=(ModelSpec(ModelKind.MLR), ModelSpec(ModelKind.PLS)))
    run = train_models(make_fm(X, y), cfg)
    mlr, pls = run.models
    assert not mlr.ok and "more training samples" in mlr.error
    assert pls.ok  # PLS copes with p > n
