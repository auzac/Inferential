from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inferential_studio.core.features import (
    GAP,
    LOW_COVERAGE,
    NO_DATA,
    OK,
    REGIME,
    TOO_FEW,
    Y_EXCLUDED,
    Y_MISSING,
    AlignmentError,
    TargetSamples,
    aggregate_windows,
    build_feature_matrix,
    data_span_s,
    lag_scan,
    process_signal,
)
from inferential_studio.core.schemas import (
    Aggregation,
    AlignmentConfig,
    FeatureSpec,
    FilterKind,
    FilterSpec,
    WindowPlacement,
)

T0 = pd.Timestamp("2026-01-01 00:00", tz="UTC")


def minutes(values, name="x", start=T0):
    idx = pd.date_range(start, periods=len(values), freq="1min", tz="UTC")
    return pd.Series(np.asarray(values, dtype=float), index=idx, name=name)


def target(times, values=None, excluded=None, name="y"):
    times = pd.DatetimeIndex(times)
    n = len(times)
    values = np.arange(n, dtype=float) if values is None else np.asarray(values, dtype=float)
    return TargetSamples(
        name,
        times,
        values,
        np.zeros(n, bool) if excluded is None else np.asarray(excluded),
        np.zeros(n, bool),
    )


def at(minute):
    return T0 + pd.Timedelta(minutes=minute)


# --------------------------------------------------------------------------- windows


@pytest.mark.parametrize(
    "agg, expected",
    [
        (Aggregation.MEAN, 6.0),  # minutes 4..8 hold values 4..8
        (Aggregation.MEDIAN, 6.0),
        (Aggregation.MIN, 4.0),
        (Aggregation.MAX, 8.0),
        (Aggregation.STD, float(np.std([4, 5, 6, 7, 8], ddof=1))),
        (Aggregation.LAST, 8.0),
    ],
)
def test_lookback_window_is_left_open_right_closed(agg, expected):
    x = minutes(range(20))  # value == minute
    spec = FeatureSpec("x", window_s=300, deadtime_s=120, aggregation=agg, min_coverage=0)
    sig = process_signal(x, spec)
    res = aggregate_windows(sig, np.array([at(10).value]), spec)
    # Window (10 − 2 − 5, 10 − 2] = (3, 8] → minutes 4, 5, 6, 7, 8.
    assert res.n_values[0] == 5
    assert res.value[0] == pytest.approx(expected)
    assert res.coverage[0] == pytest.approx(1.0)


def test_shift_and_deadtime_are_summed_once():
    spec = FeatureSpec("x", window_s=600, deadtime_s=1200, shift_s=300)
    assert spec.effective_delay_s == 1500
    assert spec.window_offsets_s == (-2100, -1500)
    assert spec.is_causal
    ahead = FeatureSpec("x", window_s=600, deadtime_s=0, shift_s=-60)
    assert not ahead.is_causal  # negative total delay reads the future


def test_centered_window_is_flagged_non_causal():
    spec = FeatureSpec("x", window_s=600, deadtime_s=120, placement=WindowPlacement.CENTERED)
    assert spec.window_offsets_s == (-420, 180)
    assert not spec.is_causal


def test_coverage_gap_and_minimum_counts():
    vals = np.arange(30, dtype=float)
    vals[10:16] = np.nan  # six missing minutes
    x = minutes(vals)
    t = np.array([at(20).value])
    spec = FeatureSpec("x", window_s=900, min_coverage=0.8)  # (5, 20]: 15 slots, 9 valid
    res = aggregate_windows(process_signal(x, spec), t, spec)
    assert res.n_values[0] == 9 and res.coverage[0] == pytest.approx(9 / 15)
    assert res.status[0] == LOW_COVERAGE and np.isnan(res.value[0])
    loose = FeatureSpec("x", window_s=900, min_coverage=0.5, max_gap_s=300)
    res = aggregate_windows(process_signal(x, loose), t, loose)
    assert res.gap_s[0] == pytest.approx(7 * 60)  # 9 → 16 minutes: seven minutes without data
    assert res.status[0] == GAP
    ok = FeatureSpec("x", window_s=900, min_coverage=0.5, max_gap_s=600)
    res = aggregate_windows(process_signal(x, ok), t, ok)
    assert res.status[0] == OK
    empty = aggregate_windows(process_signal(x, ok), np.array([at(100).value]), ok)
    assert empty.status[0] == NO_DATA
    std = FeatureSpec("x", window_s=60, aggregation=Aggregation.STD, min_coverage=0)
    assert aggregate_windows(process_signal(x, std), t, std).status[0] == TOO_FEW


def test_excluded_cells_do_not_contribute():
    x = minutes(np.ones(30))
    excluded = np.zeros(30, bool)
    excluded[15:20] = True
    x.iloc[15:20] = 1000.0  # bad values inside the exclusion
    spec = FeatureSpec("x", window_s=600, min_coverage=0.4)
    res = aggregate_windows(process_signal(x, spec, excluded), np.array([at(20).value]), spec)
    assert res.value[0] == pytest.approx(1.0)
    assert res.n_values[0] == 5


def test_regime_overlap_rejects_window():
    x = minutes(np.ones(60))
    regime = np.zeros(60, bool)
    regime[40] = True
    tgt = target([at(45), at(59)])
    spec = FeatureSpec("x", window_s=600, min_coverage=0.5)
    fm = build_feature_matrix(
        tgt, x.to_frame(), [spec], excluded=pd.DataFrame({"x": regime}, index=x.index),
        regime=regime,
    )  # fmt: skip
    assert fm.status["x"].tolist() == [REGIME, OK]
    off = build_feature_matrix(
        tgt, x.to_frame(), [spec], regime=regime,
        alignment=AlignmentConfig(reject_regime_overlap=False),
    )  # fmt: skip
    assert off.status["x"].tolist() == [OK, OK]


# --------------------------------------------------------------------------- resampling / filters


def test_resample_bins_are_labelled_at_their_end():
    x = minutes(range(10))  # minutes 0..9
    spec = FeatureSpec("x", resample_s=300, window_s=600, min_coverage=0)
    sig = process_signal(x, spec)
    s = sig.series()
    # Bins (−5, 0] → {0}; (0, 5] → {1..5}; (5, 10] → {6..9}
    assert s.index.tolist() == [at(0), at(5), at(10)]
    assert s.tolist() == [0.0, 3.0, 7.5]
    assert sig.interval_s == 300


def test_filters_are_causal():
    rng = np.random.default_rng(1)
    base = rng.normal(size=200)
    changed = base.copy()
    changed[120:] += 50.0  # future disturbance
    for f in (
        FilterSpec(FilterKind.ROLLING_MEAN, window_s=600),
        FilterSpec(FilterKind.ROLLING_MEDIAN, window_s=600),
        FilterSpec(FilterKind.EWMA, halflife_s=300),
    ):
        spec = FeatureSpec("x", filter=f)
        a = process_signal(minutes(base), spec).values
        b = process_signal(minutes(changed), spec).values
        np.testing.assert_array_equal(a[:120], b[:120])
        assert not np.allclose(a[120:], b[120:])


def test_rolling_mean_values():
    x = minutes([1, 2, 3, 4, 5])
    spec = FeatureSpec("x", filter=FilterSpec(FilterKind.ROLLING_MEAN, window_s=180))
    # Trailing (τ − 3 min, τ]: 1, 1.5, 2, 3, 4
    assert process_signal(x, spec).values.tolist() == [1, 1.5, 2, 3, 4]


def test_data_span_includes_filter_history():
    spec = FeatureSpec(
        "x", window_s=600, deadtime_s=300, filter=FilterSpec(FilterKind.ROLLING_MEAN, window_s=900)
    )
    assert data_span_s(spec) == (-1800, -300)


# --------------------------------------------------------------------------- feature matrix


def test_one_row_per_lab_sample_and_no_forward_fill():
    x = minutes(np.arange(300))
    times = [at(60), at(60), at(150), at(200), at(290)]  # a replicate pair at minute 60
    tgt = target(times, values=[1.0, 1.1, np.nan, 3.0, 4.0], excluded=[0, 0, 0, 1, 0])
    fm = build_feature_matrix(tgt, x.to_frame(), [FeatureSpec("x", window_s=600)])
    assert len(fm.y) == 5  # exactly the lab rows; nothing filled between samples
    assert list(fm.sample_time) == times
    np.testing.assert_array_equal(fm.y, [1.0, 1.1, np.nan, 3.0, 4.0])
    assert fm.y_status.tolist() == ["ok", "ok", Y_MISSING, Y_EXCLUDED, "ok"]
    assert fm.eligible.tolist() == [True, True, False, False, True]
    assert fm.X["x"].iloc[0] == fm.X["x"].iloc[1]  # replicates share their window
    summary = fm.summary()
    assert summary["eligible"] == 3 and summary["target_missing"] == 1
    assert fm.reasons()[2] == Y_MISSING


def test_no_future_leakage_in_causal_mode():
    rng = np.random.default_rng(3)
    x = minutes(rng.normal(size=600))
    times = [at(m) for m in (120, 240, 360, 480)]
    tgt = target(times)
    specs = [
        FeatureSpec("x", window_s=1800, deadtime_s=600),
        FeatureSpec(
            "x2", window_s=900, aggregation=Aggregation.LAST,
            filter=FilterSpec(FilterKind.EWMA, halflife_s=600), resample_s=300,
        ),
    ]  # fmt: skip
    frame = pd.DataFrame({"x": x, "x2": x})
    base = build_feature_matrix(tgt, frame, specs)
    for k, t in enumerate(times):
        future = frame.copy()
        future[future.index > t] = 1e6  # everything recorded after this sample changes
        out = build_feature_matrix(tgt, future, specs)
        pd.testing.assert_series_equal(out.X.iloc[k], base.X.iloc[k])
    centred = FeatureSpec("x", window_s=1800, placement=WindowPlacement.CENTERED)
    fm = build_feature_matrix(tgt, frame[["x"]], [centred])
    assert fm.non_causal == ["x"] and fm.notes


def test_deterministic():
    rng = np.random.default_rng(5)
    x = minutes(rng.normal(size=500))
    tgt = target([at(m) for m in range(100, 500, 37)])
    spec = FeatureSpec("x", window_s=1200, deadtime_s=300, aggregation=Aggregation.MEDIAN,
                       filter=FilterSpec(FilterKind.ROLLING_MEDIAN, window_s=300))  # fmt: skip
    a = build_feature_matrix(tgt, x.to_frame(), [spec])
    b = build_feature_matrix(tgt, x.to_frame(), [spec])
    pd.testing.assert_frame_equal(a.table(), b.table())


def test_errors_are_plain():
    tgt = target([at(10)])
    with pytest.raises(AlignmentError, match="at least one predictor"):
        build_feature_matrix(tgt, pd.DataFrame(), [])
    with pytest.raises(AlignmentError, match="No process data"):
        build_feature_matrix(tgt, minutes([1, 2]).to_frame(), [FeatureSpec("nope")])


def test_lag_scan_recovers_pure_deadtime():
    rng = np.random.default_rng(11)
    n = 6000
    phi = 0.97
    x = np.zeros(n)
    for i in range(1, n):  # smooth AR(1) driver
        x[i] = phi * x[i - 1] + rng.normal()
    xs = minutes(x)
    sample_min = np.arange(200, n, 47)
    y = 2.0 * x[sample_min - 20] + rng.normal(0, 0.05, sample_min.size)  # 20-minute deadtime
    tgt = target([at(int(m)) for m in sample_min], values=y)
    spec = FeatureSpec("x", window_s=60, aggregation=Aggregation.LAST)
    scan = lag_scan(tgt, xs, spec, np.arange(0, 3601, 60))
    assert scan.best_delay_s == 1200
    # The scan respects the training cutoff: later samples are ignored.
    cut = lag_scan(tgt, xs, spec, np.array([1200.0]), cutoff=at(2000))
    assert cut.n_pairs[0] == int((sample_min <= 2000).sum())


def test_synthetic_column_deadtime_recovered():
    """Truth: 20 min deadtime then a 15 min first-order lag. A 30-min mean window has its centre
    of mass 15 min before its end, so the best-fitting window deadtime should be close to 20 min."""
    from inferential_studio.core.synthetic import SyntheticConfig, generate_distillation_dataset

    ds = generate_distillation_dataset(SyntheticConfig(days=28, seed=7))
    lab = ds.lab[~ds.lab["Sample_ID"].isin(ds.truth["bad_lab_sample_ids"])]
    tgt = target(pd.DatetimeIndex(lab["Sample_Timestamp"]), lab["Product_Composition"])
    drivers = ["Tray_15_Temp", "Column_Pressure", "Feed_Temp"]
    rmse = {}
    for d in range(0, 3601, 120):
        specs = [FeatureSpec(v, window_s=1800, deadtime_s=d, min_coverage=0.5) for v in drivers]
        _, X, y = build_feature_matrix(tgt, ds.process, specs).eligible_data()
        A = np.c_[np.ones(len(y)), X.to_numpy()]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        rmse[d] = float(np.sqrt(np.mean((y - A @ coef) ** 2)))
    best = min(rmse, key=rmse.get)
    assert 12 * 60 <= best <= 28 * 60
    assert rmse[best] < 0.5 * rmse[0]


@pytest.mark.parametrize("agg", [Aggregation.MEDIAN, Aggregation.STD, Aggregation.MEAN,
                                 Aggregation.MIN, Aggregation.MAX, Aggregation.LAST])  # fmt: skip
def test_vectorised_windows_match_direct_computation(agg):
    rng = np.random.default_rng(9)
    n = 3000
    keep = rng.random(n) > 0.3  # irregular sampling: windows hold different numbers of values
    idx = pd.date_range(T0, periods=n, freq="1min")[keep]
    x = pd.Series(rng.normal(size=keep.sum()), index=idx, name="x")
    spec = FeatureSpec("x", window_s=1200, deadtime_s=300, aggregation=agg, min_coverage=0)
    times = pd.date_range(T0 + pd.Timedelta(hours=1), periods=150, freq="17min")
    res = aggregate_windows(process_signal(x, spec), times.as_unit("ns").asi8, spec)
    fn = {Aggregation.MEDIAN: np.median, Aggregation.MEAN: np.mean, Aggregation.MIN: np.min,
          Aggregation.MAX: np.max, Aggregation.LAST: lambda v: v[-1],
          Aggregation.STD: lambda v: v.std(ddof=1) if v.size > 1 else np.nan}[agg]  # fmt: skip
    for k, t in enumerate(times):
        lo, hi = t - pd.Timedelta(seconds=1500), t - pd.Timedelta(seconds=300)
        window = x[(x.index > lo) & (x.index <= hi)].to_numpy()
        expected = fn(window) if window.size else np.nan
        assert res.value[k] == pytest.approx(expected, rel=1e-12, nan_ok=True)
