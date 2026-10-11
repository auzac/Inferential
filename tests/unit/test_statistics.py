from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inferential_studio.core.statistics import (
    compute_eda,
    correlation_matrix,
    coverage,
    hierarchical_order,
    histogram,
    pair_counts,
    preliminary_xy_correlation,
    summary_statistics,
)


def frame(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC", name="time")
    return pd.DataFrame(cols, index=idx, dtype=float)


def test_summary_matches_hand_calculation():
    df = frame(a=[1.0, 2.0, 3.0, 4.0, np.nan, 10.0])
    mask = pd.DataFrame({"a": [False] * 4 + [True, False]}, index=df.index)
    s = summary_statistics(df, mask).loc["a"]
    valid = np.array([1, 2, 3, 4, 10.0])
    assert s["rows"] == 6 and s["valid"] == 5
    assert s["valid_pct"] == pytest.approx(500 / 6)
    assert s["invalid_pct"] == pytest.approx(100 / 6) and s["missing_pct"] == pytest.approx(0)
    assert s["mean"] == pytest.approx(4.0) and s["median"] == 3.0
    assert s["std"] == pytest.approx(valid.std(ddof=1))
    assert s["variance"] == pytest.approx(valid.var(ddof=1))
    assert (s["q1"], s["q3"]) == (2.0, 4.0)
    assert s["cv"] == pytest.approx(valid.std(ddof=1) / 4.0)
    assert s["skewness"] > 0


def test_cv_omitted_when_values_cross_zero_and_constant_columns():
    df = frame(t=[-1.0, 0.5, 2.0, 3.0], c=[5.0, 5.0, 5.0, 5.0], empty=[np.nan] * 4)
    s = summary_statistics(df)
    assert np.isnan(s.loc["t", "cv"])
    assert s.loc["c", "std"] == 0 and s.loc["c", "cv"] == 0
    assert np.isnan(s.loc["c", "skewness"]) and np.isnan(s.loc["c", "kurtosis"])
    assert s.loc["empty", "valid"] == 0 and s.loc["empty", "missing_pct"] == 100


def test_correlations_pairwise_complete():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = 2 * x + 1
    z = rng.normal(size=200)
    y_gap = y.copy()
    y_gap[:50] = np.nan
    df = frame(x=x, y=y_gap, z=z, w=-(x**3))
    p = correlation_matrix(df)
    assert p.loc["x", "y"] == pytest.approx(1.0)
    assert abs(p.loc["x", "z"]) < 0.2
    s = correlation_matrix(df, "spearman")
    assert s.loc["x", "w"] == pytest.approx(-1.0)  # monotonic but not linear
    assert p.loc["x", "w"] > -1.0
    assert pair_counts(df).loc["x", "y"] == 150
    with pytest.raises(ValueError):
        correlation_matrix(df, "kendall")


def test_correlation_requires_minimum_pairs():
    df = frame(a=np.arange(10.0), b=np.arange(10.0))
    assert np.isnan(correlation_matrix(df, min_periods=30).loc["a", "b"])


def test_hierarchical_order_groups_correlated_variables():
    rng = np.random.default_rng(2)
    base1, base2 = rng.normal(size=500), rng.normal(size=500)
    df = frame(
        a1=base1 + 0.05 * rng.normal(size=500),
        b1=base2 + 0.05 * rng.normal(size=500),
        a2=base1 + 0.05 * rng.normal(size=500),
        b2=base2 + 0.05 * rng.normal(size=500),
    )
    order = hierarchical_order(correlation_matrix(df))
    assert abs(order.index("a1") - order.index("a2")) == 1
    assert abs(order.index("b1") - order.index("b2")) == 1


def test_preliminary_xy_uses_only_past_or_present_values():
    idx = pd.date_range("2026-01-01", periods=1000, freq="1min", tz="UTC", name="time")
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(size=1000), index=idx)
    sample_times = idx[10:990:7] + pd.Timedelta(seconds=30)  # between process timestamps
    future_value = x.reindex(sample_times + pd.Timedelta(seconds=30)).to_numpy()
    past_value = x.reindex(sample_times - pd.Timedelta(seconds=30)).to_numpy()
    process = pd.DataFrame({"future_twin": x, "past_twin": x}, index=idx)
    # The target equals the NEXT process value: a leaky method would find r = 1 here.
    target = pd.Series(future_value, index=sample_times)
    res = preliminary_xy_correlation(process, target)
    assert abs(res.table.loc["future_twin", "pearson"]) < 0.3
    # ...while a target equal to the previous value is found exactly.
    res2 = preliminary_xy_correlation(process, pd.Series(past_value, index=sample_times))
    assert res2.table.loc["past_twin", "pearson"] == pytest.approx(1.0)
    assert res2.n_target_samples == len(sample_times)


def test_preliminary_xy_respects_tolerance():
    idx = pd.date_range("2026-01-01", periods=10, freq="1min", tz="UTC")
    process = pd.DataFrame({"x": np.arange(10.0)}, index=idx)
    target = pd.Series(
        np.arange(20.0), index=pd.date_range("2026-01-01 05:00", periods=20, freq="1min", tz="UTC")
    )
    res = preliminary_xy_correlation(process, target)
    assert res.table.loc["x", "n_pairs"] == 0  # hours after the last process value: not paired


def test_histogram_and_constant():
    counts, edges = histogram(pd.Series(np.arange(1000.0)))
    assert counts.sum() == 1000 and len(edges) == len(counts) + 1 and len(counts) <= 100
    counts, edges = histogram(pd.Series([3.0, 3.0]))
    assert counts.tolist() == [2]
    assert histogram(pd.Series([np.nan]))[0].size == 0


def test_coverage_counts_rows_and_valid_fraction():
    idx = pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC")
    a = np.ones(100)
    a[:50] = np.nan
    df = pd.DataFrame({"a": a}, index=idx)
    cov = coverage(df, n_bins=10)
    assert cov.rows.sum() == 100
    assert cov.valid_fraction["a"].iloc[0] == 0 and cov.valid_fraction["a"].iloc[-1] == 1


def test_compute_eda_on_synthetic(small_dataset):
    ds = small_dataset
    target = ds.lab.set_index("Sample_Timestamp")["Product_Composition"]
    result = compute_eda(ds.process, None, target)
    assert list(result.summary.index) == list(ds.process.columns)
    assert result.pearson.loc["Tray_15_Temp", "Tray_20_Temp"] > 0.95
    assert result.xy is not None and result.xy.table["n_pairs"].max() <= len(target)


def test_spearman_matches_pandas_with_gaps_ties_and_constants():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(size=(3000, 6)) ** 3, columns=list("abcdef"))
    df.iloc[:200, 1] = np.nan
    df.iloc[100:300, 3] = np.nan
    df.iloc[2500:, 5] = np.nan
    df["e"] = df["a"].round(1)  # many ties
    df["const"] = 5.0
    ours = correlation_matrix(df, "spearman").to_numpy()
    ref = df.corr("spearman", min_periods=30).to_numpy()
    np.testing.assert_array_equal(np.isnan(ours), np.isnan(ref))
    np.testing.assert_allclose(ours[~np.isnan(ours)], ref[~np.isnan(ref)], atol=1e-12)


def test_skew_kurtosis_match_scipy():
    from scipy import stats

    x = np.random.default_rng(3).gamma(2.0, size=1000)
    s = summary_statistics(frame(x=x)).loc["x"]
    assert s["skewness"] == pytest.approx(stats.skew(x, bias=False))
    assert s["kurtosis"] == pytest.approx(stats.kurtosis(x, bias=False))
