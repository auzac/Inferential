from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inferential_studio.core.quality import DetectorError, detect, run_detectors
from inferential_studio.core.schemas import DetectorConfig, DetectorKind, SchemaError, VariableMeta


def series(values, freq="1min", name="x"):
    idx = pd.date_range("2026-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(np.asarray(values, dtype=float), index=idx, name=name)


def det(kind, **kw):
    return DetectorConfig(DetectorKind(kind), ("x",), **kw)


RNG = np.random.default_rng(0)
NOISE = RNG.normal(0, 1, 2000)


def test_mad_finds_spikes_but_not_a_genuine_step():
    x = NOISE.copy()
    x[1000:] += 3.0  # genuine level change (valid process move)
    x[[300, 1500]] += 25  # instrument spikes
    s = series(x)
    flags = detect(det("hampel", threshold=6, window_s=1800), s).flags
    assert set(np.flatnonzero(flags)) == {300, 1500}  # step change survives the local median


def test_zscore_mad_iqr_on_outliers():
    x = NOISE.copy()
    x[[10, 20]] = [15, -15]
    s = series(x)
    for kind, thr in (("zscore", 5), ("mad", 6), ("iqr", 3)):
        res = detect(det(kind, threshold=thr), s)
        assert set(np.flatnonzero(res.flags)) == {10, 20}, kind
        assert {e.start for e in res.events} == {s.index[10], s.index[20]}


def test_constant_and_short_series_flag_nothing():
    s = series([5.0] * 100)
    for kind in ("zscore", "mad", "iqr"):
        res = detect(det(kind, threshold=3), s)
        assert not res.flags.any() and res.note
    assert not detect(det("hampel", threshold=3, window_s=600), s).flags.any()
    assert not detect(det("zscore", threshold=3), series([1.0, np.nan])).flags.any()
    assert not detect(det("mad", threshold=3), series([np.nan] * 10)).flags.any()


def test_fit_period_controls_statistics():
    x = np.r_[RNG.normal(0, 1, 500), RNG.normal(10, 1, 500)]
    s = series(x)
    whole = detect(det("zscore", threshold=4), s)
    early = detect(det("zscore", threshold=4, fit_end=s.index[499].to_pydatetime()), s)
    assert not whole.flags.any()
    assert early.flags[500:].mean() > 0.9  # judged against the early period only


def test_rate_of_change_ignores_gaps():
    s = series([1, 1, 1, 10, 10, 10, np.nan, np.nan, np.nan, 50, 50])
    res = detect(det("rate_of_change", rate_per_min=5), s)
    assert np.flatnonzero(res.flags).tolist() == [3]  # the jump; not the step across the gap


def test_flatline_detects_stuck_value_and_respects_duration():
    x = NOISE[:200].copy()
    x[50:80] = 7.0  # 30 minutes stuck
    x[120:125] = 7.0  # 5 minutes: too short
    res = detect(det("flatline", window_s=20 * 60), series(x))
    assert np.flatnonzero(res.flags).tolist() == list(range(50, 80))
    assert len(res.events) == 1 and res.events[0].score == pytest.approx(30)


def test_limits_from_detector_or_variable():
    s = series([1, 5, 9, 11, -1])
    assert np.flatnonzero(detect(det("limits", low=0, high=10), s).flags).tolist() == [3, 4]
    meta = VariableMeta("x", "p", "x", eng_low=2, eng_high=8)
    assert np.flatnonzero(detect(det("limits"), s, meta=meta).flags).tolist() == [0, 2, 3, 4]
    with pytest.raises(DetectorError):
        detect(det("limits"), s)


def test_invalid_text_flags_status_cells():
    s = series([1, np.nan, 3])
    mask = pd.Series([False, True, False], index=s.index)
    assert detect(det("invalid_text"), s, text_mask=mask).flags.tolist() == [False, True, False]


def test_all_nan_and_config_validation():
    with pytest.raises(SchemaError):
        DetectorConfig(DetectorKind.HAMPEL, ("x",), threshold=3)  # no window
    with pytest.raises(SchemaError):
        DetectorConfig(DetectorKind.ZSCORE, ("x",), threshold=-1)
    with pytest.raises(SchemaError):
        DetectorConfig(DetectorKind.MAD, (), threshold=3)


def test_run_detectors_skips_disabled_and_missing():
    s = series(NOISE[:100])
    on = det("mad", threshold=3)
    off = DetectorConfig(DetectorKind.ZSCORE, ("x", "missing"), threshold=3, enabled=False)
    out = run_detectors([on, off], {"x": s}, {}, {})
    assert list(out) == [(on.detector_id, "x")]


def test_synthetic_ground_truth_recovered(small_dataset):
    ds = small_dataset
    spikes = pd.DatetimeIndex(ds.truth["events_utc"]["pressure_spikes"])
    s = ds.process["Column_Pressure"]
    running = ds.process["Feed_Flow"] > 60
    res = detect(
        DetectorConfig(
            DetectorKind.HAMPEL,
            ("Column_Pressure",),
            threshold=4,
            window_s=1800,
            min_deviation=0.05,
        ),
        s,
    )
    flagged = s.index[res.flags]
    found = spikes[running.reindex(spikes).to_numpy()]
    assert set(found) <= set(flagged)  # every spike during normal running is flagged
    assert len(set(flagged) - set(spikes)) <= 2  # and almost nothing else


def test_example_files_truth_flatline():
    truth = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthetic_distillation/ground_truth.json"
        ).read_text()
    )
    start, end = (pd.Timestamp(t) for t in truth["events_utc"]["top_temp_flatline"])
    assert end - start == pd.Timedelta("6h")
