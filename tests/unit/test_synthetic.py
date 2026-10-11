from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inferential_studio.core.synthetic import (
    LAB_TIME_FORMAT,
    PROCESS_TIME_FORMAT,
    SyntheticConfig,
    _first_order_lag,
    create_example,
    generate_distillation_dataset,
)
from inferential_studio.core.tasks import CancelToken, TaskCancelled


def test_deterministic_for_same_seed():
    cfg = SyntheticConfig(days=4, seed=11)
    a, b = generate_distillation_dataset(cfg), generate_distillation_dataset(cfg)
    pd.testing.assert_frame_equal(a.process, b.process)
    pd.testing.assert_frame_equal(a.lab, b.lab)
    c = generate_distillation_dataset(SyntheticConfig(days=4, seed=12))
    assert not a.process.equals(c.process)


def test_process_grid_is_regular_utc(small_dataset):
    idx = small_dataset.process.index
    assert str(idx.tz) == "UTC"
    assert idx.is_monotonic_increasing and idx.is_unique
    assert (idx[1:] - idx[:-1] == pd.Timedelta("1min")).all()
    assert len(idx) == 6 * 24 * 60


def test_lab_file_has_one_timestamp_and_sample_ids(small_dataset):
    lab = small_dataset.lab
    assert list(lab.columns) == ["Sample_Timestamp", "Product_Composition", "Sample_ID"]
    assert lab["Sample_ID"].is_unique
    assert lab["Sample_Timestamp"].isin(small_dataset.process.index).all()


def test_no_lab_samples_during_shutdowns(small_dataset):
    times = small_dataset.lab["Sample_Timestamp"]
    for start, end in small_dataset.truth["events_utc"]["shutdowns"]:
        assert not times.between(pd.Timestamp(start), pd.Timestamp(end)).any()


def test_lab_values_match_truth_within_noise_except_gross_error(small_dataset):
    ds = small_dataset
    lab = ds.lab.set_index("Sample_ID")
    truth_at_sample = ds.target_truth.reindex(lab["Sample_Timestamp"]).to_numpy()
    error = lab["Product_Composition"].to_numpy() - truth_at_sample
    bad = lab.index.isin(ds.truth["bad_lab_sample_ids"])
    assert bad.sum() == 1
    assert np.abs(error[~bad]).max() < 6 * ds.truth["config"]["lab_noise_sd"]
    assert np.abs(error[bad]).max() > 0.05


def test_target_follows_documented_dynamics_and_deadtime(small_dataset):
    """Rebuild the target from measured inputs with the documented model; only the true deadtime fits."""
    ds = small_dataset
    model = ds.truth["target_model"]
    d, c, ref = model["deadtime_min"], model["coefficients"], model["reference_point"]

    # Use the period before the first shutdown; repair the injected measurement artefacts.
    first_shutdown = pd.Timestamp(ds.truth["events_utc"]["shutdowns"][0][0])
    p = ds.process.loc[: first_shutdown - pd.Timedelta("1min")].copy()
    spikes = pd.DatetimeIndex(ds.truth["events_utc"]["pressure_spikes"])
    p.loc[p.index.intersection(spikes), "Column_Pressure"] = np.nan
    p = p.interpolate(limit_direction="both")

    u = (
        c["intercept"]
        + c["Tray_15_Temp"] * (p["Tray_15_Temp"] - ref["Tray_15_Temp"])
        + c["Column_Pressure"] * (p["Column_Pressure"] - ref["Column_Pressure"])
        + c["Feed_Temp"] * (p["Feed_Temp"] - ref["Feed_Temp"])
    ).to_numpy()
    lagged = _first_order_lag(u, model["time_constant_min"])
    y = ds.target_truth.loc[p.index].to_numpy()

    def max_error(delay: int) -> float:
        k = np.arange(d + 10, len(u))
        return float(np.abs(y[k] - lagged[k - delay]).max())

    assert max_error(d) < 5e-5  # rounding of exported values + repaired spikes
    for wrong in (d - 5, d + 5, 0):
        assert max_error(wrong) > 5 * max_error(d)


def test_injected_artefacts_are_present(small_dataset):
    ds, ev = small_dataset, small_dataset.truth["events_utc"]
    p = ds.process
    spikes = pd.DatetimeIndex(ev["pressure_spikes"])
    assert (p.loc[spikes, "Column_Pressure"] - 3.15).abs().min() > 0.2
    s, e = (pd.Timestamp(t) for t in ev["top_temp_flatline"])
    assert p.loc[s : e - pd.Timedelta("1min"), "Top_Temp"].nunique() == 1
    s, e = (pd.Timestamp(t) for t in ev["feed_temp_gap"])
    assert p.loc[s : e - pd.Timedelta("1min"), "Feed_Temp"].isna().all()
    s, e = (pd.Timestamp(t) for t in ev["low_load"])
    assert p.loc[s + pd.Timedelta("1h") : e - pd.Timedelta("1h"), "Feed_Flow"].max() < 50


def test_progress_and_cancellation():
    seen: list[float] = []
    generate_distillation_dataset(SyntheticConfig(days=3), progress=lambda f, m: seen.append(f))
    assert seen == sorted(seen) and seen[-1] == 1.0

    token = CancelToken()
    token.cancel()
    with pytest.raises(TaskCancelled):
        generate_distillation_dataset(SyntheticConfig(days=3), cancel=token)


def test_written_files_roundtrip_through_local_time(tmp_path):
    cfg = SyntheticConfig(days=3, seed=5)
    paths = create_example(tmp_path, cfg)
    ds = generate_distillation_dataset(cfg)

    csv = pd.read_csv(paths["process"])
    parsed = (
        pd.to_datetime(csv["Timestamp"], format=PROCESS_TIME_FORMAT)
        .dt.tz_localize(cfg.timezone)
        .dt.tz_convert("UTC")
    )
    assert (pd.DatetimeIndex(parsed) == ds.process.index).all()
    np.testing.assert_allclose(
        csv["Tray_15_Temp"].to_numpy(), ds.process["Tray_15_Temp"].to_numpy()
    )

    lab = pd.read_excel(paths["lab"], sheet_name="Lab")
    sample = pd.to_datetime(lab["Sample_Timestamp"], format=LAB_TIME_FORMAT).dt.tz_localize(
        cfg.timezone
    )
    assert (sample.dt.tz_convert("UTC").to_numpy() == ds.lab["Sample_Timestamp"].to_numpy()).all()
