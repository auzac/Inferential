from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from inferential_studio.core.features import TargetSamples, build_feature_matrix
from inferential_studio.core.models import train_models
from inferential_studio.core.schemas import (
    Aggregation,
    FeatureSpec,
    FilterKind,
    FilterSpec,
    WindowPlacement,
)
from inferential_studio.core.synthetic import build_sample_project
from inferential_studio.core.what_if import (
    StepInput,
    StepModel,
    default_timing,
    respond,
    simulate,
    step_model,
)


@pytest.fixture(scope="module")
def run(small_dataset):
    ds = small_dataset
    project = build_sample_project(ds)
    lab = ds.lab[~ds.lab["Sample_ID"].isin(ds.truth["bad_lab_sample_ids"])]
    n = len(lab)
    target = TargetSamples(
        "Product_Composition", pd.DatetimeIndex(lab["Sample_Timestamp"]),
        lab["Product_Composition"].to_numpy(), np.zeros(n, bool), np.zeros(n, bool),
    )  # fmt: skip
    return train_models(build_feature_matrix(target, ds.process, list(project.features)),
                        project.modelling)  # fmt: skip


def one_input(spec: FeatureSpec, b: float = 2.0) -> StepModel:
    return StepModel("y", 1.0, [StepInput("x", b, 10.0, 0.0, 20.0, "degC", spec)])


def test_deadtime_then_window_ramp_matches_the_textbook_response():
    D, W, T, X, b = 600.0, 1800.0, 900.0, 3.0, 2.0  # 10 min deadtime, 30 min mean, step at 15 min
    m = one_input(FeatureSpec("x", window_s=W, deadtime_s=D), b)
    res = simulate(m, {"x": X}, T, 7200.0, dt_s=60.0)
    base = 1.0 + b * 10.0
    assert res.baseline == pytest.approx(base) and res.final_change == pytest.approx(b * X)
    # Discrete reference: the window (t−D−W, t−D] holds W/dt one-minute samples; those at or
    # after the step carry +X.
    hi = res.t_s - D
    stepped = np.clip(np.floor((hi - T) / 60.0) + 1, 0, W / 60.0)
    np.testing.assert_allclose(res.predicted, base + b * X * stepped / (W / 60.0), atol=1e-9)
    assert res.first_response_s == pytest.approx(D)  # nothing moves during the deadtime
    assert D < res.settled_s <= D + W  # 95 % within deadtime + window
    assert res.outside == [] and res.notes == []
    x = res.inputs["x"]
    assert (x[res.t_s < T] == 10.0).all() and (x[res.t_s >= T] == 13.0).all()


def test_ewma_filter_adds_a_first_order_lag():
    h = 600.0  # 10-minute half-life, then the last value of a 1-minute window
    spec = FeatureSpec("x", window_s=60.0, aggregation="last",
                       filter=FilterSpec(FilterKind.EWMA, halflife_s=h))  # fmt: skip
    res = simulate(one_input(spec, 1.0), {"x": 1.0}, 600.0, 14400.0, dt_s=60.0)
    after = res.t_s >= 600.0
    k = (res.t_s[after] - 600.0) / 60.0  # samples since the step (the step sample counts)
    change = res.predicted[after] - res.baseline
    np.testing.assert_allclose(change, 1 - 0.5 ** ((k + 1) * 60.0 / h), atol=1e-5)
    assert res.settled_s == pytest.approx(h * np.log2(20) - 60, abs=61)  # 95 % ≈ 4.3 half-lives


def test_centred_window_responds_before_the_step_and_says_so():
    spec = FeatureSpec("x", window_s=1800.0, placement=WindowPlacement.CENTERED)
    res = simulate(one_input(spec), {"x": 1.0}, 3600.0, 7200.0, dt_s=60.0)
    assert res.first_response_s < 0
    assert any("before the step" in n for n in res.notes)


def test_steps_outside_the_training_range_are_reported_and_inputs_superpose():
    spec = FeatureSpec("x", window_s=600.0)
    m = StepModel("y", 0.0, [StepInput("x", 1.0, 10.0, 0.0, 20.0, None, spec),
                             StepInput("z", -0.5, 5.0, 4.0, 6.0, None, replace(spec, variable="z"))])  # fmt: skip
    res = simulate(m, {"x": 2.0, "z": 3.0}, 600.0, 3600.0, dt_s=60.0)
    assert res.outside == ["z"]  # 5 + 3 = 8 > 6
    assert res.final_change == pytest.approx(2.0 - 1.5)
    assert res.predicted[-1] - res.baseline == pytest.approx(0.5)
    with pytest.raises(ValueError):
        simulate(m, {}, 5000.0, 3600.0)


def test_every_trained_model_settles_at_coefficient_times_step(run):
    for model in run.models:
        m = step_model(run, model)
        step_at, length = default_timing(m)
        assert step_at < length and length >= step_at + m.response_span_s()
        steps = {i.name: 0.1 * i.span for i in m.inputs[:2]}
        res = simulate(m, steps, step_at, length)
        expected = sum(i.coefficient * steps.get(i.name, 0.0) for i in m.inputs)
        assert res.predicted[-1] - res.baseline == pytest.approx(expected, rel=1e-9, abs=1e-12)
        before = res.t_s < step_at
        np.testing.assert_allclose(res.predicted[before], res.baseline, rtol=1e-12)
        assert res.settled_s is not None


def test_std_feature_is_zero_at_steady_state_and_its_step_fades():
    spec = FeatureSpec("x", window_s=600.0)
    std = StepInput("s", 3.0, 0.5, 0.1, 1.0, None,
                    replace(spec, variable="s", aggregation=Aggregation.STD))  # fmt: skip
    m = StepModel("y", 1.0, [StepInput("x", 2.0, 10.0, 5.0, 15.0, None, spec), std])
    res = simulate(m, {"x": 1.0}, 900.0, 3600.0, dt_s=60.0)
    assert res.baseline == pytest.approx(1.0 + 2.0 * 10.0)  # the std feature reads 0, not 0.5
    assert res.first_response_s == pytest.approx(0.0) and res.settled_s is not None
    res = simulate(m, {"s": 0.2}, 900.0, 3600.0, dt_s=60.0)
    assert res.final_change == 0.0 and res.predicted[-1] == pytest.approx(res.baseline)


def test_live_schedule_applies_each_change_from_its_time():
    m = one_input(FeatureSpec("x", window_s=60.0, aggregation="last"), b=2.0)
    t = np.arange(0.0, 600.0, 60.0)
    y = respond(m, [(120.0, {"x": 1.0}), (300.0, {"x": -1.0})], t, 60.0)
    np.testing.assert_allclose(y, 21.0 + 2.0 * np.array([0, 0, 1, 1, 1, -1, -1, -1, -1, -1]))
