from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inferential_studio.core.features import TargetSamples, build_feature_matrix
from inferential_studio.core.models import train_models
from inferential_studio.core.splits import TEST, TRAIN, VALIDATION
from inferential_studio.core.synthetic import build_sample_project
from inferential_studio.core.what_if import HOLD, SHIFT, evaluate, what_if_base


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


def test_unchanged_inputs_reproduce_every_models_predictions(run):
    for model in run.models:
        base = what_if_base(run, model, (TRAIN, VALIDATION))
        np.testing.assert_allclose(base.intercept + base.X @ base.coefficients, base.predicted,
                                   rtol=1e-9, atol=1e-12)  # fmt: skip
        res = evaluate(base, {})
        np.testing.assert_allclose(res.predicted, base.predicted, rtol=1e-12)
        assert not res.outside.any()  # nothing changed, nothing leaves the range
        assert TEST not in set(base.splits)  # hidden until revealed


def test_shift_moves_the_line_by_coefficient_times_offset(run):
    model = run.models[0]
    base = what_if_base(run, model, (TRAIN, VALIDATION), {model.coefficients.index[0]: "degC"})
    a, b = base.inputs[0], base.inputs[1]
    res = evaluate(base, {a.name: (SHIFT, 0.5), b.name: (SHIFT, -2.0)})
    expected = a.coefficient * 0.5 + b.coefficient * -2.0
    np.testing.assert_allclose(res.delta, expected)
    assert res.impact[a.name] == pytest.approx(a.coefficient * 0.5)
    assert base.inputs[0].units == "degC"


def test_hold_removes_that_inputs_variation(run):
    model = run.models[0]
    base = what_if_base(run, model, (TRAIN,))
    a = base.inputs[0]
    res = evaluate(base, {a.name: (HOLD, a.mean)})
    np.testing.assert_allclose(res.delta, a.coefficient * (a.mean - base.X[:, 0]))
    # Held at the training mean, the training rows' average prediction is unchanged.
    assert res.impact[a.name] == pytest.approx(0.0, abs=1e-12)
    assert not res.outside.any()


def test_leaving_the_training_range_is_reported(run):
    base = what_if_base(run, run.models[0], (TRAIN,))
    a = base.inputs[0]
    res = evaluate(base, {a.name: (SHIFT, a.span)})
    assert res.outside.any() and res.outside_inputs == [a.name]
    with pytest.raises(ValueError):
        evaluate(base, {a.name: ("scale", 2.0)})
