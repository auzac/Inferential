"""Import → clean → align → train → validate → save → reopen → identical results."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inferential_studio.core.exclusions import compute_cleaning
from inferential_studio.core.features import build_project_features
from inferential_studio.core.importers import load_project_data
from inferential_studio.core.models import train_models
from inferential_studio.core.persistence import load_project, save_project
from inferential_studio.core.quality import run_detectors
from inferential_studio.core.schemas import (
    DetectorConfig,
    DetectorKind,
    FlagDecision,
    FlagDecisionKind,
    ModelKind,
)
from inferential_studio.core.splits import TEST, VALIDATION

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "synthetic_distillation"


def pipeline(project, path):
    data = load_project_data(project, path)
    frame, mask = data.process_frame(project)
    target = data.target_series(project)
    series = {c: frame[c] for c in frame.columns}
    series[str(target.name)] = target
    results = run_detectors(
        project.detectors,
        series,
        {c: mask[c] for c in mask},
        {v.name: v for v in project.variables},
    )
    state = compute_cleaning(project, frame, mask, results, target)
    fm = build_project_features(project, data, frame, state)
    return data, state, results, fm, train_models(fm, project.modelling)


@pytest.fixture
def example(tmp_path):
    shutil.copytree(EXAMPLE_DIR, tmp_path / "ex")
    return tmp_path / "ex" / "sample_project.imsproj"


def test_full_pipeline_with_reviewed_lab_outlier_and_reload(example, tmp_path):
    project = load_project(example)
    # Review the gross analyzer error the way a user would: detect, then accept the flag.
    det = DetectorConfig(DetectorKind.MAD, ("Product_Composition",), threshold=6)
    project.detectors = [det]
    _data, _state, results, _fm, _run = pipeline(project, example)
    events = results[(det.detector_id, "Product_Composition")].events
    assert len(events) == 1
    ev = events[0]
    project.flag_decisions = [
        FlagDecision(det.detector_id, ev.variable, ev.start.to_pydatetime(), ev.end.to_pydatetime(),
                     FlagDecisionKind.ACCEPTED)
    ]  # fmt: skip
    project.validate()
    data, state, _results, fm, run = pipeline(project, example)
    assert fm.summary()["target_excluded"] == 1
    assert len(fm.y) == data.n_lab_samples  # one row per lab sample, Y never filled in
    assert not run.errors
    labels = run.split.labels
    for m in run.models:
        assert m.ok
        assert m.metric(VALIDATION, "r2") > 0.9  # with the outlier excluded, validation is good
        assert m.metric(TEST, "r2") > 0.9
        assert m.metrics.loc[TEST, "n"] == (labels == TEST).sum()  # identical test definition
    mlr = next(m for m in run.models if m.spec.kind is ModelKind.MLR)
    # Signs agree with the synthetic truth (hotter tray → lower purity; pressure ↑ → purity ↑).
    assert mlr.coefficients["Tray_15_Temp"] < 0 < mlr.coefficients["Column_Pressure"]

    # Save, reopen, recompute: the same split, hyperparameters and predictions.
    saved = tmp_path / "ex" / "reviewed.imsproj"
    save_project(project, saved)
    again = load_project(saved)
    assert again.flag_decisions == project.flag_decisions
    assert again.features == project.features and again.modelling == project.modelling
    _d, _s, _r, fm2, run2 = pipeline(again, saved)
    pd.testing.assert_frame_equal(fm.table(), fm2.table())
    np.testing.assert_array_equal(run.split.labels, run2.split.labels)
    for a, b in zip(run.models, run2.models, strict=True):
        assert a.hyperparameters == b.hyperparameters
        pd.testing.assert_frame_equal(a.predictions, b.predictions)


def test_regime_exclusion_removes_overlapping_windows(example):
    from inferential_studio.core.features import REGIME, Y_EXCLUDED
    from inferential_studio.core.schemas import AlignmentConfig, ExclusionInterval

    project = load_project(example)
    _data, _state, _r, fm, _run = pipeline(project, example)
    t = fm.sample_time[100]
    project.exclusions.append(
        ExclusionInterval(
            (t - pd.Timedelta(minutes=10)).to_pydatetime(),
            (t + pd.Timedelta(minutes=5)).to_pydatetime(),
        )
    )
    _data, _state, _r, fm2, _run = pipeline(project, example)
    assert fm2.y_status[100] == Y_EXCLUDED  # the sample time itself is excluded
    # Only that sample is lost: the next one is ~2 h later, beyond every window.
    assert fm2.n_eligible == fm.n_eligible - 1
    # Excluding a period 30 min before a sample hits its window (deadtime 20, window 30 min).
    t2 = fm.sample_time[150]
    project.exclusions.append(
        ExclusionInterval(
            (t2 - pd.Timedelta(minutes=31)).to_pydatetime(),
            (t2 - pd.Timedelta(minutes=29)).to_pydatetime(),
        )
    )
    _data, _state, _r, fm3, _run = pipeline(project, example)
    assert (fm3.status.iloc[150] == REGIME).all() and not fm3.eligible[150]
    project.alignment = AlignmentConfig(reject_regime_overlap=False)
    _data, _state, _r, fm4, _run = pipeline(project, example)
    assert fm4.eligible[150]  # 3 of 30 minutes missing still passes the 80 % coverage rule
    assert (fm4.coverage.iloc[150] < 1).all()
