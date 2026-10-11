from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from inferential_studio.core.exclusions import (
    EventStatus,
    category_table,
    compute_cleaning,
    interval_rows,
    summary_table,
)
from inferential_studio.core.quality import run_detectors
from inferential_studio.core.schemas import (
    Condition,
    ConditionGroup,
    DataSource,
    DetectorConfig,
    DetectorKind,
    ExclusionInterval,
    ExclusionReason,
    ExclusionRule,
    FlagDecision,
    FlagDecisionKind,
    Project,
    RuleAction,
    RuleOperator,
    SourceKind,
    TimestampSpec,
    VariableMeta,
    VariableRole,
)

IDX = pd.date_range("2026-01-01", periods=20, freq="1min", tz="UTC")


def t(i: int) -> pd.Timestamp:
    return IDX[i]


def make_project(**kw) -> Project:
    srcs = [
        DataSource(
            source_id="p",
            kind=SourceKind.PROCESS,
            path="p.csv",
            timestamp=TimestampSpec(column="t"),
        ),
        DataSource(
            source_id="l", kind=SourceKind.LAB, path="l.csv", timestamp=TimestampSpec(column="t")
        ),
    ]
    variables = [
        VariableMeta("A", "p", "A", VariableRole.PREDICTOR),
        VariableMeta("B", "p", "B", VariableRole.PREDICTOR),
        VariableMeta("Y", "l", "Y", VariableRole.TARGET),
    ]
    return Project(sources=srcs, variables=variables, **kw)


FRAME = pd.DataFrame({"A": np.arange(20.0), "B": np.ones(20)}, index=IDX)


def interval(i, j, **kw):
    return ExclusionInterval(start=t(i).to_pydatetime(), end=t(j).to_pydatetime(), **kw)


def test_closed_intervals_and_overlaps():
    rows = interval_rows(IDX, [interval(2, 4), interval(3, 6)])  # overlapping
    assert np.flatnonzero(rows).tolist() == [2, 3, 4, 5, 6]  # both endpoints included, union
    between = ExclusionInterval(
        start=(t(2) + pd.Timedelta("10s")).to_pydatetime(),
        end=(t(3) + pd.Timedelta("10s")).to_pydatetime(),
    )
    assert np.flatnonzero(interval_rows(IDX, [between])).tolist() == [3]


def test_global_vs_signal_specific_and_disabled():
    project = make_project(
        exclusions=[
            interval(0, 1, reason=ExclusionReason.SHUTDOWN),
            interval(5, 6, signals=("A",), reason=ExclusionReason.INSTRUMENT_FAULT),
            interval(10, 12, enabled=False),
        ]
    )
    st = compute_cleaning(project, FRAME)
    assert np.flatnonzero(st.regime).tolist() == [0, 1]
    assert np.flatnonzero(st.excluded["A"].to_numpy()).tolist() == [0, 1, 5, 6]
    assert np.flatnonzero(st.excluded["B"].to_numpy()).tolist() == [
        0,
        1,
    ]  # B unaffected by A's fault
    cleaned = st.cleaned(FRAME)
    assert np.isnan(cleaned["A"].iloc[5]) and cleaned["B"].iloc[5] == 1.0
    assert FRAME["A"].iloc[5] == 5.0  # original data untouched


def test_rules_exclude_or_flag():
    low = ExclusionRule("A low", ConditionGroup((Condition("A", RuleOperator.LT, 3),)))
    flag = ExclusionRule(
        "A high",
        ConditionGroup((Condition("A", RuleOperator.GT, 17),)),
        action=RuleAction.FLAG,
        signals=("A",),
    )
    off = ExclusionRule(
        "off", ConditionGroup((Condition("A", RuleOperator.GT, -1),)), enabled=False
    )
    st = compute_cleaning(make_project(rules=[low, flag, off]), FRAME)
    assert np.flatnonzero(st.regime).tolist() == [0, 1, 2]
    assert np.flatnonzero(st.pending["A"].to_numpy()).tolist() == [18, 19]
    assert not st.excluded.loc[IDX[18], "A"]
    labels = category_table(st)["reason"].tolist()
    assert labels == ["Rule: A low", "Rule flag: A high"]


def test_rule_on_missing_variable_reported_not_crashing():
    bad = ExclusionRule("uses Y", ConditionGroup((Condition("Y", RuleOperator.LT, 3),)))
    st = compute_cleaning(make_project(rules=[bad]), FRAME)
    assert bad.rule_id in st.rule_errors and not st.regime.any()


def test_detector_review_states():
    a = FRAME["A"].copy()
    a.iloc[[5, 15]] = [100, -100]
    frame = FRAME.assign(A=a)
    det = DetectorConfig(DetectorKind.LIMITS, ("A",), low=-10, high=50)
    project = make_project(detectors=[det])
    results = run_detectors(project.detectors, {"A": frame["A"]}, {}, {})
    st = compute_cleaning(project, frame, None, results)
    assert {e.status for e in st.events} == {EventStatus.PENDING}
    assert not st.excluded["A"].any() and st.pending["A"].sum() == 2  # flagged, not deleted

    ev5, ev15 = sorted((e.event for e in st.events), key=lambda e: e.start)
    project.flag_decisions = [
        FlagDecision(
            det.detector_id,
            "A",
            ev5.start.to_pydatetime(),
            ev5.end.to_pydatetime(),
            FlagDecisionKind.ACCEPTED,
        ),
        FlagDecision(
            det.detector_id,
            "A",
            ev15.start.to_pydatetime(),
            ev15.end.to_pydatetime(),
            FlagDecisionKind.REJECTED,
        ),
    ]
    st = compute_cleaning(project, frame, None, results)
    assert np.flatnonzero(st.excluded["A"].to_numpy()).tolist() == [5]
    assert not st.pending["A"].any()

    project.flag_decisions = []
    project.detectors = [replace(det, auto_exclude=True)]
    st = compute_cleaning(project, frame, None, results)
    assert np.flatnonzero(st.excluded["A"].to_numpy()).tolist() == [5, 15]
    assert {e.status for e in st.events} == {EventStatus.AUTO}


def test_target_samples_follow_intervals_and_flags():
    target = pd.Series([0.98, 0.90, 0.985], index=IDX[[2, 8, 14]], name="Y")
    det = DetectorConfig(DetectorKind.LIMITS, ("Y",), low=0.95)
    project = make_project(exclusions=[interval(13, 15)], detectors=[det])
    results = run_detectors(project.detectors, {"Y": target}, {}, {})
    st = compute_cleaning(project, FRAME, None, results, target)
    assert st.target.excluded.tolist() == [False, False, True]
    assert st.target.pending.tolist() == [False, True, False]


def test_summary_counts():
    project = make_project(exclusions=[interval(0, 4)])
    st = compute_cleaning(project, FRAME)
    summary = summary_table(st, FRAME)
    assert summary.loc["A", "excluded"] == 5 and summary.loc["A", "usable"] == 15
    assert summary.loc["A", "excluded_duration_h"] == pytest.approx(5 / 60)
