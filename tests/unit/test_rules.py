from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inferential_studio.core.rules import (
    RuleError,
    apply_min_duration,
    evaluate_group,
    evaluate_rule,
    mask_to_intervals,
    split_runs_at_gaps,
)
from inferential_studio.core.schemas import (
    Combinator,
    Condition,
    ConditionGroup,
    ExclusionRule,
    Project,
    RuleOperator,
    SchemaError,
    from_jsonable,
    to_jsonable,
)


def frame(n=10, **cols):
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(cols, index=idx, dtype=float)


def cond(var, op, v=None, v2=None):
    return Condition(var, RuleOperator(op), v, v2)


def group(*items, combinator="and"):
    return ConditionGroup(items, Combinator(combinator))


DF = frame(6, F=[10, 60, np.nan, 40, 70, 30], P=[3.5, 2.0, 3.2, 3.1, 1.0, 4.0])
TEXT = pd.DataFrame(
    {"F": [False, False, True, False, False, False], "P": [False] * 6}, index=DF.index
)


@pytest.mark.parametrize(
    "op, v, v2, expected",
    [
        ("<", 50, None, [1, 0, 0, 1, 0, 1]),
        ("<=", 40, None, [1, 0, 0, 1, 0, 1]),
        (">", 40, None, [0, 1, 0, 0, 1, 0]),
        (">=", 40, None, [0, 1, 0, 1, 1, 0]),
        ("==", 40, None, [0, 0, 0, 1, 0, 0]),
        ("!=", 40, None, [1, 1, 0, 0, 1, 1]),  # missing value never satisfies a comparison
        ("between", 30, 60, [0, 1, 0, 1, 0, 1]),
        ("outside", 30, 60, [1, 0, 0, 0, 1, 0]),
        ("is missing", None, None, [0, 0, 1, 0, 0, 0]),
        ("is invalid", None, None, [0, 0, 1, 0, 0, 0]),
    ],
)
def test_operators(op, v, v2, expected):
    out = evaluate_group(group(cond("F", op, v, v2)), DF, TEXT)
    assert out.astype(int).tolist() == expected


def test_and_or_and_nesting():
    low_flow = cond("F", "<", 50)
    high_p = cond("P", ">", 3)
    assert evaluate_group(group(low_flow, high_p), DF, TEXT).astype(int).tolist() == [
        1,
        0,
        0,
        1,
        0,
        1,
    ]
    assert evaluate_group(group(low_flow, high_p, combinator="or"), DF, TEXT).astype(
        int
    ).tolist() == [1, 0, 1, 1, 0, 1]
    nested = group(
        cond("F", ">", 50), group(cond("P", "<", 1.5), cond("P", ">", 3.9), combinator="or")
    )
    assert evaluate_group(nested, DF, TEXT).astype(int).tolist() == [0, 0, 0, 0, 1, 0]


def test_unknown_variable_is_reported():
    with pytest.raises(RuleError, match="not in the loaded"):
        evaluate_group(group(cond("Nope", "<", 1)), DF, TEXT)


def test_invalid_conditions_rejected():
    with pytest.raises(SchemaError):
        Condition("F", RuleOperator.BETWEEN, 5, 1)
    with pytest.raises(SchemaError):
        Condition("F", RuleOperator.LT)
    with pytest.raises(SchemaError):
        Condition("F", RuleOperator.IS_MISSING, 3)
    with pytest.raises(SchemaError):
        Condition("F", "contains", 3)  # not a whitelisted operator


def test_rule_text_cannot_execute_code():
    """Rules are data: a malicious string is just an unknown operator or variable name."""
    with pytest.raises(SchemaError):
        Condition("F", "__import__('os').system('x')", 1)
    rule = ExclusionRule("r", group(cond("__import__('os')", "<", 1)))
    with pytest.raises(RuleError):
        evaluate_rule(rule, DF)


def test_min_duration_counts_sample_intervals():
    df = frame(12, F=[0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1])
    rule = ExclusionRule("below", group(cond("F", ">", 0.5)), min_duration_s=300)
    # Run of five 1-minute samples lasts 5 min (kept); runs of two do not.
    assert evaluate_rule(rule, df).astype(int).tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]


def test_gap_breaks_a_run():
    idx = pd.DatetimeIndex(
        [
            "2026-01-01 00:00",
            "2026-01-01 00:01",
            "2026-01-01 00:02",
            "2026-01-01 01:00",
            "2026-01-01 01:01",
            "2026-01-01 01:02",
        ],
        tz="UTC",
    )
    mask = np.ones(6, dtype=bool)
    starts, ends = split_runs_at_gaps(mask, idx, 2 * 60 * 10**9)
    assert starts.tolist() == [0, 3] and ends.tolist() == [2, 5]
    # 3 + 3 minutes either side of a 58-minute gap is not a 5-minute run.
    assert not apply_min_duration(mask, idx, 300).any()
    assert apply_min_duration(mask, idx, 180).all()


def test_mask_to_intervals():
    df = frame(6, F=[0, 1, 1, 0, 1, 0])
    out = mask_to_intervals(df["F"].to_numpy() > 0, df.index)
    assert out == [(df.index[1], df.index[2]), (df.index[4], df.index[4])]


def test_rule_json_roundtrip_and_project_validation():
    rule = ExclusionRule(
        "low load",
        group(cond("F", "<", 50), group(cond("P", ">", 3), combinator="or")),
        min_duration_s=300,
    )
    assert from_jsonable(ExclusionRule, to_jsonable(rule)) == rule
    assert rule.describe() == "(F < 50) AND (P > 3) for at least 5 min"
    from inferential_studio.core.schemas import DataSource, SourceKind, TimestampSpec, VariableMeta

    src = DataSource(
        source_id="p", kind=SourceKind.PROCESS, path="x.csv", timestamp=TimestampSpec(column="t")
    )
    with pytest.raises(SchemaError, match="unknown variable"):
        Project(sources=[src], variables=[VariableMeta("F", "p", "F")], rules=[rule])
