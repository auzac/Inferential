from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from inferential_studio.core.schemas import (
    DataSource,
    ExclusionInterval,
    FeatureSpec,
    FilterKind,
    FilterSpec,
    Project,
    SchemaError,
    SourceKind,
    TimestampSpec,
    VariableMeta,
    VariableRole,
    WindowPlacement,
    from_jsonable,
    to_jsonable,
    validate_timezone,
)

# ----------------------------------------------------------------------------- feature windows


def test_lookback_window_offsets_follow_documented_convention():
    f = FeatureSpec(variable="T", window_s=1800, deadtime_s=1200)
    assert f.effective_delay_s == 1200
    assert f.window_offsets_s == (-3000, -1200)  # (t - d - W, t - d]
    assert f.is_causal


def test_shift_is_added_once_to_deadtime():
    f = FeatureSpec(variable="T", window_s=600, deadtime_s=1200, shift_s=300)
    assert f.effective_delay_s == 1500
    assert f.window_offsets_s == (-2100, -1500)


def test_negative_shift_past_sample_time_is_non_causal():
    f = FeatureSpec(variable="T", window_s=600, deadtime_s=60, shift_s=-120)
    assert f.window_offsets_s[1] == 60
    assert not f.is_causal


def test_centered_window_is_non_causal_without_enough_delay():
    f = FeatureSpec(variable="T", window_s=600, deadtime_s=0, placement=WindowPlacement.CENTERED)
    assert f.window_offsets_s == (-300, 300)
    assert not f.is_causal


def test_centered_window_ending_before_sample_time_is_causal():
    f = FeatureSpec(variable="T", window_s=600, deadtime_s=300, placement=WindowPlacement.CENTERED)
    assert f.window_offsets_s == (-600, 0)
    assert f.is_causal


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_s": 0},
        {"window_s": -60},
        {"deadtime_s": -1},
        {"min_coverage": 1.5},
        {"min_coverage": -0.1},
        {"resample_s": 0},
        {"max_gap_s": -5},
        {"window_s": float("nan")},
        {"deadtime_s": float("inf")},
    ],
)
def test_invalid_feature_specs_are_rejected(kwargs):
    with pytest.raises(SchemaError):
        FeatureSpec(variable="T", **kwargs)


def test_filters_require_their_parameters():
    with pytest.raises(SchemaError):
        FilterSpec(kind=FilterKind.ROLLING_MEAN)
    with pytest.raises(SchemaError):
        FilterSpec(kind=FilterKind.EWMA, halflife_s=0)
    assert FilterSpec(kind=FilterKind.EWMA, halflife_s=300).is_causal


# ----------------------------------------------------------------------------- timezones & times


def test_timezone_validation():
    assert validate_timezone("Asia/Singapore") == "Asia/Singapore"
    with pytest.raises(SchemaError):
        validate_timezone("Mars/Olympus_Mons")
    with pytest.raises(SchemaError):
        TimestampSpec(column="t", source_timezone="not-a-zone")


def test_exclusion_requires_aware_times_and_normalises_to_utc():
    with pytest.raises(SchemaError, match="timezone-aware"):
        ExclusionInterval(start=datetime(2026, 1, 1), end=datetime(2026, 1, 2))
    plus8 = timezone(timedelta(hours=8))
    ex = ExclusionInterval(
        start=datetime(2026, 1, 1, 8, tzinfo=plus8), end=datetime(2026, 1, 1, 9, tzinfo=plus8)
    )
    assert ex.start == datetime(2026, 1, 1, 0, tzinfo=UTC)
    assert ex.start.utcoffset() == timedelta(0)
    assert ex.is_global


def test_exclusion_end_must_follow_start():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(SchemaError):
        ExclusionInterval(start=t, end=t)


def test_only_lab_sources_have_result_timestamps():
    with pytest.raises(SchemaError):
        DataSource(
            source_id="p",
            kind=SourceKind.PROCESS,
            path="p.csv",
            timestamp=TimestampSpec(column="t"),
            result_timestamp=TimestampSpec(column="r"),
        )


def test_engineering_limits_must_be_ordered():
    with pytest.raises(SchemaError):
        VariableMeta(name="x", source_id="s", source_column="x", eng_low=5, eng_high=1)


# ----------------------------------------------------------------------------- project integrity


def _project(**overrides) -> Project:
    src = DataSource(
        source_id="p", kind=SourceKind.PROCESS, path="p.csv", timestamp=TimestampSpec(column="t")
    )
    lab = DataSource(
        source_id="l", kind=SourceKind.LAB, path="l.csv", timestamp=TimestampSpec(column="t")
    )
    data = dict(
        sources=[src, lab],
        variables=[
            VariableMeta(name="X1", source_id="p", source_column="X1", role=VariableRole.PREDICTOR),
            VariableMeta(name="X2", source_id="p", source_column="X2", role=VariableRole.UNUSED),
            VariableMeta(name="Y", source_id="l", source_column="Y", role=VariableRole.TARGET),
        ],
        features=[FeatureSpec(variable="X1")],
    )
    data.update(overrides)
    return Project(**data)


def test_valid_project_queries():
    p = _project()
    assert p.target.name == "Y"
    assert [v.name for v in p.predictors] == ["X1"]
    assert p.source("l").kind is SourceKind.LAB


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p.variables.append(p.variables[0]), "Duplicate variable"),
        (
            lambda p: p.variables.append(VariableMeta("Y2", "l", "Y2", role=VariableRole.TARGET)),
            "exactly one target",
        ),
        (lambda p: p.features.append(FeatureSpec(variable="X2")), "not refer to a predictor"),
        (lambda p: p.variables.append(VariableMeta("Z", "nope", "Z")), "unknown source"),
        (lambda p: p.sources.append(p.sources[0]), "Duplicate source"),
    ],
)
def test_project_integrity_errors(mutate, message):
    p = _project()
    mutate(p)
    with pytest.raises(SchemaError, match=message):
        p.validate()


def test_exclusion_signals_must_exist():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(SchemaError, match="unknown signal"):
        _project(
            exclusions=[ExclusionInterval(start=t, end=t + timedelta(hours=1), signals=("Nope",))]
        )


# ----------------------------------------------------------------------------- JSON conversion


def test_json_roundtrip_preserves_project():
    t = datetime(2026, 3, 1, 12, tzinfo=UTC)
    p = _project(
        exclusions=[
            ExclusionInterval(start=t, end=t + timedelta(hours=2), signals=("X1",), note="n")
        ]
    )
    data = to_jsonable(p)
    assert data["exclusions"][0]["start"] == "2026-03-01T12:00:00+00:00"
    assert from_jsonable(Project, data) == p


def test_unknown_fields_are_rejected():
    data = to_jsonable(_project())
    data["features"][0]["surprise"] = 1
    with pytest.raises(SchemaError, match="unknown field"):
        from_jsonable(Project, data)


def test_bad_enum_value_lists_allowed_values():
    data = to_jsonable(_project())
    data["variables"][0]["role"] = "boss"
    with pytest.raises(SchemaError, match="target, predictor, unused"):
        from_jsonable(Project, data)


def test_naive_timestamp_string_is_rejected():
    data = to_jsonable(_project())
    data["created_at"] = "2026-03-01T12:00:00"
    with pytest.raises(SchemaError, match="timezone-aware"):
        from_jsonable(Project, data)


def test_wrong_scalar_types_are_rejected():
    data = to_jsonable(_project())
    data["features"][0]["window_s"] = "1800"
    with pytest.raises(SchemaError, match="expected a number"):
        from_jsonable(Project, data)


def test_missing_optional_fields_take_defaults():
    data = to_jsonable(_project())
    del data["notes"]
    del data["features"][0]["min_coverage"]
    p = from_jsonable(Project, data)
    assert p.notes == ""
    assert p.features[0].min_coverage == 0.8
