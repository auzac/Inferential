from __future__ import annotations

import pandas as pd
import pytest

from inferential_studio.core.schemas import TimestampKind, TimestampSpec
from inferential_studio.core.timestamps import (
    UTC_DTYPE,
    DateOrder,
    TimestampError,
    detect_format,
    parse_timestamps,
    spec_from_candidate,
)


def detect(values):
    return detect_format(pd.Series(values, dtype="object"))


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


# ----------------------------------------------------------------------------- detection


def test_iso_without_offset_is_unambiguous():
    d = detect(["2026-10-11 08:30:00", "2026-10-11 08:31:00.250"])
    assert not d.ambiguous and not d.has_offsets
    assert d.recommended.format == "ISO8601"


def test_iso_with_offsets_detected():
    d = detect(["2026-10-11T08:30:00+08:00", "2026-10-11T00:31:00Z", "2026-10-11T02:32:00-0100"])
    assert d.has_offsets and d.recommended is not None


def test_mixed_offsets_rejected():
    d = detect(["2026-10-11T08:30:00+08:00", "2026-10-11T00:31:00"])
    assert d.mixed_offsets and d.recommended is None and "offset" in d.message


def test_day_month_ambiguity_requires_choice():
    d = detect(["11/10/2026 08:30", "12/10/2026 09:30", "01/02/2026 10:00"])
    assert d.ambiguous
    assert d.recommended is None
    orders = {c.order for c in d.candidates}
    assert {DateOrder.DAY_FIRST, DateOrder.MONTH_FIRST} <= orders
    day_first = next(c for c in d.candidates if c.order is DateOrder.DAY_FIRST)
    assert day_first.example_parsed.startswith("2026-10-11")  # example shows the difference


@pytest.mark.parametrize(
    "values, order",
    [
        (["13/10/2026 08:30", "02/10/2026 08:30"], DateOrder.DAY_FIRST),
        (["10/13/2026 08:30", "10/02/2026 08:30"], DateOrder.MONTH_FIRST),
        (["2026/10/13 08:30:15", "2026/10/02 08:30:15"], DateOrder.YEAR_FIRST),
        (["11-Oct-2026 08:30:00", "12-Oct-2026 08:30:00"], DateOrder.NAMED_MONTH),
        (["Oct 11, 2026 08:30", "Oct 12, 2026 08:30"], DateOrder.NAMED_MONTH),
    ],
)
def test_order_resolved_by_data(values, order):
    d = detect(values)
    assert not d.ambiguous
    assert d.recommended.order is order


def test_disambiguating_value_found_in_middle_of_long_column():
    values = ["01/02/2026 00:00"] * 5000
    values[2500] = "25/02/2026 00:00"  # only one value proves day-first
    d = detect(values)
    assert not d.ambiguous and d.recommended.order is DateOrder.DAY_FIRST


def test_numeric_kinds_detected_by_range():
    assert detect(["45940.5", "45941"]).recommended.kind is TimestampKind.EXCEL_SERIAL
    assert detect(["1760140800", "1760144400"]).recommended.kind is TimestampKind.UNIX_SECONDS
    assert detect(["1760140800000"]).recommended.kind is TimestampKind.UNIX_MILLISECONDS
    unknown = detect(["1", "2", "3"])
    assert unknown.recommended is None and "not in a recognised range" in unknown.message


def test_unparseable_and_empty():
    assert detect(["garbage", "x"]).recommended is None
    assert detect(["", None]).message == "The column contains no values."


def test_native_datetime_column():
    d = detect_format(pd.Series(pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:01"])))
    assert d.recommended.kind is TimestampKind.NATIVE


# ----------------------------------------------------------------------------- parsing


def spec(**kw) -> TimestampSpec:
    return TimestampSpec(column="t", **kw)


def test_parse_naive_local_to_utc():
    r = parse_timestamps(
        pd.Series(["11/10/2026 08:30", "12/10/2026 08:30"]),
        spec(format="%d/%m/%Y %H:%M", source_timezone="Asia/Singapore"),
    )
    assert str(r.utc.dtype) == UTC_DTYPE
    assert r.utc.tolist() == [utc("2026-10-11 00:30"), utc("2026-10-12 00:30")]


def test_same_text_differs_by_chosen_order():
    values = pd.Series(["11/10/2026 08:30"])
    dmy = parse_timestamps(values, spec(format="%d/%m/%Y %H:%M"))
    mdy = parse_timestamps(values, spec(format="%m/%d/%Y %H:%M"))
    assert dmy.utc.iloc[0] == utc("2026-10-11 08:30")
    assert mdy.utc.iloc[0] == utc("2026-11-10 08:30")


def test_offsets_win_over_source_timezone():
    r = parse_timestamps(
        pd.Series(["2026-10-11T08:30:00+08:00", "2026-10-11T00:31:00.5Z"]),
        spec(source_timezone="America/New_York"),
    )
    assert r.utc.tolist() == [utc("2026-10-11 00:30"), utc("2026-10-11 00:31:00.5")]


def test_mixed_offsets_raise_when_parsing():
    with pytest.raises(TimestampError, match="mixes"):
        parse_timestamps(pd.Series(["2026-10-11T08:30:00+08:00", "2026-10-11T08:31:00"]), spec())


def test_iso_fractional_seconds_with_comma():
    r = parse_timestamps(pd.Series(["2026-10-11 08:30:00,250"]), spec())
    assert r.utc.iloc[0] == utc("2026-10-11 08:30:00.250")


def test_dst_gap_and_overlap_are_flagged_not_guessed():
    values = pd.Series(
        ["2026-03-08 01:30", "2026-03-08 02:30", "2026-11-01 01:30", "2026-11-01 03:00"]
    )
    r = parse_timestamps(values, spec(format="%Y-%m-%d %H:%M", source_timezone="America/New_York"))
    assert r.dst_nonexistent.tolist() == [False, True, False, False]
    assert r.dst_ambiguous.tolist() == [False, False, True, False]
    assert r.utc.isna().tolist() == [False, True, True, False]
    assert r.utc.iloc[0] == utc("2026-03-08 06:30")  # EST, UTC-5
    assert r.utc.iloc[3] == utc("2026-11-01 08:00")  # EST again after fall-back
    assert r.n_invalid == 2


def test_blank_and_bad_values_reported_separately():
    r = parse_timestamps(pd.Series(["2026-01-01 00:00", "", None, "31/31/2026"]), spec())
    assert r.blank.tolist() == [False, True, True, False]
    assert r.unparseable.tolist() == [False, False, False, True]
    assert r.valid.tolist() == [True, False, False, False]


@pytest.mark.parametrize(
    "system, serial, expected_local",
    [
        (1900, "45940.5", "2025-10-10 12:00"),
        (1900, "61", "1900-03-01 00:00"),
        (1904, "44478.5", "2025-10-10 12:00"),
    ],
)
def test_excel_serial_date_systems(system, serial, expected_local):
    r = parse_timestamps(
        pd.Series([serial]),
        spec(kind=TimestampKind.EXCEL_SERIAL, excel_date_system=system, source_timezone="UTC"),
    )
    assert r.utc.iloc[0] == utc(expected_local)


def test_excel_serial_float_noise_is_rounded():
    # 08:30 as a day fraction is not exactly representable in binary floating point.
    serial = 45940 + (8 * 60 + 30) / 1440
    r = parse_timestamps(
        pd.Series([repr(serial)]), spec(kind=TimestampKind.EXCEL_SERIAL, excel_date_system=1900)
    )
    assert r.utc.iloc[0] == utc("2025-10-10 08:30")


def test_excel_1900_leap_bug_region_rejected():
    r = parse_timestamps(
        pd.Series(["59"]), spec(kind=TimestampKind.EXCEL_SERIAL, excel_date_system=1900)
    )
    assert r.unparseable.iloc[0]


def test_unix_epochs_ignore_source_timezone():
    s = parse_timestamps(
        pd.Series(["1760140800"]),
        spec(kind=TimestampKind.UNIX_SECONDS, source_timezone="Asia/Tokyo"),
    )
    ms = parse_timestamps(pd.Series(["1760140800500"]), spec(kind=TimestampKind.UNIX_MILLISECONDS))
    assert s.utc.iloc[0] == utc("2025-10-11 00:00")
    assert ms.utc.iloc[0] == utc("2025-10-11 00:00:00.5")


def test_native_cells_localized():
    values = pd.Series(pd.to_datetime(["2026-10-11 08:30"]))
    r = parse_timestamps(values, spec(kind=TimestampKind.NATIVE, source_timezone="Asia/Singapore"))
    assert r.utc.iloc[0] == utc("2026-10-11 00:30")


def test_reparsing_is_deterministic():
    values = pd.Series(["02/03/2026 00:00", "02/03/2026 00:01"])
    s = spec(format="%d/%m/%Y %H:%M", source_timezone="Asia/Singapore")
    pd.testing.assert_series_equal(parse_timestamps(values, s).utc, parse_timestamps(values, s).utc)


def test_spec_from_candidate_records_user_choice():
    d = detect(["11/10/2026 08:30"])
    chosen = next(c for c in d.candidates if c.order is DateOrder.MONTH_FIRST)
    s = spec_from_candidate("Time", chosen, "Asia/Singapore")
    assert (
        s.dayfirst is False
        and s.format == "%m/%d/%Y %H:%M"
        and s.source_timezone == "Asia/Singapore"
    )
