"""Timestamp format detection and parsing to timezone-aware UTC.

Policy (see docs/methodology.md):

* Day/month order is never guessed. :func:`detect_format` reports when both orders fit the
  data, and the importer refuses to continue until the user chooses.
* Values without a UTC offset are local times in the source timezone chosen by the user.
  Values with an offset are converted using that offset. A column mixing both is an error.
* Local times that fall in a DST transition (repeated or skipped hour) are not resolved by
  guessing: they become invalid (NaT) and are counted so the user can see them.
* Unix epoch values are absolute instants; the source timezone does not apply to them.
* Excel serial dates are local times; the workbook's 1900/1904 date system must be known.

All results use nanosecond resolution so that every timestamp series in the application has
the same dtype.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import product

import numpy as np
import pandas as pd

from inferential_studio.core.schemas import TimestampKind, TimestampSpec

UTC_DTYPE = "datetime64[ns, UTC]"
ISO_FORMAT = "ISO8601"

_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d{1,9})?)?)?"
    r"\s*(?:Z|[+-]\d{2}(?::?\d{2})?)?$"
)
_OFFSET_RE = re.compile(r"(?:Z|[+-]\d{2}(?::?\d{2})?)$")

EXCEL_ORIGINS = {1900: pd.Timestamp("1899-12-30"), 1904: pd.Timestamp("1904-01-01")}

# Plausible numeric ranges: 1954..2119 for Excel serials, 1973..2103 for Unix epochs.
_EXCEL_RANGE = (20000.0, 80000.0)
_UNIX_S_RANGE = (1.0e8, 4.2e9)
_UNIX_MS_RANGE = (1.0e11, 4.2e12)


class TimestampError(ValueError):
    """A timestamp column cannot be interpreted. The message is shown to the user."""


class DateOrder(StrEnum):
    DAY_FIRST = "day_first"
    MONTH_FIRST = "month_first"
    YEAR_FIRST = "year_first"
    NAMED_MONTH = "named_month"  # month written as text, so no day/month ambiguity


@dataclass(frozen=True)
class FormatCandidate:
    kind: TimestampKind
    format: str | None  # strptime format or ISO_FORMAT for TEXT; None for numeric kinds
    order: DateOrder | None
    description: str
    example_input: str = ""
    example_parsed: str = ""  # ISO text, naive local (offset kept if present)

    @property
    def dayfirst(self) -> bool | None:
        if self.order is DateOrder.DAY_FIRST:
            return True
        if self.order is DateOrder.MONTH_FIRST:
            return False
        return None


@dataclass
class FormatDetection:
    candidates: list[FormatCandidate] = field(default_factory=list)
    ambiguous: bool = False  # both day-first and month-first fit: the user must choose
    has_offsets: bool = False  # every value carries an explicit UTC offset
    mixed_offsets: bool = False  # some values have offsets and some do not
    sample_size: int = 0
    message: str = ""

    @property
    def recommended(self) -> FormatCandidate | None:
        """The single candidate to preselect, or None when the user must decide."""
        if self.ambiguous or self.mixed_offsets or not self.candidates:
            return None
        return self.candidates[0]


# --------------------------------------------------------------------------- candidate formats


def _text_formats() -> list[tuple[str, DateOrder]]:
    dates: list[tuple[str, DateOrder]] = []
    for sep in ("/", "-", "."):
        dates += [
            (f"%d{sep}%m{sep}%Y", DateOrder.DAY_FIRST),
            (f"%m{sep}%d{sep}%Y", DateOrder.MONTH_FIRST),
            (f"%Y{sep}%m{sep}%d", DateOrder.YEAR_FIRST),
            (f"%d{sep}%m{sep}%y", DateOrder.DAY_FIRST),
            (f"%m{sep}%d{sep}%y", DateOrder.MONTH_FIRST),
        ]
    dates += [
        ("%d-%b-%Y", DateOrder.NAMED_MONTH),
        ("%d-%b-%y", DateOrder.NAMED_MONTH),
        ("%d %b %Y", DateOrder.NAMED_MONTH),
        ("%d %B %Y", DateOrder.NAMED_MONTH),
        ("%b %d %Y", DateOrder.NAMED_MONTH),
        ("%b %d, %Y", DateOrder.NAMED_MONTH),
        ("%B %d %Y", DateOrder.NAMED_MONTH),
        ("%B %d, %Y", DateOrder.NAMED_MONTH),
        ("%Y-%b-%d", DateOrder.NAMED_MONTH),
    ]
    times = ["", " %H:%M", " %H:%M:%S", " %H:%M:%S.%f", " %I:%M %p", " %I:%M:%S %p"]
    return [(d + t, order) for (d, order), t in product(dates, times)]


_TEXT_FORMATS = _text_formats()

_ORDER_TEXT = {
    DateOrder.DAY_FIRST: "day first (dd/mm)",
    DateOrder.MONTH_FIRST: "month first (mm/dd)",
    DateOrder.YEAR_FIRST: "year first",
    DateOrder.NAMED_MONTH: "month name",
}


def _describe(fmt: str, order: DateOrder | None) -> str:
    readable = (
        fmt.replace("%d", "dd")
        .replace("%m", "mm")
        .replace("%Y", "yyyy")
        .replace("%y", "yy")
        .replace("%H", "HH")
        .replace("%I", "hh")
        .replace("%M", "MM")
        .replace("%S", "SS")
        .replace("%f", "ffffff")
        .replace("%p", "AM/PM")
        .replace("%b", "Mon")
        .replace("%B", "Month")
    )
    return f"{readable} — {_ORDER_TEXT[order]}" if order else readable


# --------------------------------------------------------------------------- detection


def sample_values(values: pd.Series, max_samples: int = 2000) -> pd.Series:
    """Non-blank values spread evenly over the column (head, middle and tail)."""
    text = values.astype("string").str.strip()
    text = text[text.notna() & (text != "")]
    if len(text) <= max_samples:
        return text
    positions = np.unique(np.linspace(0, len(text) - 1, max_samples).round().astype(int))
    return text.iloc[positions]


def detect_format(values: pd.Series, *, max_samples: int = 2000) -> FormatDetection:
    """Find the timestamp formats that fit every sampled value of a column."""
    if pd.api.types.is_datetime64_any_dtype(values):
        tz = getattr(values.dtype, "tz", None)
        return FormatDetection(
            candidates=[
                FormatCandidate(
                    TimestampKind.NATIVE,
                    None,
                    None,
                    "Date/time cells" + (" with timezone" if tz else " (local time)"),
                )
            ],
            has_offsets=tz is not None,
            sample_size=int(values.notna().sum()),
        )

    sample = sample_values(values, max_samples)
    detection = FormatDetection(sample_size=len(sample))
    if sample.empty:
        detection.message = "The column contains no values."
        return detection

    numeric = pd.to_numeric(sample, errors="coerce")
    if numeric.notna().all():
        detection.candidates = _numeric_candidates(numeric, sample)
        if not detection.candidates:
            detection.message = (
                "The column is numeric but not in a recognised range for Excel serial dates "
                "or Unix seconds/milliseconds."
            )
        return detection

    with_offset = sample.str.contains(_OFFSET_RE, regex=True)
    is_iso = sample.str.match(_ISO_RE)
    if is_iso.all():
        if with_offset.any() and not with_offset.all():
            detection.mixed_offsets = True
            detection.message = (
                f"{int(with_offset.sum())} of {len(sample)} sampled values carry a UTC offset and "
                "the rest do not. Split or correct the column so all values are consistent."
            )
            return detection
        detection.has_offsets = bool(with_offset.all())
        parsed = _parse_iso(sample, utc=detection.has_offsets)
        if parsed.notna().all():
            detection.candidates = [
                _candidate(
                    TimestampKind.TEXT, ISO_FORMAT, DateOrder.YEAR_FIRST, sample, parsed, "ISO 8601"
                )
            ]
            return detection

    fitting: list[FormatCandidate] = []
    seen_orders: set[DateOrder] = set()
    for fmt, order in _TEXT_FORMATS:
        if order in seen_orders and order is not DateOrder.NAMED_MONTH:
            continue
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        if parsed.notna().all():
            fitting.append(_candidate(TimestampKind.TEXT, fmt, order, sample, parsed))
            seen_orders.add(order)

    orders = {c.order for c in fitting}
    if (
        DateOrder.DAY_FIRST in orders
        and DateOrder.MONTH_FIRST in orders
        and len(sample) < values.notna().sum()
    ):
        # The sample cannot tell; check the whole column for a value that proves one order.
        fitting = _resolve_with_full_column(values, fitting)
        orders = {c.order for c in fitting}
    detection.candidates = fitting
    detection.ambiguous = DateOrder.DAY_FIRST in orders and DateOrder.MONTH_FIRST in orders
    if detection.ambiguous:
        detection.message = (
            "Every sampled date could be read either day-first or month-first "
            "(no day above 12 was found). Choose the order used by the source system."
        )
    elif not fitting:
        bad = sample.iloc[0]
        detection.message = (
            f"No supported date/time format matches all values (for example '{bad}')."
        )
    return detection


def _resolve_with_full_column(
    values: pd.Series, fitting: list[FormatCandidate]
) -> list[FormatCandidate]:
    """Drop the day-first or month-first candidate if some value in the full column rules it out.

    Only a value that one order cannot parse (e.g. day 25) settles the question; if both orders
    parse everything, both candidates remain and the user must choose.
    """
    text = values.astype("string").str.strip()
    text = text[text.notna() & (text != "")]
    keep = []
    for cand in fitting:
        order_matters = cand.order in (DateOrder.DAY_FIRST, DateOrder.MONTH_FIRST)
        if order_matters and pd.to_datetime(text, format=cand.format, errors="coerce").isna().any():
            continue
        keep.append(cand)
    remaining = {c.order for c in keep}
    if DateOrder.DAY_FIRST not in remaining and DateOrder.MONTH_FIRST not in remaining:
        return fitting  # neither order fits every value: keep both and let the user see why
    return keep


def _candidate(
    kind: TimestampKind,
    fmt: str | None,
    order: DateOrder | None,
    sample: pd.Series,
    parsed: pd.Series,
    description: str | None = None,
) -> FormatCandidate:
    # Prefer an example whose day and month differ so the user can see the interpretation.
    idx = 0
    if hasattr(parsed, "dt"):
        distinct = np.flatnonzero((parsed.dt.day != parsed.dt.month).to_numpy())
        if distinct.size:
            idx = int(distinct[0])
    example = parsed.iloc[idx]
    return FormatCandidate(
        kind=kind,
        format=fmt,
        order=order,
        description=description or _describe(fmt or "", order),
        example_input=str(sample.iloc[idx]),
        example_parsed=example.isoformat() if pd.notna(example) else "",
    )


def _numeric_candidates(numeric: pd.Series, sample: pd.Series) -> list[FormatCandidate]:
    lo, hi = float(numeric.min()), float(numeric.max())
    out: list[FormatCandidate] = []

    def within(rng: tuple[float, float]) -> bool:
        return rng[0] <= lo and hi <= rng[1]

    if within(_EXCEL_RANGE):
        parsed = EXCEL_ORIGINS[1900] + pd.to_timedelta(numeric, unit="D")
        out.append(
            _candidate(TimestampKind.EXCEL_SERIAL, None, None, sample, parsed, "Excel serial date")
        )
    if within(_UNIX_S_RANGE):
        parsed = pd.to_datetime(numeric, unit="s", utc=True)
        out.append(
            _candidate(
                TimestampKind.UNIX_SECONDS, None, None, sample, parsed, "Unix time, seconds (UTC)"
            )
        )
    if within(_UNIX_MS_RANGE):
        parsed = pd.to_datetime(numeric, unit="ms", utc=True)
        out.append(
            _candidate(
                TimestampKind.UNIX_MILLISECONDS,
                None,
                None,
                sample,
                parsed,
                "Unix time, milliseconds (UTC)",
            )
        )
    return out


def _parse_iso(text: pd.Series, *, utc: bool) -> pd.Series:
    return pd.to_datetime(
        text.str.replace(",", ".", regex=False), format=ISO_FORMAT, errors="coerce", utc=utc
    )


# --------------------------------------------------------------------------- parsing


@dataclass
class ParsedTimestamps:
    """Result of parsing one column. All Series share the input's index."""

    utc: pd.Series  # datetime64[ns, UTC]; NaT where blank, unparseable or DST-unresolvable
    blank: pd.Series  # bool: cell was empty
    unparseable: pd.Series  # bool: non-empty cell that does not match the format
    dst_ambiguous: pd.Series  # bool: local time occurs twice (clocks set back)
    dst_nonexistent: pd.Series  # bool: local time does not exist (clocks set forward)

    @property
    def valid(self) -> pd.Series:
        return self.utc.notna()

    @property
    def n_invalid(self) -> int:
        return int((self.unparseable | self.dst_ambiguous | self.dst_nonexistent).sum())


def parse_timestamps(values: pd.Series, spec: TimestampSpec) -> ParsedTimestamps:
    """Parse a column according to an explicit, user-confirmed :class:`TimestampSpec`."""
    index = values.index
    false = pd.Series(False, index=index)

    if pd.api.types.is_datetime64_any_dtype(values):
        blank = values.isna()
        naive_or_aware = values
        unparseable = false.copy()
    else:
        text = values.astype("string").str.strip()
        blank = text.isna() | (text == "")
        naive_or_aware, unparseable = _parse_by_kind(text, blank, spec)

    if getattr(naive_or_aware.dtype, "tz", None) is not None:
        utc = naive_or_aware.dt.tz_convert("UTC")
        ambiguous = nonexistent = false.copy()
    else:
        utc, ambiguous, nonexistent = _localize(naive_or_aware, spec.source_timezone)

    utc = utc.astype(UTC_DTYPE)
    return ParsedTimestamps(
        utc=utc,
        blank=blank.astype(bool),
        unparseable=unparseable.astype(bool),
        dst_ambiguous=ambiguous.astype(bool),
        dst_nonexistent=nonexistent.astype(bool),
    )


def _parse_by_kind(
    text: pd.Series, blank: pd.Series, spec: TimestampSpec
) -> tuple[pd.Series, pd.Series]:
    kind = spec.kind
    if kind is TimestampKind.NATIVE:
        raise TimestampError(
            f"Column '{spec.column}' was expected to contain date/time cells but contains text. "
            "Choose a text format instead."
        )

    if kind is TimestampKind.TEXT:
        fmt = spec.format or ISO_FORMAT
        if fmt == ISO_FORMAT:
            has_offset = text.str.contains(_OFFSET_RE, regex=True).fillna(False)
            nonblank = ~blank
            if has_offset[nonblank].any() and not has_offset[nonblank].all():
                raise TimestampError(
                    f"Column '{spec.column}' mixes values with and without UTC offsets; "
                    "they cannot be interpreted consistently."
                )
            parsed = _parse_iso(text, utc=bool(has_offset[nonblank].any()))
        else:
            parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        return parsed, parsed.isna() & ~blank

    numeric = pd.to_numeric(text, errors="coerce")
    unparseable = numeric.isna() & ~blank
    if kind is TimestampKind.EXCEL_SERIAL:
        system = spec.excel_date_system or 1900
        if system == 1900:
            # Serials below 61 fall before Excel's fictitious 29 Feb 1900; reject them.
            unparseable |= numeric < 61
            numeric = numeric.where(numeric >= 61)
        parsed = EXCEL_ORIGINS[system] + pd.to_timedelta(numeric, unit="D")
        # Excel stores times as day fractions; round away binary floating-point noise.
        parsed = parsed.dt.round("ms")
        return parsed, unparseable
    unit = "s" if kind is TimestampKind.UNIX_SECONDS else "ms"
    parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return parsed, unparseable | (parsed.isna() & numeric.notna())


def _localize(naive: pd.Series, timezone: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Attach ``timezone`` to naive local times, flagging DST gaps and overlaps instead of guessing."""
    as_dst = naive.dt.tz_localize(
        timezone, ambiguous=np.ones(len(naive), dtype=bool), nonexistent="NaT"
    )
    as_std = naive.dt.tz_localize(
        timezone, ambiguous=np.zeros(len(naive), dtype=bool), nonexistent="NaT"
    )
    present = naive.notna()
    nonexistent = present & as_dst.isna()
    ambiguous = present & as_dst.notna() & (as_dst != as_std)
    utc = as_dst.where(~ambiguous).dt.tz_convert("UTC")
    return utc, ambiguous, nonexistent


def spec_from_candidate(
    column: str,
    candidate: FormatCandidate,
    source_timezone: str,
    *,
    dayfirst: bool | None = None,
    excel_date_system: int | None = None,
) -> TimestampSpec:
    """Build the TimestampSpec the user confirmed in the import wizard."""
    return TimestampSpec(
        column=column,
        kind=candidate.kind,
        format=candidate.format,
        dayfirst=candidate.dayfirst if dayfirst is None else dayfirst,
        source_timezone=source_timezone,
        excel_date_system=excel_date_system
        if candidate.kind is TimestampKind.EXCEL_SERIAL
        else None,
    )
