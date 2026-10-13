"""Typed configuration objects that make up a project.

Everything here is plain data (frozen dataclasses + enums) with validation in
``__post_init__``. Objects are converted to/from JSON with :func:`to_jsonable` and
:func:`from_jsonable`; unknown keys are rejected so corrupt or foreign files fail loudly.

Conventions fixed here and relied on everywhere else (see docs/methodology.md):

* All instants are timezone-aware and stored in UTC. Display timezone is separate.
* All durations are float **seconds**.
* Feature windows are **left-open, right-closed**: ``(t + start_offset, t + end_offset]``.
* ``deadtime_s`` (>= 0) and ``shift_s`` (signed) both mean "how far *before* the target
  sample time the window ends". Positive shift moves the window further into the past;
  negative shift moves it towards (or past) the sample time. They are summed once into
  ``effective_delay_s`` so the offset is never applied twice.
* Manual exclusion intervals are **closed**: ``[start, end]``.
"""

from __future__ import annotations

import dataclasses
import math
import types
import typing
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA_VERSION = 6  # 1 = M0; 2 = native timestamps, duplicate policy, sample ids;
# 3 = rules, detectors, flag reviews; 4 = alignment options, splits and model configurations;
# 5 = optional embedded data (portable projects), saved training results, report details;
# 6 = optional predictor-subset model search


class SchemaError(ValueError):
    """Raised when configuration data is structurally invalid."""


# --------------------------------------------------------------------------- enums


class SourceKind(StrEnum):
    PROCESS = "process"  # dense historian data, one row per timestamp
    LAB = "lab"  # sparse laboratory / analyzer target samples


class TimestampKind(StrEnum):
    TEXT = "text"
    EXCEL_SERIAL = "excel_serial"
    UNIX_SECONDS = "unix_seconds"
    UNIX_MILLISECONDS = "unix_milliseconds"
    NATIVE = "native"  # the file already stores date/time cells (e.g. Excel dates)


class DuplicatePolicy(StrEnum):
    """What to do when several rows share a timestamp."""

    ERROR = "error"  # refuse until the user decides (process default)
    KEEP_FIRST = "keep_first"
    MEAN = "mean"  # average numeric values of rows with the same time
    KEEP_ALL = "keep_all"  # keep separate rows; lab sources only (replicate samples)


class VariableRole(StrEnum):
    TARGET = "target"
    PREDICTOR = "predictor"
    UNUSED = "unused"


class WindowPlacement(StrEnum):
    LOOKBACK = "lookback"
    CENTERED = "centered"


class Aggregation(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    STD = "std"
    LAST = "last"  # last valid value in the window


class FilterKind(StrEnum):
    """All filters are trailing (causal). Centred smoothing is deliberately not offered."""

    NONE = "none"
    ROLLING_MEAN = "rolling_mean"
    ROLLING_MEDIAN = "rolling_median"
    EWMA = "ewma"


class ExclusionReason(StrEnum):
    SHUTDOWN = "shutdown"
    STARTUP = "startup"
    TRANSITION = "transition"
    INSTRUMENT_FAULT = "instrument_fault"
    ANALYZER_FAULT = "analyzer_fault"
    MAINTENANCE = "maintenance"
    OPERATOR_JUDGEMENT = "operator_judgement"
    OTHER = "other"


# --------------------------------------------------------------------------- helpers


def validate_timezone(name: str) -> str:
    """Return ``name`` if it is a valid IANA timezone, else raise SchemaError."""
    if not isinstance(name, str) or not name:
        raise SchemaError("Timezone must be a non-empty IANA name such as 'Asia/Singapore'.")
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise SchemaError(
            f"Unknown timezone '{name}'. Use an IANA name such as 'Europe/London'."
        ) from exc
    return name


def ensure_utc(value: datetime, what: str = "timestamp") -> datetime:
    """Require a timezone-aware datetime and return it converted to UTC."""
    if not isinstance(value, datetime):
        raise SchemaError(f"{what} must be a datetime, got {type(value).__name__}.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise SchemaError(f"{what} must be timezone-aware; naive datetimes are ambiguous.")
    return value.astimezone(UTC)


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def new_id() -> str:
    return uuid.uuid4().hex


def _coerce_enum(obj: object, name: str, enum: type[StrEnum], *, optional: bool = False) -> None:
    """Accept enum members or their string values (e.g. from Qt widgets) on frozen dataclasses."""
    value = getattr(obj, name)
    if value is None and optional:
        return
    try:
        object.__setattr__(obj, name, enum(value))
    except ValueError as exc:
        allowed = ", ".join(m.value for m in enum)
        raise SchemaError(f"'{value}' is not a valid {name}; expected one of: {allowed}.") from exc


def _finite(value: float | None, what: str, *, allow_none: bool = True) -> None:
    if value is None:
        if not allow_none:
            raise SchemaError(f"{what} is required.")
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise SchemaError(f"{what} must be a finite number, got {value!r}.")


# --------------------------------------------------------------------------- data sources


@dataclass(frozen=True)
class TimestampSpec:
    """How to interpret one timestamp column of a source file."""

    column: str
    kind: TimestampKind = TimestampKind.TEXT
    format: str | None = None  # strptime format when known, e.g. "%d/%m/%Y %H:%M"
    dayfirst: bool | None = None  # None = not resolved; importer refuses ambiguous data
    source_timezone: str = "UTC"  # applied to naive values; offsets in the data win
    excel_date_system: int | None = None  # 1900 or 1904, for EXCEL_SERIAL

    def __post_init__(self) -> None:
        _coerce_enum(self, "kind", TimestampKind)
        if not self.column:
            raise SchemaError("Timestamp column name must not be empty.")
        validate_timezone(self.source_timezone)
        if self.excel_date_system not in (None, 1900, 1904):
            raise SchemaError("Excel date system must be 1900 or 1904.")


@dataclass(frozen=True)
class DataSource:
    """One imported file. ``path`` may be relative to the project file's folder."""

    source_id: str
    kind: SourceKind
    path: str
    timestamp: TimestampSpec
    result_timestamp: TimestampSpec | None = None  # lab result-availability time, if supplied
    sha256: str | None = None
    sheet: str | None = None
    delimiter: str | None = None
    encoding: str = "utf-8"
    decimal: str = "."
    header_row: int = 0
    duplicate_policy: DuplicatePolicy | None = None  # None = default for the source kind
    sample_id_column: str | None = None  # lab only

    def __post_init__(self) -> None:
        _coerce_enum(self, "kind", SourceKind)
        _coerce_enum(self, "duplicate_policy", DuplicatePolicy, optional=True)
        if not self.source_id:
            raise SchemaError("Source id must not be empty.")
        if not self.path:
            raise SchemaError(f"Source '{self.source_id}' has no file path.")
        if self.result_timestamp is not None and self.kind is not SourceKind.LAB:
            raise SchemaError("Only lab sources can have a result (availability) timestamp.")
        if self.header_row < 0:
            raise SchemaError("Header row must be >= 0.")
        if self.decimal not in (".", ","):
            raise SchemaError("Decimal separator must be '.' or ','.")
        if self.kind is SourceKind.PROCESS and self.duplicate_policy is DuplicatePolicy.KEEP_ALL:
            raise SchemaError(
                "Process data needs one row per timestamp; 'keep all' is for lab data."
            )
        if self.sample_id_column is not None and self.kind is not SourceKind.LAB:
            raise SchemaError("Only lab sources have a sample id column.")

    @property
    def effective_duplicate_policy(self) -> DuplicatePolicy:
        if self.duplicate_policy is not None:
            return self.duplicate_policy
        return DuplicatePolicy.KEEP_ALL if self.kind is SourceKind.LAB else DuplicatePolicy.ERROR

    @property
    def timestamp_columns(self) -> tuple[str, ...]:
        cols = [self.timestamp.column]
        if self.result_timestamp is not None:
            cols.append(self.result_timestamp.column)
        if self.sample_id_column is not None:
            cols.append(self.sample_id_column)
        return tuple(cols)


# --------------------------------------------------------------------------- variables


@dataclass(frozen=True)
class VariableMeta:
    name: str
    source_id: str
    source_column: str
    role: VariableRole = VariableRole.UNUSED
    units: str | None = None
    description: str = ""
    eng_low: float | None = None
    eng_high: float | None = None

    def __post_init__(self) -> None:
        _coerce_enum(self, "role", VariableRole)
        if not self.name:
            raise SchemaError("Variable name must not be empty.")
        _finite(self.eng_low, f"{self.name}: engineering low limit")
        _finite(self.eng_high, f"{self.name}: engineering high limit")
        if self.eng_low is not None and self.eng_high is not None and self.eng_low >= self.eng_high:
            raise SchemaError(f"{self.name}: engineering low limit must be below the high limit.")


# --------------------------------------------------------------------------- features


@dataclass(frozen=True)
class FilterSpec:
    kind: FilterKind = FilterKind.NONE
    window_s: float | None = None  # rolling filters: trailing window length
    halflife_s: float | None = None  # EWMA: time-based half-life

    def __post_init__(self) -> None:
        _coerce_enum(self, "kind", FilterKind)
        _finite(self.window_s, "Filter window")
        _finite(self.halflife_s, "EWMA half-life")
        rolling = self.kind in (FilterKind.ROLLING_MEAN, FilterKind.ROLLING_MEDIAN)
        if rolling and (self.window_s is None or self.window_s <= 0):
            raise SchemaError("Rolling filters need a positive window length.")
        if self.kind is FilterKind.EWMA and (self.halflife_s is None or self.halflife_s <= 0):
            raise SchemaError("EWMA filter needs a positive half-life.")

    @property
    def is_causal(self) -> bool:
        return True  # every offered filter is trailing; see FilterKind


@dataclass(frozen=True)
class FeatureSpec:
    """How one predictor X is turned into a feature value at each target sample time ``t``.

    Window = ``(t + start_offset_s, t + end_offset_s]`` where, with D = effective delay
    and W = window length:

    * lookback: ``(t - D - W, t - D]``
    * centered: ``(t - D - W/2, t - D + W/2]``
    """

    variable: str
    window_s: float = 1800.0
    deadtime_s: float = 0.0
    shift_s: float = 0.0
    placement: WindowPlacement = WindowPlacement.LOOKBACK
    aggregation: Aggregation = Aggregation.MEAN
    filter: FilterSpec = field(default_factory=FilterSpec)
    resample_s: float | None = None  # None = use original sampling
    min_coverage: float = 0.8  # minimum fraction of the window that must hold valid data
    max_gap_s: float | None = None  # longest permitted run of missing data inside the window

    def __post_init__(self) -> None:
        _coerce_enum(self, "placement", WindowPlacement)
        _coerce_enum(self, "aggregation", Aggregation)
        if not self.variable:
            raise SchemaError("Feature must reference a variable.")
        for value, what in (
            (self.window_s, "window"),
            (self.deadtime_s, "deadtime"),
            (self.shift_s, "shift"),
        ):
            _finite(value, f"{self.variable}: {what}", allow_none=False)
        _finite(self.resample_s, f"{self.variable}: resample interval")
        _finite(self.max_gap_s, f"{self.variable}: maximum gap")
        if self.window_s <= 0:
            raise SchemaError(f"{self.variable}: aggregation window must be positive.")
        if self.deadtime_s < 0:
            raise SchemaError(f"{self.variable}: deadtime cannot be negative (use shift instead).")
        if self.resample_s is not None and self.resample_s <= 0:
            raise SchemaError(f"{self.variable}: resample interval must be positive.")
        if self.max_gap_s is not None and self.max_gap_s < 0:
            raise SchemaError(f"{self.variable}: maximum gap cannot be negative.")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise SchemaError(f"{self.variable}: minimum coverage must be between 0 and 1.")

    @property
    def effective_delay_s(self) -> float:
        """Total time the window end sits before the sample time (deadtime + shift)."""
        return self.deadtime_s + self.shift_s

    @property
    def window_offsets_s(self) -> tuple[float, float]:
        """``(start, end)`` offsets in seconds relative to the sample time; negative = past."""
        delay = self.effective_delay_s
        if self.placement is WindowPlacement.LOOKBACK:
            return (-(delay + self.window_s), -delay)
        half = self.window_s / 2.0
        return (-(delay + half), -delay + half)

    @property
    def is_causal(self) -> bool:
        """True when the feature reads no data recorded after the sample time."""
        return self.window_offsets_s[1] <= 0.0 and self.filter.is_causal


# --------------------------------------------------------------------------- exclusions


@dataclass(frozen=True)
class ExclusionInterval:
    """A manually drawn exclusion over the closed interval ``[start, end]`` (UTC)."""

    start: datetime
    end: datetime
    reason: ExclusionReason = ExclusionReason.OTHER
    note: str = ""
    signals: tuple[str, ...] = ()  # empty = applies to every signal (operating-regime exclusion)
    enabled: bool = True
    author: str = ""
    created_at: datetime = field(default_factory=utc_now)
    interval_id: str = field(default_factory=new_id)

    def __post_init__(self) -> None:
        _coerce_enum(self, "reason", ExclusionReason)
        object.__setattr__(self, "start", ensure_utc(self.start, "Exclusion start"))
        object.__setattr__(self, "end", ensure_utc(self.end, "Exclusion end"))
        object.__setattr__(self, "created_at", ensure_utc(self.created_at, "Creation time"))
        object.__setattr__(self, "signals", tuple(self.signals))
        if self.end <= self.start:
            raise SchemaError("Exclusion end must be after its start.")

    @property
    def is_global(self) -> bool:
        return not self.signals


# --------------------------------------------------------------------------- exclusion rules


class RuleOperator(StrEnum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="
    BETWEEN = "between"  # lo <= x <= hi
    OUTSIDE = "outside"  # x < lo or x > hi
    IS_MISSING = "is missing"  # no valid number (blank or text)
    IS_INVALID = "is invalid"  # the cell held non-numeric text such as "Bad"


RANGE_OPERATORS = frozenset({RuleOperator.BETWEEN, RuleOperator.OUTSIDE})
VALUELESS_OPERATORS = frozenset({RuleOperator.IS_MISSING, RuleOperator.IS_INVALID})


class Combinator(StrEnum):
    AND = "and"
    OR = "or"


class RuleAction(StrEnum):
    EXCLUDE = "exclude"
    FLAG = "flag"  # highlight for review only


@dataclass(frozen=True)
class Condition:
    """One comparison on one variable, e.g. ``Feed_Flow < 50`` or ``Pressure between 2 and 4``."""

    variable: str
    operator: RuleOperator
    value: float | None = None
    value2: float | None = None  # upper bound for between/outside

    def __post_init__(self) -> None:
        _coerce_enum(self, "operator", RuleOperator)
        if not self.variable:
            raise SchemaError("A condition needs a variable.")
        op = self.operator
        if op in VALUELESS_OPERATORS:
            if self.value is not None or self.value2 is not None:
                raise SchemaError(f"'{op.value}' takes no value.")
            return
        _finite(self.value, f"{self.variable} {op.value}: value", allow_none=False)
        if op in RANGE_OPERATORS:
            _finite(self.value2, f"{self.variable} {op.value}: upper value", allow_none=False)
            if self.value2 < self.value:
                raise SchemaError(
                    f"{self.variable} {op.value}: the lower value must not exceed the upper value."
                )
        elif self.value2 is not None:
            raise SchemaError(f"'{op.value}' takes a single value.")

    def describe(self) -> str:
        op = self.operator
        if op in VALUELESS_OPERATORS:
            return f"{self.variable} {op.value}"
        if op in RANGE_OPERATORS:
            return f"{self.variable} {op.value} {self.value:g} and {self.value2:g}"
        return f"{self.variable} {op.value} {self.value:g}"


@dataclass(frozen=True)
class ConditionGroup:
    """Conditions (or nested groups) combined with one AND/OR."""

    items: tuple[Condition | ConditionGroup, ...]
    combinator: Combinator = Combinator.AND

    def __post_init__(self) -> None:
        _coerce_enum(self, "combinator", Combinator)
        object.__setattr__(self, "items", tuple(self.items))
        if not self.items:
            raise SchemaError("A rule needs at least one condition.")

    def variables(self) -> set[str]:
        out: set[str] = set()
        for item in self.items:
            out |= {item.variable} if isinstance(item, Condition) else item.variables()
        return out

    def describe(self, top: bool = True) -> str:
        parts = [
            f"({item.describe()})"
            if isinstance(item, Condition) and len(self.items) > 1
            else item.describe()
            if isinstance(item, Condition)
            else f"({item.describe(top=False)})"
            for item in self.items
        ]
        return f" {self.combinator.value.upper()} ".join(parts)


@dataclass(frozen=True)
class ExclusionRule:
    """A reusable logical exclusion, evaluated on the process data's own timestamps.

    ``signals`` empty = operating-regime rule that excludes whole rows (every variable);
    otherwise only the listed variables are excluded where the condition holds.
    ``min_duration_s``: the condition must hold continuously at least this long (see
    docs/methodology.md for how gaps end a run).
    """

    name: str
    condition: ConditionGroup
    action: RuleAction = RuleAction.EXCLUDE
    reason: ExclusionReason = ExclusionReason.OTHER
    signals: tuple[str, ...] = ()
    min_duration_s: float | None = None
    enabled: bool = True
    note: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=utc_now)
    rule_id: str = field(default_factory=new_id)

    def __post_init__(self) -> None:
        _coerce_enum(self, "action", RuleAction)
        _coerce_enum(self, "reason", ExclusionReason)
        object.__setattr__(self, "signals", tuple(self.signals))
        object.__setattr__(self, "created_at", ensure_utc(self.created_at, "Rule creation time"))
        if not self.name.strip():
            raise SchemaError("A rule needs a name.")
        _finite(self.min_duration_s, f"Rule '{self.name}': minimum duration")
        if self.min_duration_s is not None and self.min_duration_s <= 0:
            raise SchemaError(f"Rule '{self.name}': minimum duration must be positive.")

    def describe(self) -> str:
        text = self.condition.describe()
        if self.min_duration_s:
            text += f" for at least {self.min_duration_s / 60:g} min"
        return text


# --------------------------------------------------------------------------- anomaly detection


class DetectorKind(StrEnum):
    ZSCORE = "zscore"
    MAD = "mad"  # robust (modified) z-score
    IQR = "iqr"
    HAMPEL = "hampel"
    RATE_OF_CHANGE = "rate_of_change"
    FLATLINE = "flatline"
    LIMITS = "limits"  # physical / engineering limits
    INVALID_TEXT = "invalid_text"  # cells that held status text such as "Bad"


@dataclass(frozen=True)
class DetectorConfig:
    """Settings of one automatic data-quality detector. Results are flags for review by default."""

    kind: DetectorKind
    variables: tuple[str, ...]
    threshold: float | None = None  # sigmas (z, MAD, Hampel) or IQR multiplier k
    window_s: float | None = None  # Hampel window; flatline minimum duration
    rate_per_min: float | None = None  # rate of change limit, units per minute
    tolerance: float = 0.0  # flatline: changes up to this count as "no change"
    low: float | None = None  # limits; None = use the variable's engineering limit
    high: float | None = None
    min_deviation: float = 0.0  # Hampel: ignore deviations smaller than this (units)
    fit_start: datetime | None = None  # optional period for fitting statistics (z, MAD, IQR)
    fit_end: datetime | None = None
    auto_exclude: bool = False  # opt-in: treat every flag as accepted unless rejected
    enabled: bool = True
    detector_id: str = field(default_factory=new_id)

    def __post_init__(self) -> None:
        _coerce_enum(self, "kind", DetectorKind)
        object.__setattr__(self, "variables", tuple(self.variables))
        if not self.variables:
            raise SchemaError("A detector needs at least one variable.")
        for name in ("threshold", "window_s", "rate_per_min", "low", "high"):
            _finite(getattr(self, name), name)
        _finite(self.tolerance, "tolerance", allow_none=False)
        _finite(self.min_deviation, "minimum deviation", allow_none=False)
        k = self.kind
        needs_threshold = k in (
            DetectorKind.ZSCORE,
            DetectorKind.MAD,
            DetectorKind.IQR,
            DetectorKind.HAMPEL,
        )
        if needs_threshold and (self.threshold is None or self.threshold <= 0):
            raise SchemaError(f"{k.value}: the threshold must be positive.")
        if k in (DetectorKind.HAMPEL, DetectorKind.FLATLINE) and (
            self.window_s is None or self.window_s <= 0
        ):
            raise SchemaError(f"{k.value}: the window/duration must be positive.")
        if k is DetectorKind.RATE_OF_CHANGE and (
            self.rate_per_min is None or self.rate_per_min <= 0
        ):
            raise SchemaError("Rate of change: the limit must be positive.")
        if self.tolerance < 0 or self.min_deviation < 0:
            raise SchemaError("Tolerances cannot be negative.")
        if self.low is not None and self.high is not None and self.low >= self.high:
            raise SchemaError("Limits: low must be below high.")
        for name in ("fit_start", "fit_end"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, ensure_utc(value, name))
        if self.fit_start and self.fit_end and self.fit_end <= self.fit_start:
            raise SchemaError("The fitting period must end after it starts.")

    def describe(self) -> str:
        k = self.kind
        if k is DetectorKind.ZSCORE:
            return f"|z| > {self.threshold:g}"
        if k is DetectorKind.MAD:
            return f"robust |z| > {self.threshold:g}"
        if k is DetectorKind.IQR:
            return f"outside Q1/Q3 ± {self.threshold:g}·IQR"
        if k is DetectorKind.HAMPEL:
            return f"Hampel: > {self.threshold:g}σ from the {self.window_s / 60:g}-min local median"
        if k is DetectorKind.RATE_OF_CHANGE:
            return f"changes faster than {self.rate_per_min:g} units/min"
        if k is DetectorKind.FLATLINE:
            return f"unchanged (±{self.tolerance:g}) for ≥ {self.window_s / 60:g} min"
        if k is DetectorKind.LIMITS:
            lo = "eng. low" if self.low is None else f"{self.low:g}"
            hi = "eng. high" if self.high is None else f"{self.high:g}"
            return f"outside [{lo}, {hi}]"
        return "cells holding status text"


class FlagDecisionKind(StrEnum):
    ACCEPTED = "accepted"  # confirmed bad: excluded for that variable
    REJECTED = "rejected"  # confirmed valid (e.g. a genuine process upset): kept


@dataclass(frozen=True)
class FlagDecision:
    """The user's review of one flagged event, identified by detector, variable and time span."""

    detector_id: str
    variable: str
    start: datetime
    end: datetime
    decision: FlagDecisionKind
    note: str = ""
    author: str = ""
    decided_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _coerce_enum(self, "decision", FlagDecisionKind)
        object.__setattr__(self, "start", ensure_utc(self.start, "Flag start"))
        object.__setattr__(self, "end", ensure_utc(self.end, "Flag end"))
        object.__setattr__(self, "decided_at", ensure_utc(self.decided_at, "Decision time"))
        if self.end < self.start:
            raise SchemaError("A flag cannot end before it starts.")

    @property
    def key(self) -> tuple[str, str, datetime, datetime]:
        return (self.detector_id, self.variable, self.start, self.end)


# --------------------------------------------------------------------------- alignment and modelling


@dataclass(frozen=True)
class AlignmentConfig:
    """Project-wide rules for turning process data into one feature row per lab sample."""

    # A sample is not model-eligible when any predictor's aggregation window overlaps an
    # operating-regime exclusion (shutdown, start-up …), even if coverage would still pass.
    reject_regime_overlap: bool = True


class Scaling(StrEnum):
    STANDARD = "standard"  # (x - mean) / std, fitted on training samples only
    ROBUST = "robust"  # (x - median) / IQR, fitted on training samples only
    NONE = "none"


class ModelKind(StrEnum):
    MLR = "mlr"
    PLS = "pls"
    PCR = "pcr"
    RIDGE = "ridge"


MODEL_LABELS = {
    ModelKind.MLR: "MLR (ordinary least squares)",
    ModelKind.PLS: "PLS regression",
    ModelKind.PCR: "PCR (PCA + regression)",
    ModelKind.RIDGE: "Ridge regression",
}


class SplitMode(StrEnum):
    FRACTIONS = "fractions"  # by share of eligible samples, in time order
    DATES = "dates"  # explicit boundary instants


class CvKind(StrEnum):
    EXPANDING = "expanding"  # rolling origin: train on the past, validate on the next block
    BLOCKED = "blocked"  # contiguous k-fold blocks with a purge gap either side


@dataclass(frozen=True)
class SplitConfig:
    """Chronological train / validation / final test split and the training-set CV.

    Fractions: the first ``train_fraction`` of eligible samples (in time order) train, the next
    ``validation_fraction`` validate, the rest is the untouched final test.
    Dates: train ``t < train_end``, validation ``train_end <= t < validation_end``, test after.
    ``purge_s`` = None picks the purge gap automatically (longest feature history).
    """

    mode: SplitMode = SplitMode.FRACTIONS
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    train_end: datetime | None = None
    validation_end: datetime | None = None
    purge_s: float | None = None
    cv_kind: CvKind = CvKind.EXPANDING
    cv_folds: int = 5

    def __post_init__(self) -> None:
        _coerce_enum(self, "mode", SplitMode)
        _coerce_enum(self, "cv_kind", CvKind)
        _finite(self.train_fraction, "Training fraction", allow_none=False)
        _finite(self.validation_fraction, "Validation fraction", allow_none=False)
        _finite(self.purge_s, "Purge gap")
        if not 0.0 < self.train_fraction < 1.0:
            raise SchemaError("The training fraction must be between 0 and 1.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise SchemaError("The validation fraction must be between 0 and 1.")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise SchemaError("Training and validation together must leave samples for the test.")
        for name in ("train_end", "validation_end"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, ensure_utc(value, name))
        if self.mode is SplitMode.DATES:
            if self.train_end is None or self.validation_end is None:
                raise SchemaError("Date-based splits need both a training end and a test start.")
            if self.validation_end < self.train_end:
                raise SchemaError("The test period must start at or after the training end.")
        if self.purge_s is not None and self.purge_s < 0:
            raise SchemaError("The purge gap cannot be negative.")
        if not isinstance(self.cv_folds, int) or isinstance(self.cv_folds, bool):
            raise SchemaError("The number of CV folds must be a whole number.")
        if not 2 <= self.cv_folds <= 20:
            raise SchemaError("Use between 2 and 20 cross-validation folds.")


@dataclass(frozen=True)
class ModelSpec:
    """One candidate model. ``None`` hyperparameters are tuned by time-aware CV on training data."""

    kind: ModelKind
    enabled: bool = True
    n_components: int | None = None  # PLS / PCR
    max_components: int = 10  # PLS / PCR tuning range 1..max
    alpha: float | None = None  # Ridge
    alpha_min: float = 1e-3
    alpha_max: float = 1e3
    alpha_steps: int = 13

    def __post_init__(self) -> None:
        _coerce_enum(self, "kind", ModelKind)
        _finite(self.alpha, "Ridge alpha")
        _finite(self.alpha_min, "Smallest alpha", allow_none=False)
        _finite(self.alpha_max, "Largest alpha", allow_none=False)
        for name in ("n_components", "max_components", "alpha_steps"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise SchemaError(f"{name.replace('_', ' ')} must be a whole number.")
        if self.n_components is not None and self.n_components < 1:
            raise SchemaError("The number of components must be at least 1.")
        if not 1 <= self.max_components <= 50:
            raise SchemaError("The component search range must be 1–50.")
        if self.alpha is not None and self.alpha <= 0:
            raise SchemaError("Ridge alpha must be positive.")
        if not 0 < self.alpha_min < self.alpha_max:
            raise SchemaError("The alpha search range must satisfy 0 < smallest < largest.")
        if not 2 <= self.alpha_steps <= 50:
            raise SchemaError("Use 2–50 alpha values in the search.")

    @property
    def label(self) -> str:
        return MODEL_LABELS[self.kind]


def default_model_specs() -> tuple[ModelSpec, ...]:
    return tuple(ModelSpec(kind) for kind in ModelKind)


@dataclass(frozen=True)
class ModellingConfig:
    scaling: Scaling = Scaling.STANDARD
    split: SplitConfig = field(default_factory=SplitConfig)
    models: tuple[ModelSpec, ...] = field(default_factory=default_model_specs)
    # Tuning picks the simplest setting whose CV error is within one standard error of the best.
    one_standard_error: bool = True
    search_subsets: bool = False

    def __post_init__(self) -> None:
        _coerce_enum(self, "scaling", Scaling)
        object.__setattr__(self, "models", tuple(self.models))
        _require_unique([m.kind.value for m in self.models], "model type")

    def spec(self, kind: ModelKind) -> ModelSpec | None:
        return next((m for m in self.models if m.kind is kind), None)


@dataclass(frozen=True)
class ReportInfo:
    """Model-card details entered by the user (Export & Report page)."""

    title: str = ""
    model_version: str = "1.0"
    owner: str = ""
    purpose: str = ""
    limitations: str = ""


# --------------------------------------------------------------------------- project


@dataclass
class Project:
    """The complete editable state of one modelling project (excluding bulk data)."""

    name: str = "Untitled project"
    display_timezone: str = "UTC"
    sources: list[DataSource] = field(default_factory=list)
    variables: list[VariableMeta] = field(default_factory=list)
    features: list[FeatureSpec] = field(default_factory=list)
    exclusions: list[ExclusionInterval] = field(default_factory=list)
    rules: list[ExclusionRule] = field(default_factory=list)
    detectors: list[DetectorConfig] = field(default_factory=list)
    flag_decisions: list[FlagDecision] = field(default_factory=list)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    modelling: ModellingConfig = field(default_factory=ModellingConfig)
    report: ReportInfo = field(default_factory=ReportInfo)
    notes: str = ""
    created_at: datetime = field(default_factory=utc_now)
    modified_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        validate_timezone(self.display_timezone)
        self.created_at = ensure_utc(self.created_at, "Project creation time")
        self.modified_at = ensure_utc(self.modified_at, "Project modification time")
        self.validate()

    # -- queries ---------------------------------------------------------------

    @property
    def target(self) -> VariableMeta | None:
        targets = [v for v in self.variables if v.role is VariableRole.TARGET]
        return targets[0] if targets else None

    @property
    def predictors(self) -> list[VariableMeta]:
        return [v for v in self.variables if v.role is VariableRole.PREDICTOR]

    def variable(self, name: str) -> VariableMeta:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)

    def source(self, source_id: str) -> DataSource:
        for s in self.sources:
            if s.source_id == source_id:
                return s
        raise KeyError(source_id)

    # -- integrity -------------------------------------------------------------

    def validate(self) -> None:
        """Raise SchemaError if cross-references between parts of the project are broken."""
        source_ids = [s.source_id for s in self.sources]
        _require_unique(source_ids, "source id")
        names = [v.name for v in self.variables]
        _require_unique(names, "variable name")

        for v in self.variables:
            if v.source_id not in source_ids:
                raise SchemaError(f"Variable '{v.name}' refers to unknown source '{v.source_id}'.")
        if sum(v.role is VariableRole.TARGET for v in self.variables) > 1:
            raise SchemaError("A project model has exactly one target variable; several were set.")

        predictor_names = {v.name for v in self.predictors}
        _require_unique([f.variable for f in self.features], "feature variable")
        for f in self.features:
            if f.variable not in predictor_names:
                raise SchemaError(f"Feature '{f.variable}' does not refer to a predictor variable.")
        known = set(names)

        def check(refs, what: str) -> None:  # noqa: ANN001
            unknown = sorted(set(refs) - known)
            if unknown:
                raise SchemaError(f"{what} refers to unknown variable(s): {', '.join(unknown)}.")

        for ex in self.exclusions:
            check(ex.signals, "An exclusion")
        _require_unique([r.rule_id for r in self.rules], "rule id")
        for rule in self.rules:
            check(rule.signals, f"Rule '{rule.name}'")
            check(rule.condition.variables(), f"Rule '{rule.name}'")
        _require_unique([d.detector_id for d in self.detectors], "detector id")
        for det in self.detectors:
            check(det.variables, f"Detector '{det.kind.value}'")
        detector_ids = {d.detector_id for d in self.detectors}
        _require_unique([str(f.key) for f in self.flag_decisions], "flag decision")
        for f in self.flag_decisions:
            if f.detector_id not in detector_ids:
                raise SchemaError("A flag decision refers to a detector that no longer exists.")
            check([f.variable], "A flag decision")


def _require_unique(values: list[str], what: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise SchemaError(f"Duplicate {what} '{value}'.")
        seen.add(value)


# --------------------------------------------------------------------------- JSON conversion


def to_jsonable(obj: typing.Any) -> typing.Any:
    """Convert schema objects into JSON-compatible builtins."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, StrEnum):
        return obj.value
    if isinstance(obj, datetime):
        return ensure_utc(obj).isoformat()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, bool, int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):
            raise SchemaError("Non-finite numbers cannot be saved.")
        return obj
    raise SchemaError(f"Cannot serialise value of type {type(obj).__name__}.")


T = typing.TypeVar("T")


def from_jsonable(cls: type[T], data: typing.Any) -> T:
    """Build ``cls`` (a schema dataclass) from JSON data, validating types strictly."""
    return typing.cast(T, _convert(cls, data, cls.__name__))


def _convert(tp: typing.Any, value: typing.Any, where: str) -> typing.Any:
    origin = typing.get_origin(tp)

    if origin in (typing.Union, types.UnionType):
        options = typing.get_args(tp)
        if value is None:
            if type(None) in options:
                return None
            raise SchemaError(f"{where}: value is required.")
        non_none = [o for o in options if o is not type(None)]
        errors = []
        for option in non_none:
            try:
                return _convert(option, value, where)
            except SchemaError as exc:
                errors.append(str(exc))
        raise SchemaError("; ".join(errors))

    if origin is list:
        (item_tp,) = typing.get_args(tp)
        if not isinstance(value, list):
            raise SchemaError(f"{where}: expected a list.")
        return [_convert(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value)]

    if origin is tuple:
        item_tp = typing.get_args(tp)[0]
        if not isinstance(value, list):
            raise SchemaError(f"{where}: expected a list.")
        return tuple(_convert(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value))

    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise SchemaError(f"{where}: expected an object.")
        hints = typing.get_type_hints(tp)
        known = {f.name for f in dataclasses.fields(tp) if f.init}
        unknown = set(value) - known
        if unknown:
            raise SchemaError(f"{where}: unknown field(s) {sorted(unknown)}.")
        kwargs = {k: _convert(hints[k], v, f"{where}.{k}") for k, v in value.items()}
        try:
            return tp(**kwargs)
        except TypeError as exc:  # missing required fields
            raise SchemaError(f"{where}: {exc}") from exc

    if isinstance(tp, type) and issubclass(tp, StrEnum):
        try:
            return tp(value)
        except ValueError as exc:
            allowed = ", ".join(m.value for m in tp)
            raise SchemaError(f"{where}: '{value}' is not one of: {allowed}.") from exc

    if tp is datetime:
        if not isinstance(value, str):
            raise SchemaError(f"{where}: expected an ISO 8601 timestamp string.")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise SchemaError(f"{where}: invalid timestamp '{value}'.") from exc
        return ensure_utc(parsed, where)

    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaError(f"{where}: expected a number.")
        return float(value)

    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaError(f"{where}: expected an integer.")
        return value

    if tp is bool:
        if not isinstance(value, bool):
            raise SchemaError(f"{where}: expected true/false.")
        return value

    if tp is str:
        if not isinstance(value, str):
            raise SchemaError(f"{where}: expected text.")
        return value

    raise SchemaError(f"{where}: unsupported field type {tp!r}.")
