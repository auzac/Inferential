"""In-memory data layers for imported sources.

Layers (kept separate, never overwritten by later steps):

* **Raw** (:class:`SourceData`): one entry per file row, in file order. Original timestamp text,
  parsed UTC times (NaT where invalid), numeric values, and a mask of cells that held text
  instead of a number. The source file itself, identified by its SHA-256, is the ultimate raw
  record; nothing is ever written back to it.
* **Normalized** (:meth:`SourceData.normalized`): rows with a valid time, sorted by UTC time,
  duplicates resolved by the source's explicit policy. Each row keeps its file row number.

Exclusion masks, features, splits and models are later layers built on top of these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property

import numpy as np
import pandas as pd

from inferential_studio.core.schemas import DataSource, DuplicatePolicy, Project, SourceKind


class DataImportError(ValueError):
    """A file cannot be imported as configured. The message is shown to the user."""


class Severity(StrEnum):
    ERROR = "error"  # import cannot be used until resolved
    WARNING = "warning"  # usable, but the user should look
    INFO = "info"


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str


@dataclass
class ColumnQuality:
    name: str
    n_numeric: int
    n_blank: int
    n_text: int  # cells holding text that is not a number (e.g. "Bad", "I/O Timeout")
    text_examples: dict[str, int] = field(default_factory=dict)  # most common tokens -> count

    @property
    def n_rows(self) -> int:
        return self.n_numeric + self.n_blank + self.n_text

    @property
    def valid_fraction(self) -> float:
        return self.n_numeric / self.n_rows if self.n_rows else 0.0


@dataclass
class Gap:
    start: pd.Timestamp  # last valid time before the gap
    end: pd.Timestamp  # first valid time after the gap

    @property
    def duration(self) -> pd.Timedelta:
        return self.end - self.start


@dataclass
class ImportReport:
    source_id: str
    kind: SourceKind
    n_rows: int
    n_valid_times: int
    n_blank_times: int
    n_unparseable_times: int
    n_dst_ambiguous: int
    n_dst_nonexistent: int
    invalid_time_examples: list[tuple[int, str]]  # (file line number, original text)
    n_duplicate_times: int  # rows sharing a time with an earlier row
    n_out_of_order: int  # rows whose time is earlier than the previous valid row's
    time_start: pd.Timestamp | None
    time_end: pd.Timestamp | None
    median_interval: pd.Timedelta | None
    irregular_fraction: float  # share of intervals deviating > 10 % from the median
    gaps: list[Gap]  # longest gaps first (at most 20 listed)
    n_gaps: int  # intervals longer than the gap threshold
    gap_threshold: pd.Timedelta | None
    columns: list[ColumnQuality]
    issues: list[Issue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity is Severity.ERROR for i in self.issues)

    def column(self, name: str) -> ColumnQuality:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)


@dataclass
class NormalizedData:
    """Time-indexed view of one source (process: unique sorted index; lab: sorted, may repeat)."""

    values: pd.DataFrame  # float64, index = UTC time named "time"
    text_mask: pd.DataFrame  # bool, same shape: cell held non-numeric text
    file_line: pd.Series  # int, original file line number of each row (first row for merged rows)
    result_time: pd.Series | None = None  # lab result availability time (UTC)
    sample_id: pd.Series | None = None


@dataclass(eq=False)
class SourceData:
    """Raw layer for one imported file. Treat as immutable."""

    source: DataSource
    sha256: str
    file_line: np.ndarray  # file line number (1-based, header counted) of each data row
    raw_time_text: pd.Series  # original timestamp text as read
    time: pd.Series  # parsed UTC time, NaT where invalid
    values: pd.DataFrame  # float64 numeric values, NaN where blank or text
    text_mask: pd.DataFrame  # True where the cell held non-numeric text
    report: ImportReport
    result_time: pd.Series | None = None
    sample_id: pd.Series | None = None

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def columns(self) -> list[str]:
        return list(self.values.columns)

    @cached_property
    def normalized(self) -> NormalizedData:
        return normalize(self)


def normalize(data: SourceData) -> NormalizedData:
    """Drop rows without a valid time, sort chronologically and apply the duplicate policy."""
    valid = data.time.notna().to_numpy()
    order = np.argsort(data.time.to_numpy()[valid], kind="stable")
    rows = np.flatnonzero(valid)[order]

    index = pd.DatetimeIndex(data.time.iloc[rows], name="time")  # keeps the UTC dtype
    values = pd.DataFrame(data.values.to_numpy()[rows], index=index, columns=data.values.columns)
    mask = pd.DataFrame(
        data.text_mask.to_numpy()[rows], index=index, columns=data.text_mask.columns
    )
    lines = pd.Series(data.file_line[rows], index=index, name="file_line")
    result = data.result_time.iloc[rows].set_axis(index) if data.result_time is not None else None
    sample_id = data.sample_id.iloc[rows].set_axis(index) if data.sample_id is not None else None

    policy = data.source.effective_duplicate_policy
    duplicated = index.duplicated(keep="first")
    if duplicated.any():
        if policy is DuplicatePolicy.ERROR:
            raise DataImportError(
                f"'{data.source.path}' has {int(duplicated.sum())} rows that repeat an earlier "
                "timestamp. Choose how duplicates should be handled (keep first or average)."
            )
        if policy is DuplicatePolicy.KEEP_FIRST:
            keep = ~duplicated
            values, mask, lines = values[keep], mask[keep], lines[keep]
            result = result[keep] if result is not None else None
            sample_id = sample_id[keep] if sample_id is not None else None
        elif policy is DuplicatePolicy.MEAN:
            values = values.groupby(level=0, sort=True).mean()
            mask = mask.groupby(level=0, sort=True).any()
            lines = lines.groupby(level=0, sort=True).first()
            result = result.groupby(level=0, sort=True).max() if result is not None else None
            sample_id = (
                sample_id.groupby(level=0, sort=True).agg(lambda s: "+".join(map(str, s)))
                if sample_id is not None
                else None
            )
        # KEEP_ALL: lab replicates stay as separate rows, still sorted by sample time.

    return NormalizedData(
        values=values, text_mask=mask, file_line=lines, result_time=result, sample_id=sample_id
    )


@dataclass
class ProjectData:
    """All imported sources of a project, keyed by source id."""

    sources: dict[str, SourceData] = field(default_factory=dict)
    failures: dict[str, DataImportError] = field(default_factory=dict)  # source id -> why it failed

    def of_kind(self, kind: SourceKind) -> list[SourceData]:
        return [s for s in self.sources.values() if s.source.kind is kind]

    @property
    def n_process_rows(self) -> int:
        return sum(s.report.n_rows for s in self.of_kind(SourceKind.PROCESS))

    @property
    def n_lab_samples(self) -> int:
        return sum(s.report.n_rows for s in self.of_kind(SourceKind.LAB))

    def series(self, source_id: str, column: str) -> pd.Series:
        """Normalized numeric series of one column (UTC index)."""
        return self.sources[source_id].normalized.values[column]

    def process_frame(self, project: Project) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process variables (named as in the project) on their own timestamps, plus text mask.

        Several process sources are combined by outer join on time: rows exist wherever any
        source has a timestamp and are **not** filled in, so cross-source statistics use only
        exactly matching times.
        """
        values, masks = [], []
        for src in self.of_kind(SourceKind.PROCESS):
            names = {
                v.source_column: v.name
                for v in project.variables
                if v.source_id == src.source_id and v.source_column in src.columns
            }
            norm = src.normalized
            values.append(norm.values[list(names)].rename(columns=names))
            masks.append(norm.text_mask[list(names)].rename(columns=names))
        if not values:
            return pd.DataFrame(), pd.DataFrame()
        if len(values) == 1:
            return values[0], masks[0]
        frame = pd.concat(values, axis=1, join="outer", sort=True)
        mask = (
            pd.concat(masks, axis=1, join="outer", sort=True)
            .reindex(frame.index)
            .fillna(False)
            .astype(bool)
        )
        return frame, mask

    def target_series(self, project: Project) -> pd.Series | None:
        """The project's target at its sample times (sparse; never forward-filled)."""
        target = project.target
        if target is None or target.source_id not in self.sources:
            return None
        src = self.sources[target.source_id]
        if target.source_column not in src.columns:
            return None
        return src.normalized.values[target.source_column].rename(target.name)
