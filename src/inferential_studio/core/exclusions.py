"""One authoritative exclusion decision, assembled from every source of exclusion.

Inputs (all persisted in the project, all reversible):

* manual intervals — global (operating regime) or for listed variables;
* enabled rules — global (whole rows) or for listed variables; action exclude or flag;
* detector flags — per variable, excluded only when the user accepted them (or the detector is
  set to auto-exclude and the event was not rejected); otherwise *pending review*.

Outputs keep the categories separate so the user can always see *why* a value is excluded.
Raw data is never touched: exclusions are masks over the normalized data.

A flag on one variable excludes only that variable's value; it does not invalidate other
variables measured at the same time. (A later supervised observation that needs that variable
may still become unusable — decided during alignment in Milestone 3.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from inferential_studio.core.quality import DetectorResult, FlagEvent, event_mask
from inferential_studio.core.rules import evaluate_rule, fill_runs, median_interval_ns, ns
from inferential_studio.core.schemas import (
    ExclusionInterval,
    FlagDecisionKind,
    Project,
    RuleAction,
)


class EventStatus:
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AUTO = "auto-excluded"
    COVERED = "covered"  # awaiting review, but already excluded by another reason (e.g. a shutdown)


@dataclass
class Category:
    """One reason for exclusion, e.g. 'Manual: shutdown' or 'Rule: low load'."""

    key: str
    label: str
    kind: str  # "manual", "rule", "detector", "rule-flag"
    excluding: bool  # False for flag-only categories
    rows: np.ndarray  # bool per row: where this category applies
    signals: tuple[str, ...]  # variables affected; empty = every variable (operating regime)


@dataclass
class ReviewedEvent:
    event: FlagEvent
    status: str


@dataclass
class TargetMask:
    """Exclusion state of the sparse target samples (lab sample times)."""

    excluded: pd.Series  # bool, index = sample time
    pending: pd.Series


@dataclass
class CleaningState:
    index: pd.DatetimeIndex
    regime: np.ndarray  # rows excluded for every variable (operating-regime exclusions)
    excluded: pd.DataFrame  # bool per row x variable: final exclusion (includes regime)
    pending: (
        pd.DataFrame
    )  # bool: flagged for review (detector events awaiting a decision, rule flags)
    categories: list[Category] = field(default_factory=list)
    events: list[ReviewedEvent] = field(default_factory=list)
    target: TargetMask | None = None
    rule_errors: dict[str, str] = field(default_factory=dict)  # rule id -> message

    @property
    def median_interval(self) -> pd.Timedelta:
        return pd.Timedelta(median_interval_ns(self.index), unit="ns")

    def cleaned(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` with excluded cells set to NaN (a view for display and statistics)."""
        return frame.mask(self.excluded.reindex(columns=frame.columns, fill_value=False))


def interval_rows(index: pd.DatetimeIndex, intervals: list[ExclusionInterval]) -> np.ndarray:
    """Rows inside any of the closed intervals [start, end]."""
    if not intervals or len(index) == 0:
        return np.zeros(len(index), dtype=bool)
    i8 = ns(index)
    starts = np.searchsorted(
        i8, np.array([pd.Timestamp(iv.start).value for iv in intervals]), side="left"
    )
    ends = (
        np.searchsorted(
            i8, np.array([pd.Timestamp(iv.end).value for iv in intervals]), side="right"
        )
        - 1
    )
    keep = ends >= starts
    return fill_runs(len(i8), starts[keep], ends[keep])


def event_status(project: Project, event: FlagEvent) -> str:
    decisions = {d.key: d.decision for d in project.flag_decisions}
    det = next((d for d in project.detectors if d.detector_id == event.detector_id), None)
    decision = decisions.get(
        (event.detector_id, event.variable, event.start.to_pydatetime(), event.end.to_pydatetime())
    )
    if decision is FlagDecisionKind.ACCEPTED:
        return EventStatus.ACCEPTED
    if decision is FlagDecisionKind.REJECTED:
        return EventStatus.REJECTED
    if det is not None and det.auto_exclude:
        return EventStatus.AUTO
    return EventStatus.PENDING


def compute_cleaning(
    project: Project,
    frame: pd.DataFrame,
    text_mask: pd.DataFrame | None = None,
    detector_results: dict[tuple[str, str], DetectorResult] | None = None,
    target: pd.Series | None = None,
) -> CleaningState:
    """Combine manual intervals, rules and reviewed detector flags into final masks."""
    index = frame.index
    columns = list(frame.columns)
    col_pos = {c: i for i, c in enumerate(columns)}
    n = len(index)
    excluded = np.zeros((n, len(columns)), dtype=bool)
    pending = np.zeros((n, len(columns)), dtype=bool)
    regime = np.zeros(n, dtype=bool)
    categories: list[Category] = []
    rule_errors: dict[str, str] = {}

    def add_category(
        key: str, label: str, kind: str, rows: np.ndarray, signals: tuple[str, ...], excluding: bool
    ):
        targets = (
            [col_pos[s] for s in signals if s in col_pos] if signals else list(range(len(columns)))
        )
        if not targets or not rows.any():
            return
        categories.append(Category(key, label, kind, excluding, rows, tuple(signals)))
        block = excluded if excluding else pending
        hit = np.flatnonzero(rows)
        block[np.ix_(hit, targets)] = True
        if excluding and not signals:
            regime[:] |= rows

    # Manual intervals, grouped by reason and scope so the breakdown stays readable.
    for iv in (iv for iv in project.exclusions if iv.enabled):
        rows = interval_rows(index, [iv])
        scope = "all signals" if iv.is_global else ", ".join(iv.signals)
        add_category(
            f"manual:{iv.interval_id}",
            f"Manual ({iv.reason.value}) — {scope}",
            "manual",
            rows,
            iv.signals,
            True,
        )

    for rule in (r for r in project.rules if r.enabled):
        try:
            rows = evaluate_rule(rule, frame, text_mask)
        except ValueError as exc:
            rule_errors[rule.rule_id] = str(exc)
            continue
        is_flag = rule.action is RuleAction.FLAG
        add_category(
            f"rule:{rule.rule_id}",
            f"{'Rule flag' if is_flag else 'Rule'}: {rule.name}",
            "rule-flag" if is_flag else "rule",
            rows,
            rule.signals,
            not is_flag,
        )

    # Manual intervals also apply to target samples (global ones, or ones naming the target).
    target_excluded = target_pending = None
    if target is not None:
        tname = str(target.name)
        relevant = [
            iv for iv in project.exclusions if iv.enabled and (iv.is_global or tname in iv.signals)
        ]
        target_excluded = pd.Series(interval_rows(target.index, relevant), index=target.index)
        target_pending = pd.Series(False, index=target.index)

    # Detector events. Exclusions decided so far (manual, rules) tell us which pending events
    # fall entirely inside data that is excluded anyway; those are reported separately.
    base = excluded.copy()
    reviewed: list[ReviewedEvent] = []
    for (det_id, variable), result in (detector_results or {}).items():
        det = next((d for d in project.detectors if d.detector_id == det_id), None)
        if det is None or not det.enabled:
            continue
        is_target = target is not None and variable == target.name
        if is_target:
            covered_by = target_excluded.reindex(result.index, fill_value=False).to_numpy()
        elif variable in col_pos and result.index.equals(index):
            covered_by = base[:, col_pos[variable]]
        else:
            covered_by = None
        i8 = ns(result.index)
        by_status: dict[str, list[FlagEvent]] = {}
        for ev in result.events:
            status = event_status(project, ev)
            if status == EventStatus.PENDING and covered_by is not None:
                s0 = int(np.searchsorted(i8, ev.start.value, side="left"))
                s1 = int(np.searchsorted(i8, ev.end.value, side="right"))
                rows = result.flags[s0:s1]
                if rows.any() and covered_by[s0:s1][rows].all():
                    status = EventStatus.COVERED
            reviewed.append(ReviewedEvent(ev, status))
            by_status.setdefault(status, []).append(ev)
        exclude_events = by_status.get(EventStatus.ACCEPTED, []) + by_status.get(
            EventStatus.AUTO, []
        )
        pending_events = by_status.get(EventStatus.PENDING, [])
        if is_target:
            target_excluded |= pd.Series(
                event_mask(result, exclude_events), index=result.index
            ).reindex(target.index, fill_value=False)
            target_pending |= pd.Series(
                event_mask(result, pending_events), index=result.index
            ).reindex(target.index, fill_value=False)
            continue
        if variable not in columns:
            continue
        aligned = result.index.equals(index)
        ex_rows = event_mask(result, exclude_events)
        pe_rows = event_mask(result, pending_events)
        if not aligned:  # defensive: map onto the frame's rows by timestamp
            ex_rows = (
                pd.Series(ex_rows, index=result.index).reindex(index, fill_value=False).to_numpy()
            )
            pe_rows = (
                pd.Series(pe_rows, index=result.index).reindex(index, fill_value=False).to_numpy()
            )
        label = f"Flag ({det.kind.value}) — {variable}"
        add_category(f"detector:{det_id}:{variable}", label, "detector", ex_rows, (variable,), True)
        if pe_rows.any():
            pending[pe_rows, col_pos[variable]] = True

    target_mask = None
    if target is not None:
        target_mask = TargetMask(target_excluded, target_pending & ~target_excluded)

    pending &= ~excluded
    return CleaningState(
        index=index,
        regime=regime,
        excluded=pd.DataFrame(excluded, index=index, columns=columns),
        pending=pd.DataFrame(pending, index=index, columns=columns),
        categories=categories,
        events=reviewed,
        target=target_mask,
        rule_errors=rule_errors,
    )


def summary_table(
    state: CleaningState, frame: pd.DataFrame, text_mask: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-variable counts: rows, valid values, excluded, pending review, invalid text, usable."""
    rows = []
    step = state.median_interval
    for col in frame.columns:
        present = frame[col].notna().to_numpy()
        ex = state.excluded[col].to_numpy()
        pe = state.pending[col].to_numpy()
        usable = present & ~ex
        rows.append(
            {
                "variable": col,
                "rows": len(frame),
                "valid_values": int(present.sum()),
                "invalid_text": int(text_mask[col].sum())
                if text_mask is not None and col in text_mask
                else 0,
                "excluded": int((ex & present).sum()),
                "pending_review": int((pe & present).sum()),
                "usable": int(usable.sum()),
                "usable_pct": 100.0 * usable.sum() / len(frame) if len(frame) else np.nan,
                "excluded_duration_h": float((ex.sum() * step).total_seconds() / 3600),
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def category_table(state: CleaningState) -> pd.DataFrame:
    """Rows affected and approximate duration per exclusion reason (categories can overlap)."""
    step = state.median_interval
    out = []
    for cat in state.categories:
        n = int(cat.rows.sum())
        out.append(
            {
                "reason": cat.label,
                "kind": cat.kind,
                "excludes": cat.excluding,
                "rows": n,
                "approx_hours": float((n * step).total_seconds() / 3600),
            }
        )
    return pd.DataFrame(out, columns=["reason", "kind", "excludes", "rows", "approx_hours"])
