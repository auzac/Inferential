"""Evaluation of logical exclusion rules on a time-indexed frame.

Rules are typed data (see :class:`~inferential_studio.core.schemas.ExclusionRule`); they are
interpreted here with whitelisted NumPy operations. Nothing is ever passed to ``eval``.

Semantics (also in docs/methodology.md):

* Time basis: the rows of the frame passed in (the process data's own timestamps). Nothing is
  interpolated; a variable without a value on a row does not satisfy a comparison there.
* Comparisons (``<``, ``between`` …) are False where the value is missing.
  ``is missing`` is True where there is no valid number (blank or text); ``is invalid`` is True
  where the cell held non-numeric text.
* Minimum duration: consecutive rows where the condition holds form a run. A run is broken by
  a row where it does not hold *or* by a time gap longer than ``max_gap`` (default 2× the
  median sampling interval). A run counts as lasting ``last − first + median interval``, so
  five 1-minute samples last 5 minutes. Runs shorter than the minimum are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from inferential_studio.core.schemas import (
    Combinator,
    Condition,
    ConditionGroup,
    ExclusionRule,
    RuleOperator,
)


class RuleError(ValueError):
    """A rule refers to data that is not available. The message is shown to the user."""


def ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Timestamps as int64 nanoseconds whatever the index resolution (pandas 3 may use µs).

    Scales the integers directly: ``as_unit("ns")`` copies through a slow path on big indexes.
    """
    idx = pd.DatetimeIndex(index)
    factor = _NS_PER_UNIT[idx.unit]
    i8 = idx.asi8
    if factor == 1:
        return i8
    nat = i8 == np.iinfo(np.int64).min
    out = i8 * factor
    if nat.any():
        out[nat] = np.iinfo(np.int64).min
    return out


_NS_PER_UNIT = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}


def median_interval_ns(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 0
    diffs = np.diff(ns(index))
    diffs = diffs[diffs > 0]
    return int(np.median(diffs)) if diffs.size else 0


def evaluate_condition(
    cond: Condition, frame: pd.DataFrame, text_mask: pd.DataFrame | None
) -> np.ndarray:
    if cond.variable not in frame.columns:
        raise RuleError(f"Variable '{cond.variable}' is not in the loaded process data.")
    x = frame[cond.variable].to_numpy(dtype=float)
    valid = ~np.isnan(x)
    op = cond.operator
    if op is RuleOperator.IS_MISSING:
        return ~valid
    if op is RuleOperator.IS_INVALID:
        if text_mask is None or cond.variable not in text_mask:
            return np.zeros(len(x), dtype=bool)
        return text_mask[cond.variable].to_numpy(dtype=bool)
    v, v2 = cond.value, cond.value2
    with np.errstate(invalid="ignore"):
        if op is RuleOperator.LT:
            out = x < v
        elif op is RuleOperator.LE:
            out = x <= v
        elif op is RuleOperator.GT:
            out = x > v
        elif op is RuleOperator.GE:
            out = x >= v
        elif op is RuleOperator.EQ:
            out = x == v
        elif op is RuleOperator.NE:
            out = x != v
        elif op is RuleOperator.BETWEEN:
            out = (x >= v) & (x <= v2)
        elif op is RuleOperator.OUTSIDE:
            out = (x < v) | (x > v2)
        else:  # pragma: no cover - the enum is exhaustive
            raise RuleError(f"Unsupported operator {op!r}.")
    return out & valid


def evaluate_group(
    group: ConditionGroup, frame: pd.DataFrame, text_mask: pd.DataFrame | None
) -> np.ndarray:
    results = [
        evaluate_condition(item, frame, text_mask)
        if isinstance(item, Condition)
        else evaluate_group(item, frame, text_mask)
        for item in group.items
    ]
    combine = np.logical_and if group.combinator is Combinator.AND else np.logical_or
    out = results[0].copy()
    for r in results[1:]:
        out = combine(out, r)
    return out


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (inclusive) positions of consecutive True runs."""
    if mask.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return edges[0::2], edges[1::2] - 1


def split_runs_at_gaps(
    mask: np.ndarray, index: pd.DatetimeIndex, max_gap_ns: int
) -> tuple[np.ndarray, np.ndarray]:
    """Runs of True that never span a time gap longer than ``max_gap_ns`` (vectorized)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    if max_gap_ns <= 0:
        return runs(mask)
    gap = np.diff(ns(index)) > max_gap_ns  # gap[i]: step from row i to row i+1 is a gap
    prev_true = np.concatenate([[False], mask[:-1] & ~gap])
    next_true = np.concatenate([mask[1:] & ~gap, [False]])
    starts = np.flatnonzero(mask & ~prev_true)
    ends = np.flatnonzero(mask & ~next_true)
    return starts, ends


def apply_min_duration(
    mask: np.ndarray,
    index: pd.DatetimeIndex,
    min_duration_s: float,
    *,
    max_gap_ns: int | None = None,
) -> np.ndarray:
    """Keep only runs that last at least ``min_duration_s`` (see module docstring)."""
    step = median_interval_ns(index)
    gap = 2 * step if max_gap_ns is None else max_gap_ns
    starts, ends = split_runs_at_gaps(mask, index, gap)
    t = ns(index)
    need = int(round(min_duration_s * 1e9))
    keep = (t[ends] - t[starts] + step) >= need
    return fill_runs(len(mask), starts[keep], ends[keep])


def fill_runs(n: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Boolean array of length ``n`` that is True on every [start, end] run (inclusive)."""
    delta = np.zeros(n + 1, dtype=np.int64)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends + 1, -1)
    return np.cumsum(delta[:-1]) > 0


def evaluate_rule(
    rule: ExclusionRule, frame: pd.DataFrame, text_mask: pd.DataFrame | None = None
) -> np.ndarray:
    """Boolean array over ``frame``'s rows: True where the rule applies."""
    mask = evaluate_group(rule.condition, frame, text_mask)
    if rule.min_duration_s:
        mask = apply_min_duration(mask, frame.index, rule.min_duration_s)
    return mask


def mask_to_intervals(
    mask: np.ndarray, index: pd.DatetimeIndex
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Closed [first, last] time spans of consecutive True rows (for display and reports)."""
    starts, ends = runs(mask)
    return [(index[s], index[e]) for s, e in zip(starts, ends, strict=True)]
