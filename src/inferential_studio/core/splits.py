"""Chronological train / validation / test splits and time-aware cross-validation.

Rules (docs/methodology.md):

* Splits follow time. Rows are never shuffled. Replicate lab samples (same sample time) always
  land in the same split and the same CV fold.
* A **purge gap** separates neighbouring splits and CV folds. Two samples whose feature windows
  (including filter history) overlap share process data; the *earlier* block therefore loses
  its samples closer than the purge gap to the first sample of the next block. The default gap
  is the longest feature history, so no process value feeds both sides of a boundary. The
  final test set itself is never trimmed.
* Cross-validation runs inside the training split only:

  - *expanding* (rolling origin, default): training times are cut into k + 1 contiguous blocks;
    fold i trains on blocks 0..i−1 and validates on block i. Only the past predicts the future.
  - *blocked*: k contiguous blocks; each is held out in turn and the rest (minus the purge gap
    on both sides of the held-out block) trains.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from inferential_studio.core.rules import ns
from inferential_studio.core.schemas import CvKind, SplitConfig, SplitMode

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
PURGED = "purge gap"
NOT_ELIGIBLE = "not eligible"
SPLITS = (TRAIN, VALIDATION, TEST)

NS = 1_000_000_000


class SplitError(ValueError):
    """The requested split or CV is impossible. The message is shown to the user."""


@dataclass
class SplitResult:
    labels: np.ndarray  # per feature-matrix row: TRAIN / VALIDATION / TEST / PURGED / NOT_ELIGIBLE
    validation_start: pd.Timestamp | None
    test_start: pd.Timestamp | None
    purge_s: float
    notes: list[str] = field(default_factory=list)

    def mask(self, label: str) -> np.ndarray:
        return self.labels == label

    def counts(self) -> dict[str, int]:
        return {lab: int((self.labels == lab).sum()) for lab in (*SPLITS, PURGED, NOT_ELIGIBLE)}


def _purge_before(
    times_ns: np.ndarray, rows: np.ndarray, boundary_ns: int, purge_ns: int
) -> np.ndarray:
    """Rows (bool) of the earlier block that sit within the purge gap before ``boundary_ns``."""
    return rows & (times_ns > boundary_ns - purge_ns)


def chronological_split(
    times: pd.DatetimeIndex,
    eligible: np.ndarray,
    config: SplitConfig,
    history_s: float = 0.0,
) -> SplitResult:
    """Assign every row to train / validation / test (or purge gap / not eligible)."""
    t = ns(pd.DatetimeIndex(times))
    eligible = np.asarray(eligible, dtype=bool)
    labels = np.full(t.size, NOT_ELIGIBLE, dtype=object)
    purge_s = float(history_s if config.purge_s is None else config.purge_s)
    purge_ns = int(round(purge_s * NS))
    notes: list[str] = []
    uniq = np.unique(t[eligible])
    if uniq.size == 0:
        raise SplitError("There are no model-eligible samples to split.")

    if config.mode is SplitMode.FRACTIONS:
        n = uniq.size
        n_train = int(round(config.train_fraction * n))
        n_val = int(round(config.validation_fraction * n))
        n_train = min(max(n_train, 1), n)
        n_val = min(n_val, n - n_train)
        b1 = uniq[n_train] if n_train < n else np.iinfo(np.int64).max
        b2 = uniq[n_train + n_val] if n_train + n_val < n else np.iinfo(np.int64).max
    else:
        b1 = pd.Timestamp(config.train_end).value
        b2 = pd.Timestamp(config.validation_end).value

    train = eligible & (t < b1)
    val = eligible & (t >= b1) & (t < b2)
    test = eligible & (t >= b2)
    labels[train], labels[val], labels[test] = TRAIN, VALIDATION, TEST

    # Purge: the earlier block loses samples whose windows overlap the next block's first one.
    blocks = [m for m in (train, val, test) if m.any()]
    for earlier, later in zip(blocks, blocks[1:], strict=False):
        first_later = int(t[later].min())
        drop = _purge_before(t, earlier, first_later, purge_ns)
        labels[drop] = PURGED
    n_purged = int((labels == PURGED).sum())
    if n_purged:
        notes.append(
            f"{n_purged} sample(s) dropped in purge gaps of {purge_s / 60:g} min before split "
            "boundaries, so no process data is shared across splits."
        )
    if not (labels == TRAIN).any():
        raise SplitError("The training period has no eligible samples. Move the boundary later.")
    if not val.any():
        notes.append("No validation samples: models are ranked by cross-validation error.")
    if not test.any():
        notes.append("No final test samples: performance on unseen data cannot be reported.")
    vstart = pd.Timestamp(int(t[val].min()), tz="UTC") if val.any() else None
    tstart = pd.Timestamp(int(t[test].min()), tz="UTC") if test.any() else None
    return SplitResult(labels, vstart, tstart, purge_s, notes)


def cv_folds(
    times: pd.DatetimeIndex,
    kind: CvKind,
    k: int,
    purge_s: float,
    *,
    min_train: int = 3,
    min_validation: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-aware folds over the given (training) rows: list of (train positions, val positions)."""
    t = ns(pd.DatetimeIndex(times))
    if t.size and np.any(np.diff(t) < 0):
        raise SplitError("Cross-validation rows must be in time order.")
    uniq = np.unique(t)
    n_blocks = k + 1 if kind is CvKind.EXPANDING else k
    if uniq.size < n_blocks * min_validation:
        raise SplitError(
            f"{k}-fold {kind.value} cross-validation needs at least {n_blocks * min_validation} "
            f"training samples; there are {uniq.size}. Use fewer folds or fix the hyperparameters."
        )
    purge_ns = int(round(purge_s * NS))
    blocks = np.array_split(uniq, n_blocks)
    folds = []
    first_val = 1 if kind is CvKind.EXPANDING else 0
    for i in range(first_val, n_blocks):
        v0, v1 = blocks[i][0], blocks[i][-1]
        val = (t >= v0) & (t <= v1)
        if kind is CvKind.EXPANDING:
            train = (t < v0) & (t <= v0 - purge_ns)
        else:
            train = ~val & ((t <= v0 - purge_ns) | (t >= v1 + purge_ns))
        if train.sum() < min_train or val.sum() < min_validation:
            raise SplitError(
                f"Cross-validation fold {i + 1 - first_val} would train on {int(train.sum())} and "
                f"validate on {int(val.sum())} samples. Use fewer folds, a shorter purge gap or "
                "fixed hyperparameters."
            )
        folds.append((np.flatnonzero(train), np.flatnonzero(val)))
    return folds
