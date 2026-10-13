"""What-if analysis: how the selected model's prediction changes when its inputs change.

Every model is affine in its features (``ŷ = b₀ + Σ bⱼ fⱼ``), so the effect of an input is
exact and needs no refitting:

* **shift** input j by δ: every prediction changes by ``bⱼ · δ`` (the line moves in parallel);
* **hold** input j at a value v: each prediction changes by ``bⱼ · (v − fⱼ(t))``, which removes
  that input's own variation from the line.

The inputs are the model's *features* (windowed, filtered values in the variable's units), not
raw tag readings. Moving one input while the others stay unchanged is a model sensitivity; in
the plant, correlated inputs usually move together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from inferential_studio.core.models import ModelResult, TrainingRun

SHIFT = "shift"
HOLD = "hold"


@dataclass(frozen=True)
class WhatIfInput:
    name: str
    coefficient: float
    low: float  # training minimum of the feature
    high: float  # training maximum
    mean: float  # training mean
    units: str | None = None

    @property
    def span(self) -> float:
        return self.high - self.low


@dataclass
class WhatIfBase:
    """The model and the samples it is explored on (training/validation, test once revealed)."""

    target: str
    intercept: float
    inputs: list[WhatIfInput]
    times: pd.DatetimeIndex
    splits: np.ndarray  # split label per row
    actual: np.ndarray
    X: np.ndarray  # rows x inputs, feature values in input order
    predicted: np.ndarray  # the model's prediction, b₀ + X·b

    @property
    def coefficients(self) -> np.ndarray:
        return np.array([i.coefficient for i in self.inputs])


@dataclass
class WhatIfResult:
    predicted: np.ndarray  # what-if prediction per row
    delta: np.ndarray  # what-if − original, per row
    impact: dict[str, float]  # mean change of the prediction caused by each input
    outside: np.ndarray  # bool per row: some changed input is outside its training range
    outside_inputs: list[str]  # inputs that leave the training range on at least one row


def what_if_base(
    run: TrainingRun,
    model: ModelResult,
    splits: tuple[str, ...],
    units: dict[str, str | None] | None = None,
) -> WhatIfBase:
    """Collect what the analysis needs for ``model`` on the rows of the given splits."""
    if not model.ok:
        raise ValueError(f"{model.name} has no fitted model.")
    names = list(model.coefficients.index)
    pred = model.predictions[model.predictions["split"].isin(splits)].sort_values("sample_time")
    rows = pred["row"].to_numpy()
    X = run.feature_values.iloc[rows][names].to_numpy(dtype=float)
    ranges = run.train_ranges.loc[names]
    units = units or {}
    inputs = [
        WhatIfInput(
            name,
            float(model.coefficients[name]),
            float(ranges.loc[name, "min"]),
            float(ranges.loc[name, "max"]),
            float(ranges.loc[name, "mean"]),
            units.get(name),
        )
        for name in names
    ]
    return WhatIfBase(
        target=run.target,
        intercept=float(model.intercept),
        inputs=inputs,
        times=pd.DatetimeIndex(pred["sample_time"]),
        splits=pred["split"].to_numpy(),
        actual=pred["actual"].to_numpy(dtype=float),
        X=X,
        predicted=pred["predicted"].to_numpy(dtype=float),
    )


def evaluate(base: WhatIfBase, changes: dict[str, tuple[str, float]]) -> WhatIfResult:
    """Prediction with ``changes`` = {input: (SHIFT, δ) or (HOLD, value)}; others unchanged."""
    X = base.X.copy()
    impact: dict[str, float] = {}
    outside = np.zeros(len(X), dtype=bool)
    leaving: list[str] = []
    for j, inp in enumerate(base.inputs):
        mode, amount = changes.get(inp.name, (SHIFT, 0.0))
        if mode == HOLD:
            X[:, j] = amount
        elif mode == SHIFT:
            if amount == 0.0:
                impact[inp.name] = 0.0
                continue
            X[:, j] = X[:, j] + amount
        else:
            raise ValueError(f"Unknown what-if mode {mode!r}.")
        impact[inp.name] = float(np.mean(inp.coefficient * (X[:, j] - base.X[:, j])))
        out = (X[:, j] < inp.low) | (X[:, j] > inp.high)
        if out.any():
            outside |= out
            leaving.append(inp.name)
    predicted = base.intercept + X @ base.coefficients
    return WhatIfResult(predicted, predicted - base.predicted, impact, outside, leaving)
