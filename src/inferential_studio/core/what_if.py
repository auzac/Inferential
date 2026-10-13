"""What-if step test: how the inferential's prediction responds when inputs are bumped.

Like a step test on a simulator at steady state:

1. every input sits at a constant **baseline** (its training mean);
2. at the step time, chosen inputs change by their step size and stay there;
3. the inputs pass through the model's **own feature pipeline** (deadtime, trailing filter,
   aggregation window, coverage checks — the same code as training and batch prediction) and
   the model predicts on a fine time grid.

The response is therefore the soft sensor's response as it would appear online: flat during
the deadtime, then a ramp over the averaging window (an EWMA adds a first-order lag), settling
at ``Σ bⱼ·Δⱼ`` (the model is linear in its features). Its shape comes from the feature design,
not from the real process dynamics. A non-causal (centred) window responds *before* the step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from inferential_studio.core.features import TargetSamples, build_feature_matrix, data_span_s
from inferential_studio.core.models import ModelResult, TrainingRun
from inferential_studio.core.schemas import Aggregation, FeatureSpec, FilterKind

EPOCH = pd.Timestamp("2000-01-01", tz="UTC")  # arbitrary origin; results are in seconds from it
SETTLED = 0.95  # "settled" = 95 % of the final change
# An EWMA weighs its whole history. On real data the history is long; the steady-state lead-in
# must be too, or the normalised weights over a short history distort the response. After 20
# half-lives the remaining weight is below 1e-6.
EWMA_WARMUP_HALFLIVES = 20.0


@dataclass(frozen=True)
class StepInput:
    name: str
    coefficient: float
    baseline: float  # steady-state value: the training mean of the feature
    low: float  # training range of the feature
    high: float
    units: str | None = None
    spec: FeatureSpec | None = None

    @property
    def span(self) -> float:
        return self.high - self.low


@dataclass
class StepModel:
    target: str
    intercept: float
    inputs: list[StepInput]

    @property
    def coefficients(self) -> np.ndarray:
        return np.array([i.coefficient for i in self.inputs])

    @property
    def baseline_prediction(self) -> float:
        return float(self.intercept + self.coefficients @ [i.baseline for i in self.inputs])

    def response_span_s(self) -> float:
        """Longest time from a step until every feature has fully taken it in."""
        spans = [data_span_s(i.spec)[1] - data_span_s(i.spec)[0] for i in self.inputs if i.spec]
        delays = [i.spec.effective_delay_s for i in self.inputs if i.spec]
        return max(spans, default=0.0) + max(delays, default=0.0)


@dataclass
class StepResult:
    t_s: np.ndarray  # seconds from the start of the run
    inputs: dict[str, np.ndarray]  # input value over time (baseline + step)
    predicted: np.ndarray  # NaN where a feature failed its checks
    baseline: float  # prediction at steady state
    final_change: float  # Σ bⱼ·Δⱼ: where the output settles
    first_response_s: float | None  # first time the prediction moves (relative to the step)
    settled_s: float | None  # time to reach 95 % of the final change (relative to the step)
    outside: list[str] = field(default_factory=list)  # inputs stepped outside the training range
    notes: list[str] = field(default_factory=list)


def step_model(
    run: TrainingRun, model: ModelResult, units: dict[str, str | None] | None = None
) -> StepModel:
    """The selected model with each input's coefficient, baseline, training range and feature."""
    if not model.ok:
        raise ValueError(f"{model.name} has no fitted model.")
    units = units or {}
    ranges = run.train_ranges
    inputs = [
        StepInput(
            name,
            float(model.coefficients[name]),
            float(ranges.loc[name, "mean"]),
            float(ranges.loc[name, "min"]),
            float(ranges.loc[name, "max"]),
            units.get(name),
            run.feature_specs.get(name),
        )
        for name in model.coefficients.index
    ]
    return StepModel(run.target, float(model.intercept), inputs)


def default_timing(model: StepModel) -> tuple[float, float]:
    """(step time, run length) in seconds: a quiet lead-in, then room for the full response."""
    span = model.response_span_s()
    step_at = _round_up(max(900.0, 0.25 * span), 300.0)
    length = _round_up(max(7200.0, step_at + 2.0 * span), 1800.0)
    return step_at, length


def simulate(
    model: StepModel,
    steps: dict[str, float],
    step_at_s: float,
    length_s: float,
    *,
    dt_s: float | None = None,
) -> StepResult:
    """Run the step test. ``steps`` = {input: change}; inputs not listed stay at baseline."""
    if length_s <= 0 or not 0 <= step_at_s <= length_s:
        raise ValueError("The step must happen within the run.")
    dt = dt_s or _grid_step(model)
    warmup = _round_up(_warmup_s(model) + dt, dt)  # windows and filters start at steady state
    t_raw = np.arange(-warmup, length_s + dt / 2, dt)
    t_pred = np.arange(0.0, length_s + dt / 2, dt)
    index = EPOCH + pd.to_timedelta(t_raw, unit="s")
    after = t_raw >= step_at_s
    frame = {}
    for inp in model.inputs:
        x = np.full(t_raw.size, inp.baseline)
        x[after] += steps.get(inp.name, 0.0)
        frame[inp.name] = x
    process = pd.DataFrame(frame, index=index)
    times = EPOCH + pd.to_timedelta(t_pred, unit="s")
    n = t_pred.size
    dummy = TargetSamples("__y__", pd.DatetimeIndex(times), np.zeros(n), np.zeros(n, bool),
                          np.zeros(n, bool))  # fmt: skip
    specs = [i.spec or _default_spec(i.name) for i in model.inputs]
    fm = build_feature_matrix(dummy, process, specs)
    names = [i.name for i in model.inputs]
    X = fm.X[names].to_numpy(dtype=float)
    ok = (fm.status[names] == "ok").all(axis=1).to_numpy()
    predicted = np.where(ok, model.intercept + X @ model.coefficients, np.nan)
    baseline = model.baseline_prediction
    final = float(sum(i.coefficient * steps.get(i.name, 0.0) for i in model.inputs))
    first, settled = _timing(t_pred, predicted, baseline, final, step_at_s)
    outside = [
        i.name
        for i in model.inputs
        if steps.get(i.name, 0.0) and not i.low <= i.baseline + steps[i.name] <= i.high
    ]
    notes = []
    if any(s.aggregation is Aggregation.STD for s in specs):
        notes.append(
            "Standard-deviation features are 0 at steady state (no variation), which is usually "
            "outside their training range; they react to a step only while it passes through "
            "the window."
        )
    if fm.non_causal:
        notes.append(
            "Centred windows read data after the sample time, so the prediction moves before "
            "the step: " + ", ".join(fm.non_causal) + "."
        )
    if not ok.all():
        notes.append("Some times have no prediction: a feature failed its coverage checks.")
    return StepResult(
        t_pred, {k: v[t_raw >= 0] for k, v in frame.items()}, predicted, baseline, final,
        first, settled, outside, notes,
    )  # fmt: skip


# --------------------------------------------------------------------------- internals


def _round_up(value: float, step: float) -> float:
    return float(np.ceil(value / step) * step)


def _warmup_s(model: StepModel) -> float:
    out = model.response_span_s()
    for i in model.inputs:
        f = i.spec.filter if i.spec else None
        if f is not None and f.kind is FilterKind.EWMA:
            start = -data_span_s(i.spec)[0]
            out = max(out, start + EWMA_WARMUP_HALFLIVES * float(f.halflife_s))
    return out


def _grid_step(model: StepModel) -> float:
    """1 minute, finer for windows shorter than 10 minutes (at least 10 points per window)."""
    windows = [i.spec.window_s for i in model.inputs if i.spec]
    return float(max(1.0, min(60.0, min(windows, default=600.0) / 10)))


def _default_spec(name: str) -> FeatureSpec:
    return FeatureSpec(name)


def _timing(
    t: np.ndarray, y: np.ndarray, baseline: float, final: float, step_at: float
) -> tuple[float | None, float | None]:
    if final == 0 or not np.isfinite(y).any():
        return None, None
    change = (y - baseline) / final  # 0 before, 1 when settled
    tol = 1e-9
    moved = np.flatnonzero(np.isfinite(change) & (np.abs(change) > tol))
    first = float(t[moved[0]] - step_at) if moved.size else None
    # Settled: from this time on the change stays at or beyond 95 % of the final value.
    below = np.flatnonzero(~(np.isfinite(change) & (change >= SETTLED)))
    if below.size == 0:
        settled = float(t[0] - step_at)
    elif below[-1] + 1 < t.size:
        settled = float(t[below[-1] + 1] - step_at)
    else:
        settled = None  # not within the run
    return first, settled
