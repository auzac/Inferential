"""The nine workflow stages shown in the sidebar (plain data, no widgets)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    key: str
    number: int
    title: str
    goal: str
    milestone: int  # milestone that delivers the working page


STAGES: tuple[Stage, ...] = (
    Stage(
        "import",
        1,
        "Data Import",
        "Load process and lab data and establish valid, traceable UTC timestamps.",
        1,
    ),
    Stage(
        "initial_eda",
        2,
        "Initial EDA",
        "Inspect raw data: trends, summary statistics, correlations and coverage.",
        1,
    ),
    Stage(
        "cleaning",
        3,
        "Cleaning & Exclusions",
        "Exclude invalid periods manually or by rule; review automatic anomaly flags.",
        2,
    ),
    Stage(
        "variables",
        4,
        "Variable Selection",
        "Reassess cleaned data, choose the target and candidate predictors, check VIF.",
        3,
    ),
    Stage(
        "processing",
        5,
        "Signal Processing",
        "Per-predictor deadtime, lookback window, aggregation, filtering and lab alignment.",
        3,
    ),
    Stage(
        "model_setup",
        6,
        "Model Setup",
        "Choose MLR, PLS, PCR or Ridge, chronological splits and time-aware CV.",
        4,
    ),
    Stage(
        "training",
        7,
        "Training & Interpretation",
        "Fit on training data only; coefficients, loadings, scores and VIP.",
        4,
    ),
    Stage(
        "validation",
        8,
        "Validation",
        "Compare models on unseen chronological data with residual diagnostics.",
        4,
    ),
    Stage(
        "export",
        9,
        "Export & Report",
        "Save the project, export predictions, model card and model artifact.",
        5,
    ),
)
