"""The nine workflow stages shown in the navigation rail, in workflow order, and how far along
each one a project is (for the progress bar)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inferential_studio.ui.session import ProjectSession


@dataclass(frozen=True)
class Stage:
    key: str
    number: int
    title: str
    goal: str
    milestone: int  # milestone that delivers the working page
    icon: str = ""  # icon file name in assets/icons/fluent (see ui/icons.py)


STAGES: tuple[Stage, ...] = (
    Stage(
        "import",
        1,
        "Data Import",
        "Load process and lab data and establish valid, traceable UTC timestamps.",
        1,
        icon="database",
    ),
    Stage(
        "initial_eda",
        2,
        "Initial EDA",
        "Inspect raw data: trends, summary statistics, correlations and coverage.",
        1,
        icon="data_trending",
    ),
    Stage(
        "cleaning",
        3,
        "Cleaning & Exclusions",
        "Exclude invalid periods manually or by rule; review automatic anomaly flags.",
        2,
        icon="broom",
    ),
    Stage(
        "variables",
        4,
        "Variable Selection",
        "Reassess cleaned data, choose the target and candidate predictors, check VIF.",
        3,
        icon="checkbox_checked",
    ),
    Stage(
        "processing",
        5,
        "Signal Processing",
        "Per-predictor deadtime, lookback window, aggregation, filtering and lab alignment.",
        3,
        icon="pulse",
    ),
    Stage(
        "model_setup",
        6,
        "Model Setup",
        "Choose MLR, PLS, PCR or Ridge, chronological splits and time-aware CV.",
        4,
        icon="options",
    ),
    Stage(
        "training",
        7,
        "Training & Interpretation",
        "Fit on training data only; coefficients, loadings, scores and VIP.",
        4,
        icon="brain_circuit",
    ),
    Stage(
        "validation",
        8,
        "Validation",
        "Compare models on unseen chronological data with residual diagnostics.",
        4,
        icon="target_arrow",
    ),
    Stage(
        "export",
        9,
        "Export & Report",
        "Save the project, export predictions, model card and model artifact.",
        5,
        icon="share",
    ),
)


DONE, ATTENTION, TODO = "done", "attention", "todo"


def stage_states(session: ProjectSession) -> list[tuple[str, str]]:
    """``(state, note)`` per stage, in workflow order, for the progress bar.

    Stages that leave something behind are done when it is there (data loaded, target and
    predictors chosen, features aligned, a current training run, an export written); review
    stages are done once visited. Visited but not done needs attention.
    """
    p, visited = session.project, set(session.project.progress)
    trained = session.training is not None
    met = {
        "import": bool(session.data.sources) and not session.data.failures,
        "initial_eda": "initial_eda" in visited,
        "cleaning": "cleaning" in visited,
        "variables": p.target is not None and bool(p.predictors),
        "processing": session.features is not None,
        "model_setup": "model_setup" in visited and any(m.enabled for m in p.modelling.models),
        "training": trained and not session.training_stale,
        "validation": "validation" in visited and trained,
        "export": "export" in visited,
    }
    todo = {
        "import": "Load process and lab data.",
        "variables": "Choose the target and at least one predictor.",
        "processing": session.features_error or "Features are not aligned yet.",
        "model_setup": "Enable at least one model.",
        "training": "Inputs changed since training: train again." if trained else "Train models.",
        "validation": "Train models first.",
        "export": "Save a model card, predictions or model file.",
    }
    out = []
    for stage in STAGES:
        if met[stage.key]:
            out.append((DONE, "Done."))
        elif stage.key in visited or (stage.key == "training" and trained):
            out.append((ATTENTION, todo.get(stage.key, "")))
        else:
            out.append((TODO, "Not started."))
    return out
