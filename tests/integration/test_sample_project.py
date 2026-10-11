"""Milestone 0 gate: the committed sample project fixture loads into core code."""

from __future__ import annotations

import json
from pathlib import Path

from inferential_studio.core.persistence import load_project, save_project, verify_source
from inferential_studio.core.schemas import SourceKind, VariableRole
from inferential_studio.core.synthetic import create_example

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "synthetic_distillation"


def test_committed_sample_project_loads_and_sources_verify():
    project_path = EXAMPLE_DIR / "sample_project.imsproj"
    project = load_project(project_path)
    assert project.target is not None and project.target.name == "Product_Composition"
    assert {s.kind for s in project.sources} == {SourceKind.PROCESS, SourceKind.LAB}
    assert project.source("lab").result_timestamp is not None
    assert project.source("process").timestamp.dayfirst is True
    for source in project.sources:
        assert verify_source(source, project_path) is None, source.path
    assert all(f.is_causal for f in project.features)
    truth = json.loads((EXAMPLE_DIR / "ground_truth.json").read_text())
    assert all(f.deadtime_s == truth["target_model"]["deadtime_min"] * 60 for f in project.features)


def test_generate_load_save_reload(tmp_path):
    paths = create_example(tmp_path / "ex")
    project = load_project(paths["project"])
    assert len(project.predictors) == 5
    assert sum(v.role is VariableRole.TARGET for v in project.variables) == 1
    copy = tmp_path / "copy.imsproj"
    saved = save_project(project, copy)
    assert load_project(copy) == saved


def test_committed_example_data_is_reproducible_from_code(tmp_path):
    """The committed example files must be exactly what the current generator produces."""
    import pandas as pd

    paths = create_example(tmp_path / "regen")
    assert paths["process"].read_bytes() == (EXAMPLE_DIR / "tower_process.csv").read_bytes()
    pd.testing.assert_frame_equal(
        pd.read_excel(paths["lab"], sheet_name="Lab"),
        pd.read_excel(EXAMPLE_DIR / "lab_results.xlsx", sheet_name="Lab"),
    )
    assert json.loads(paths["truth"].read_text()) == json.loads(
        (EXAMPLE_DIR / "ground_truth.json").read_text()
    )
