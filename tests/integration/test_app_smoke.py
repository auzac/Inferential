"""The packaged-app smoke tests, run from source: plain start and the full example workflow."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "synthetic_distillation"


def _run(args, home):  # noqa: ANN001
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "INFERENTIAL_STUDIO_HOME": str(home)}
    return subprocess.run(
        [sys.executable, "-m", "inferential_studio", *args],
        env=env, capture_output=True, text=True, timeout=240,
    )  # fmt: skip


def test_smoke_test_flag(tmp_path):
    result = _run(["--smoke-test"], tmp_path / "home")
    assert result.returncode == 0, result.stderr[-2000:]


def test_workflow_smoke_test_writes_model_card(tmp_path):
    shutil.copytree(EXAMPLE_DIR, tmp_path / "ex")
    home = tmp_path / "home"
    result = _run([str(tmp_path / "ex" / "sample_project.imsproj"), "--smoke-test"], home)
    assert result.returncode == 0, result.stderr[-2000:]
    card = (home / "smoke-model-card.html").read_text(encoding="utf-8")
    assert "Product_Composition" in card and "<svg" in card


def test_workflow_smoke_test_fails_on_missing_data(tmp_path):
    shutil.copytree(EXAMPLE_DIR, tmp_path / "ex")
    (tmp_path / "ex" / "tower_process.csv").unlink()
    result = _run([str(tmp_path / "ex" / "sample_project.imsproj"), "--smoke-test"], tmp_path / "h")
    assert result.returncode != 0
