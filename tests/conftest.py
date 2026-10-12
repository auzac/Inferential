"""Shared test configuration.

* Qt runs with the offscreen platform unless QT_QPA_PLATFORM is already set, so GUI tests
  work headless (CI) and do not flash windows on a developer desktop.
* The offscreen plugin has no system font database; on Windows, point it at the installed
  fonts so text metrics match a real session (Qt no longer ships its own fonts).
* Every test gets an isolated application home (settings + logs) in a temp folder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    if fonts.is_dir():
        os.environ.setdefault("QT_QPA_FONTDIR", str(fonts))

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "synthetic_distillation"


@pytest.fixture(autouse=True)
def isolated_app_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    home = tmp_path_factory.mktemp("ims_home")
    monkeypatch.setenv("INFERENTIAL_STUDIO_HOME", str(home))
    return home


@pytest.fixture(scope="session")
def small_dataset():
    """A short synthetic dataset shared by tests that only read it."""
    from inferential_studio.core.synthetic import SyntheticConfig, generate_distillation_dataset

    return generate_distillation_dataset(SyntheticConfig(days=6, seed=7))
