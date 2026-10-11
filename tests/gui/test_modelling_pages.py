from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from inferential_studio.core.schemas import CvKind, ModelKind, Scaling, SplitMode
from inferential_studio.core.splits import TEST
from inferential_studio.infrastructure.settings import AppSettings
from inferential_studio.ui.commands import edit_lists
from inferential_studio.ui.main_window import MainWindow

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "synthetic_distillation"

pytestmark = pytest.mark.gui


def idle(w) -> bool:
    s = w.session
    return not w.tasks.busy and not s._cleaning_running and not s._features_running


@pytest.fixture
def loaded(qtbot, tmp_path, monkeypatch):
    shutil.copytree(EXAMPLE_DIR, tmp_path / "ex")
    w = MainWindow(AppSettings(display_timezone="Asia/Singapore", author_name="Tester"))
    w.errors = []
    monkeypatch.setattr(w, "maybe_save", lambda: True)
    monkeypatch.setattr(w, "show_error", lambda title, msg, details="": w.errors.append(msg))
    qtbot.addWidget(w)
    w.show()
    with qtbot.waitSignal(w.tasks.task_ended, timeout=30000):
        w.open_project_file(tmp_path / "ex" / "sample_project.imsproj")
    qtbot.waitUntil(lambda: idle(w) and w.session.features is not None, timeout=30000)
    w.sidebar.setCurrentRow(5)
    yield w
    w.session._set_dirty(False)
    w.tasks.cancel_all()
    w.tasks.wait_for_done(5000)


def train(w, qtbot):
    w.run_training()
    qtbot.waitUntil(lambda: w.session.training is not None and idle(w), timeout=60000)


def test_setup_form_edits_config_with_undo(loaded, qtbot):
    page = loaded.pages["model_setup"]
    page.scaling.setCurrentIndex(page.scaling.findData(Scaling.ROBUST.value))
    page.cv_folds.setValue(4)
    page.cv_kind.setCurrentIndex(page.cv_kind.findData(CvKind.BLOCKED.value))
    page.rows[ModelKind.PCR].enabled.setChecked(False)
    assert page.apply()
    cfg = loaded.session.project.modelling
    assert cfg.scaling is Scaling.ROBUST and cfg.split.cv_folds == 4
    assert cfg.split.cv_kind is CvKind.BLOCKED
    assert not cfg.spec(ModelKind.PCR).enabled
    loaded.undo_stack.undo()
    assert loaded.session.project.modelling.scaling is Scaling.STANDARD
    assert page.scaling.currentData() == Scaling.STANDARD.value  # the form follows undo
    page.train_pct.setValue(80)
    page.val_pct.setValue(25)
    assert not page.apply()
    assert "leave samples for the test" in page.config_error.text()


def test_switching_to_dates_starts_from_current_boundaries(loaded, qtbot):
    page = loaded.pages["model_setup"]
    page._update_preview()
    before = page.split_note.text()
    page.mode.setCurrentIndex(page.mode.findData(SplitMode.DATES.value))
    assert page.apply()
    split = loaded.session.project.modelling.split
    assert split.mode is SplitMode.DATES and split.train_end < split.validation_end
    page._update_preview()
    # Same samples per period as the share-based split (boundaries rounded to minutes).
    assert page.split_note.text().split(";")[0] == before.split(";")[0]


def test_training_and_guarded_test_reveal(loaded, qtbot):
    train(loaded, qtbot)
    run = loaded.session.training
    assert not run.errors and len(run.models) == 4 and all(m.ok for m in run.models)
    assert loaded.sidebar.currentRow() == 6  # jumps to Training & Interpretation
    tpage = loaded.pages["training"]
    assert tpage.model_combo.count() == 4
    tpage.model_combo.setCurrentIndex(1)  # PLS
    assert tpage.coef_table.model_.rowCount() == 6  # intercept + 5 predictors
    assert tpage.load_kind.count() == 3  # VIP, weights, loadings
    vpage = loaded.pages["validation"]
    cols = list(vpage.comparison.model_.frame.columns)
    assert not any(c.startswith("test_") for c in cols)
    assert vpage.reveal_button.isEnabled()
    shown = vpage.metrics.model_.frame.index.tolist()
    assert "Final test" not in shown
    loaded.session.reveal_test()
    assert any(c.startswith("test_") for c in vpage.comparison.model_.frame.columns)
    assert "Final test" in vpage.metrics.model_.frame.index.tolist()
    assert not vpage.reveal_button.isEnabled()
    n_test = int(run.split.mask(TEST).sum())
    assert all(m.metric(TEST, "n") == n_test for m in run.models)
    # Editing an input after training marks the results out of date (kept, not discarded).
    specs = loaded.session.project.features
    edit_lists(loaded.undo_stack, loaded.session, "edit",
               features=[replace(specs[0], window_s=600)] + specs[1:])  # fmt: skip
    assert loaded.session.training_stale and loaded.session.training is run
    assert not vpage.banner.isHidden() and "out of date" in vpage.banner.text()
    qtbot.waitUntil(lambda: idle(loaded), timeout=30000)


def test_training_reports_impossible_configuration(loaded, qtbot):
    cfg = loaded.session.project.modelling
    models = tuple(
        replace(m, n_components=20) if m.kind is ModelKind.PLS else m for m in cfg.models
    )
    edit_lists(loaded.undo_stack, loaded.session, "edit", modelling=replace(cfg, models=models))
    train(loaded, qtbot)
    pls = next(m for m in loaded.session.training.models if m.spec.kind is ModelKind.PLS)
    assert not pls.ok and "at most 5" in pls.error
    tpage = loaded.pages["training"]
    assert any(
        "not trained" in tpage.model_combo.itemText(i) for i in range(tpage.model_combo.count())
    )
    vpage = loaded.pages["validation"]
    assert vpage.model_combo.count() == 3  # only trained models have diagnostics
