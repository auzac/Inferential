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


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_compact_validation_keeps_diagnostics_and_controls_reachable(loaded, qtbot, theme):
    from PySide6.QtCore import QPoint

    train(loaded, qtbot)
    loaded.set_theme(theme)
    loaded.resize(960, 640)
    loaded.sidebar.setCurrentRow(7)
    page = loaded.pages["validation"]
    page.tabs.setCurrentWidget(page.what_if)
    qtbot.wait(200)
    assert not page.comparison_toggle.isChecked()
    assert not page.comparison_panel.isVisible() and not page.metrics.isVisible()
    assert page.tabs.height() >= 260
    assert not page.what_if.inputs_chart.isVisibleTo(page.what_if)
    assert page.what_if.output_chart.height() >= 150
    assert page.what_if.length.isVisibleTo(page)
    viewport = page.what_if._list.parentWidget()
    list_top = viewport.mapTo(page.what_if.summary.parentWidget(), QPoint()).y()
    assert page.what_if.summary.geometry().bottom() < list_top
    page.comparison_toggle.click()
    qtbot.wait(20)
    assert page.metrics.isVisible() and page.comparison_panel.isVisible()
    assert page.content_scroll.verticalScrollBar().maximum() > 0
    loaded.resize(1440, 900)
    qtbot.wait(20)
    assert page.comparison_toggle.isChecked()


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
    assert (
        "leave samples for the test" in loaded.session.messages.condition("model_setup.config").text
    )


def test_switching_to_dates_starts_from_current_boundaries(loaded, qtbot):
    page = loaded.pages["model_setup"]
    page._update_preview()
    split_text = lambda: loaded.session.messages.condition("model_setup.split").text  # noqa: E731
    before = split_text()
    page.mode.setCurrentIndex(page.mode.findData(SplitMode.DATES.value))
    assert page.apply()
    split = loaded.session.project.modelling.split
    assert split.mode is SplitMode.DATES and split.train_end < split.validation_end
    page._update_preview()
    # Same samples per period as the share-based split (boundaries rounded to minutes).
    assert split_text().split(";")[0] == before.split(";")[0]


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
    state = loaded.session.messages.condition("training.state")
    assert "out of date" in state.text and state.visible_in("validation")
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


def test_what_if_live_trend_responds_like_the_soft_sensor(loaded, qtbot):
    import numpy as np

    train(loaded, qtbot)
    vpage = loaded.pages["validation"]
    view = vpage.what_if
    assert vpage.tabs.indexOf(view) >= 0
    assert view._model is not None and view.steps() == {}
    assert "steady state" in view.summary.text()
    baseline = view._baseline
    assert np.allclose(view._y, baseline)  # starts settled
    top = view._controls[0]  # largest possible effect first
    top.slider.setValue(750)  # +50 % of the training range: changes now
    step = top.step()
    assert step == pytest.approx(0.5 * top.inp.span) and "Δŷ" in top.effect.text()
    expected = top.inp.coefficient * step
    assert view._target - baseline == pytest.approx(expected, rel=1e-9)
    assert "settling at" in view.summary.text()
    stepped_at = view._schedule[-1][0]
    # The clock runs: nothing moves before the change, then it settles at coefficient × step.
    while view._t[-1] < stepped_at + view._view:
        view._tick()
    before = view._t < stepped_at
    assert np.allclose(view._y[before], baseline)
    assert view._y[-1] - baseline == pytest.approx(expected, rel=1e-9)
    assert view._t[-1] - view._t[0] <= view._view + view._dt  # the trend scrolls
    # Another model of the same run keeps the changes and starts settled at them.
    vpage.model_combo.setCurrentIndex(1)
    same = next(c for c in view._controls if c.inp.name == top.inp.name)
    assert same.step() == pytest.approx(step)
    assert view._y[-1] == pytest.approx(view._target, rel=1e-9)
    view.reset()
    assert view.steps() == {} and np.allclose(view._y, view._baseline)
    from inferential_studio.ui.widgets.what_if import COMPACT_HEIGHT

    # Short screens: the inputs chart gives way to the prediction.
    assert view.inputs_chart.isVisibleTo(view) == (view.height() >= COMPACT_HEIGHT)
