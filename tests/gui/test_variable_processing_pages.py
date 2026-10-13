from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from inferential_studio.core.schemas import Aggregation, VariableRole, WindowPlacement
from inferential_studio.infrastructure.settings import AppSettings
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
    yield w
    w.session._set_dirty(False)
    w.tasks.cancel_all()
    w.tasks.wait_for_done(5000)


def test_features_follow_the_project(loaded, qtbot):
    fm = loaded.session.features
    assert fm.features == [v.name for v in loaded.session.project.predictors]
    assert len(fm.y) == loaded.session.data.n_lab_samples  # one row per lab sample
    assert fm.n_eligible > 250


def test_variable_page_selection_is_undoable(loaded, qtbot):
    loaded.sidebar.setCurrentRow(3)
    page = loaded.pages["variables"]
    qtbot.waitUntil(lambda: page.ranking.model_.rowCount() == 5, timeout=10000)
    assert page.ranking.model_.frame["vif"].notna().all()
    page.predictors.set_checked("Feed_Temp", True)
    page._apply_predictors()
    qtbot.waitUntil(lambda: idle(loaded), timeout=30000)
    project = loaded.session.project
    assert project.variable("Feed_Temp").role is VariableRole.PREDICTOR
    assert "Feed_Temp" in loaded.session.features.features
    loaded.undo_stack.undo()
    qtbot.waitUntil(lambda: idle(loaded), timeout=30000)
    assert loaded.session.project.variable("Feed_Temp").role is VariableRole.UNUSED
    assert "Feed_Temp" not in loaded.session.features.features
    # Raw vs cleaned comparison runs in the background while the page is visible.
    qtbot.waitUntil(lambda: page.raw_clean.model_.rowCount() > 0, timeout=30000)
    assert "r_y_cleaned" in page.raw_clean.model_.frame


def test_processing_page_batch_edit_and_causality(loaded, qtbot):
    loaded.sidebar.setCurrentRow(4)
    page = loaded.pages["processing"]
    assert page.table.rowCount() == 5
    page.table.selectRow(0)
    page.window.setValue(15)
    page.aggregation.setCurrentIndex(page.aggregation.findData(Aggregation.MEDIAN.value))
    assert page.apply_form(all_rows=True)
    qtbot.waitUntil(lambda: idle(loaded), timeout=30000)
    specs = loaded.session.project.features
    assert all(s.window_s == 900 and s.aggregation is Aggregation.MEDIAN for s in specs)
    assert loaded.session.messages.condition("processing.noncausal") is None
    page.table.selectRow(1)
    page.placement.setCurrentIndex(page.placement.findData(WindowPlacement.CENTERED.value))
    page.deadtime.setValue(5)  # centred 15-min window ending 2.5 min after the sample time
    assert page.apply_form(all_rows=False)
    qtbot.waitUntil(lambda: idle(loaded), timeout=30000)
    assert "Non-causal" in loaded.session.messages.condition("processing.noncausal").text
    assert loaded.session.features.non_causal == [specs[1].variable]
    loaded.undo_stack.undo()
    qtbot.waitUntil(lambda: idle(loaded), timeout=30000)
    assert not loaded.session.features.non_causal
    page.coverage.setValue(150)  # clamps to 100 %
    page.filter_kind.setCurrentIndex(1)  # rolling mean without a length → invalid
    page.filter_len.setValue(0)
    assert not page.apply_form(all_rows=False)
    assert "positive window" in loaded.session.messages.events[0].text


def test_lag_scan_runs_on_training_period(loaded, qtbot):
    loaded.sidebar.setCurrentRow(4)
    page = loaded.pages["processing"]
    page.table.selectRow(3)  # Tray_15_Temp
    page.lag_max.setValue(60)
    page.lag_step.setValue(10)
    page.run_lag_scan()
    qtbot.waitUntil(lambda: page._lag is not None, timeout=30000)
    scan = page._lag
    assert scan.variable == "Tray_15_Temp"
    assert scan.cutoff is not None and scan.cutoff < loaded.session.features.sample_time.max()
    assert np.isfinite(scan.pearson).all()
    assert page.lag_use.isEnabled()


def test_lag_scan_without_training_scope_does_not_submit(loaded, monkeypatch):
    page = loaded.pages["processing"]
    monkeypatch.setattr(page, "training_cutoff", lambda: None)
    monkeypatch.setattr(page._tasks, "submit", lambda *a, **k: pytest.fail("unexpected scan"))
    page.run_lag_scan()
    assert page._lag is None
    assert not page.lag_use.isEnabled()
    assert "training period" in loaded.session.messages.events[0].text
