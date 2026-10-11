"""Milestone 5 GUI: the whole dataset-to-report workflow, reproduction, portable projects."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QMessageBox

from inferential_studio.core.artifacts import load_artifact
from inferential_studio.core.persistence import read_manifest
from inferential_studio.core.splits import TEST
from inferential_studio.infrastructure.recovery import find_recovery_copies
from inferential_studio.infrastructure.settings import AppSettings
from inferential_studio.ui.help import GuideDialog
from inferential_studio.ui.main_window import MainWindow
from inferential_studio.ui.messages import Level

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "synthetic_distillation"

pytestmark = pytest.mark.gui


def idle(w) -> bool:
    s = w.session
    return not w.tasks.busy and not s._cleaning_running and not s._features_running


def make_window(qtbot, monkeypatch):
    w = MainWindow(AppSettings(display_timezone="Asia/Singapore", author_name="Tester"))
    w.errors = []
    monkeypatch.setattr(w, "maybe_save", lambda: True)
    monkeypatch.setattr(w, "show_error", lambda title, msg, details="": w.errors.append(msg))
    qtbot.addWidget(w)
    w.show()
    return w


def open_project(w, qtbot, path):
    with qtbot.waitSignal(w.tasks.task_ended, timeout=30000):
        assert w.open_project_file(path)
    qtbot.waitUntil(lambda: idle(w) and w.session.features is not None, timeout=30000)


@pytest.fixture
def trained(qtbot, tmp_path, monkeypatch):
    shutil.copytree(EXAMPLE_DIR, tmp_path / "ex")
    w = make_window(qtbot, monkeypatch)
    open_project(w, qtbot, tmp_path / "ex" / "sample_project.imsproj")
    w.run_training()
    qtbot.waitUntil(lambda: w.session.training is not None and idle(w), timeout=60000)
    yield w
    w.session._set_dirty(False)
    w.tasks.cancel_all()
    w.tasks.wait_for_done(5000)


def test_dataset_to_report_workflow(trained, qtbot, tmp_path):
    w = trained
    w.sidebar.setCurrentRow(8)
    page = w.pages["export"]
    assert page.btn_report.isEnabled()
    assert w.session.messages.condition("training.state").level == Level.INFO
    assert not page.include_test.isEnabled()  # final test not revealed yet
    page.title.setText("Top purity soft sensor")
    page.purpose.setPlainText("Estimate purity between lab samples.")
    page._store_info()
    assert w.session.project.report.title == "Top purity soft sensor"

    report = page.save_report(tmp_path / "card.html")
    html = report.read_text(encoding="utf-8")
    assert "Top purity soft sensor" in html and "has not been evaluated" in html
    preds = page.export_predictions(tmp_path / "pred.xlsx")
    table = pd.read_excel(preds, sheet_name="Predictions")
    assert TEST not in set(table["period"])
    w.session.reveal_test()
    assert page.include_test.isChecked()
    page.export_predictions(tmp_path / "pred.csv")
    assert TEST in set(pd.read_csv(tmp_path / "pred.csv", encoding="utf-8-sig")["period"])
    assert page.export_features(tmp_path / "features.xlsx").is_file()
    model_file = page.save_model(tmp_path / "model.imsmodel")
    art = load_artifact(model_file)
    assert art.name == page.model_combo.currentText() and TEST in art.data["metrics"]
    assert page.equation.toPlainText().startswith("Product_Composition = ")

    # Apply the saved model to this project's data at the lab sample times.
    assert page.load_model_file(model_file)
    assert page.btn_apply.isEnabled()
    page.apply_model()
    qtbot.waitUntil(lambda: page._prediction is not None and idle(w), timeout=30000)
    pred = page._prediction.table
    m = next(x for x in w.session.training.models if x.name == art.name)
    merged = pred.set_index("time")["prediction"].reindex(
        pd.DatetimeIndex(m.predictions["sample_time"])
    )
    np.testing.assert_allclose(merged.to_numpy(), m.predictions["predicted"].to_numpy(), rtol=1e-9)
    assert "Against the lab" in page.apply_note.text()
    assert page.export_applied(tmp_path / "batch.csv").is_file()
    assert w.errors == []


def test_saved_results_are_reproduced_on_reopen(trained, qtbot, tmp_path):
    w = trained
    w.session.reveal_test()
    path = tmp_path / "ex" / "trained.imsproj"
    assert w.save_project_to(path)
    w.session.new()
    open_project(w, qtbot, path)
    qtbot.waitUntil(lambda: w.session.training is not None and idle(w), timeout=60000)
    assert w.session.reproduction and w.session.reproduction.startswith("✓"), w.session.reproduction
    assert w.session.test_revealed  # having looked at the test is remembered
    shown = w.session.messages.conditions("validation")
    assert any(m.text == w.session.reproduction for m in shown)


def test_portable_copy_opens_without_the_original_files(trained, qtbot, tmp_path):
    w = trained
    target = tmp_path / "elsewhere" / "portable.imsproj"
    assert w.save_project_to(target, portable=True)
    assert read_manifest(target)["storage"] == "portable"
    for f in (tmp_path / "ex").iterdir():  # the original data disappears
        f.unlink()
    w.session.new()
    open_project(w, qtbot, target)
    assert w.session.storage == "portable"
    assert set(w.session.data.sources) == {"process", "lab"} and not w.session.data.failures
    assert w.errors == []


def test_recovery_copy_is_offered_after_a_crash(qtbot, tmp_path, monkeypatch):
    shutil.copytree(EXAMPLE_DIR, tmp_path / "ex")
    w = make_window(qtbot, monkeypatch)
    open_project(w, qtbot, tmp_path / "ex" / "sample_project.imsproj")
    w.session.project.notes = "unsaved edit"
    w.session.mark_dirty()
    w._write_recovery()
    assert len(find_recovery_copies()) == 1
    # A new window after a "crash" offers the copy; accepting restores it as unsaved changes.
    w2 = make_window(qtbot, monkeypatch)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    with qtbot.waitSignal(w2.tasks.task_ended, timeout=30000):
        w2.offer_recovery()
    assert w2.session.project.notes == "unsaved edit"
    assert w2.session.dirty and w2.session.path == tmp_path / "ex" / "sample_project.imsproj"
    assert find_recovery_copies() == []
    for win in (w, w2):
        win.session._set_dirty(False)
        win.tasks.cancel_all()
        win.tasks.wait_for_done(5000)


def test_contextual_help_and_guide(qtbot, monkeypatch):
    w = make_window(qtbot, monkeypatch)
    w.sidebar.setCurrentRow(4)
    assert "Deadtime" in w.help_panel.toPlainText()
    dialog = GuideDialog(w)
    qtbot.addWidget(dialog)
    assert "User guide" in dialog.browser.toPlainText()
    dialog.choice.setCurrentText("Methodology and definitions")
    assert "VIF" in dialog.browser.toPlainText()
