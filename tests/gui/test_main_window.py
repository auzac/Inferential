from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox

from inferential_studio.infrastructure.settings import AppSettings
from inferential_studio.ui.main_window import MainWindow
from inferential_studio.ui.pages.import_page import ImportPage
from inferential_studio.ui.pages.stage_page import PlaceholderPage
from inferential_studio.ui.stages import STAGES

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "synthetic_distillation"

pytestmark = pytest.mark.gui


@pytest.fixture
def window(qtbot, monkeypatch):
    w = MainWindow(AppSettings(display_timezone="Asia/Singapore"))
    w.errors = []
    monkeypatch.setattr(
        w, "show_error", lambda title, msg, details="": w.errors.append((title, msg))
    )
    qtbot.addWidget(w)
    w.show()
    yield w
    w.session._set_dirty(False)  # avoid the unsaved-changes prompt on teardown
    w.tasks.cancel_all()
    w.tasks.wait_for_done(5000)


@pytest.fixture
def example_copy(tmp_path):
    dest = tmp_path / "example"
    shutil.copytree(EXAMPLE_DIR, dest)
    return dest


def test_shell_has_nine_stages_and_navigation(window):
    assert window.sidebar.count() == len(STAGES) == 9
    assert isinstance(window.stack.currentWidget(), ImportPage)
    window.sidebar.setCurrentRow(6)
    page = window.stack.currentWidget()
    assert isinstance(page, PlaceholderPage) and page.stage.key == "training"
    assert window.windowTitle().startswith("Untitled project")
    assert not window.act_import.isEnabled()  # honest: import is not implemented yet


def test_new_project_uses_display_timezone_from_settings(window):
    assert window.session.project.display_timezone == "Asia/Singapore"


def test_open_sample_project(window, example_copy):
    assert window.open_project_file(example_copy / "sample_project.imsproj")
    assert window.errors == []  # source files present with matching checksums
    assert window.session.project.target.name == "Product_Composition"
    assert window.lbl_exclusions.text() == "Manual exclusions: 2"
    overview = window.pages["import"].overview
    assert overview.topLevelItem(0).text(1) == "Synthetic distillation example"
    assert "sample_project.imsproj" in window.windowTitle()


def test_open_reports_changed_source_file(window, example_copy):
    (example_copy / "tower_process.csv").write_text("tampered\n")
    assert window.open_project_file(example_copy / "sample_project.imsproj")
    assert window.errors and "checksum mismatch" in window.errors[0][1]


def test_open_corrupt_project_shows_error(window, tmp_path):
    bad = tmp_path / "bad.imsproj"
    bad.write_text("not a zip")
    assert not window.open_project_file(bad)
    assert window.errors[0][0] == "Cannot open project"
    assert window.session.path is None


def test_save_as_and_reopen(window, example_copy, tmp_path):
    window.open_project_file(example_copy / "sample_project.imsproj")
    window.session.mark_dirty()
    assert window.isWindowModified()
    out = tmp_path / "saved"  # extension added automatically
    assert window.save_project_to(out)
    assert not window.isWindowModified()
    assert (tmp_path / "saved.imsproj").exists()
    assert window.settings.recent_projects[0].endswith("saved.imsproj")

    reopened = MainWindow(AppSettings())
    try:
        reopened.show_error = lambda *a, **k: None
        assert reopened.open_project_file(tmp_path / "saved.imsproj")
        assert reopened.session.project.features == window.session.project.features
        assert reopened.session.project.exclusions == window.session.project.exclusions
    finally:
        reopened.close()


def test_generate_example_in_background(window, qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    with qtbot.waitSignal(window.tasks.task_ended, timeout=30000) as ended:
        window.generate_example_to(tmp_path / "ex")
        assert window.task_widget.isVisible()
        assert not window.act_open.isEnabled()  # project actions locked while busy
    assert ended.args[2] == "succeeded"
    assert (tmp_path / "ex" / "sample_project.imsproj").exists()
    assert window.act_open.isEnabled()
    assert not window.task_widget.isVisible()
