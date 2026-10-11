from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox

from inferential_studio.infrastructure.settings import AppSettings
from inferential_studio.ui.main_window import MainWindow
from inferential_studio.ui.pages.eda_page import EdaPage
from inferential_studio.ui.pages.import_page import ImportPage
from inferential_studio.ui.pages.stage_page import PlaceholderPage
from inferential_studio.ui.stages import STAGES

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "synthetic_distillation"

pytestmark = pytest.mark.gui


@pytest.fixture
def window(qtbot, monkeypatch):
    w = MainWindow(AppSettings(display_timezone="Asia/Singapore"))
    w.errors = []
    # pytest-qt closes widgets before fixture teardown: never block on "save changes?" there.
    monkeypatch.setattr(w, "maybe_save", lambda: True)
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


def open_and_wait(qtbot, window, path) -> None:
    with qtbot.waitSignal(window.tasks.task_ended, timeout=30000):
        assert window.open_project_file(path)


def test_shell_has_nine_stages_and_navigation(window):
    assert window.sidebar.count() == len(STAGES) == 9
    assert isinstance(window.stack.currentWidget(), ImportPage)
    window.sidebar.setCurrentRow(1)
    assert isinstance(window.stack.currentWidget(), EdaPage)
    window.sidebar.setCurrentRow(6)
    page = window.stack.currentWidget()
    assert isinstance(page, PlaceholderPage) and page.stage.key == "training"
    assert window.windowTitle().startswith("Untitled project")
    assert window.act_import.isEnabled()
    assert window.lbl_records.text() == "Process records: —"


def test_new_project_uses_display_timezone_from_settings(window):
    assert window.session.project.display_timezone == "Asia/Singapore"


def test_frequent_actions_live_on_menu_bar_not_toolbar(window):
    from PySide6.QtWidgets import QToolBar, QToolButton

    # One command row: the frequent actions sit on the menu bar's right end
    # instead of a separate toolbar.
    assert window.menuBar().findChildren(QToolBar) == []
    corner = window.menuBar().cornerWidget()
    buttons = corner.findChildren(QToolButton)
    assert [b.defaultAction() for b in buttons] == [
        window.act_new,
        window.act_open,
        window.act_save,
        window.act_save_as,
        window.act_import,
        window.act_export,
        window.act_undo,
        window.act_redo,
        window.act_settings,
    ]


def test_trend_hover_values_show_below_variable_list(qtbot, window, example_copy):
    open_and_wait(qtbot, window, example_copy / "sample_project.imsproj")
    window.sidebar.setCurrentRow(2)  # Cleaning & Exclusions
    page = window.pages["cleaning"]
    qtbot.waitUntil(lambda: bool(page.trends._series), timeout=10000)
    frame = window.session.frames()[0]
    for ts in frame.index:
        if frame.loc[ts].notna().any():
            t = ts.timestamp()
            break
    else:
        pytest.fail("example data has no finite values")
    # The readout sits below the variable list and is a static box: hovered text
    # (however long) must never resize it or push the charts around.
    assert page.cursor.parent() is not None
    assert page.cursor.minimumHeight() == page.cursor.maximumHeight()
    page.trends._show_cursor(t)
    qtbot.wait(0)
    assert "<b>" in page.cursor.time_label.text()
    assert " = " in page.cursor.values_label.text()
    assert page.cursor.height() == page.cursor.maximumHeight()


def test_open_sample_project_loads_data_in_background(qtbot, window, example_copy):
    open_and_wait(qtbot, window, example_copy / "sample_project.imsproj")
    assert window.errors == []
    data = window.session.data
    assert set(data.sources) == {"process", "lab"}
    assert window.lbl_records.text() == "Process records: 40,320"
    assert window.lbl_samples.text() == "Lab samples: 316"
    page = window.pages["import"]
    assert page.sources.rowCount() == 2 and page.sources.item(0, 6).text() == "OK"
    assert "sample_project.imsproj" in window.windowTitle()


def test_changed_source_reported_and_can_be_accepted(qtbot, window, example_copy, monkeypatch):
    csv = example_copy / "tower_process.csv"
    csv.write_bytes(csv.read_bytes().replace(b"104.59", b"104.60", 1))
    answers = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: answers.append(1) or QMessageBox.StandardButton.Yes
    )
    with qtbot.waitSignal(window.tasks.task_ended, timeout=30000):
        window.open_project_file(example_copy / "sample_project.imsproj")
    assert window.errors and "checksum mismatch" in window.errors[0][1]
    assert "process" in window.session.data.failures and "lab" in window.session.data.sources
    assert answers  # user was asked whether to accept the new contents
    qtbot.waitUntil(lambda: "process" in window.session.data.sources, timeout=30000)
    assert window.session.dirty  # new checksum recorded in the project
    assert (
        window.session.project.source("process").sha256
        == window.session.data.sources["process"].sha256
    )


def test_open_corrupt_project_shows_error(window, tmp_path):
    bad = tmp_path / "bad.imsproj"
    bad.write_text("not a zip")
    assert not window.open_project_file(bad)
    assert window.errors[0][0] == "Cannot open project"
    assert window.session.path is None


def test_save_as_elsewhere_keeps_sources_reachable(qtbot, window, example_copy, tmp_path):
    open_and_wait(qtbot, window, example_copy / "sample_project.imsproj")
    window.session.mark_dirty()
    assert window.isWindowModified()
    out_dir = tmp_path / "elsewhere"
    assert window.save_project_to(out_dir / "saved")  # extension added automatically
    assert not window.isWindowModified()
    assert window.settings.recent_projects[0].endswith("saved.imsproj")

    reopened = MainWindow(AppSettings())
    qtbot.addWidget(reopened)
    reopened.errors = []
    reopened.show_error = lambda title, msg, details="": reopened.errors.append(msg)
    reopened.maybe_save = lambda: True
    open_and_wait(qtbot, reopened, out_dir / "saved.imsproj")
    assert reopened.errors == []
    assert set(reopened.session.data.sources) == {"process", "lab"}
    assert reopened.session.project.features == window.session.project.features
    assert reopened.session.project.exclusions == window.session.project.exclusions


def test_remove_source(qtbot, window, example_copy, monkeypatch):
    open_and_wait(qtbot, window, example_copy / "sample_project.imsproj")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window._remove_source("lab")
    project = window.session.project
    assert [s.source_id for s in project.sources] == ["process"]
    assert project.target is None
    assert "lab" not in window.session.data.sources
    assert window.session.dirty


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


def test_eda_page_computes_in_background(qtbot, window, example_copy):
    open_and_wait(qtbot, window, example_copy / "sample_project.imsproj")
    eda: EdaPage = window.pages["initial_eda"]
    with qtbot.waitSignal(window.tasks.task_ended, timeout=30000):
        window.sidebar.setCurrentRow(1)
    qtbot.waitUntil(lambda: eda._result is not None, timeout=10000)
    assert eda.stats_table.model_.rowCount() == 8
    assert eda.var_list.count() == 8
    assert len(eda.trends.plots) == 5  # 4 checked variables + lab target markers
    assert eda._result.xy is not None
    eda.heatmap.cell_clicked.emit("Tray_15_Temp", "Tray_20_Temp")
    assert "r = " in eda.pair_note.text()
    eda.search.setText("tray")
    visible = [
        eda.var_list.item(i).text()
        for i in range(eda.var_list.count())
        if not eda.var_list.item(i).isHidden()
    ]
    assert all("Tray" in v for v in visible) and len(visible) == 2


def test_project_settings_change_display_timezone(qtbot, window):
    from inferential_studio.ui.project_dialog import ProjectSettingsDialog

    dialog = ProjectSettingsDialog(window.session.project, window)
    qtbot.addWidget(dialog)
    dialog.timezone.setCurrentText("Europe/London")
    dialog.name.setText("Column C-101")
    assert dialog.apply_to(window.session.project)
    assert window.session.project.display_timezone == "Europe/London"
    assert not dialog.apply_to(window.session.project)  # no change the second time
