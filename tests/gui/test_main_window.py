from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMessageBox

from inferential_studio.infrastructure.settings import AppSettings
from inferential_studio.ui.main_window import MainWindow
from inferential_studio.ui.pages.eda_page import MATRIX_DISPLAY_MAX, EdaPage
from inferential_studio.ui.pages.export_page import ExportPage
from inferential_studio.ui.pages.import_page import ImportPage
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
    window.sidebar.setCurrentRow(8)
    page = window.stack.currentWidget()
    assert isinstance(page, ExportPage) and page.stage.key == "export"
    assert not page.btn_report.isEnabled()  # nothing trained yet
    assert window.windowTitle().startswith("Untitled project")
    assert window.act_import.isEnabled()
    assert window.lbl_records.text() == "Process records: —"


def test_window_remembers_size_and_help_panel(qtbot, monkeypatch):
    first = MainWindow(AppSettings())
    qtbot.addWidget(first)
    monkeypatch.setattr(first, "maybe_save", lambda: True)
    # The offscreen test screen is narrow: the help panel starts closed, the toggle says so.
    assert first.properties_dock.isHidden() and not first.act_help_panel.isChecked()
    assert first.title_bar.help_button.defaultAction() is first.act_help_panel
    first.show()
    first.act_help_panel.trigger()  # the user opens it
    first.resize(960, 690)
    first.close()
    settings = first.settings
    assert settings.window_geometry and settings.help_panel is True
    second = MainWindow(settings)
    qtbot.addWidget(second)
    monkeypatch.setattr(second, "maybe_save", lambda: True)
    second.show()
    assert second.size().height() == 690 and second.properties_dock.isVisible()
    second.act_help_panel.trigger()  # closed again = back to the automatic default
    second.close()
    assert second.settings.help_panel is None


def test_new_project_uses_display_timezone_from_settings(window):
    assert window.session.project.display_timezone == "Asia/Singapore"


def test_title_bar_replaces_menu_bar_and_toolbar(window):
    from PySide6.QtWidgets import QMenuBar, QToolBar

    # One seamless title bar: no menu bar, no toolbar.
    assert window.findChildren(QMenuBar) == [] and window.findChildren(QToolBar) == []
    assert window.menuWidget() is window.title_bar
    bar = window.title_bar
    assert [b.defaultAction() for b in bar.command_buttons] == [
        window.act_open,
        window.act_save,
        window.act_save_as,
        window.act_import,
        window.act_undo,
        window.act_redo,
    ]
    icon_only = Qt.ToolButtonStyle.ToolButtonIconOnly
    assert all(b.toolButtonStyle() == icon_only for b in bar.command_buttons)
    assert not window.act_save.icon().isNull()
    assert window.act_save.toolTip() == "Save (Ctrl+S)"
    # Everything else lives in the main menu; shortcuts work without a menu bar.
    menu = bar.menu_button.menu()
    flat = menu.actions() + window.help_menu.actions()
    commands = [
        window.act_new, window.act_open, window.act_save, window.act_save_as,
        window.act_save_portable, window.act_import, window.act_export, window.act_project,
        window.act_settings, window.act_exit, window.act_guide, window.act_example,
        window.act_about, window.act_logs, window.act_help_panel,
    ]  # fmt: skip
    for act in commands:
        assert act in flat, act.text()
        assert act in window.actions(), act.text()  # shortcut registered on the window
    assert window.recent_menu.menuAction() in menu.actions()


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
    assert page.cursor.sizeHint().height() == page.cursor.maximumHeight()  # not content-driven
    before = page.cursor.height()
    page.trends._show_cursor(t)
    qtbot.wait(0)
    assert "<b>" in page.cursor.time_label.text()
    assert " = " in page.cursor.values_label.text()
    assert page.cursor.height() == before


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


def test_import_page_has_one_primary_import_button(window, monkeypatch):
    page = window.pages["import"]
    # One entry point: the wizard resolves process vs lab from the user's choices.
    assert not hasattr(page, "import_process") and not hasattr(page, "import_lab")
    assert page.import_button.text() == "Import data…"
    assert page.import_button.property("primary") is True
    calls = []
    monkeypatch.setattr(window, "import_dialog", lambda *a, **k: calls.append(1))
    page.import_button.click()
    assert calls == [1]


def test_import_page_example_box_only_for_empty_project(qtbot, window, example_copy):
    page = window.pages["import"]
    assert page.example_box.isVisibleTo(page)
    assert window.act_example in window.help_menu.actions()  # always reachable from Help
    open_and_wait(qtbot, window, example_copy / "sample_project.imsproj")
    assert not page.example_box.isVisibleTo(page)
    assert "Process data" in page.details.text() and "Checksum" in page.details.text()


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


def test_eda_correlation_follows_checked_variables(qtbot, window, example_copy):
    open_and_wait(qtbot, window, example_copy / "sample_project.imsproj")
    eda: EdaPage = window.pages["initial_eda"]
    with qtbot.waitSignal(window.tasks.task_ended, timeout=30000):
        window.sidebar.setCurrentRow(1)
    qtbot.waitUntil(lambda: eda._result is not None, timeout=10000)
    assert not hasattr(eda, "recalc") and not hasattr(eda, "only_selected")
    assert eda.tabs.cornerWidget() is None
    eda.tabs.setCurrentWidget(eda._corr_tab)
    checked = eda.selected_variables()
    assert list(eda.heatmap._corr.columns) == checked and len(checked) == 4
    assert eda.count_label.text() == "4 of 8"

    def apply() -> None:
        eda._selection_timer.stop()
        eda._apply_selection()

    eda.var_list.item(0).setCheckState(Qt.CheckState.Unchecked)  # untick one
    apply()
    assert (
        list(eda.heatmap._corr.columns) == eda.selected_variables() and len(eda.heatmap._corr) == 3
    )
    eda.check_all.click()
    apply()
    assert len(eda.heatmap._corr) == 8 and eda.count_label.text() == "8 of 8"
    eda.heatmap.cell_clicked.emit("Tray_15_Temp", "Tray_20_Temp")
    assert eda._pair == ("Tray_15_Temp", "Tray_20_Temp")
    eda.search.setText("tray")
    eda.check_none.click()  # only the filtered (visible) variables are unticked
    apply()
    assert len(eda.selected_variables()) == 6
    assert eda._pair is None  # the shown pair left the matrix
    eda.search.setText("")
    eda.check_none.click()
    apply()
    assert eda.heatmap._corr.empty
    assert "at least two" in eda.heatmap._placeholder.textItem.toPlainText()


def test_eda_scatter_matrix_reuses_its_grid_and_aligns_the_axes(
    qtbot, window, example_copy, monkeypatch
):
    open_and_wait(qtbot, window, example_copy / "sample_project.imsproj")
    eda: EdaPage = window.pages["initial_eda"]
    with qtbot.waitSignal(window.tasks.task_ended, timeout=30000):
        window.sidebar.setCurrentRow(1)
    qtbot.waitUntil(lambda: eda._result is not None, timeout=10000)

    def apply() -> None:
        eda._selection_timer.stop()
        eda._apply_selection()

    eda.tabs.setCurrentWidget(eda._matrix_tab)  # 4 variables checked by default
    eda._check_listed(True)  # 6 fit; the example has 8
    apply()
    assert eda._matrix_side == 6 and len(eda._matrix_cells) == 36
    cells = dict(eda._matrix_cells)
    # Every row of the left column starts its plot area at the same x: one shared y gutter.
    # (Wait for the layout: the rectangles are empty until the grid has been sized.)
    qtbot.waitUntil(lambda: cells[(0, 0)].vb.sceneBoundingRect().width() > 1, timeout=2000)
    lefts = {round(cells[(row, 0)].vb.sceneBoundingRect().left(), 3) for row in range(6)}
    assert len(lefts) == 1 and lefts.pop() > 0
    # The cells are drawn once, with a bounded number of points each.
    points = {}
    for (row, col), item in eda._matrix_items.items():
        if row != col:
            points[(row, col)] = len(item.getData()[0])
    assert points and max(points.values()) <= MATRIX_DISPLAY_MAX
    # The grid survives a change of variables, and a tick that changes nothing redraws nothing.
    draws = []
    original = eda._even_y_axes
    monkeypatch.setattr(eda, "_even_y_axes", lambda: draws.append(1) or original())
    eda.var_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    apply()
    assert draws == [1]
    assert all(eda._matrix_cells[key] is cells[key] for key in cells)  # same plots, new data
    eda.var_list.item(0).setCheckState(Qt.CheckState.Checked)
    apply()
    assert draws == [1, 1]
    apply()
    assert draws == [1, 1]  # nothing changed: the scene is left as it is
    # Fewer than two variables take the grid down, and it comes back on the next tick.
    eda.check_none.click()
    apply()
    assert eda._matrix_side == 0 and not eda._matrix_cells
    assert "Check 2–6 variables" in eda.matrix_note.text()


def test_project_settings_change_display_timezone(qtbot, window):
    from inferential_studio.ui.project_dialog import ProjectSettingsDialog

    dialog = ProjectSettingsDialog(window.session.project, window)
    qtbot.addWidget(dialog)
    dialog.timezone.setCurrentText("Europe/London")
    dialog.name.setText("Column C-101")
    assert dialog.apply_to(window.session.project)
    assert window.session.project.display_timezone == "Europe/London"
    assert not dialog.apply_to(window.session.project)  # no change the second time


def test_title_shows_project_name_and_unsaved_state(qtbot, window, example_copy):
    # The full title needs room: the bar elides it on narrow windows (checked at the end).
    window.resize(1280, 800)
    title = window.title_bar.title
    assert (window.session.project.name or "Untitled project") in title.text()
    open_and_wait(qtbot, window, example_copy / "sample_project.imsproj")
    name = window.session.project.name
    assert name in title.text() and "unsaved" not in title.text()
    assert "Inferential Model Studio" in title.text()
    # The label is transparent to mouse events (dragging works everywhere), so the tooltip
    # with the full path sits on the bar itself.
    assert window.title_bar.toolTip().endswith("sample_project.imsproj")
    window.session._set_dirty(True)
    assert f"{name} (unsaved changes)" in title.text()
    window.session._set_dirty(False)
    assert window.pages["import"].findChild(QLabel, "pageEyebrow").text() == "Stage 1 of 9"
    # A narrow window elides the title instead of clipping it; the full path stays in the tooltip.
    window.resize(900, 700)
    qtbot.waitUntil(lambda: title.text().endswith("…"), timeout=2000)
    assert title.text().startswith(name[:10])
    assert title.geometry().right() < window.title_bar.help_button.geometry().left()
    assert window.title_bar.toolTip().endswith("sample_project.imsproj")
