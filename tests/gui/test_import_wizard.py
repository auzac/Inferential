from __future__ import annotations

import pandas as pd
import pytest

from inferential_studio.core.schemas import DuplicatePolicy, Project, SourceKind, VariableRole
from inferential_studio.core.timestamps import DateOrder
from inferential_studio.infrastructure.settings import AppSettings
from inferential_studio.ui.import_wizard import ImportWizard
from inferential_studio.ui.main_window import MainWindow

pytestmark = pytest.mark.gui


@pytest.fixture
def ambiguous_csv(tmp_path):
    # Every day is <= 12, so day-first vs month-first cannot be told from the data.
    times = pd.date_range("2026-03-01", periods=12 * 24, freq="1h")
    lines = ["Time;Flow;Temp;Status"]
    for i, t in enumerate(times):
        lines.append(
            f"{t:%d/%m/%Y %H:%M};{100 + i % 7},5;{145 + i % 5},{i % 10};{'Bad' if i == 3 else 'Good'}"
        )
    path = tmp_path / "plant.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def walk_to(wizard: ImportWizard, page) -> None:
    if wizard.currentPage() is None:
        wizard.show()
    while wizard.currentPage() is not page:
        current = wizard.currentPage()
        assert current.isComplete(), type(current).__name__
        wizard.next()
        assert wizard.currentPage() is not current, f"could not leave {type(current).__name__}"


def test_wizard_requires_explicit_day_month_choice(qtbot, ambiguous_csv):
    wizard = ImportWizard(Project(), "Asia/Singapore", source_id="process")
    qtbot.addWidget(wizard)
    wizard.show()
    wizard.file_page.path_edit.setText(str(ambiguous_csv))
    walk_to(wizard, wizard.read_page)
    assert wizard.state.opts.delimiter == ";"  # sniffed
    assert wizard.state.opts.decimal == ","
    walk_to(wizard, wizard.time_page)

    tp = wizard.time_page
    assert tp.time.selected_column == "Time"
    assert tp.time.detection.ambiguous
    assert not tp.isComplete()  # cannot proceed without choosing
    day_first = next(
        i
        for i in range(tp.time.format.count())
        if getattr(tp.time.format.itemData(i), "order", None) is DateOrder.DAY_FIRST
    )
    tp.time.format.setCurrentIndex(day_first)
    assert tp.isComplete()
    assert tp.tz.timezone() == "Asia/Singapore"
    utc_first = tp.preview.model_.frame["UTC (stored)"].iloc[0]
    assert utc_first == "2026-02-28 16:00:00"  # 01/03 00:00 Singapore = 28 Feb 16:00 UTC

    walk_to(wizard, wizard.columns_page)
    cols = wizard.columns_page
    assert cols.table.rowCount() == 3  # Flow, Temp, Status
    assert cols.table.item(2, 5).text() == "0%"  # Status column is text
    cols.table.item(2, 0).setCheckState(cols.table.item(2, 0).checkState().Unchecked)
    cols.table.item(0, 4).setText("t/h")
    walk_to(wizard, wizard.finish_page)
    wizard.finish_page.policy.setCurrentIndex(1)  # keep first
    plan = wizard.plan()
    assert (
        plan.source.timestamp.format == "%d/%m/%Y %H:%M" and plan.source.timestamp.dayfirst is True
    )
    assert plan.source.timestamp.source_timezone == "Asia/Singapore"
    assert plan.source.decimal == "," and plan.source.delimiter == ";"
    assert plan.source.duplicate_policy is DuplicatePolicy.KEEP_FIRST
    assert [(v.name, v.role, v.units) for v in plan.variables] == [
        ("Flow", VariableRole.PREDICTOR, "t/h"),
        ("Temp", VariableRole.PREDICTOR, None),
    ]


def test_import_through_main_window(qtbot, ambiguous_csv, tmp_path):
    window = MainWindow(AppSettings(display_timezone="Asia/Singapore"))
    qtbot.addWidget(window)
    wizard = ImportWizard(window.session.project, "Asia/Singapore", source_id="process")
    qtbot.addWidget(wizard)
    wizard.file_page.path_edit.setText(str(ambiguous_csv))
    walk_to(wizard, wizard.time_page)
    tp = wizard.time_page
    idx = next(
        i
        for i in range(tp.time.format.count())
        if getattr(tp.time.format.itemData(i), "order", None) is DateOrder.DAY_FIRST
    )
    tp.time.format.setCurrentIndex(idx)
    walk_to(wizard, wizard.finish_page)
    plan = wizard.plan()

    with qtbot.waitSignal(window.tasks.task_ended, timeout=30000):
        window.run_import(plan)
    data = window.session.data.sources["process"]
    assert data.report.n_rows == 288
    assert data.report.column("Status").n_text == 288  # every Status cell is text
    assert data.values["Flow"].iloc[0] == 100.5  # decimal comma read correctly
    assert window.session.dirty
    assert [v.name for v in window.session.project.variables] == ["Flow", "Temp", "Status"]
    assert window.session.project.source("process").sha256 == data.sha256
    window.session._set_dirty(False)


def test_lab_wizard_has_result_time_and_target_default(qtbot, tmp_path):
    lab = pd.DataFrame(
        {
            "Sampled": pd.to_datetime(["2026-03-13 08:00", "2026-03-13 10:00"]),
            "Reported": pd.to_datetime(["2026-03-13 09:10", "2026-03-13 11:05"]),
            "Purity": [0.985, 0.987],
            "LIMS id": ["A1", "A2"],
        }
    )
    path = tmp_path / "lab.xlsx"
    lab.to_excel(path, index=False)
    wizard = ImportWizard(Project(), "UTC", kind=SourceKind.LAB, source_id="lab")
    qtbot.addWidget(wizard)
    wizard.file_page.path_edit.setText(str(path))
    walk_to(wizard, wizard.time_page)
    tp = wizard.time_page
    assert tp.result.isVisible() or not wizard.isVisible()
    tp.result.column.setCurrentText("Reported")
    tp.sample_id.setCurrentText("LIMS id")
    assert tp.isComplete()
    walk_to(wizard, wizard.finish_page)
    plan = wizard.plan()
    assert plan.source.kind is SourceKind.LAB
    assert plan.source.result_timestamp.column == "Reported"
    assert plan.source.sample_id_column == "LIMS id"
    assert [(v.name, v.role) for v in plan.variables] == [("Purity", VariableRole.TARGET)]
    assert plan.source.duplicate_policy is DuplicatePolicy.KEEP_ALL
