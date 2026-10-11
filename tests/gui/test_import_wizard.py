from __future__ import annotations

import pandas as pd
import pytest

from inferential_studio.core.schemas import (
    DataSource,
    DuplicatePolicy,
    Project,
    SourceKind,
    TimestampKind,
    TimestampSpec,
    VariableMeta,
    VariableRole,
)
from inferential_studio.core.timestamps import DateOrder
from inferential_studio.infrastructure.settings import AppSettings
from inferential_studio.ui.import_wizard import ImportPlan, ImportWizard, resolve_kind
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


def set_role(wizard: ImportWizard, column: str, role: VariableRole) -> None:
    """Set the role combo of the columns page for the column named ``column``."""
    page = wizard.columns_page
    for row in range(page.table.rowCount()):
        if page.table.item(row, 1).text() == column:
            combo = page.table.cellWidget(row, 3)
            combo.setCurrentIndex(combo.findData(role))
            return
    raise AssertionError(f"column {column!r} is not on the columns page")


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
    wizard = ImportWizard(Project(), "UTC", source_id="lab")
    qtbot.addWidget(wizard)
    wizard.file_page.path_edit.setText(str(path))
    walk_to(wizard, wizard.time_page)
    tp = wizard.time_page
    assert tp.result.isVisible()  # lab extras are always offered now
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


def test_resolve_kind_rules():
    spec = TimestampSpec(column="T", kind=TimestampKind.TEXT)

    def variables(*roles: VariableRole) -> list[VariableMeta]:
        return [
            VariableMeta(name=f"V{i}", source_id="s", source_column=f"V{i}", role=role)
            for i, role in enumerate(roles)
        ]

    # Lab markers (result time / sample id) always mean lab data.
    assert resolve_kind(spec, None, variables()) is SourceKind.LAB
    assert resolve_kind(None, "ID", variables()) is SourceKind.LAB
    assert resolve_kind(spec, None, variables(VariableRole.PREDICTOR)) is SourceKind.LAB
    # Without markers: target-only is a sparse lab file; anything else is process.
    assert resolve_kind(None, None, variables(VariableRole.TARGET)) is SourceKind.LAB
    assert (
        resolve_kind(
            None, None, variables(VariableRole.TARGET, VariableRole.PREDICTOR)
        )
        is SourceKind.PROCESS
    )
    assert resolve_kind(None, None, variables(VariableRole.PREDICTOR)) is SourceKind.PROCESS
    assert resolve_kind(None, None, variables()) is SourceKind.PROCESS


def test_combined_csv_target_and_predictors_resolves_process(qtbot, tmp_path):
    times = pd.date_range("2026-03-01", periods=100, freq="1h")
    path = tmp_path / "plant_with_lab.csv"
    pd.DataFrame(
        {"Time": times, "Pressure": range(100), "Purity": [0.98 + i / 10_000 for i in range(100)]}
    ).to_csv(path, index=False)
    wizard = ImportWizard(Project(), "UTC", source_id="process")
    qtbot.addWidget(wizard)
    wizard.file_page.path_edit.setText(str(path))
    walk_to(wizard, wizard.columns_page)
    set_role(wizard, "Purity", VariableRole.TARGET)
    walk_to(wizard, wizard.finish_page)
    plan = wizard.plan()
    # One CSV holding both the target and the predictors is the common case: process data.
    assert plan.source.kind is SourceKind.PROCESS
    assert {v.name: v.role for v in plan.variables} == {
        "Pressure": VariableRole.PREDICTOR,
        "Purity": VariableRole.TARGET,
    }
    assert plan.source.result_timestamp is None and plan.source.sample_id_column is None
    assert plan.source.duplicate_policy is DuplicatePolicy.ERROR


def test_target_only_file_resolves_lab(qtbot, tmp_path):
    path = tmp_path / "sparse.csv"
    pd.DataFrame(
        {
            "Sampled": pd.to_datetime(["2026-03-13 08:00", "2026-03-13 10:00"]),
            "Purity": [0.985, 0.987],
        }
    ).to_csv(path, index=False)
    wizard = ImportWizard(Project(), "UTC", source_id="process")
    qtbot.addWidget(wizard)
    wizard.file_page.path_edit.setText(str(path))
    walk_to(wizard, wizard.columns_page)
    set_role(wizard, "Purity", VariableRole.TARGET)
    walk_to(wizard, wizard.finish_page)
    plan = wizard.plan()
    # A target with no predictors is a sparse lab file even without lab markers.
    assert plan.source.kind is SourceKind.LAB
    assert plan.source.duplicate_policy is DuplicatePolicy.KEEP_ALL


def test_result_time_marker_resolves_lab(qtbot, tmp_path):
    path = tmp_path / "analyzer.csv"
    pd.DataFrame(
        {
            "Sampled": pd.to_datetime(["2026-03-13 08:00", "2026-03-13 10:00"]),
            "Reported": pd.to_datetime(["2026-03-13 09:10", "2026-03-13 11:05"]),
            "Purity": [0.985, 0.987],
        }
    ).to_csv(path, index=False)
    wizard = ImportWizard(Project(), "UTC", source_id="process")
    qtbot.addWidget(wizard)
    wizard.file_page.path_edit.setText(str(path))
    walk_to(wizard, wizard.time_page)
    wizard.time_page.result.column.setCurrentText("Reported")
    walk_to(wizard, wizard.finish_page)
    plan = wizard.plan()
    assert plan.source.kind is SourceKind.LAB
    assert plan.source.result_timestamp.column == "Reported"
    assert [v.role for v in plan.variables] == [VariableRole.TARGET]  # lab default


def test_lab_source_with_predictors_is_blocked(qtbot, tmp_path):
    path = tmp_path / "analyzer.csv"
    pd.DataFrame(
        {
            "Sampled": pd.to_datetime(["2026-03-13 08:00", "2026-03-13 10:00"]),
            "Reported": pd.to_datetime(["2026-03-13 09:10", "2026-03-13 11:05"]),
            "V1": [1.0, 2.0],
            "V2": [3.0, 4.0],
        }
    ).to_csv(path, index=False)
    wizard = ImportWizard(Project(), "UTC", source_id="process")
    qtbot.addWidget(wizard)
    wizard.file_page.path_edit.setText(str(path))
    walk_to(wizard, wizard.time_page)
    wizard.time_page.result.column.setCurrentText("Reported")
    walk_to(wizard, wizard.finish_page)
    fp = wizard.finish_page
    assert fp.isComplete()  # V1 defaults to TARGET, V2 is not used
    set_role(wizard, "V2", VariableRole.PREDICTOR)
    assert not fp.isComplete()
    assert "Predictor columns must come from process data" in fp.error.text()
    set_role(wizard, "V2", VariableRole.UNUSED)
    assert fp.isComplete()


def test_import_dialog_id_follows_resolved_kind(qtbot, monkeypatch):
    window = MainWindow(AppSettings())
    qtbot.addWidget(window)
    captured: list[ImportPlan] = []
    monkeypatch.setattr(window, "run_import", captured.append)

    def make_plan(kind: SourceKind, source_id: str) -> ImportPlan:
        return ImportPlan(
            DataSource(
                source_id=source_id,
                kind=kind,
                path="x.csv",
                timestamp=TimestampSpec(column="T", kind=TimestampKind.TEXT),
            ),
            [
                VariableMeta(
                    name="Purity", source_id=source_id, source_column="V", role=VariableRole.TARGET
                )
            ],
        )

    class StubWizard:
        plan_: ImportPlan

        def __init__(self, *args, **kwargs) -> None:
            pass

        def exec(self) -> bool:
            return True

        def plan(self) -> ImportPlan:
            return self.plan_

    monkeypatch.setattr("inferential_studio.ui.main_window.ImportWizard", StubWizard)
    StubWizard.plan_ = make_plan(SourceKind.LAB, "process")  # provisional id, resolved kind
    window.import_dialog()
    assert captured[-1].source.source_id == "lab"
    assert captured[-1].variables[0].source_id == "lab"
    # The retry path keeps an id that already matches the resolved kind.
    StubWizard.plan_ = make_plan(SourceKind.LAB, "lab")
    window.import_dialog(initial=StubWizard.plan_)
    assert captured[-1].source.source_id == "lab"

    window.session._set_dirty(False)
