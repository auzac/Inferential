from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inferential_studio.core.exclusions import EventStatus
from inferential_studio.core.schemas import (
    Condition,
    ConditionGroup,
    DetectorConfig,
    DetectorKind,
    ExclusionInterval,
    ExclusionRule,
    FlagDecisionKind,
    RuleOperator,
)
from inferential_studio.infrastructure.settings import AppSettings
from inferential_studio.ui.commands import edit_lists
from inferential_studio.ui.main_window import MainWindow
from inferential_studio.ui.widgets.overlays import IntervalShade

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "synthetic_distillation"

pytestmark = pytest.mark.gui


@pytest.fixture
def loaded(qtbot, tmp_path, monkeypatch):
    shutil.copytree(EXAMPLE_DIR, tmp_path / "ex")
    w = MainWindow(AppSettings(display_timezone="Asia/Singapore", author_name="Tester"))
    w.errors = []
    # pytest-qt closes widgets before fixture teardown: never block on "save changes?" there.
    monkeypatch.setattr(w, "maybe_save", lambda: True)
    monkeypatch.setattr(w, "show_error", lambda title, msg, details="": w.errors.append(msg))
    qtbot.addWidget(w)
    w.show()
    with qtbot.waitSignal(w.tasks.task_ended, timeout=30000):
        w.open_project_file(tmp_path / "ex" / "sample_project.imsproj")
    w.sidebar.setCurrentRow(2)
    yield w
    w.session._set_dirty(False)
    w.tasks.cancel_all()
    w.tasks.wait_for_done(5000)


def low_load_rule() -> ExclusionRule:
    return ExclusionRule(
        "low load",
        ConditionGroup((Condition("Feed_Flow", RuleOperator.LT, 50),)),
        min_duration_s=600,
    )


def wait_detection(qtbot, w):
    qtbot.waitUntil(
        lambda: not w.tasks.busy and w.session.detector_results_version == w.session.data_version,
        timeout=30000,
    )


def test_sample_project_starts_with_two_shutdowns(loaded):
    state = loaded.session.cleaning
    assert state is not None and int(state.regime.sum()) == 1081 + 721
    page = loaded.pages["cleaning"]
    assert page.manual_table.rowCount() == 2
    assert loaded.lbl_exclusions.text().startswith("Excluded rows: 1,802")


def test_add_interval_undo_redo(loaded):
    page = loaded.pages["cleaning"]
    session = loaded.session
    t0 = pd.Timestamp("2026-03-12 00:00", tz="UTC")
    before = int(session.cleaning.regime.sum())
    page.add_interval(
        ExclusionInterval(
            start=t0.to_pydatetime(), end=(t0 + pd.Timedelta("59min")).to_pydatetime()
        )
    )
    assert int(session.cleaning.regime.sum()) == before + 60  # closed interval, 1-minute data
    assert session.dirty and loaded.undo_stack.count() == 1
    loaded.undo_stack.undo()
    assert int(session.cleaning.regime.sum()) == before and len(session.project.exclusions) == 2
    loaded.undo_stack.redo()
    assert int(session.cleaning.regime.sum()) == before + 60


def test_signal_specific_interval_only_affects_that_variable(loaded):
    page = loaded.pages["cleaning"]
    t0 = pd.Timestamp("2026-03-12 00:00", tz="UTC")
    page.add_interval(
        ExclusionInterval(
            start=t0.to_pydatetime(),
            end=(t0 + pd.Timedelta("9min")).to_pydatetime(),
            signals=("Tray_15_Temp",),
        )
    )
    state = loaded.session.cleaning
    rows = state.index.slice_indexer(t0, t0 + pd.Timedelta("9min"))
    assert state.excluded["Tray_15_Temp"].iloc[rows].all()
    assert not state.excluded["Tray_20_Temp"].iloc[rows].any()
    assert not state.regime[rows].any()


def test_rule_toggle_through_table(loaded, qtbot):
    page = loaded.pages["cleaning"]
    session = loaded.session
    edit_lists(loaded.undo_stack, session, "Add rule", rules=[low_load_rule()])
    with_rule = int(session.cleaning.regime.sum())
    assert with_rule > 1802
    assert page.rules_table.item(0, 6).text() == "2,182"
    page.rules_table.item(0, 0).setCheckState(page.rules_table.item(0, 0).checkState().Unchecked)
    assert int(session.cleaning.regime.sum()) == 1802
    assert not session.project.rules[0].enabled
    loaded.undo_stack.undo()
    assert int(session.cleaning.regime.sum()) == with_rule


def test_detect_review_and_decide(loaded, qtbot):
    page = loaded.pages["cleaning"]
    session = loaded.session
    det = DetectorConfig(DetectorKind.HAMPEL, ("Column_Pressure",), threshold=6, window_s=3600)
    edit_lists(loaded.undo_stack, session, "Add detector", detectors=[det])
    wait_detection(qtbot, loaded)
    state = session.cleaning
    pending = [e for e in state.events if e.status == EventStatus.PENDING]
    assert len(pending) >= 15
    assert not state.excluded["Column_Pressure"].iloc[:0].any()
    assert int(state.pending["Column_Pressure"].sum()) == sum(e.event.n_points for e in pending)

    page.event_filter.setCurrentIndex(1)  # pending
    page.events_table.selectRow(0)
    first = page._selected_data(page.events_table)[0]
    page._decide(FlagDecisionKind.ACCEPTED)
    state = session.cleaning
    row = state.index.get_loc(first.start)
    assert state.excluded["Column_Pressure"].iloc[row]
    assert not state.excluded["Tray_15_Temp"].iloc[row]  # only the flagged variable
    assert session.project.flag_decisions[0].author == "Tester"

    # Reject: kept as valid, no longer pending.
    page.event_filter.setCurrentIndex(1)
    page.events_table.selectRow(0)
    second = page._selected_data(page.events_table)[0]
    page._decide(FlagDecisionKind.REJECTED)
    row2 = session.cleaning.index.get_loc(second.start)
    assert not session.cleaning.excluded["Column_Pressure"].iloc[row2]
    assert not session.cleaning.pending["Column_Pressure"].iloc[row2]

    loaded.undo_stack.undo()
    loaded.undo_stack.undo()
    assert session.project.flag_decisions == []


def test_masks_roundtrip_through_save_and_reopen(loaded, qtbot, tmp_path):
    session = loaded.session
    det = DetectorConfig(DetectorKind.MAD, ("Product_Composition",), threshold=5)
    edit_lists(loaded.undo_stack, session, "setup", rules=[low_load_rule()], detectors=[det])
    wait_detection(qtbot, loaded)
    ev = session.cleaning.events[0].event
    page = loaded.pages["cleaning"]
    page.event_filter.setCurrentIndex(0)
    page.events_table.selectRow(0)
    page._decide(FlagDecisionKind.ACCEPTED)
    before = session.cleaning
    assert before.target.excluded.loc[ev.start]
    assert loaded.save_project_to(tmp_path / "cleaned.imsproj")

    other = MainWindow(AppSettings())
    qtbot.addWidget(other)
    other.show_error = lambda *a, **k: None
    with qtbot.waitSignal(other.tasks.task_ended, timeout=30000):
        other.open_project_file(tmp_path / "cleaned.imsproj")
    wait_detection(qtbot, other)
    after = other.session.cleaning
    np.testing.assert_array_equal(after.regime, before.regime)
    pd.testing.assert_frame_equal(after.excluded, before.excluded)
    pd.testing.assert_frame_equal(after.pending, before.pending)
    pd.testing.assert_series_equal(after.target.excluded, before.target.excluded)


def test_range_selection_and_overlay_items(loaded, qtbot):
    page = loaded.pages["cleaning"]
    page.select_button.setChecked(True)
    assert page._regions and len(page._regions) == len(page.trends.plots)
    page._regions[0].setRegion(
        (page._regions[0].getRegion()[0], page._regions[0].getRegion()[0] + 3600)
    )
    assert page._regions[-1].getRegion() == page._regions[0].getRegion()  # charts stay in sync
    lo, hi = page.selection_utc()
    assert (hi - lo) == pd.Timedelta("1h")
    # One shading item per plot, however many intervals are excluded.
    for plot in page.trends.plots:
        shades = [i for i in plot.items if isinstance(i, IntervalShade)]
        assert len(shades) <= 2  # regime + variable-specific


def test_views_switch(loaded):
    page = loaded.pages["cleaning"]
    page.view_cleaned.setChecked(True)
    name = page.trends.groups[0][1][0].name
    y = page.trends.groups[0][1][0].y
    state = loaded.session.cleaning
    assert np.isnan(y[state.excluded[name].to_numpy()]).all()
    page.view_raw.setChecked(True)
    assert not any(isinstance(i, IntervalShade) for plot in page.trends.plots for i in plot.items)


def test_recommended_detectors(loaded):
    page = loaded.pages["cleaning"]
    kinds = [d.kind for d in page.recommended_detectors()]
    assert (
        DetectorKind.HAMPEL in kinds
        and DetectorKind.FLATLINE in kinds
        and DetectorKind.MAD in kinds
    )
