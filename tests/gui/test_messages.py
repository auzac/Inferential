"""Message centre and the fixed message panel."""

from __future__ import annotations

import pytest

from inferential_studio.infrastructure.settings import AppSettings
from inferential_studio.ui.main_window import MainWindow
from inferential_studio.ui.messages import Level, MessageCenter
from inferential_studio.ui.widgets.message_panel import PANEL_HEIGHT, MessagePanel, panel_height

pytestmark = pytest.mark.gui


def test_conditions_and_events(qtbot):
    center = MessageCenter()
    changes = []
    center.changed.connect(lambda: changes.append(1))
    center.set("processing.x", Level.WARNING, "careful")
    center.set("processing.x", Level.WARNING, "careful")  # unchanged: no signal
    assert len(changes) == 1
    center.set("variables.y", Level.ERROR, "broken", also=("processing",))
    center.error("save failed", "app")
    center.info("cleaning event", "cleaning")
    # Most severe first; stage filter keeps the stage's own, app-wide and "also" messages.
    assert [m.text for m in center.conditions("processing")] == ["broken", "careful"]
    assert "cleaning event" not in center.texts("processing")
    assert "save failed" in center.texts("processing")
    counts = center.counts("processing")
    assert counts[Level.ERROR] == 2 and counts[Level.WARNING] == 1
    center.set("processing.x", Level.WARNING, "")  # empty text clears
    assert center.condition("processing.x") is None
    center.clear_events()
    assert center.events == [] and center.condition("variables.y") is not None


def test_panel_is_fixed_and_filters_by_stage(qtbot):
    center = MessageCenter()
    panel = MessagePanel(center)
    qtbot.addWidget(panel)
    panel.set_stage("processing")
    assert panel.height() == PANEL_HEIGHT and panel.minimumHeight() == panel.maximumHeight()
    assert panel.list.topLevelItemCount() == 1  # "No messages."
    center.set("processing.a", Level.INFO, "a")
    center.warning("w", "cleaning")
    assert [m.text for m in panel.visible_messages()] == ["a"]
    assert not panel.badges[Level.WARNING].isVisibleTo(panel)
    panel.scope.setCurrentIndex(1)  # all stages
    assert [m.text for m in panel.visible_messages()] == ["a", "w"]
    assert panel.badges[Level.WARNING].text() == "1 warning"


def test_theme_switch_keeps_selection_and_scroll(qtbot):
    center = MessageCenter()
    panel = MessagePanel(center)
    qtbot.addWidget(panel)
    panel.show()
    for i in range(15):
        center.info(f"message number {i}", "cleaning")
    qtbot.waitUntil(lambda: panel.list.topLevelItemCount() == 15, timeout=2000)
    bar = panel.list.verticalScrollBar()
    qtbot.waitUntil(lambda: bar.maximum() > 0, timeout=2000)
    bar.setValue(bar.maximum())
    panel.list.setCurrentItem(panel.list.topLevelItem(12))
    expected = panel.list.currentItem().text(3)  # newest first: index 12 ≠ "message number 12"
    panel.apply_theme()  # a theme switch rebuilds the list: keep where the user was
    current = panel.list.currentItem()
    assert current is not None and current.text(3) == expected
    assert bar.value() == bar.maximum()


def test_errors_go_to_the_panel_not_a_dialog(qtbot):
    w = MainWindow(AppSettings())
    qtbot.addWidget(w)
    w.show()
    w.show_error("Cannot save project", "disk full", "trace")
    msg = w.session.messages.events[0]
    assert msg.level is Level.ERROR and msg.text.startswith("Cannot save project")
    assert msg.details == "trace"
    assert w.message_panel.height() == panel_height(w.message_panel.rows)  # fixed
    assert w.message_panel.rows in (3, 4)  # one row less on short screens
    assert w.session.messages.condition("training.state") is not None  # stages 7–9 guidance
