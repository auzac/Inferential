from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QGraphicsSceneMouseEvent

from inferential_studio.ui.widgets.cursor_readout import CursorReadout
from inferential_studio.ui.widgets.time_axis import TzDateAxisItem, format_time, to_epoch_seconds
from inferential_studio.ui.widgets.trend_view import BINS, MAX_PLOTS, TrendSeries, TrendView

pytestmark = pytest.mark.gui


def _click_event(view: TrendView, x: float, button: Qt.MouseButton) -> MouseClickEvent:
    """A synthetic mouse click on the first plot at time x (press event wrapped by pyqtgraph)."""
    press = QGraphicsSceneMouseEvent(QEvent.Type.GraphicsSceneMousePress)
    press.setScenePos(view.plots[0].vb.mapViewToScene(pg.Point(x, 0.0)))
    press.setButton(button)
    return MouseClickEvent(press)


def test_epoch_seconds_requires_aware_times():
    idx = pd.DatetimeIndex(["2026-01-01 00:00"], tz="UTC")
    assert to_epoch_seconds(idx)[0] == pd.Timestamp("2026-01-01", tz="UTC").timestamp()
    with pytest.raises(ValueError):
        to_epoch_seconds(pd.DatetimeIndex(["2026-01-01"]))


def test_axis_labels_follow_timezone_and_dst(qtbot):
    import pyqtgraph as pg

    axis = TzDateAxisItem("Europe/London")
    widget = pg.PlotWidget(axisItems={"bottom": axis})
    qtbot.addWidget(widget)
    widget.show()
    widget.resize(600, 300)
    widget.grab()  # paint once so the axis knows its font metrics
    winter = pd.Timestamp("2026-01-15 12:00", tz="UTC").timestamp()
    summer = pd.Timestamp("2026-07-15 12:00", tz="UTC").timestamp()
    levels = axis.tickValues(winter - 6 * 3600, winter + 6 * 3600, 500)
    hourly = next(sp for sp, _ in levels if sp == 3600)  # the hour tick level ("%H:%M")
    assert axis.tickStrings([winter], 1, hourly)[0] == "12:00"  # GMT
    assert axis.tickStrings([summer], 1, hourly)[0] == "13:00"  # BST, though the view is in winter
    axis.set_timezone("Asia/Singapore")
    assert axis.tickStrings([winter], 1, hourly)[0] == "20:00"
    assert format_time(winter, "Asia/Singapore") == "2026-01-15 20:00:00"


def test_trend_view_stacks_series_and_reads_cursor(qtbot):
    view = TrendView("UTC")
    qtbot.addWidget(view)
    x = to_epoch_seconds(pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC"))
    y = np.arange(100.0)
    y[50:60] = np.nan
    view.set_series(
        [
            TrendSeries("A", x, y, "degC"),
            TrendSeries("B", x, y * 2, "degC"),
            TrendSeries("Lab", x[::10], y[::10], points=True),
        ]
    )
    assert len(view.plots) == 3
    view.mode.setCurrentIndex(view.mode.findData("units"))
    assert len(view.plots) == 2  # A and B share degC
    seen: list[tuple[str, dict]] = []
    view.hover.connect(lambda t, values: seen.append((t, values)))
    view._show_cursor(float(x[10]) + 30)
    time_text, values = seen[-1]
    assert values["A"] == "10 degC" and values["B"] == "20 degC"  # last value at or before cursor
    assert time_text.startswith("2026-01-01 00:10")
    view._show_cursor(float(x[55]))
    assert "A" not in seen[-1][1]  # inside the gap: no value invented
    view.set_series([])
    assert view.plots == [] and view.empty_label.isVisibleTo(view)


def test_trend_view_updates_only_changed_plots(qtbot):
    view = TrendView("UTC")
    qtbot.addWidget(view)
    x = to_epoch_seconds(pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC"))
    view.set_series([TrendSeries("A", x, np.arange(100.0))])
    plot_a = view.plots[0]
    # Adding a series reuses the existing plot instead of rebuilding everything.
    view.set_series(
        [TrendSeries("A", x, np.arange(100.0)), TrendSeries("B", x, np.arange(100.0) * 2)]
    )
    assert len(view.plots) == 2 and view.plots[0] is plot_a
    # Removing one keeps the other (with its zoom preserved by keep_range).
    view.set_x_range(float(x[10]), float(x[20]))
    view.set_series([TrendSeries("B", x, np.arange(100.0) * 2)], keep_range=True)
    assert len(view.plots) == 1 and view.plots[0] is not plot_a
    assert view.x_range() == (float(x[10]), float(x[20]))


def test_trend_view_caps_plot_count(qtbot):
    view = TrendView("UTC")
    qtbot.addWidget(view)
    x = to_epoch_seconds(pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC"))
    view.set_series([TrendSeries(f"V{i}", x, np.arange(100.0)) for i in range(15)])
    assert len(view.plots) == MAX_PLOTS == 12
    assert "12 of 15" in view.limit_label.text() and view.limit_label.isVisibleTo(view)


def test_trend_view_draws_envelope_then_raw_slice(qtbot):
    view = TrendView("UTC")
    qtbot.addWidget(view)
    x = to_epoch_seconds(pd.date_range("2026-01-01", periods=10_000, freq="1s", tz="UTC"))
    view.set_series([TrendSeries("A", x, np.arange(10_000.0))])
    plot = view.plots[0]
    lo_item, hi_item, _sig = view._items[view._plot_keys[0]]["A"]
    # Zoomed out: the curves hold the precomputed min/max envelope, not the raw points.
    assert len(lo_item.xData) == BINS == 2048
    assert hi_item.isVisible()
    # Zoomed in: the visible window is drawn from the raw samples.
    view.set_x_range(float(x[100]), float(x[130]))
    assert len(lo_item.xData) == 33  # 99..131, one point of slack on each side
    assert not hi_item.isVisible()
    # Zooming back out restores the envelope.
    plot.setXRange(float(x[0]), float(x[-1]), padding=0)
    assert len(lo_item.xData) == BINS and hi_item.isVisible()


def test_trend_view_click_pins_point_and_second_click_unlocks(qtbot):
    view = TrendView("UTC")
    qtbot.addWidget(view)
    x = to_epoch_seconds(pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC"))
    view.set_series([TrendSeries("A", x, np.arange(100.0), "degC")])
    seen: list[tuple[str, dict]] = []
    cleared: list[int] = []
    view.point_selected.connect(lambda t, values: seen.append((t, values)))
    view.point_cleared.connect(lambda: cleared.append(1))
    view._toggle_point(float(x[10]))
    assert view._selected_x == float(x[10])
    assert seen[-1][1] == {"A": "10 degC"}
    assert view._sel_line_of[view.plots[0]].isVisible()
    view._toggle_point(float(x[20]))  # a second click anywhere releases the pin
    assert view._selected_x is None and cleared == [1]
    assert not view._sel_line_of[view.plots[0]].isVisible()
    assert len(seen) == 1
    # Dragging a highlight band also releases the pin.
    view._toggle_point(float(x[10]))
    view._set_band(float(x[30]), float(x[40]))
    assert view._selected_x is None and cleared == [1, 1]


def test_trend_view_drag_creates_synced_band(qtbot):
    view = TrendView("UTC")
    qtbot.addWidget(view)
    x = to_epoch_seconds(pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC"))
    view.set_series(
        [TrendSeries("A", x, np.arange(100.0)), TrendSeries("B", x, np.arange(100.0) * 2)]
    )
    view._set_band(float(x[5]), float(x[25]))
    assert len(view._bands) == len(view.plots) == 2
    for _plot, region in view._bands:
        assert region.getRegion() == (float(x[5]), float(x[25]))
    # Moving one band keeps the others in step.
    view._bands[0][1].setRegion((float(x[8]), float(x[20])))
    assert all(r.getRegion() == (float(x[8]), float(x[20])) for _plot, r in view._bands)
    # A click anywhere else removes the highlight.
    view._click_point(
        view.plots[0].vb, _click_event(view, float(x[60]), Qt.MouseButton.LeftButton)
    )
    assert view._bands == []


def test_trend_view_right_click_raises_plot_menu(qtbot, monkeypatch):
    view = TrendView("UTC")
    qtbot.addWidget(view)
    x = to_epoch_seconds(pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC"))
    view.set_series([TrendSeries("A", x, np.arange(100.0))])
    calls: list[Qt.MouseButton] = []
    monkeypatch.setattr(
        pg.ViewBox, "raiseContextMenu", lambda self, ev: calls.append(ev.button())
    )
    view.plots[0].vb.mouseClickEvent(_click_event(view, float(x[50]), Qt.MouseButton.RightButton))
    assert calls == [Qt.MouseButton.RightButton]


def test_cursor_readout_locks_on_selected_point():
    box = CursorReadout()
    assert box.minimumHeight() == box.maximumHeight() == 220  # static, taller than before
    box.show_hover("t1", {"A": "1"})
    assert "t1" in box.time_label.text() and box.title() == "At cursor"
    box.show_point("t2", {"A": "2"})
    assert box.title() == "Selected point" and "t2" in box.time_label.text()
    box.show_hover("t3", {"A": "3"})  # ignored while pinned
    assert "t2" in box.time_label.text() and "t3" not in box.time_label.text()
    box.unlock()
    box.show_hover("t3", {"A": "3"})
    assert box.title() == "At cursor" and "t3" in box.time_label.text()
