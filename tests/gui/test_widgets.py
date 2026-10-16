from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent, MouseDragEvent
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
    # Starting a drag also releases the pin.
    view._toggle_point(float(x[10]))
    view._set_band(float(x[30]), float(x[40]))
    assert view._selected_x is None and cleared == [1, 1]
    view._finish_band(float(x[30]), float(x[40]))


def _drag(view: TrendView, x0: float, x1: float) -> None:
    """A left-drag on the first plot from time x0 to x1 (start, then finish)."""
    vb = view.plots[0].vb
    a, b = vb.mapViewToScene(pg.Point(x0, 0.0)), vb.mapViewToScene(pg.Point(x1, 0.0))
    press = QGraphicsSceneMouseEvent(QEvent.Type.GraphicsSceneMousePress)
    press.setScenePos(a)
    press.setButton(Qt.MouseButton.LeftButton)
    move = QGraphicsSceneMouseEvent(QEvent.Type.GraphicsSceneMouseMove)
    move.setScenePos(b)
    move.setButtonDownScenePos(Qt.MouseButton.LeftButton, a)
    move.setButtons(Qt.MouseButton.LeftButton)
    vb.mouseDragEvent(MouseDragEvent(move, press, None, start=True))
    assert len(view._bands) == len(view.plots)  # highlighted on every chart while dragging
    vb.mouseDragEvent(MouseDragEvent(move, press, press, finish=True))


def _shown_view(qtbot, series: list[TrendSeries]) -> TrendView:
    view = TrendView("UTC")
    qtbot.addWidget(view)
    view.resize(900, 600)
    view.show()
    view.set_series(series)
    qtbot.wait(50)  # lay out the charts so view <-> pixel mapping is real
    return view


def test_trend_view_drag_zooms_to_period_and_back_steps_out(qtbot):
    x = to_epoch_seconds(pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC"))
    view = _shown_view(
        qtbot, [TrendSeries("A", x, np.arange(100.0)), TrendSeries("B", x, np.arange(100.0) * 2)]
    )
    full = view.visible_x_range()
    assert full[0] <= x[0] and full[1] >= x[-1]
    assert not view.back_button.isEnabled()
    _drag(view, float(x[10]), float(x[50]))
    assert view._bands == []  # the highlight becomes the new view
    lo, hi = view.visible_x_range()
    assert lo == pytest.approx(float(x[10]), abs=1.0) and hi == pytest.approx(float(x[50]), abs=1.0)
    # Every chart follows, and each y-axis fits the data now on screen (deferred refit).
    assert view.plots[1].viewRange()[0] == pytest.approx([lo, hi])
    qtbot.wait(20)
    y_lo, y_hi = view.plots[0].viewRange()[1]
    assert 5 <= y_lo <= 12 and 48 <= y_hi <= 55
    assert view.back_button.isEnabled()
    view.zoom_to(float(x[20]), float(x[30]))
    view.zoom_back()
    assert view.visible_x_range() == pytest.approx((lo, hi))
    view.zoom_back()  # bottom of the history: the view before the first zoom
    assert view.visible_x_range() == pytest.approx(full)
    assert not view.back_button.isEnabled()
    view.zoom_to(float(x[20]), float(x[30]))
    view.fit_all()
    lo, hi = view.visible_x_range()
    assert lo <= x[0] and hi >= x[-1] and not view.back_button.isEnabled()


def test_trend_view_short_drag_does_not_zoom(qtbot):
    x = to_epoch_seconds(pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC"))
    view = _shown_view(qtbot, [TrendSeries("A", x, np.arange(100.0))])
    before = view.visible_x_range()
    _drag(view, float(x[10]), float(x[10]) + 1.0)  # far less than a few pixels
    assert view._bands == [] and view.visible_x_range() == pytest.approx(before)
    assert not view.back_button.isEnabled()


def test_trend_view_right_click_opens_standard_chart_menu(qtbot, monkeypatch):
    from inferential_studio.ui.widgets import chart_menu

    view = TrendView("UTC")
    qtbot.addWidget(view)
    x = to_epoch_seconds(pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC"))
    view.set_series([TrendSeries("A", x, np.arange(100.0))])
    raw: list[Qt.MouseButton] = []
    shown: list[object] = []
    monkeypatch.setattr(pg.ViewBox, "raiseContextMenu", lambda self, ev: raw.append(ev.button()))
    monkeypatch.setattr(chart_menu, "show_chart_menu", lambda v, pos: shown.append(v))
    view.plots[0].vb.mouseClickEvent(_click_event(view, float(x[50]), Qt.MouseButton.RightButton))
    assert raw == []  # PyQtGraph's own menu (View All, axes, mouse mode …) never opens
    assert shown == [view.canvas]  # the standard menu acts on every stacked chart
    assert not view.plots[0].menuEnabled() and not view.plots[0].autoBtn.isVisible()
    assert view._selected_x is None  # a right-click does not pin a point
    assert not hasattr(view, "save_button")  # saving lives in the standard menu
    assert view.fit_button.text() == "Fit all"


def test_cursor_readout_locks_on_selected_point():
    box = CursorReadout()
    # 220 px when there is room, down to 110 px on short screens; never sized by its content.
    assert (box.minimumHeight(), box.maximumHeight(), box.sizeHint().height()) == (110, 220, 220)
    box.show_hover("t0", {f"V{i}": "1" for i in range(40)})
    assert box.sizeHint().height() == 220
    box.show_hover("t1", {"A": "1"})
    assert "t1" in box.time_label.text() and box.title() == "At cursor"
    box.show_point("t2", {"A": "2"})
    assert box.title() == "Selected point" and "t2" in box.time_label.text()
    box.show_hover("t3", {"A": "3"})  # ignored while pinned
    assert "t2" in box.time_label.text() and "t3" not in box.time_label.text()
    box.unlock()
    box.show_hover("t3", {"A": "3"})
    assert box.title() == "At cursor" and "t3" in box.time_label.text()


def test_table_numbers_are_readable():
    from inferential_studio.ui.widgets.data_table import format_value

    assert format_value(1.227e-17) == "0"  # floating-point residue, e.g. an OLS training bias
    assert format_value(-0.392929) == "-0.39293"
    assert format_value(0.000637) == "0.000637"
    assert format_value(6.4e-05) == "6.400e-05"
    assert format_value(316.0) == "316"
    assert format_value(40320.0) == format_value(np.int64(40320)) == "40,320"
    assert format_value(2026) == "2026"  # four-digit ids and years stay ungrouped
    assert format_value(123456.7) == "123,457"
    assert format_value(True) == "Yes" and format_value(float("nan")) == ""


def test_charts_show_a_message_until_they_hold_data(qtbot):
    from inferential_studio.ui.widgets.plots import (
        EMPTY_TEXT,
        bar_chart,
        chart_item,
        legend,
        plot_widget,
        show_placeholder,
    )

    plot = plot_widget("Residual vs predicted")
    qtbot.addWidget(plot)
    legend(plot)
    item = chart_item(plot)
    # A new chart is empty: message instead of bare 0-1 axes, no key in the header.
    assert item.is_empty and item.empty_text == EMPTY_TEXT
    assert not item.getAxis("left").isVisible() and item.legend is None
    plot.addItem(pg.InfiniteLine(0.5))  # reference lines alone are not data
    plot.plot([], [])
    assert item.is_empty
    plot.plot([0, 1, 2], [1, 3, 2], name="data")
    assert not item.is_empty
    assert item.getAxis("left").isVisible() and item.key_names == ["data"]
    assert "data" in item.titleLabel.text and "Residual vs predicted" in item.titleLabel.text
    assert item.vb.autoRangeEnabled() == [True, True]
    show_placeholder(plot, "Train models first.")
    assert item.is_empty and item.empty_text == "Train models first."
    assert item.key_names == [] and "data" not in item.titleLabel.text
    plot.clear()  # the page's own message stays until data arrives
    assert item.is_empty and item.empty_text == "Train models first."
    plot.setLabel("left", "Mean RMSE")  # labelling an empty chart must not bring its axis back
    assert not item.getAxis("left").isVisible()
    # Clearing an already-empty chart must not lose its axes (the coefficient chart did).
    bar_chart(plot, ["a", "b"], np.array([1.0, -0.5]))
    assert not item.is_empty and item.getAxis("left").isVisible()
    assert item.getAxis("bottom").isVisible()


def test_series_colours_match_the_exported_report():
    from inferential_studio.core import colours
    from inferential_studio.ui.widgets.plots import BLUE, NEGATIVE, ORANGE, POSITIVE, VERMILION

    assert (BLUE, ORANGE, VERMILION) == (colours.TRAIN, colours.VALIDATION, colours.TEST)
    assert (POSITIVE, NEGATIVE) == (colours.POSITIVE, colours.NEGATIVE)


def test_chart_key_marks_follow_the_series_style(qtbot):
    from inferential_studio.ui.widgets.plots import chart_item, legend, plot_widget, scatter

    plot = plot_widget()
    qtbot.addWidget(plot)
    legend(plot, "bottom-right")  # the corner is ignored: the key sits in the header
    scatter(plot, [1, 2], [1, 2], "#0072B2", "Training")
    plot.plot([0, 1], [0, 1], pen=pg.mkPen("#666666", style=Qt.PenStyle.DashLine), name="identity")
    plot.plot([0, 1], [1, 0], pen=pg.mkPen("#D55E00"), name="Prediction")
    plot.plot([0, 1], [1, 0], pen=pg.mkPen("#999999"))  # unnamed: not in the key
    item = chart_item(plot)
    assert item.key_names == ["Training", "identity", "Prediction"]
    header = item.titleLabel.text
    assert "●" in header and "┄" in header and "━" in header and "#0072b2" in header.lower()


def test_every_chart_uses_the_standard_menu(qtbot, monkeypatch, tmp_path):
    from inferential_studio.ui.widgets import chart_menu
    from inferential_studio.ui.widgets.plots import plot_widget, show_placeholder

    plot = plot_widget("Residual vs predicted")
    qtbot.addWidget(plot)
    plot.resize(400, 300)
    plot.plot([0, 1, 2], [1, 3, 2])
    assert isinstance(plot.getPlotItem().vb, chart_menu.ChartViewBox)
    assert not plot.getPlotItem().autoBtn.isVisible()
    shown: list[object] = []
    monkeypatch.setattr(chart_menu, "show_chart_menu", lambda v, pos: shown.append(v))
    vb = plot.getPlotItem().vb
    press = QGraphicsSceneMouseEvent(QEvent.Type.GraphicsSceneMousePress)
    press.setScenePos(vb.mapViewToScene(pg.Point(1.0, 2.0)))
    press.setButton(Qt.MouseButton.RightButton)
    vb.mouseClickEvent(MouseClickEvent(press))
    assert shown == [plot]
    assert chart_menu.chart_name(plot) == "residual_vs_predicted"
    assert chart_menu.save_chart(plot, tmp_path / "chart.png")
    assert (tmp_path / "chart.png").stat().st_size > 0
    show_placeholder(plot, "Nothing yet")  # no menu on an empty chart
    vb.mouseClickEvent(MouseClickEvent(press))
    assert shown == [plot]


def test_chart_menu_has_copy_and_save_only(qtbot):
    from inferential_studio.ui.widgets import chart_menu
    from inferential_studio.ui.widgets.plots import plot_widget

    plot = plot_widget(name="lag_sensitivity")
    qtbot.addWidget(plot)
    menu = chart_menu.show_chart_menu(plot, plot.mapToGlobal(plot.rect().center()))
    assert [a.text() for a in menu.actions()] == ["Copy image", "Save image…"]
    menu.close()
    assert chart_menu.chart_name(plot) == "lag_sensitivity"
    chart_menu.copy_chart(plot)
    from PySide6.QtGui import QGuiApplication

    assert not QGuiApplication.clipboard().pixmap().isNull()


def test_summary_table_fits_its_rows(qtbot):
    from inferential_studio.ui.widgets.data_table import DataTableView

    table = DataTableView()
    qtbot.addWidget(table)
    table.resize(600, 400)
    table.fit_to_rows(max_rows=6)
    table.set_frame(pd.DataFrame({"RMSE": [0.1, 0.2, 0.3, 0.4]}, index=list("ABCD")))
    table.show()
    qtbot.wait(10)
    header = table.horizontalHeader().sizeHint().height()
    assert table.height() == header + 4 * 22 + 2 * table.frameWidth()  # no empty band
    table.set_frame(pd.DataFrame({"RMSE": np.arange(20.0)}))
    assert table.height() == header + 6 * 22 + 2 * table.frameWidth()  # then it scrolls


def test_table_columns_fill_a_wide_view(qtbot):
    from PySide6.QtCore import Qt

    from inferential_studio.ui.widgets.data_table import DataTableView

    table = DataTableView()
    qtbot.addWidget(table)
    table.show()
    frame = pd.DataFrame(
        {"Variable": ["Column Pressure (barg)", "Tray 20 Temperature"], "r": [-0.3552, 0.0389]}
    )
    table.resize(100, 300)  # far narrower than the content
    table.set_frame(frame)
    qtbot.wait(10)
    content = [table.columnWidth(c) for c in range(2)]
    assert sum(content) > table.viewport().width()
    assert table.horizontalScrollBar().maximum() > 0  # it scrolls rather than eliding to fit
    table.resize(1200, 300)
    qtbot.wait(10)
    wide = [table.columnWidth(c) for c in range(2)]
    assert sum(wide) == table.viewport().width()  # no empty band beside the last column
    assert all(w > c for w, c in zip(wide, content, strict=True))  # every column grew
    assert wide[0] > wide[1]  # "Variable" holds more text, so it stays the wider column
    share = wide[0] / sum(wide)
    table.setColumnWidth(0, 60)  # as if the user dragged the divider to narrow "Variable"
    table.resize(900, 300)  # the next resize spreads the widths the user chose
    qtbot.wait(10)
    assert sum(table.columnWidth(c) for c in range(2)) == table.viewport().width()
    assert table.columnWidth(0) / table.viewport().width() < share  # the drag became the ratio
    # Each header sits over its own values: text left, numbers right.
    horizontal, role = Qt.Orientation.Horizontal, Qt.ItemDataRole.TextAlignmentRole
    assert table.model_.headerData(0, horizontal, role) & int(Qt.AlignmentFlag.AlignLeft)
    assert table.model_.headerData(1, horizontal, role) & int(Qt.AlignmentFlag.AlignRight)


def test_stacked_trends_scroll_with_readable_axes_on_short_screens(qtbot):
    from inferential_studio.ui.widgets.trend_view import MIN_PLOT_HEIGHT

    view = TrendView("UTC")
    qtbot.addWidget(view)
    x = to_epoch_seconds(pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC"))
    view.set_series([TrendSeries(f"Tag {i}", x, np.arange(100.0)) for i in range(5)])
    view.resize(650, 180)
    view.show()
    qtbot.wait(50)
    assert view.canvas.height() >= 5 * MIN_PLOT_HEIGHT
    assert view.chart_scroll.verticalScrollBar().maximum() > 0
    assert all(p.height() >= 100 for p in view.plots)
    assert all(p.getAxis("bottom").style["showValues"] for p in view.plots)
    view.chart_scroll.verticalScrollBar().setValue(view.chart_scroll.verticalScrollBar().maximum())
    qtbot.wait(10)
    assert view.plots[-1].height() >= 100  # the last signal is reachable, not squeezed out


def test_flow_row_reserves_its_wrapped_height(qtbot):
    from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

    from inferential_studio.ui.widgets.flow_layout import flow_row

    holder = QWidget()
    qtbot.addWidget(holder)
    row = flow_row(*(QPushButton(f"Button number {i}") for i in range(6)))
    layout = QVBoxLayout(holder)
    layout.addWidget(row)
    layout.addStretch(1)
    holder.resize(260, 400)
    holder.show()
    qtbot.wait(10)
    buttons = row.findChildren(QPushButton)
    assert len({b.y() for b in buttons}) > 1  # wrapped onto several lines
    assert all(b.geometry().bottom() <= row.height() for b in buttons)  # none spills out


def test_value_axis_never_draws_overlapping_labels(qtbot):
    from PySide6.QtGui import QImage, QPainter

    from inferential_studio.ui.widgets.plots import ValueAxis, plot_widget

    widget = plot_widget()
    qtbot.addWidget(widget)
    widget.plot([0, 1], [-44, 358])
    widget.resize(400, 60)  # a stacked-trend sized chart: ~38 px of axis for four ticks
    widget.show()
    qtbot.waitExposed(widget)
    widget.setYRange(-44, 358, padding=0)
    qtbot.wait(50)
    axis = widget.getAxis("left")
    assert isinstance(axis, ValueAxis)

    def label_rects(generate) -> list:
        image = QImage(10, 10, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        _, _, texts = generate(painter)
        painter.end()
        return [spec[0] for spec in texts]

    def overlapping(rects: list) -> bool:
        return any(a.intersects(b) for i, a in enumerate(rects) for b in rects[i + 1 :])

    assert overlapping(label_rects(lambda p: pg.AxisItem.generateDrawSpecs(axis, p)))
    rects = label_rects(axis.generateDrawSpecs)
    assert rects and not overlapping(rects)  # still labelled, nothing on top of each other


def test_variable_name_and_checkbox_each_toggle_once(qtbot):
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QStyle, QStyleOptionViewItem

    from inferential_studio.ui.widgets.variable_list import VariableList

    widget = VariableList()
    qtbot.addWidget(widget)
    widget.set_variables(["Tray temperature"], {}, set())
    widget.show()
    item = widget.list.item(0)
    rect = widget.list.visualItemRect(item)
    # Click the name, then click the same name again to deselect.
    pos = QPoint(rect.left() + 70, rect.center().y())
    with qtbot.waitSignal(widget.selection_changed):
        qtbot.mouseClick(widget.list.viewport(), Qt.MouseButton.LeftButton, pos=pos)
    assert widget.checked() == ["Tray temperature"]
    qtbot.mouseClick(widget.list.viewport(), Qt.MouseButton.LeftButton, pos=pos)
    assert widget.checked() == []
    option = QStyleOptionViewItem()
    option.initFrom(widget.list)
    option.rect = rect
    option.features |= QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
    checkbox = widget.list.style().subElementRect(
        QStyle.SubElement.SE_ItemViewItemCheckIndicator, option, widget.list
    )
    events = []
    widget.selection_changed.connect(lambda: events.append(True))
    qtbot.mouseClick(widget.list.viewport(), Qt.MouseButton.LeftButton, pos=checkbox.center())
    assert widget.checked() == ["Tray temperature"] and len(events) == 1
    qtbot.keyClick(widget.list, Qt.Key.Key_Space)
    assert widget.checked() == []
