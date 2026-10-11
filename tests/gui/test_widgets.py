from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inferential_studio.ui.widgets.time_axis import TzDateAxisItem, format_time, to_epoch_seconds
from inferential_studio.ui.widgets.trend_view import TrendSeries, TrendView

pytestmark = pytest.mark.gui


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
    view._show_cursor(float(x[10]) + 30)
    text = view.readout.text()
    assert "A = 10" in text and "B = 20" in text  # last value at or before the cursor
    view._show_cursor(float(x[55]))
    assert "A =" not in view.readout.text()  # inside the gap: no value invented
    view.set_series([])
    assert view.plots == [] and view.empty_label.isVisibleTo(view)
