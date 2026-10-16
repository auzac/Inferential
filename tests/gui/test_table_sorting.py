from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QPersistentModelIndex, Qt, QTimer
from PySide6.QtGui import QColor

from inferential_studio.ui.widgets import data_table
from inferential_studio.ui.widgets.data_table import DataTableView

pytestmark = pytest.mark.gui


def test_sort_preserves_record_selection_labels_colours_and_precision(qtbot):
    table = DataTableView()
    qtbot.addWidget(table)
    frame = pd.DataFrame({"value": pd.array([10, None, 2, 2], dtype="Int64")},
                         index=["ten", "missing", "first two", "second two"])
    table.set_frame(frame)
    table.model_.set_row_colour(0, QColor("red"))
    table.selectRow(0)
    saved = QPersistentModelIndex(table.model().index(0, 0))
    table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    assert table.model_._order.tolist() == [1, 2, 3, 0]
    assert saved.row() == 3 and saved.data() == "10"
    assert table.selectionModel().selectedRows()[0].row() == 3
    assert table.model().headerData(3, Qt.Orientation.Vertical) == "ten"
    assert saved.data(Qt.ItemDataRole.BackgroundRole) == QColor("red")
    table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    assert table.model_._order.tolist() == [0, 2, 3, 1]
    pd.testing.assert_frame_equal(frame, table.model_.frame)
    table.set_frame(pd.DataFrame({"value": [2**60 + 1, 2**60]}))
    table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    assert table.model_._order.tolist() == [1, 0]


def test_large_sort_leaves_event_loop_running_and_ignores_replaced_data(qtbot, monkeypatch):
    table = DataTableView()
    qtbot.addWidget(table)
    table.set_frame(pd.DataFrame({"x": np.arange(50_000, 0, -1)}))
    entered, release = threading.Event(), threading.Event()
    original = data_table._sorted_rows
    threads = []

    def held(*args, **kwargs):
        threads.append(threading.get_ident())
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(data_table, "_sorted_rows", held)
    heartbeat = []
    try:
        table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        qtbot.waitUntil(entered.is_set)
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        qtbot.waitUntil(lambda: bool(heartbeat))
        assert threads[0] != threading.get_ident()
        table.set_frame(pd.DataFrame({"x": [7, 9]}))
    finally:
        release.set()
    qtbot.waitUntil(lambda: not table.model_._sort_tasks.busy)
    assert table.model_._order.tolist() == [0, 1]
    assert table.model().index(0, 0).data() == "7"


def test_latest_sort_wins_for_large_table(qtbot):
    table = DataTableView()
    qtbot.addWidget(table)
    table.set_frame(pd.DataFrame({"x": np.arange(50_000)}))
    table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    qtbot.waitUntil(lambda: not table.model_._sort_tasks.busy)
    assert table.model().index(0, 0).data() == "49,999"
    assert table.model().index(49_999, 0).data() == "0"
