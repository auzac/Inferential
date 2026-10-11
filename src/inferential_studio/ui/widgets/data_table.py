"""Read-only, sortable table view for pandas DataFrames."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHeaderView, QTableView, QWidget

NUMERIC_ZERO = 1e-14  # smaller magnitudes are floating-point residue (e.g. an OLS bias)
GROUP_FROM = 10_000  # thousands separators from five digits up (years and ids stay plain)


def format_value(value: object) -> str:
    """Display text for one table cell: five significant digits, plain zeros, grouped counts.

    Formatting is for reading only; sorting uses the raw value and exports use full precision.
    """
    if (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or value is pd.NA
        or value is pd.NaT
    ):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}" if abs(int(value)) >= GROUP_FROM else str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isinf(v):
            return "∞" if v > 0 else "−∞"
        a = abs(v)
        if a < NUMERIC_ZERO:
            return "0"
        if v.is_integer() and a < 1e15:
            return f"{int(v):,}" if a >= GROUP_FROM else str(int(v))
        if a >= 1e5:
            return f"{v:,.0f}" if a < 1e12 else f"{v:.4e}"
        if a < 1e-4:
            return f"{v:.3e}"
        return f"{v:.5g}"
    return str(value)


class DataFrameModel(QAbstractTableModel):
    def __init__(
        self, frame: pd.DataFrame | None = None, headers: dict[str, str] | None = None, parent=None
    ) -> None:
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._columns: list[np.ndarray] = []  # column values as arrays: fast cell access
        self._headers = headers or {}
        self._row_colours: dict[int, QColor] = {}
        if frame is not None:
            self.set_frame(frame)

    def set_frame(self, frame: pd.DataFrame, headers: dict[str, str] | None = None) -> None:
        self.beginResetModel()
        self._frame = frame
        self._columns = [frame.iloc[:, j].to_numpy() for j in range(frame.shape[1])]
        if headers is not None:
            self._headers = headers
        self._row_colours = {}
        self.endResetModel()

    def set_row_colour(self, row: int, colour: QColor) -> None:
        self._row_colours[row] = colour

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self._frame)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self._frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._columns[index.column()][index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return format_value(value)
        if role == Qt.ItemDataRole.UserRole:  # sort key
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
                value, bool
            ):
                v = float(value)
                return -math.inf if math.isnan(v) else v
            return format_value(value)
        numeric = isinstance(value, (int, float, np.integer, np.floating))
        if role == Qt.ItemDataRole.TextAlignmentRole and numeric:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.BackgroundRole and index.row() in self._row_colours:
            return self._row_colours[index.row()]
        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            name = str(self._frame.columns[section])
            return self._headers.get(name, name)
        return format_value(self._frame.index[section])


class DataTableView(QTableView):
    def __init__(self, parent: QWidget | None = None, *, sortable: bool = True) -> None:
        super().__init__(parent)
        self.model_ = DataFrameModel(parent=self)
        if sortable:
            proxy = QSortFilterProxyModel(self)
            proxy.setSourceModel(self.model_)
            proxy.setSortRole(Qt.ItemDataRole.UserRole)
            self.setModel(proxy)
            # No initial sort: rows keep the order of the frame (e.g. a ranking) until the
            # user clicks a column header.
            self.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
            self.setSortingEnabled(True)
        else:
            self.setModel(self.model_)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.verticalHeader().setDefaultSectionSize(22)
        # Size columns from the first rows only: measuring every cell of a long table is slow.
        self.horizontalHeader().setResizeContentsPrecision(200)
        self._fit_rows: int | None = None

    def set_frame(self, frame: pd.DataFrame, headers: dict[str, str] | None = None) -> None:
        self.model_.set_frame(frame, headers)
        self.resizeColumnsToContents()
        for col in range(self.model_.columnCount()):
            self.setColumnWidth(col, min(self.columnWidth(col), 260))
        self._apply_fit()

    def fit_to_rows(self, max_rows: int = 8, min_rows: int = 2) -> None:
        """Make the table exactly as tall as its rows (up to ``max_rows``; then it scrolls).

        For short summary tables (model comparison, metrics): no empty band under the rows,
        and the page below keeps the space.
        """
        self._fit_rows = max_rows
        self._min_rows = min_rows
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._apply_fit()

    def _apply_fit(self) -> None:
        if self._fit_rows is None:
            return
        rows = max(min(self.model_.rowCount(), self._fit_rows), self._min_rows)
        header = self.horizontalHeader().sizeHint().height()
        height = header + rows * self.verticalHeader().defaultSectionSize() + 2 * self.frameWidth()
        if self.horizontalScrollBar().maximum() > 0:  # columns wider than the view
            height += self.horizontalScrollBar().sizeHint().height()
        if height != self.height() or self.minimumHeight() != height:
            self.setFixedHeight(height)

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().resizeEvent(event)
        self._apply_fit()  # a narrower view may now need the horizontal scroll bar
