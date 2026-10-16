"""Read-only, sortable table view for pandas DataFrames."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from PySide6.QtCore import (
    QAbstractTableModel,
    QEvent,
    QModelIndex,
    QObject,
    QSize,
    QSortFilterProxyModel,
    Qt,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHeaderView, QTableView, QWidget

NUMERIC_ZERO = 1e-14  # smaller magnitudes are floating-point residue (e.g. an OLS bias)
GROUP_FROM = 10_000  # thousands separators from five digits up (years and ids stay plain)
COLUMN_MAX = 260  # px: no column is measured wider than this (long text elides instead)


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
        self._numeric_cols: list[bool] = []
        self._headers = headers or {}
        self._row_colours: dict[int, QColor] = {}
        if frame is not None:
            self.set_frame(frame)

    def set_frame(self, frame: pd.DataFrame, headers: dict[str, str] | None = None) -> None:
        self.beginResetModel()
        self._frame = frame
        self._columns = [frame.iloc[:, j].to_numpy() for j in range(frame.shape[1])]
        self._numeric_cols = [
            bool(pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype))
            for dtype in frame.dtypes
        ]
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
        if role == Qt.ItemDataRole.TextAlignmentRole:
            # Track the cells: as the columns widen, a header that sits at the other end of its
            # column from its values looks detached. Qt's default here is centred.
            if orientation == Qt.Orientation.Horizontal and section < len(self._numeric_cols):
                side = (
                    Qt.AlignmentFlag.AlignRight
                    if self._numeric_cols[section]
                    else Qt.AlignmentFlag.AlignLeft
                )
                return int(side | Qt.AlignmentFlag.AlignVCenter)
            return None
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            name = str(self._frame.columns[section])
            return self._headers.get(name, name)
        return format_value(self._frame.index[section])


class ColumnFiller(QObject):
    """Grow a table's columns, in proportion, until they fill the view.

    No Qt resize mode does this. ``ResizeToContents`` and ``Interactive`` leave the spare width
    of a wide window as an empty band to the right of the last column; ``Stretch`` gives every
    column the same width, which reads as wrong as soon as one column holds a variable name and
    the next holds a flag. Growing all of them in proportion keeps the table's rhythm at any
    window size, and no single column balloons.

    A table narrower than its content keeps its content widths and scrolls, rather than eliding
    text to fit. Widths the user drags become the proportions used from then on.
    """

    def __init__(self, table: QTableView) -> None:
        super().__init__(table)
        self._table = table
        self._weights: list[float] = []
        self._applying = False
        table.horizontalHeader().sectionResized.connect(self._section_resized)
        # The viewport's own resize is the signal to use: during the view's resizeEvent the
        # viewport may not have been laid out yet, and the width read would be the old one.
        table.viewport().installEventFilter(self)

    def refit(self) -> None:
        """Re-measure the columns from their content (capped), then fill the view.

        Call after the table's content changes.
        """
        self._table.resizeColumnsToContents()
        for col in range(self._table.model().columnCount()):
            self._table.setColumnWidth(col, min(self._table.columnWidth(col), COLUMN_MAX))
        self.measure()

    def measure(self) -> None:
        """Adopt the current column widths (just measured from content) as the proportions."""
        self.set_weights(
            [self._table.columnWidth(c) for c in range(self._table.model().columnCount())]
        )

    def set_weights(self, weights: list[float]) -> None:
        self._weights = [max(1.0, float(w)) for w in weights]
        self.apply()

    def apply(self) -> None:
        count = self._table.model().columnCount()
        view = self._table.viewport().width()
        total = sum(self._weights)
        if not self._weights or len(self._weights) != count or view <= 0 or total <= 0:
            return
        if view <= total:  # narrower than the content: keep the widths and let it scroll
            widths = [int(w) for w in self._weights]
        else:
            widths = [max(1, round(w * view / total)) for w in self._weights]
            widths[widths.index(max(widths))] += view - sum(widths)  # rounding: fill exactly
        if all(self._table.columnWidth(c) == widths[c] for c in range(count)):
            return
        self._applying = True
        try:
            for col, width in enumerate(widths):
                self._table.setColumnWidth(col, width)
        finally:
            self._applying = False

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        if event.type() == QEvent.Type.Resize:
            self.apply()
        return False

    def _section_resized(self, *_args) -> None:
        if self._applying:
            return  # our own distribution, not the user dragging a divider
        # Remember the widths the user chose, but leave them alone: re-distributing under the
        # mouse would drag the column back out from under the pointer. The next resize of the
        # view scales these proportions to fill it.
        self._weights = [
            max(1.0, float(self._table.columnWidth(c)))
            for c in range(self._table.model().columnCount())
        ]


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
        self._filler = ColumnFiller(self)  # keeps the columns filling a wide window
        self._fit_rows: int | None = None

    def set_frame(self, frame: pd.DataFrame, headers: dict[str, str] | None = None) -> None:
        self.model_.set_frame(frame, headers)
        self._filler.refit()
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

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        hint = super().sizeHint()
        if self._fit_rows is not None:  # containers (tab widgets) size to the fitted height
            hint.setHeight(self.maximumHeight())
        return hint

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().resizeEvent(event)
        self._apply_fit()  # a narrower view may now need the horizontal scroll bar
