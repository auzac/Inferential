"""Toolbar layout that wraps onto the next line instead of squeezing or clipping its buttons.

Used for the button rows above tables and charts: on a wide window they sit on one line, on a
laptop screen they wrap, and a button's text is never cut off. The minimum width of a row is
the widest single item, so a page's minimum width no longer grows with its number of buttons.
"""

from __future__ import annotations

from PySide6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QWidget


class FlowLayout(QLayout):
    def __init__(self, parent: QWidget | None = None, *, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    # -- QLayout interface ------------------------------------------------------

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        # Preferred: everything on one line.
        m = self.contentsMargins()
        width = sum(i.sizeHint().width() for i in self._visible()) + self._spacing * max(
            len(self._visible()) - 1, 0
        )
        height = max((i.sizeHint().height() for i in self._visible()), default=0)
        return QSize(width + m.left() + m.right(), height + m.top() + m.bottom())

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._visible():
            size = size.expandedTo(item.minimumSize())
        m: QMargins = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    # -- helpers -----------------------------------------------------------------

    def _visible(self) -> list[QLayoutItem]:
        return [i for i in self._items if not (i.widget() and i.widget().isHidden())]

    def _arrange(self, rect: QRect, *, apply: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_height = area.x(), area.y(), 0
        for item in self._visible():
            hint = item.sizeHint()
            next_x = x + hint.width()
            if next_x > area.right() + 1 and line_height > 0:
                x, y = area.x(), y + line_height + self._spacing
                next_x, line_height = x + hint.width(), 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x + self._spacing
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + m.bottom()


def flow_row(*widgets: QWidget, spacing: int = 6) -> QWidget:
    """A wrapping row of widgets (e.g. a toolbar of buttons) as one widget."""
    row = QWidget()
    layout = FlowLayout(row, spacing=spacing)
    for w in widgets:
        layout.addWidget(w)
    row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    return row
