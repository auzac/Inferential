"""Fixed-size readout of the trend values under the mouse cursor.

The text lives inside a scroll area and the box has a fixed height, so long
values or many checked variables never change its size hint: the surrounding
layout (and therefore the charts) stays put no matter what is hovered.

Clicking a trend pins the readout ("Selected point") until the next click;
while pinned, hover updates are ignored.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

PREFERRED_HEIGHT = 220
MIN_HEIGHT = 110


class CursorReadout(QGroupBox):
    """The "At cursor" box shown below the variable selection on the trend pages."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("At cursor", parent)
        self._locked = False
        self.time_label = QLabel("Move the mouse over a trend to read its values here.")
        self.time_label.setWordWrap(True)
        self.values_label = QLabel(" ")
        self.values_label.setWordWrap(True)
        self.values_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.values_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(self.values_label)
        body_layout.addStretch(1)
        self._scroll = QScrollArea(self)
        self._scroll.setWidget(body)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.addWidget(self.time_label)
        layout.addWidget(self._scroll, 1)

        # Static box: content grows inside the scroll area, never the group. It is 220 px tall
        # when the page has room and gives up to 90 px to the variable list on short screens.
        self.setMinimumHeight(MIN_HEIGHT)
        self.setMaximumHeight(PREFERRED_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(super().sizeHint().width(), PREFERRED_HEIGHT)  # never depends on content

    def show_hover(self, time_text: str, values: dict[str, str]) -> None:
        """Display the values of all series at the hovered time (ignored while pinned)."""
        if self._locked:
            return
        self._show(time_text, values)

    def show_point(self, time_text: str, values: dict[str, str]) -> None:
        """Pin the readout to a clicked point in time until :meth:`unlock`."""
        self._locked = True
        self.setTitle("Selected point")
        self._show(time_text, values)

    def unlock(self) -> None:
        self._locked = False
        self.setTitle("At cursor")

    def _show(self, time_text: str, values: dict[str, str]) -> None:
        self.time_label.setText(f"<b>{time_text}</b>")
        if values:
            self.values_label.setText(
                "<br>".join(f"{name} = {text}" for name, text in values.items())
            )
        else:
            self.values_label.setText("(no values at this time)")
