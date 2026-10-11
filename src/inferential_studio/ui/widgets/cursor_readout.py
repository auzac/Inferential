"""Fixed-size readout of the trend values under the mouse cursor.

The text lives inside a scroll area and the box has a fixed height, so long
values or many checked variables never change its size hint: the surrounding
layout (and therefore the charts) stays put no matter what is hovered.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class CursorReadout(QGroupBox):
    """"At cursor" box shown below the variable selection on the trend pages."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("At cursor", parent)
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

        # Static box: content grows inside the scroll area, never the group.
        self.setFixedHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def show_hover(self, time_text: str, values: dict[str, str]) -> None:
        """Display the values of all series at the hovered time."""
        self.time_label.setText(f"<b>{time_text}</b>")
        if values:
            self.values_label.setText(
                "<br>".join(f"{name} = {text}" for name, text in values.items())
            )
        else:
            self.values_label.setText("(no values at this time)")
