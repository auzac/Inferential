"""Common frame for a workflow stage page: header with goal, then the stage content."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from inferential_studio.ui.stages import Stage


class StagePage(QWidget):
    def __init__(self, stage: Stage, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.stage = stage
        self.setObjectName(f"page_{stage.key}")

        title = QLabel(f"Stage {stage.number} — {stage.title}")
        title.setObjectName("pageTitle")
        goal = QLabel(stage.goal)
        goal.setObjectName("pageGoal")
        goal.setWordWrap(True)

        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFrameShadow(QFrame.Shadow.Sunken)

        self.body = QVBoxLayout()
        self.body.setSpacing(12)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.addWidget(title)
        layout.addWidget(goal)
        layout.addWidget(rule)
        layout.addLayout(self.body, 1)


class PlaceholderPage(StagePage):
    """Stage that is not implemented yet. Says so plainly instead of showing mock controls."""

    def __init__(self, stage: Stage, parent: QWidget | None = None) -> None:
        super().__init__(stage, parent)
        note = QLabel(
            f"This stage is delivered in Milestone {stage.milestone}. "
            "No functionality is available here yet."
        )
        note.setObjectName("placeholderNote")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.body.addWidget(note)
        self.body.addStretch(1)
