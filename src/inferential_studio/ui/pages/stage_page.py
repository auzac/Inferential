"""Common frame for a workflow stage page: header with goal, then the stage content."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from inferential_studio.ui.stages import STAGES, Stage
from inferential_studio.ui.widgets.stage_progress import StageProgress


class StagePage(QWidget):
    """Header strip (stage, title, goal; workflow progress on the right) above the content.

    Pages put their content in ``self.body``. Status, warnings and errors do not go into the
    page: they are reported to the message panel (``session.messages``), so the content never
    moves when a message appears.
    """

    def __init__(self, stage: Stage, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.stage = stage
        self.setObjectName(f"page_{stage.key}")

        header = QFrame()
        header.setObjectName("pageHeader")
        eyebrow = QLabel(f"Stage {stage.number} of {len(STAGES)}")
        eyebrow.setObjectName("pageEyebrow")
        title = QLabel(stage.title)
        title.setObjectName("pageTitle")
        goal = QLabel(stage.goal)
        goal.setObjectName("pageGoal")
        goal.setWordWrap(True)
        self.progress = StageProgress(stage.number - 1)
        self.progress_label = QLabel()
        self.progress_label.setObjectName("pageProgress")
        text = QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(eyebrow)
        text.addWidget(title)
        text.addWidget(goal)
        side = QVBoxLayout()
        side.setSpacing(4)
        side.addStretch(1)
        side.addWidget(self.progress_label, 0, Qt.AlignmentFlag.AlignRight)
        side.addWidget(self.progress)
        side.addStretch(1)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 10, 24, 10)
        hl.addLayout(text, 1)
        hl.addLayout(side)

        content = QWidget()
        self.body = QVBoxLayout(content)
        self.body.setContentsMargins(16, 12, 16, 12)
        self.body.setSpacing(10)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(content, 1)

    def set_progress(self, states: list[tuple[str, str]]) -> None:
        self.progress.set_states(states)
        self.progress_label.setText(f"{self.progress.done_count()} of {len(STAGES)} done")


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
