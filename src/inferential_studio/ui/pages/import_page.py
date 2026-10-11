"""Stage 1 page. Milestone 0: project overview and example-data generation.

The CSV/Excel import wizard arrives in Milestone 1.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.core.schemas import Project
from inferential_studio.ui.pages.stage_page import StagePage
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.stages import Stage


class ImportPage(StagePage):
    generate_example_requested = Signal()
    open_project_requested = Signal()

    def __init__(
        self, stage: Stage, session: ProjectSession, parent: QWidget | None = None
    ) -> None:
        super().__init__(stage, parent)
        self._session = session

        # -- example data ------------------------------------------------------
        example_box = QGroupBox("Example data")
        intro = QLabel(
            "Generate an artificial distillation-column dataset (1-minute process data plus "
            "2-hourly lab samples) with a matching sample project. Useful for learning the "
            "workflow. All values are synthetic."
        )
        intro.setWordWrap(True)
        self.generate_button = QPushButton("Generate example dataset…")
        self.generate_button.clicked.connect(self.generate_example_requested)
        self.open_button = QPushButton("Open project…")
        self.open_button.clicked.connect(self.open_project_requested)
        buttons = QHBoxLayout()
        buttons.addWidget(self.generate_button)
        buttons.addWidget(self.open_button)
        buttons.addStretch(1)
        box_layout = QVBoxLayout(example_box)
        box_layout.addWidget(intro)
        box_layout.addLayout(buttons)

        # -- project overview ------------------------------------------------------
        overview_box = QGroupBox("Project overview")
        self.overview = QTreeWidget()
        self.overview.setHeaderLabels(["Item", "Details"])
        self.overview.setColumnWidth(0, 260)
        QVBoxLayout(overview_box).addWidget(self.overview)

        pending = QLabel("The CSV/Excel import wizard is delivered in Milestone 1.")
        pending.setObjectName("placeholderNote")

        self.body.addWidget(example_box)
        self.body.addWidget(overview_box, 1)
        self.body.addWidget(pending)

        session.changed.connect(self.refresh)
        self.refresh()

    @Slot()
    def refresh(self) -> None:
        self.overview.clear()
        project: Project = self._session.project

        def section(title: str, count: int) -> QTreeWidgetItem:
            item = QTreeWidgetItem([title, str(count)])
            self.overview.addTopLevelItem(item)
            return item

        top = QTreeWidgetItem(["Project", project.name])
        self.overview.addTopLevelItem(top)
        QTreeWidgetItem(top, ["Display timezone", project.display_timezone])

        sources = section("Data sources", len(project.sources))
        for s in project.sources:
            QTreeWidgetItem(sources, [s.source_id, f"{s.kind.value}: {s.path}"])

        variables = section("Variables", len(project.variables))
        for v in project.variables:
            units = f" [{v.units}]" if v.units else ""
            QTreeWidgetItem(variables, [v.name + units, v.role.value])

        features = section("Predictor features", len(project.features))
        for f in project.features:
            start, end = f.window_offsets_s
            causal = "causal" if f.is_causal else "NON-CAUSAL"
            QTreeWidgetItem(
                features,
                [
                    f.variable,
                    f"{f.aggregation.value} over ({start / 60:g}, {end / 60:g}] min, {causal}",
                ],
            )

        tz = ZoneInfo(project.display_timezone)
        exclusions = section("Manual exclusions", len(project.exclusions))
        for ex in project.exclusions:
            start, end = ex.start.astimezone(tz), ex.end.astimezone(tz)
            QTreeWidgetItem(
                exclusions,
                [
                    ex.reason.value,
                    f"{start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M} ({project.display_timezone})",
                ],
            )

        self.overview.expandToDepth(0)
