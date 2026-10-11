"""Stage 1 page: imported sources, their validation reports, and example data."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.core.dataset import Issue, Severity
from inferential_studio.core.importers import format_duration, lab_coverage_issue
from inferential_studio.core.schemas import SourceKind, VariableRole
from inferential_studio.ui.pages.stage_page import StagePage
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.stages import Stage

SEVERITY_STYLE = {
    Severity.ERROR: ("Error", QColor("#b00020")),
    Severity.WARNING: ("Warning", QColor("#9a6700")),
    Severity.INFO: ("Info", QColor("#3a4553")),
}


class ImportPage(StagePage):
    import_requested = Signal(object)  # SourceKind
    remove_requested = Signal(str)  # source id
    generate_example_requested = Signal()
    open_project_requested = Signal()

    def __init__(
        self, stage: Stage, session: ProjectSession, parent: QWidget | None = None
    ) -> None:
        super().__init__(stage, parent)
        self._session = session

        buttons = QHBoxLayout()
        self.import_process = QPushButton("Import process data…")
        self.import_process.clicked.connect(lambda: self.import_requested.emit(SourceKind.PROCESS))
        self.import_lab = QPushButton("Import lab / analyzer data…")
        self.import_lab.clicked.connect(lambda: self.import_requested.emit(SourceKind.LAB))
        self.remove_button = QPushButton("Remove source")
        self.remove_button.setToolTip(
            "Remove the selected source and its variables from the project (the file is not touched)"
        )
        self.remove_button.clicked.connect(self._remove)
        for b in (self.import_process, self.import_lab, self.remove_button):
            buttons.addWidget(b)
        buttons.addStretch(1)

        self.sources = QTableWidget(0, 7)
        self.sources.setHorizontalHeaderLabels(
            ["Source", "Type", "File", "Rows", "From", "To", "Status"]
        )
        self.sources.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.sources.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.sources.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.sources.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.sources.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sources.verticalHeader().setVisible(False)
        self.sources.itemSelectionChanged.connect(self._show_issues)

        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.issues = QListWidget()
        self.issues.setWordWrap(True)
        report_box = QGroupBox("Validation report")
        rl = QVBoxLayout(report_box)
        rl.addWidget(self.details)
        rl.addWidget(self.issues, 1)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.sources)
        split.addWidget(report_box)
        split.setSizes([200, 300])

        example_box = QGroupBox("Example data")
        intro = QLabel(
            "Generate an artificial distillation-column dataset (1-minute process data plus "
            "2-hourly lab samples) with a matching sample project. All values are synthetic."
        )
        intro.setWordWrap(True)
        self.generate_button = QPushButton("Generate example dataset…")
        self.generate_button.clicked.connect(self.generate_example_requested)
        self.open_button = QPushButton("Open project…")
        self.open_button.clicked.connect(self.open_project_requested)
        eb = QHBoxLayout()
        eb.addWidget(intro, 1)
        eb.addWidget(self.generate_button)
        eb.addWidget(self.open_button)
        example_box.setLayout(eb)

        self.body.addLayout(buttons)
        self.body.addWidget(split, 1)
        self.body.addWidget(example_box)

        session.changed.connect(self.refresh)
        session.data_changed.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------ display

    def _tz(self) -> ZoneInfo:
        return ZoneInfo(self._session.project.display_timezone)

    def _fmt(self, ts: pd.Timestamp | None) -> str:
        return "" if ts is None else ts.tz_convert(self._tz()).strftime("%Y-%m-%d %H:%M")

    @Slot()
    def refresh(self) -> None:
        project, data = self._session.project, self._session.data
        selected = self.selected_source_id()
        self.sources.setRowCount(len(project.sources))
        for row, src in enumerate(project.sources):
            loaded = data.sources.get(src.source_id)
            failure = data.failures.get(src.source_id)
            rep = loaded.report if loaded else None
            if failure is not None:
                status = "Not loaded"
            elif rep is None:
                status = "Loading…"
            else:
                n_err = sum(i.severity is Severity.ERROR for i in rep.issues)
                n_warn = sum(i.severity is Severity.WARNING for i in rep.issues)
                status = (
                    f"{n_err} error(s)" if n_err else (f"{n_warn} warning(s)" if n_warn else "OK")
                )
            cells = [
                src.source_id,
                "Lab" if src.kind is SourceKind.LAB else "Process",
                src.path,
                f"{rep.n_rows:,}" if rep else "",
                self._fmt(rep.time_start) if rep else "",
                self._fmt(rep.time_end) if rep else "",
                status,
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, src.source_id)
                if col == 6 and (failure is not None or (rep and rep.has_errors)):
                    item.setForeground(SEVERITY_STYLE[Severity.ERROR][1])
                self.sources.setItem(row, col, item)
        self.remove_button.setEnabled(bool(project.sources))
        if project.sources:
            ids = [s.source_id for s in project.sources]
            self.sources.selectRow(ids.index(selected) if selected in ids else 0)
        self._show_issues()

    def selected_source_id(self) -> str | None:
        rows = self.sources.selectionModel().selectedRows() if self.sources.selectionModel() else []
        if not rows:
            return None
        item = self.sources.item(rows[0].row(), 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    @Slot()
    def _show_issues(self) -> None:
        self.issues.clear()
        sid = self.selected_source_id()
        if sid is None:
            self.details.setText(
                "No data imported yet. Use the buttons above, or generate the example dataset."
            )
            return
        project, data = self._session.project, self._session.data
        src = project.source(sid)
        failure = data.failures.get(sid)
        loaded = data.sources.get(sid)
        variables = [v for v in project.variables if v.source_id == sid]
        roles = {r: sum(v.role is r for v in variables) for r in VariableRole}
        ts = src.timestamp
        lines = [
            f"<b>{src.path}</b>",
            f"Timestamp column '{ts.column}' ({ts.kind.value}{', ' + ts.format if ts.format else ''}), "
            f"source timezone {ts.source_timezone}"
            + (f"; result time '{src.result_timestamp.column}'" if src.result_timestamp else ""),
            f"Variables: {len(variables)} ({roles[VariableRole.PREDICTOR]} candidate predictors, "
            f"{roles[VariableRole.TARGET]} target, {roles[VariableRole.UNUSED]} not used)",
        ]
        if loaded:
            rep = loaded.report
            interval = (
                f"typical interval {format_duration(rep.median_interval)}"
                if rep.median_interval is not None
                else ""
            )
            lines.append(
                f"{rep.n_valid_times:,} of {rep.n_rows:,} rows have a valid time; {interval}"
            )
            lines.append(f"SHA-256 {loaded.sha256[:16]}…")
        self.details.setText("<br>".join(lines))

        issues: list[Issue] = []
        if failure is not None:
            issues.append(Issue(Severity.ERROR, "load_failed", str(failure)))
        if loaded:
            issues.extend(loaded.report.issues)
            if src.kind is SourceKind.LAB:
                for proc in data.of_kind(SourceKind.PROCESS):
                    issue = lab_coverage_issue(proc, loaded)
                    if issue:
                        issues.append(issue)
        if loaded and not issues:
            issues.append(Issue(Severity.INFO, "ok", "No problems found."))
        for issue in issues:
            label, colour = SEVERITY_STYLE[issue.severity]
            item = QListWidgetItem(f"{label}: {issue.message}")
            item.setForeground(colour)
            self.issues.addItem(item)

    def _remove(self) -> None:
        sid = self.selected_source_id()
        if sid:
            self.remove_requested.emit(sid)
