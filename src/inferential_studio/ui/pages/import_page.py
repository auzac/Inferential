"""Stage 1 page: imported sources, their validation reports, and example data."""

from __future__ import annotations

from html import escape
from zoneinfo import ZoneInfo

import pandas as pd
from PySide6.QtCore import Qt, Signal, Slot
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
from inferential_studio.ui.messages import Level
from inferential_studio.ui.pages.stage_page import StagePage
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.stages import Stage
from inferential_studio.ui.theme import TOKENS, mark_primary, theme_colour
from inferential_studio.ui.widgets.message_panel import level_icon, ok_icon

SEVERITY_STYLE = {
    Severity.ERROR: ("Error", "error", Level.ERROR),
    Severity.WARNING: ("Warning", "warning", Level.WARNING),
    Severity.INFO: ("Info", "text", Level.INFO),
}


class ImportPage(StagePage):
    import_requested = Signal()
    remove_requested = Signal(str)  # source id
    generate_example_requested = Signal()
    open_project_requested = Signal()

    def __init__(
        self, stage: Stage, session: ProjectSession, parent: QWidget | None = None
    ) -> None:
        super().__init__(stage, parent)
        self._session = session

        buttons = QHBoxLayout()
        self.import_button = QPushButton("Import data…")
        mark_primary(self.import_button)
        self.import_button.setToolTip(
            "Import data from CSV or Excel — process tags, lab / analyzer results, or both in "
            "one file. The wizard works out the source type from your choices."
        )
        self.import_button.clicked.connect(lambda: self.import_requested.emit())
        self.remove_button = QPushButton("Remove source")
        self.remove_button.setToolTip(
            "Remove the selected source and its variables from the project (the file is not touched)"
        )
        self.remove_button.clicked.connect(self._remove)
        for b in (self.import_button, self.remove_button):
            buttons.addWidget(b)
        buttons.addStretch(1)

        self.sources = QTableWidget(0, 7)
        self.sources.setHorizontalHeaderLabels(
            ["Source", "Type", "File", "Rows", "From", "To", "Status"]
        )
        header = self.sources.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # File paths are the long content here, so that column takes the spare width...
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        # ...and its header stays over the paths instead of floating mid-column.
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.sources.horizontalHeaderItem(3).setTextAlignment(  # Rows are right-aligned
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.sources.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.sources.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.sources.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sources.verticalHeader().setVisible(False)
        self.sources.setAlternatingRowColors(True)
        self.sources.setShowGrid(False)
        self.sources.itemSelectionChanged.connect(self._show_issues)

        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.issues = QListWidget()
        self.issues.setWordWrap(True)
        self._icons = {lvl: level_icon(lvl) for lvl in Level}
        self._ok_icon = ok_icon()
        report_box = QGroupBox("Validation report")
        rl = QVBoxLayout(report_box)
        rl.addWidget(self.details)
        rl.addWidget(self.issues, 1)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.sources)
        split.addWidget(report_box)
        split.setSizes([170, 380])

        self.example_box = example_box = QGroupBox("Example data")
        intro = QLabel(
            "Generate an artificial distillation-column dataset (1-minute process data plus "
            "2-hourly lab samples) with a matching sample project. All values are synthetic. "
            "Also available later from the menu (≡) → Help → Generate example dataset."
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
                if col == 3:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if col == 6:
                    if failure is not None or (rep and rep.has_errors):
                        item.setIcon(self._icons[Level.ERROR])
                        item.setForeground(theme_colour("error"))
                    elif rep is not None and status != "OK":
                        item.setIcon(self._icons[Level.WARNING])
                    elif rep is not None:
                        item.setIcon(self._ok_icon)
                self.sources.setItem(row, col, item)
        self.remove_button.setEnabled(bool(project.sources))
        # Getting-started box only for an empty project: once data is in, the sources and
        # their validation report are the content of this page.
        self.example_box.setVisible(not project.sources)
        if project.sources:
            ids = [s.source_id for s in project.sources]
            self.sources.selectRow(ids.index(selected) if selected in ids else 0)
        self._show_issues()

    def apply_theme(self) -> None:
        self._icons = {lvl: level_icon(lvl) for lvl in Level}
        self._ok_icon = ok_icon()
        self.refresh()

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
                "No data imported yet. Use Import data… above, or generate the example dataset below."
            )
            return
        project, data = self._session.project, self._session.data
        src = project.source(sid)
        failure = data.failures.get(sid)
        loaded = data.sources.get(sid)
        variables = [v for v in project.variables if v.source_id == sid]
        roles = {r: sum(v.role is r for v in variables) for r in VariableRole}
        ts = src.timestamp
        rows = [
            ("File", f"<b>{escape(src.path)}</b>"),
            ("Type", "Lab / analyzer results" if src.kind is SourceKind.LAB else "Process data"),
            (
                "Timestamp",
                f"column '{escape(ts.column)}' ({ts.kind.value}"
                f"{', ' + escape(ts.format) if ts.format else ''}), "
                f"source timezone {ts.source_timezone}",
            ),
            (
                "Variables",
                f"{len(variables)}: {roles[VariableRole.PREDICTOR]} candidate predictors, "
                f"{roles[VariableRole.TARGET]} target, {roles[VariableRole.UNUSED]} not used",
            ),
        ]
        if src.sample_id_column:
            rows.append(("Sample ID", f"column '{escape(src.sample_id_column)}'"))
        if loaded:
            rep = loaded.report
            interval = (
                f"; typical interval {format_duration(rep.median_interval)}"
                if rep.median_interval is not None
                else ""
            )
            rows.append(
                ("Rows", f"{rep.n_valid_times:,} of {rep.n_rows:,} with a valid time{interval}")
            )
            rows.append(("Checksum", f"SHA-256 {loaded.sha256[:16]}…"))
        cells = "".join(
            f"<tr><td style='color:{TOKENS['text_muted']}; padding:1px 16px 1px 0'>{k}</td>"
            f"<td style='padding:1px 0'>{v}</td></tr>"
            for k, v in rows
        )
        self.details.setText(f"<table cellspacing='0'>{cells}</table>")

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
            label, token, level = SEVERITY_STYLE[issue.severity]
            if issue.code == "ok":
                item = QListWidgetItem(self._ok_icon, issue.message)
            else:
                item = QListWidgetItem(self._icons[level], f"{label}: {issue.message}")
                item.setForeground(theme_colour(token))
            self.issues.addItem(item)

    def _remove(self) -> None:
        sid = self.selected_source_id()
        if sid:
            self.remove_requested.emit(sid)
