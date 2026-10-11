"""Import wizard: file → read options → timestamps → columns → duplicates.

The wizard only *configures* an import and returns an :class:`ImportPlan`. Reading the whole
file happens afterwards in a background task. Timestamp interpretation is never guessed: when
the data allow both day-first and month-first readings the user must pick one before Next is
enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QGuiApplication
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from inferential_studio.core.dataset import DataImportError
from inferential_studio.core.importers import (
    LARGE_FILE_BYTES,
    ReadOptions,
    excel_date_system,
    file_kind,
    list_sheets,
    read_column,
    read_header,
    read_preview,
    sniff_csv,
)
from inferential_studio.core.schemas import (
    DataSource,
    DuplicatePolicy,
    Project,
    SchemaError,
    SourceKind,
    TimestampKind,
    TimestampSpec,
    VariableMeta,
    VariableRole,
)
from inferential_studio.core.timestamps import (
    DateOrder,
    FormatCandidate,
    FormatDetection,
    TimestampError,
    detect_format,
    parse_timestamps,
    spec_from_candidate,
)
from inferential_studio.ui.widgets.data_table import DataTableView
from inferential_studio.ui.widgets.timezone_combo import TimezoneCombo

DELIMITERS = [("Comma ( , )", ","), ("Semicolon ( ; )", ";"), ("Tab", "\t"), ("Pipe ( | )", "|")]
ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "utf-16"]
NONE_ITEM = "(none)"


@dataclass
class ImportPlan:
    source: DataSource
    variables: list[VariableMeta] = field(default_factory=list)


class _State:
    """Values shared between wizard pages."""

    def __init__(self, project: Project, default_timezone: str) -> None:
        self.project = project
        self.default_timezone = default_timezone
        self.kind = SourceKind.PROCESS
        self.path: Path | None = None
        self.opts = ReadOptions()
        self.names: list[str] = []
        self.preview = pd.DataFrame()
        self.excel_system = 1900
        self.time_spec: TimestampSpec | None = None
        self.result_spec: TimestampSpec | None = None
        self.sample_id_column: str | None = None


# --------------------------------------------------------------------------- page 1: file


class FilePage(QWizardPage):
    def __init__(self, state: _State) -> None:
        super().__init__()
        self.s = state
        self.setTitle("Choose a file")
        self.setSubTitle(
            "CSV/TSV text files or Excel workbooks (.xlsx). The file is never modified."
        )
        self.process_radio = QRadioButton(
            "Process data — historian export, one timestamp column and many tags"
        )
        self.lab_radio = QRadioButton("Lab / analyzer results — sparse samples of the target")
        self.process_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.process_radio)
        group.addButton(self.lab_radio)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Browse or drag a file onto this window")
        self.path_edit.textChanged.connect(self.completeChanged)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.message = QLabel()
        self.message.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(self.path_edit, 1)
        row.addWidget(browse)
        layout = QVBoxLayout(self)
        layout.addWidget(self.process_radio)
        layout.addWidget(self.lab_radio)
        layout.addSpacing(12)
        layout.addLayout(row)
        layout.addWidget(self.message)
        layout.addStretch(1)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose data file", str(self.s.path.parent if self.s.path else ""),
            "Data files (*.csv *.tsv *.txt *.xlsx *.xlsm);;All files (*)",
        )  # fmt: skip
        if path:
            self.path_edit.setText(path)

    def isComplete(self) -> bool:  # noqa: N802
        text = self.path_edit.text().strip()
        return bool(text) and Path(text).is_file()

    def validatePage(self) -> bool:  # noqa: N802
        path = Path(self.path_edit.text().strip())
        try:
            kind = file_kind(path)
            if kind == "csv":
                opts = sniff_csv(path)
            else:
                sheets = list_sheets(path)
                opts = ReadOptions(sheet=sheets[0] if sheets else None)
                self.s.excel_system = excel_date_system(path)
        except (DataImportError, OSError) as exc:
            self.message.setText(f"<span style='color:#b00020'>{exc}</span>")
            return False
        if self.s.path != path:  # keep options when coming back to the same file
            self.s.opts = opts
        self.s.path = path
        self.s.kind = SourceKind.LAB if self.lab_radio.isChecked() else SourceKind.PROCESS
        if path.stat().st_size > LARGE_FILE_BYTES:
            self.message.setText(
                "This is a very large file; the full import may take several minutes."
            )
        return True


# --------------------------------------------------------------------------- page 2: read options


class ReadPage(QWizardPage):
    def __init__(self, state: _State) -> None:
        super().__init__()
        self.s = state
        self.setTitle("How should the file be read?")
        self.setSubTitle(
            "Check the preview: each column should hold one field and the header must be the column names."
        )
        self._ok = False

        self.delimiter = QComboBox()
        for label, value in DELIMITERS:
            self.delimiter.addItem(label, value)
        self.encoding = QComboBox()
        self.encoding.setEditable(True)
        self.encoding.addItems(ENCODINGS)
        self.decimal = QComboBox()
        self.decimal.addItem("Point ( 1.5 )", ".")
        self.decimal.addItem("Comma ( 1,5 )", ",")
        self.sheet = QComboBox()
        self.header = QSpinBox()
        self.header.setRange(1, 1000)
        self.header.setToolTip(
            "Line (CSV) or row (Excel) that holds the column names; earlier lines are skipped."
        )

        self.csv_box = QGroupBox("Text file options")
        form = QFormLayout(self.csv_box)
        form.addRow("Delimiter:", self.delimiter)
        form.addRow("Encoding:", self.encoding)
        form.addRow("Decimal separator:", self.decimal)
        self.excel_box = QGroupBox("Workbook options")
        QFormLayout(self.excel_box).addRow("Worksheet:", self.sheet)
        header_row = QFormLayout()
        header_row.addRow("Column names are on line/row:", self.header)

        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color:#b00020")
        self.table = DataTableView(sortable=False)

        top = QHBoxLayout()
        top.addWidget(self.csv_box)
        top.addWidget(self.excel_box)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(header_row)
        layout.addWidget(self.error)
        layout.addWidget(self.table, 1)

        for w in (self.delimiter, self.decimal, self.sheet):
            w.currentIndexChanged.connect(self._refresh)
        self.encoding.currentTextChanged.connect(self._refresh)
        self.header.valueChanged.connect(self._refresh)
        self._loading = False

    def initializePage(self) -> None:  # noqa: N802
        self._loading = True
        is_csv = file_kind(self.s.path) == "csv"
        self.csv_box.setVisible(is_csv)
        self.excel_box.setVisible(not is_csv)
        o = self.s.opts
        if is_csv:
            idx = self.delimiter.findData(o.delimiter or ",")
            self.delimiter.setCurrentIndex(max(idx, 0))
            self.encoding.setCurrentText(o.encoding)
            self.decimal.setCurrentIndex(self.decimal.findData(o.decimal))
        else:
            self.sheet.clear()
            self.sheet.addItems(list_sheets(self.s.path))
            if o.sheet:
                self.sheet.setCurrentText(o.sheet)
        self.header.setValue(o.header_row + 1)
        self._loading = False
        self._refresh()

    def _options(self) -> ReadOptions:
        if file_kind(self.s.path) == "csv":
            return ReadOptions(
                delimiter=self.delimiter.currentData(),
                encoding=self.encoding.currentText().strip() or "utf-8",
                decimal=self.decimal.currentData(),
                header_row=self.header.value() - 1,
            )
        return ReadOptions(
            sheet=self.sheet.currentText() or None, header_row=self.header.value() - 1
        )

    def _refresh(self) -> None:
        if self._loading or self.s.path is None:
            return
        opts = self._options()
        try:
            preview = read_preview(self.s.path, opts)
            names = read_header(self.s.path, opts)
        except (DataImportError, LookupError, ValueError, OSError) as exc:
            self._ok = False
            self.error.setText(str(exc))
            self.table.set_frame(pd.DataFrame())
        else:
            self._ok = True
            self.error.setText(
                "" if len(names) > 1 else "Only one column found — check the delimiter."
            )
            self.s.opts, self.s.names, self.s.preview = opts, names, preview
            self.table.set_frame(preview.head(100).astype(str).replace({"nan": "", "NaT": ""}))
        self.completeChanged.emit()

    def isComplete(self) -> bool:  # noqa: N802
        return self._ok and len(self.s.names) > 1


# --------------------------------------------------------------------------- timestamp chooser


class TimestampChooser(QGroupBox):
    """Column + format + (shared) timezone selection with a live parse preview."""

    changed = Signal()

    def __init__(self, title: str, state: _State, *, optional: bool = False) -> None:
        super().__init__(title)
        self.s = state
        self.optional = optional
        self.detection: FormatDetection | None = None
        self.timezone: TimezoneCombo | None = None
        self.column = QComboBox()
        self.format = QComboBox()
        self.format.setMinimumContentsLength(30)
        self.note = QLabel()
        self.note.setWordWrap(True)
        form = QFormLayout(self)
        form.addRow("Column:", self.column)
        form.addRow("Format:", self.format)
        form.addRow(self.note)
        self.column.currentIndexChanged.connect(self._detect)
        self.format.currentIndexChanged.connect(lambda _i: self.changed.emit())

    def populate(self, names: list[str], preferred: str | None) -> None:
        self.column.blockSignals(True)
        self.column.clear()
        if self.optional:
            self.column.addItem(NONE_ITEM)
        self.column.addItems(names)
        if preferred and preferred in names:
            self.column.setCurrentText(preferred)
        self.column.blockSignals(False)
        self._detect()

    @property
    def selected_column(self) -> str | None:
        text = self.column.currentText()
        return None if not text or text == NONE_ITEM else text

    def candidate(self) -> FormatCandidate | None:
        data = self.format.currentData()
        return data if isinstance(data, FormatCandidate) else None

    def _detect(self) -> None:
        self.format.blockSignals(True)
        self.format.clear()
        self.detection = None
        col = self.selected_column
        if col is None:
            self.note.setText("")
            self.format.setEnabled(False)
            self.format.blockSignals(False)
            self.changed.emit()
            return
        self.format.setEnabled(True)
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            values = read_column(self.s.path, self.s.opts, col)
            self.detection = detect_format(values)
        except (DataImportError, ValueError, OSError) as exc:
            self.note.setText(f"<span style='color:#b00020'>{exc}</span>")
            self.format.blockSignals(False)
            self.changed.emit()
            return
        finally:
            QGuiApplication.restoreOverrideCursor()

        d = self.detection
        if d.recommended is None and d.candidates:
            self.format.addItem("— choose the format —", None)
        for c in d.candidates:
            label = c.description
            if c.example_input:
                label += f"    e.g. {c.example_input} → {c.example_parsed}"
            self.format.addItem(label, c)
        if d.ambiguous:
            self.note.setText(
                "<b style='color:#b00020'>Day/month order is ambiguous.</b> " + d.message
            )
        elif d.message:
            self.note.setText(f"<span style='color:#b00020'>{d.message}</span>")
        elif d.recommended and d.recommended.order in (DateOrder.DAY_FIRST, DateOrder.MONTH_FIRST):
            order = "day-first" if d.recommended.order is DateOrder.DAY_FIRST else "month-first"
            self.note.setText(
                f"The order is {order}: some values in the file can only be read that way. "
                "Check the preview below."
            )
        elif d.has_offsets:
            self.note.setText(
                "Values carry UTC offsets, so the source timezone below is not needed for this column."
            )
        elif d.recommended and d.recommended.kind in (
            TimestampKind.UNIX_SECONDS,
            TimestampKind.UNIX_MILLISECONDS,
        ):
            self.note.setText(
                "Unix timestamps are absolute (UTC); the source timezone does not apply."
            )
        else:
            self.note.setText("")
        self.format.blockSignals(False)
        self.changed.emit()

    def select_format(self, fmt: str | None, kind: TimestampKind) -> None:
        for i in range(self.format.count()):
            c = self.format.itemData(i)
            if isinstance(c, FormatCandidate) and c.format == fmt and c.kind is kind:
                self.format.setCurrentIndex(i)
                return

    def spec(self, timezone: str) -> TimestampSpec | None:
        col, cand = self.selected_column, self.candidate()
        if col is None or cand is None:
            return None
        return spec_from_candidate(col, cand, timezone, excel_date_system=self.s.excel_system)

    def is_complete(self) -> bool:
        if self.selected_column is None:
            return self.optional
        return self.candidate() is not None


# --------------------------------------------------------------------------- page 3: timestamps


class TimePage(QWizardPage):
    def __init__(self, state: _State) -> None:
        super().__init__()
        self.s = state
        self.setTitle("Timestamps")
        self.setSubTitle(
            "Confirm how times are written and which timezone the source system used. "
            "All data is stored in UTC."
        )
        self.time = TimestampChooser("Timestamp", state)
        self.result = TimestampChooser("Result available time (optional)", state, optional=True)
        self.result.setToolTip(
            "When the lab result was reported. Recorded for audit; models align to the sample time."
        )
        self.sample_id = QComboBox()
        self.tz = TimezoneCombo(state.default_timezone)
        self.tz.setToolTip("Timezone of values without an explicit UTC offset (plant local time).")
        self.preview = DataTableView(sortable=False)
        self.status = QLabel()
        self.status.setWordWrap(True)

        tz_form = QFormLayout()
        tz_form.addRow("Source timezone:", self.tz)
        self.lab_box = QGroupBox("Lab sample identification")
        lab_form = QFormLayout(self.lab_box)
        lab_form.addRow("Sample ID column (optional):", self.sample_id)

        top = QHBoxLayout()
        top.addWidget(self.time, 1)
        top.addWidget(self.result, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.lab_box)
        layout.addLayout(tz_form)
        layout.addWidget(self.status)
        layout.addWidget(self.preview, 1)

        self.time.changed.connect(self._update)
        self.result.changed.connect(self._update)
        self.tz.currentTextChanged.connect(self._update)

    def initializePage(self) -> None:  # noqa: N802
        is_lab = self.s.kind is SourceKind.LAB
        self.time.setTitle("Sample (collection) time" if is_lab else "Timestamp")
        self.result.setVisible(is_lab)
        self.lab_box.setVisible(is_lab)
        names = self.s.names
        guess = next(
            (n for n in names if any(k in n.lower() for k in ("time", "date", "zeit"))), names[0]
        )
        prev = self.s.time_spec.column if self.s.time_spec else guess
        self.time.populate(names, prev)
        if self.s.time_spec:
            self.time.select_format(self.s.time_spec.format, self.s.time_spec.kind)
            self.tz.setCurrentText(self.s.time_spec.source_timezone)
        if is_lab:
            self.result.populate(names, self.s.result_spec.column if self.s.result_spec else None)
            if self.s.result_spec:
                self.result.select_format(self.s.result_spec.format, self.s.result_spec.kind)
            self.sample_id.clear()
            self.sample_id.addItem(NONE_ITEM)
            self.sample_id.addItems(names)
            if self.s.sample_id_column:
                self.sample_id.setCurrentText(self.s.sample_id_column)
        self._update()

    def _update(self) -> None:
        self.status.setText("")
        spec = self.time.spec(self.tz.timezone()) if self.tz.is_valid() else None
        if not self.tz.is_valid():
            self.status.setText("<span style='color:#b00020'>Choose a valid timezone.</span>")
        if spec is None:
            self.preview.set_frame(pd.DataFrame())
            self.completeChanged.emit()
            return
        raw = self.s.preview[spec.column].head(50)
        try:
            parsed = parse_timestamps(raw, spec)
        except (TimestampError, SchemaError) as exc:
            self.status.setText(f"<span style='color:#b00020'>{exc}</span>")
            self.preview.set_frame(pd.DataFrame())
            self.completeChanged.emit()
            return
        local = parsed.utc.dt.tz_convert(self.tz.timezone())
        display_tz = self.s.project.display_timezone
        frame = pd.DataFrame(
            {
                "In file": raw.astype(str).to_numpy(),
                f"Local ({self.tz.timezone()})": local.dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                .fillna("INVALID")
                .to_numpy(),
                "UTC (stored)": parsed.utc.dt.strftime("%Y-%m-%d %H:%M:%S")
                .fillna("INVALID")
                .to_numpy(),
                f"Display ({display_tz})": parsed.utc.dt.tz_convert(display_tz)
                .dt.strftime("%Y-%m-%d %H:%M")
                .fillna("")
                .to_numpy(),
            }
        )
        self.preview.set_frame(frame)
        if parsed.n_invalid:
            self.status.setText(
                f"{parsed.n_invalid} of the first {len(raw)} values cannot be interpreted with this format."
            )
        self.completeChanged.emit()

    def isComplete(self) -> bool:  # noqa: N802
        if (
            not self.tz.is_valid()
            or not self.time.is_complete()
            or self.time.spec(self.tz.timezone()) is None
        ):
            return False
        return self.result.is_complete() if self.s.kind is SourceKind.LAB else True

    def validatePage(self) -> bool:  # noqa: N802
        tz = self.tz.timezone()
        self.s.time_spec = self.time.spec(tz)
        self.s.result_spec = self.result.spec(tz) if self.s.kind is SourceKind.LAB else None
        sid = self.sample_id.currentText()
        self.s.sample_id_column = (
            sid if self.s.kind is SourceKind.LAB and sid and sid != NONE_ITEM else None
        )
        if self.s.result_spec and self.s.result_spec.column == self.s.time_spec.column:
            self.status.setText(
                "<span style='color:#b00020'>Sample and result time must be different columns.</span>"
            )
            return False
        if self.s.time_spec is None:
            return False
        time_columns = {self.s.time_spec.column, getattr(self.s.result_spec, "column", None)}
        if self.s.sample_id_column is not None and self.s.sample_id_column in time_columns:
            self.status.setText(
                "<span style='color:#b00020'>The sample ID column cannot also be a time column.</span>"
            )
            return False
        return True


# --------------------------------------------------------------------------- page 4: columns


ROLE_LABELS = {
    VariableRole.PREDICTOR: "Candidate predictor (X)",
    VariableRole.TARGET: "Target (Y)",
    VariableRole.UNUSED: "Not used",
}


class ColumnsPage(QWizardPage):
    def __init__(self, state: _State) -> None:
        super().__init__()
        self.s = state
        self.setTitle("Columns")
        self.setSubTitle(
            "Choose the columns to import, their names in the project and units. Roles can be changed later "
            "(Variable Selection)."
        )
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Import", "Column in file", "Variable name", "Role", "Units", "Numeric in preview"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.message = QLabel()
        self.message.setWordWrap(True)
        buttons = QHBoxLayout()
        for text, state_ in (("Select all", True), ("Select none", False)):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, v=state_: self._check_all(v))
            buttons.addWidget(b)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.message)
        self.table.itemChanged.connect(lambda _i: self.completeChanged.emit())
        self.previous: dict[str, VariableMeta] = {}

    def initializePage(self) -> None:  # noqa: N802
        self.table.blockSignals(True)
        s = self.s
        used = {s.time_spec.column, s.sample_id_column, getattr(s.result_spec, "column", None)}
        columns = [n for n in s.names if n not in used]
        existing = {v.name for v in s.project.variables}
        has_target = s.project.target is not None
        self.table.setRowCount(len(columns))
        for row, name in enumerate(columns):
            prev = self.previous.get(name)
            include = QTableWidgetItem()
            include.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            include.setCheckState(
                Qt.CheckState.Checked
                if (prev is not None or not self.previous)
                else Qt.CheckState.Unchecked
            )
            self.table.setItem(row, 0, include)
            col_item = QTableWidgetItem(name)
            col_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 1, col_item)
            var_name = (
                prev.name
                if prev
                else (name if name not in existing else f"{name} ({s.kind.value})")
            )
            self.table.setItem(row, 2, QTableWidgetItem(var_name))
            role = QComboBox()
            for r, label in ROLE_LABELS.items():
                role.addItem(label, r)
            if prev:
                default = prev.role
            elif s.kind is SourceKind.LAB:
                default = (
                    VariableRole.TARGET if (row == 0 and not has_target) else VariableRole.UNUSED
                )
            else:
                default = VariableRole.PREDICTOR
            role.setCurrentIndex(role.findData(default))
            role.currentIndexChanged.connect(lambda _i: self.completeChanged.emit())
            self.table.setCellWidget(row, 3, role)
            self.table.setItem(row, 4, QTableWidgetItem(prev.units or "" if prev else ""))
            numeric = pd.to_numeric(
                s.preview[name].astype(str).str.replace(",", ".", regex=False)
                if s.opts.decimal == ","
                else s.preview[name],
                errors="coerce",
            )
            share = numeric.notna().mean() if len(numeric) else 0.0
            pct = QTableWidgetItem(f"{share:.0%}")
            pct.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if share < 0.5:
                pct.setForeground(Qt.GlobalColor.darkRed)
            self.table.setItem(row, 5, pct)
        self.table.blockSignals(False)
        self.completeChanged.emit()

    def _check_all(self, checked: bool) -> None:
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )

    def variables(self, source_id: str) -> list[VariableMeta]:
        out = []
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() != Qt.CheckState.Checked:
                continue
            units = self.table.item(row, 4).text().strip() or None
            out.append(
                VariableMeta(
                    name=self.table.item(row, 2).text().strip(),
                    source_id=source_id,
                    source_column=self.table.item(row, 1).text(),
                    role=self.table.cellWidget(row, 3).currentData(),
                    units=units,
                )
            )
        return out

    def _problem(self) -> str:
        chosen = self.variables("_")
        if not chosen:
            return "Select at least one column to import."
        names = [v.name for v in chosen]
        if any(not n for n in names):
            return "Every imported column needs a variable name."
        if len(set(names)) != len(names):
            return "Variable names must be unique."
        clash = set(names) & {v.name for v in self.s.project.variables}
        if clash:
            return f"Already used in this project: {', '.join(sorted(clash))}. Rename them."
        targets = sum(v.role is VariableRole.TARGET for v in chosen)
        if targets > 1 or (targets and self.s.project.target is not None):
            return "A project has exactly one target (Y). Mark other values as 'Not used'."
        return ""

    def isComplete(self) -> bool:  # noqa: N802
        problem = self._problem()
        self.message.setText(f"<span style='color:#b00020'>{problem}</span>" if problem else "")
        return not problem


# --------------------------------------------------------------------------- page 5: duplicates


class FinishPage(QWizardPage):
    def __init__(self, state: _State) -> None:
        super().__init__()
        self.s = state
        self.setTitle("Duplicate timestamps and summary")
        self.policy = QComboBox()
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        form = QFormLayout()
        form.addRow("If several rows share a timestamp:", self.policy)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addSpacing(10)
        layout.addWidget(self.summary)
        layout.addStretch(1)
        self.setFinalPage(True)

    def initializePage(self) -> None:  # noqa: N802
        self.policy.clear()
        if self.s.kind is SourceKind.LAB:
            options = [
                (DuplicatePolicy.KEEP_ALL, "Keep all rows as separate samples (replicates)"),
                (DuplicatePolicy.KEEP_FIRST, "Keep the first row only"),
                (DuplicatePolicy.MEAN, "Average the values"),
            ]
        else:
            options = [
                (DuplicatePolicy.ERROR, "Stop and tell me (recommended until you know why)"),
                (DuplicatePolicy.KEEP_FIRST, "Keep the first row only"),
                (DuplicatePolicy.MEAN, "Average the values"),
            ]
        for policy, label in options:
            self.policy.addItem(label, policy)
        ts = self.s.time_spec
        self.summary.setText(
            f"<b>File:</b> {self.s.path}<br>"
            f"<b>Type:</b> {'Lab / analyzer results' if self.s.kind is SourceKind.LAB else 'Process data'}<br>"
            f"<b>Timestamp:</b> column '{ts.column}', {ts.kind.value}"
            f"{', format ' + ts.format if ts.format else ''}, source timezone {ts.source_timezone}<br>"
            "<br>The file is read in the background after you click Finish. A validation report "
            "(invalid times, gaps, non-numeric cells) is shown on the Data Import page."
        )

    @property
    def duplicate_policy(self) -> DuplicatePolicy:
        return self.policy.currentData()


# --------------------------------------------------------------------------- wizard


class ImportWizard(QWizard):
    def __init__(
        self,
        project: Project,
        default_timezone: str,
        *,
        kind: SourceKind | None = None,
        source_id: str,
        initial: ImportPlan | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import data")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.resize(960, 680)
        self.setAcceptDrops(True)
        self.source_id = source_id
        self.state = _State(project, default_timezone)

        self.file_page = FilePage(self.state)
        self.read_page = ReadPage(self.state)
        self.time_page = TimePage(self.state)
        self.columns_page = ColumnsPage(self.state)
        self.finish_page = FinishPage(self.state)
        for page in (
            self.file_page,
            self.read_page,
            self.time_page,
            self.columns_page,
            self.finish_page,
        ):
            self.addPage(page)

        if kind is SourceKind.LAB:
            self.file_page.lab_radio.setChecked(True)
        if initial is not None:
            self._prefill(initial)

    def _prefill(self, plan: ImportPlan) -> None:
        src = plan.source
        self.file_page.lab_radio.setChecked(src.kind is SourceKind.LAB)
        self.file_page.process_radio.setChecked(src.kind is SourceKind.PROCESS)
        self.file_page.path_edit.setText(src.path)
        self.state.path = Path(src.path)
        self.state.opts = ReadOptions.from_source(src)
        self.state.time_spec = src.timestamp
        self.state.result_spec = src.result_timestamp
        self.state.sample_id_column = src.sample_id_column
        self.columns_page.previous = {v.source_column: v for v in plan.variables}

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls and self.currentPage() is self.file_page:
            self.file_page.path_edit.setText(urls[0].toLocalFile())

    def plan(self) -> ImportPlan:
        s = self.state
        source = DataSource(
            source_id=self.source_id,
            kind=s.kind,
            path=str(s.path),
            timestamp=s.time_spec,
            result_timestamp=s.result_spec,
            sheet=s.opts.sheet,
            delimiter=s.opts.delimiter,
            encoding=s.opts.encoding,
            decimal=s.opts.decimal,
            header_row=s.opts.header_row,
            duplicate_policy=self.finish_page.duplicate_policy,
            sample_id_column=s.sample_id_column,
        )
        return ImportPlan(source=source, variables=self.columns_page.variables(self.source_id))


__all__ = ["ImportPlan", "ImportWizard"]
