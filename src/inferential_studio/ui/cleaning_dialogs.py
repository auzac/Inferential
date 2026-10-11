"""Dialogs for manual exclusions, logical rules and anomaly detectors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from PySide6.QtCore import QDate, QDateTime, Qt, QTime
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.core.schemas import (
    RANGE_OPERATORS,
    VALUELESS_OPERATORS,
    Combinator,
    Condition,
    ConditionGroup,
    DetectorConfig,
    DetectorKind,
    ExclusionInterval,
    ExclusionReason,
    ExclusionRule,
    RuleAction,
    RuleOperator,
    SchemaError,
)

REASON_LABELS = {
    ExclusionReason.SHUTDOWN: "Shutdown",
    ExclusionReason.STARTUP: "Startup",
    ExclusionReason.TRANSITION: "Transition / grade change",
    ExclusionReason.INSTRUMENT_FAULT: "Instrument fault",
    ExclusionReason.ANALYZER_FAULT: "Analyzer fault",
    ExclusionReason.MAINTENANCE: "Maintenance",
    ExclusionReason.OPERATOR_JUDGEMENT: "Operator judgement",
    ExclusionReason.OTHER: "Other",
}

DETECTOR_HELP = {
    DetectorKind.HAMPEL: (
        "Hampel spike filter",
        "Compares each value with the median of its neighbours in a time window. Flags isolated "
        "spikes but follows genuine step changes in the process. Good first choice for transmitter "
        "spikes. Threshold in robust standard deviations; window in minutes.",
    ),
    DetectorKind.MAD: (
        "Robust z-score (MAD)",
        "Distance from the overall median in robust standard deviations (median absolute deviation). "
        "Not distorted by the outliers it is looking for. Good for lab results.",
    ),
    DetectorKind.ZSCORE: (
        "Z-score",
        "Distance from the mean in standard deviations. Simple, but large outliers inflate the "
        "standard deviation and can hide themselves; prefer the robust z-score.",
    ),
    DetectorKind.IQR: (
        "Interquartile range",
        "Flags values more than k × IQR below the first quartile or above the third (k = 1.5 is the "
        "box-plot rule; 3 is 'far out').",
    ),
    DetectorKind.RATE_OF_CHANGE: (
        "Rate of change",
        "Flags jumps faster than a physically possible rate (units per minute). Steps across data "
        "gaps are not compared.",
    ),
    DetectorKind.FLATLINE: (
        "Flatline / stuck value",
        "Flags periods where the value does not move (within the tolerance) for at least the given "
        "duration — typical of a frozen transmitter or a historian holding the last value.",
    ),
    DetectorKind.LIMITS: (
        "Engineering limits",
        "Flags values outside physically possible limits. Leave a limit blank to use the variable's "
        "engineering limit.",
    ),
    DetectorKind.INVALID_TEXT: (
        "Status text in the file",
        "Flags cells that contained text such as 'Bad' or 'I/O Timeout' instead of a number.",
    ),
}

DEFAULT_THRESHOLD = {
    DetectorKind.ZSCORE: 4.0,
    DetectorKind.MAD: 5.0,
    DetectorKind.IQR: 3.0,
    DetectorKind.HAMPEL: 6.0,
}


# --------------------------------------------------------------------------- time helpers


def to_qdatetime(moment: datetime, timezone: str) -> QDateTime:
    local = moment.astimezone(ZoneInfo(timezone))
    return QDateTime(
        QDate(local.year, local.month, local.day), QTime(local.hour, local.minute, local.second)
    )


def from_qdatetime(value: QDateTime, timezone: str) -> datetime:
    naive = value.toPython()
    return naive.replace(tzinfo=ZoneInfo(timezone)).astimezone(UTC)


def datetime_edit(moment: datetime, timezone: str) -> QDateTimeEdit:
    edit = QDateTimeEdit(to_qdatetime(moment, timezone))
    edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
    edit.setCalendarPopup(True)
    return edit


def parse_number(text: str) -> float | None:
    text = text.strip().replace(",", ".")
    if not text:
        return None
    return float(text)


# --------------------------------------------------------------------------- variable scope


class ScopeSelector(QGroupBox):
    """Choose 'all variables' (operating regime) or specific variables."""

    def __init__(
        self, title: str, variables: list[str], selected: tuple[str, ...], *, all_label: str
    ) -> None:
        super().__init__(title)
        self.all_radio = QRadioButton(all_label)
        self.some_radio = QRadioButton("Only these variables:")
        group = QButtonGroup(self)
        group.addButton(self.all_radio)
        group.addButton(self.some_radio)
        self.list = QListWidget()
        self.list.setMaximumHeight(140)
        for name in variables:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if name in selected else Qt.CheckState.Unchecked
            )
            self.list.addItem(item)
        (self.some_radio if selected else self.all_radio).setChecked(True)
        self.list.setEnabled(bool(selected))
        self.some_radio.toggled.connect(self.list.setEnabled)
        layout = QVBoxLayout(self)
        layout.addWidget(self.all_radio)
        layout.addWidget(self.some_radio)
        layout.addWidget(self.list)

    def signals(self) -> tuple[str, ...]:
        if self.all_radio.isChecked():
            return ()
        return tuple(
            self.list.item(i).text()
            for i in range(self.list.count())
            if self.list.item(i).checkState() == Qt.CheckState.Checked
        )


def reason_combo(current: ExclusionReason) -> QComboBox:
    combo = QComboBox()
    for reason, label in REASON_LABELS.items():
        combo.addItem(label, reason.value)
    combo.setCurrentIndex(combo.findData(current.value))
    return combo


def _buttons(dialog: QDialog) -> QDialogButtonBox:
    box = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    return box


# --------------------------------------------------------------------------- manual exclusion


class ExclusionDialog(QDialog):
    def __init__(
        self,
        interval: ExclusionInterval,
        variables: list[str],
        timezone: str,
        author: str,
        parent: QWidget | None = None,
        *,
        title: str = "Exclude period",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._interval = interval
        self._tz = timezone
        self._author = author
        self.start = datetime_edit(interval.start, timezone)
        self.end = datetime_edit(interval.end, timezone)
        self.reason = reason_combo(interval.reason)
        self.scope = ScopeSelector(
            "Applies to", variables, interval.signals, all_label="All variables (operating regime)"
        )
        self.note = QLineEdit(interval.note)
        form = QFormLayout(self)
        form.addRow(f"Start ({timezone}):", self.start)
        form.addRow(f"End ({timezone}):", self.end)
        form.addRow("Reason:", self.reason)
        form.addRow(self.scope)
        form.addRow("Note:", self.note)
        hint = QLabel(
            "Both the start and end instants are included. The original data is never changed."
        )
        hint.setObjectName("placeholderNote")
        hint.setWordWrap(True)
        form.addRow(hint)
        form.addRow(_buttons(self))

    def result_interval(self) -> ExclusionInterval:
        return replace(
            self._interval,
            start=from_qdatetime(self.start.dateTime(), self._tz),
            end=from_qdatetime(self.end.dateTime(), self._tz),
            reason=ExclusionReason(self.reason.currentData()),
            signals=self.scope.signals(),
            note=self.note.text().strip(),
            author=self._interval.author or self._author,
        )

    def accept(self) -> None:
        try:
            interval = self.result_interval()
        except SchemaError as exc:
            QMessageBox.warning(self, "Invalid period", str(exc))
            return
        if self.scope.some_radio.isChecked() and not interval.signals:
            QMessageBox.warning(
                self, "No variables", "Tick at least one variable, or choose all variables."
            )
            return
        super().accept()


# --------------------------------------------------------------------------- rule editor


OPERATOR_LABELS = {op: op.value for op in RuleOperator}


class ConditionRow(QWidget):
    def __init__(
        self,
        variables: list[str],
        on_change: Callable[[], None],
        on_remove: Callable[[ConditionRow], None],
        cond: Condition | None = None,
    ) -> None:
        super().__init__()
        self.variable = QComboBox()
        self.variable.addItems(variables)
        self.operator = QComboBox()
        for op in RuleOperator:
            self.operator.addItem(OPERATOR_LABELS[op], op.value)
        self.value = QLineEdit()
        self.value.setPlaceholderText("value")
        self.value.setMaximumWidth(110)
        self.and_label = QLabel("and")
        self.value2 = QLineEdit()
        self.value2.setPlaceholderText("upper")
        self.value2.setMaximumWidth(110)
        remove = QPushButton("✕")
        remove.setToolTip("Remove this condition")
        remove.setFixedWidth(28)
        remove.clicked.connect(lambda: on_remove(self))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for w in (self.variable, self.operator, self.value, self.and_label, self.value2, remove):
            layout.addWidget(w)
        layout.addStretch(1)
        if cond is not None:
            self.variable.setCurrentText(cond.variable)
            self.operator.setCurrentIndex(self.operator.findData(cond.operator.value))
            self.value.setText("" if cond.value is None else f"{cond.value:g}")
            self.value2.setText("" if cond.value2 is None else f"{cond.value2:g}")
        self.operator.currentIndexChanged.connect(self._update_fields)
        for w in (self.variable, self.operator):
            w.currentIndexChanged.connect(lambda _i: on_change())
        for w in (self.value, self.value2):
            w.textChanged.connect(lambda _t: on_change())
        self._update_fields()

    def _update_fields(self) -> None:
        op = RuleOperator(self.operator.currentData())
        self.value.setVisible(op not in VALUELESS_OPERATORS)
        self.and_label.setVisible(op in RANGE_OPERATORS)
        self.value2.setVisible(op in RANGE_OPERATORS)

    def condition(self) -> Condition:
        op = RuleOperator(self.operator.currentData())
        try:
            v = None if op in VALUELESS_OPERATORS else parse_number(self.value.text())
            v2 = parse_number(self.value2.text()) if op in RANGE_OPERATORS else None
        except ValueError as exc:
            raise SchemaError(
                f"'{self.variable.currentText()}': enter numbers like 50 or 3.15."
            ) from exc
        return Condition(self.variable.currentText(), op, v, v2)


class GroupBox(QGroupBox):
    def __init__(
        self,
        variables: list[str],
        on_change: Callable[[], None],
        on_remove: Callable[[GroupBox], None],
        group: ConditionGroup | None = None,
    ) -> None:
        super().__init__()
        self._variables = variables
        self._on_change = on_change
        self.combinator = QComboBox()
        self.combinator.addItem("ALL of (AND)", Combinator.AND.value)
        self.combinator.addItem("ANY of (OR)", Combinator.OR.value)
        self.combinator.currentIndexChanged.connect(lambda _i: on_change())
        self.rows: list[ConditionRow] = []
        self.rows_layout = QVBoxLayout()
        add = QPushButton("+ Condition")
        add.clicked.connect(lambda: self.add_row())
        remove = QPushButton("Remove group")
        remove.clicked.connect(lambda: on_remove(self))
        head = QHBoxLayout()
        head.addWidget(QLabel("Match"))
        head.addWidget(self.combinator)
        head.addStretch(1)
        head.addWidget(remove)
        layout = QVBoxLayout(self)
        layout.addLayout(head)
        layout.addLayout(self.rows_layout)
        layout.addWidget(add, 0, Qt.AlignmentFlag.AlignLeft)
        if group is not None:
            self.combinator.setCurrentIndex(self.combinator.findData(group.combinator.value))
            for item in group.items:
                self.add_row(item)
        else:
            self.add_row()

    def add_row(self, cond: Condition | None = None) -> None:
        row = ConditionRow(self._variables, self._on_change, self._remove_row, cond)
        self.rows.append(row)
        self.rows_layout.addWidget(row)
        self._on_change()

    def _remove_row(self, row: ConditionRow) -> None:
        if len(self.rows) == 1:
            return
        self.rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self._on_change()

    def group(self) -> ConditionGroup:
        return ConditionGroup(
            tuple(r.condition() for r in self.rows), Combinator(self.combinator.currentData())
        )


def split_groups(cond: ConditionGroup) -> tuple[Combinator, list[ConditionGroup]] | None:
    """Express a rule as (top combinator, groups of plain conditions), or None if nested deeper."""
    if all(isinstance(i, Condition) for i in cond.items):
        return Combinator.AND, [cond]
    groups = []
    for item in cond.items:
        if isinstance(item, Condition):
            groups.append(ConditionGroup((item,)))
        elif all(isinstance(i, Condition) for i in item.items):
            groups.append(item)
        else:
            return None
    return cond.combinator, groups


class RuleDialog(QDialog):
    """Visual rule builder: groups of conditions; no free-text expressions are ever evaluated."""

    def __init__(
        self,
        rule: ExclusionRule | None,
        variables: list[str],
        preview: Callable[[ExclusionRule], str],
        author: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Exclusion rule" if rule is None else f"Edit rule — {rule.name}")
        self.resize(760, 640)
        self._rule = rule
        self._variables = variables
        self._preview = preview
        self._author = author
        default_reason = rule.reason if rule else ExclusionReason.TRANSITION

        self.name = QLineEdit(rule.name if rule else "")
        self.name.setPlaceholderText("e.g. low-load exclusion")
        self.action = QComboBox()
        self.action.addItem("Exclude matching data", RuleAction.EXCLUDE.value)
        self.action.addItem("Only flag for review", RuleAction.FLAG.value)
        if rule:
            self.action.setCurrentIndex(self.action.findData(rule.action.value))
        self.reason = reason_combo(default_reason)
        self.scope = ScopeSelector(
            "Exclude",
            variables,
            rule.signals if rule else (),
            all_label="Whole rows — every variable (operating regime)",
        )
        self.use_duration = QCheckBox("Only when it holds continuously for at least")
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.1, 100000)
        self.duration.setSuffix(" min")
        self.duration.setValue((rule.min_duration_s or 300) / 60 if rule else 5)
        self.use_duration.setChecked(bool(rule and rule.min_duration_s))
        self.duration.setEnabled(self.use_duration.isChecked())
        self.use_duration.toggled.connect(self.duration.setEnabled)
        self.use_duration.toggled.connect(lambda _c: self._update_preview())
        self.duration.valueChanged.connect(lambda _v: self._update_preview())
        self.note = QLineEdit(rule.note if rule else "")

        self.top = QComboBox()
        self.top.addItem("ALL groups (AND)", Combinator.AND.value)
        self.top.addItem("ANY group (OR)", Combinator.OR.value)
        self.top.currentIndexChanged.connect(lambda _i: self._update_preview())
        self.groups: list[GroupBox] = []
        self.groups_layout = QVBoxLayout()
        holder = QWidget()
        hv = QVBoxLayout(holder)
        hv.addLayout(self.groups_layout)
        hv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        add_group = QPushButton("+ Group")
        add_group.setToolTip("Add a group of conditions, e.g. for (A or B) and C")
        add_group.clicked.connect(lambda: self._add_group())
        top_row = QHBoxLayout()
        self.top_label = QLabel("Combine groups:")
        top_row.addWidget(self.top_label)
        top_row.addWidget(self.top)
        top_row.addStretch(1)
        top_row.addWidget(add_group)

        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextFormat(Qt.TextFormat.RichText)

        self._editable = True
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("Name:", self.name)
        form.addRow("Action:", self.action)
        form.addRow("Reason:", self.reason)
        layout.addLayout(form)
        cond_box = QGroupBox("Conditions")
        cl = QVBoxLayout(cond_box)
        cl.addLayout(top_row)
        cl.addWidget(scroll, 1)
        dur_row = QHBoxLayout()
        dur_row.addWidget(self.use_duration)
        dur_row.addWidget(self.duration)
        dur_row.addStretch(1)
        cl.addLayout(dur_row)
        layout.addWidget(cond_box, 1)
        layout.addWidget(self.scope)
        nf = QFormLayout()
        nf.addRow("Note:", self.note)
        layout.addLayout(nf)
        layout.addWidget(self.preview_label)
        layout.addWidget(_buttons(self))

        if rule is not None:
            split = split_groups(rule.condition)
            if split is None:
                self._editable = False
                cond_box.setEnabled(False)
                self.preview_label.setText(
                    "This rule has deeper nesting than the editor supports; its conditions are shown read-only."
                )
            else:
                combinator, groups = split
                self.top.setCurrentIndex(self.top.findData(combinator.value))
                for g in groups:
                    self._add_group(g)
        if not self.groups and self._editable:
            self._add_group()
        self._update_preview()

    def _add_group(self, group: ConditionGroup | None = None) -> None:
        box = GroupBox(self._variables, self._update_preview, self._remove_group, group)
        self.groups.append(box)
        self.groups_layout.addWidget(box)
        self._update_preview()

    def _remove_group(self, box: GroupBox) -> None:
        if len(self.groups) == 1:
            return
        self.groups.remove(box)
        box.setParent(None)
        box.deleteLater()
        self._update_preview()

    def _condition(self) -> ConditionGroup:
        if not self._editable:
            return self._rule.condition
        groups = [g.group() for g in self.groups]
        if len(groups) == 1:
            return groups[0]
        return ConditionGroup(tuple(groups), Combinator(self.top.currentData()))

    def result_rule(self) -> ExclusionRule:
        fields = dict(
            name=self.name.text().strip() or "Unnamed rule",
            condition=self._condition(),
            action=RuleAction(self.action.currentData()),
            reason=ExclusionReason(self.reason.currentData()),
            signals=self.scope.signals(),
            min_duration_s=self.duration.value() * 60 if self.use_duration.isChecked() else None,
            note=self.note.text().strip(),
        )
        if self._rule is None:
            return ExclusionRule(author=self._author, **fields)
        return replace(self._rule, **fields)

    def _update_preview(self) -> None:
        if not hasattr(self, "preview_label"):
            return
        many = len(self.groups) > 1
        self.top.setVisible(many)
        self.top_label.setVisible(many)
        try:
            rule = self.result_rule()
        except SchemaError as exc:
            self.preview_label.setText(f"<span style='color:#b00020'>{exc}</span>")
            return
        self.preview_label.setText(f"<b>{rule.describe()}</b><br>{self._preview(rule)}")

    def accept(self) -> None:
        if not self.name.text().strip():
            QMessageBox.warning(
                self, "Rule name", "Give the rule a name, e.g. 'low-load exclusion'."
            )
            return
        try:
            rule = self.result_rule()
        except SchemaError as exc:
            QMessageBox.warning(self, "Rule", str(exc))
            return
        if self.scope.some_radio.isChecked() and not rule.signals:
            QMessageBox.warning(
                self, "No variables", "Tick at least one variable, or choose whole rows."
            )
            return
        super().accept()


# --------------------------------------------------------------------------- detector editor


class DetectorDialog(QDialog):
    def __init__(
        self,
        detector: DetectorConfig | None,
        variables: list[str],
        timezone: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Anomaly detector")
        self.resize(620, 680)
        self._detector = detector
        self._tz = timezone
        d = detector

        self.kind = QComboBox()
        for kind, (label, _text) in DETECTOR_HELP.items():
            self.kind.addItem(label, kind.value)
        if d:
            self.kind.setCurrentIndex(self.kind.findData(d.kind.value))
        self.help = QLabel()
        self.help.setWordWrap(True)
        self.help.setObjectName("placeholderNote")

        self.variables = QListWidget()
        self.variables.setMaximumHeight(150)
        chosen = set(d.variables) if d else set()
        for name in variables:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name in chosen else Qt.CheckState.Unchecked)
            self.variables.addItem(item)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.1, 1000)
        self.threshold.setDecimals(2)
        self.window = QDoubleSpinBox()
        self.window.setRange(1, 100000)
        self.window.setSuffix(" min")
        self.rate = QLineEdit()
        self.rate.setPlaceholderText("units per minute")
        self.tolerance = QLineEdit("0")
        self.low = QLineEdit()
        self.low.setPlaceholderText("blank = variable's engineering low limit")
        self.high = QLineEdit()
        self.high.setPlaceholderText("blank = variable's engineering high limit")
        self.min_dev = QLineEdit("0")
        self.min_dev.setToolTip(
            "Ignore deviations smaller than this, in the variable's units (avoids flagging tiny noise on very quiet signals)"
        )
        self.use_fit = QCheckBox("Fit the statistics on this period only")
        now = datetime.now(UTC)
        self.fit_start = datetime_edit(d.fit_start if d and d.fit_start else now, timezone)
        self.fit_end = datetime_edit(d.fit_end if d and d.fit_end else now, timezone)
        self.auto = QCheckBox("Exclude every flag automatically (still reviewable)")
        self.auto.setToolTip("Off by default: flags are only highlighted until you accept them.")

        if d:
            self.threshold.setValue(d.threshold or DEFAULT_THRESHOLD.get(d.kind, 4.0))
            self.window.setValue((d.window_s or 1800) / 60)
            self.rate.setText("" if d.rate_per_min is None else f"{d.rate_per_min:g}")
            self.tolerance.setText(f"{d.tolerance:g}")
            self.low.setText("" if d.low is None else f"{d.low:g}")
            self.high.setText("" if d.high is None else f"{d.high:g}")
            self.min_dev.setText(f"{d.min_deviation:g}")
            self.use_fit.setChecked(bool(d.fit_start or d.fit_end))
            self.auto.setChecked(d.auto_exclude)
        else:
            self.threshold.setValue(4.0)
            self.window.setValue(30)

        self.form = QFormLayout()
        self.form.addRow("Method:", self.kind)
        self.form.addRow(self.help)
        self.form.addRow("Variables:", self.variables)
        self.form.addRow("Threshold:", self.threshold)
        self.form.addRow("Window / duration:", self.window)
        self.form.addRow("Rate limit:", self.rate)
        self.form.addRow("Flatline tolerance:", self.tolerance)
        self.form.addRow("Low limit:", self.low)
        self.form.addRow("High limit:", self.high)
        self.form.addRow("Minimum deviation:", self.min_dev)
        self.form.addRow(self.use_fit)
        self.form.addRow(f"Fit from ({timezone}):", self.fit_start)
        self.form.addRow(f"Fit to ({timezone}):", self.fit_end)
        self.form.addRow(self.auto)
        layout = QVBoxLayout(self)
        layout.addLayout(self.form)
        layout.addWidget(_buttons(self))

        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.use_fit.toggled.connect(lambda _c: self._kind_changed(keep=True))
        self._kind_changed(keep=True)

    def _kind_changed(self, _index: int | None = None, *, keep: bool = False) -> None:
        kind = DetectorKind(self.kind.currentData())
        self.help.setText(DETECTOR_HELP[kind][1])
        statistical = kind in (DetectorKind.ZSCORE, DetectorKind.MAD, DetectorKind.IQR)
        enabled = {
            self.threshold: kind in DEFAULT_THRESHOLD,
            self.window: kind in (DetectorKind.HAMPEL, DetectorKind.FLATLINE),
            self.rate: kind is DetectorKind.RATE_OF_CHANGE,
            self.tolerance: kind is DetectorKind.FLATLINE,
            self.low: kind is DetectorKind.LIMITS,
            self.high: kind is DetectorKind.LIMITS,
            self.min_dev: kind is DetectorKind.HAMPEL,
            self.use_fit: statistical,
            self.fit_start: statistical and self.use_fit.isChecked(),
            self.fit_end: statistical and self.use_fit.isChecked(),
        }
        for widget, on in enabled.items():
            widget.setEnabled(on)
            label = self.form.labelForField(widget)
            if label is not None:
                label.setEnabled(on)
        if not keep and kind in DEFAULT_THRESHOLD:
            self.threshold.setValue(DEFAULT_THRESHOLD[kind])
        if not keep and kind is DetectorKind.FLATLINE:
            self.window.setValue(60)

    def result_detector(self) -> DetectorConfig:
        kind = DetectorKind(self.kind.currentData())
        variables = tuple(
            self.variables.item(i).text()
            for i in range(self.variables.count())
            if self.variables.item(i).checkState() == Qt.CheckState.Checked
        )
        statistical = kind in (DetectorKind.ZSCORE, DetectorKind.MAD, DetectorKind.IQR)
        try:
            fields = dict(
                kind=kind,
                variables=variables,
                threshold=self.threshold.value() if kind in DEFAULT_THRESHOLD else None,
                window_s=self.window.value() * 60
                if kind in (DetectorKind.HAMPEL, DetectorKind.FLATLINE)
                else None,
                rate_per_min=parse_number(self.rate.text())
                if kind is DetectorKind.RATE_OF_CHANGE
                else None,
                tolerance=parse_number(self.tolerance.text()) or 0.0,
                low=parse_number(self.low.text()) if kind is DetectorKind.LIMITS else None,
                high=parse_number(self.high.text()) if kind is DetectorKind.LIMITS else None,
                min_deviation=parse_number(self.min_dev.text()) or 0.0,
                fit_start=from_qdatetime(self.fit_start.dateTime(), self._tz)
                if statistical and self.use_fit.isChecked()
                else None,
                fit_end=from_qdatetime(self.fit_end.dateTime(), self._tz)
                if statistical and self.use_fit.isChecked()
                else None,
                auto_exclude=self.auto.isChecked(),
            )
        except ValueError as exc:
            raise SchemaError("Enter numbers like 0.5 or 12 in the numeric fields.") from exc
        if self._detector is None:
            return DetectorConfig(**fields)
        return replace(self._detector, **fields)

    def accept(self) -> None:
        try:
            self.result_detector()
        except SchemaError as exc:
            QMessageBox.warning(self, "Detector", str(exc))
            return
        super().accept()


__all__ = [
    "DetectorDialog",
    "ExclusionDialog",
    "RuleDialog",
    "from_qdatetime",
    "to_qdatetime",
]
