"""Stage 3 page: manual exclusions, logical rules, anomaly detection with review, summary.

Every edit goes through the undo stack and changes only the project's exclusion lists; the
masks are recomputed from them (core.exclusions) and the raw data is never modified.
"""

from __future__ import annotations

from dataclasses import replace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QUndoStack
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.core.exclusions import EventStatus, category_table, summary_table
from inferential_studio.core.rules import RuleError, evaluate_rule
from inferential_studio.core.schemas import (
    DetectorConfig,
    DetectorKind,
    ExclusionInterval,
    ExclusionReason,
    ExclusionRule,
    FlagDecision,
    FlagDecisionKind,
    RuleAction,
    utc_now,
)
from inferential_studio.infrastructure.settings import AppSettings
from inferential_studio.ui.cleaning_dialogs import (
    REASON_LABELS,
    DetectorDialog,
    ExclusionDialog,
    RuleDialog,
)
from inferential_studio.ui.commands import edit_lists
from inferential_studio.ui.messages import Level
from inferential_studio.ui.pages.stage_page import StagePage
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.stages import Stage
from inferential_studio.ui.theme import mark_primary, theme_colour, tinted
from inferential_studio.ui.widgets.cursor_readout import CursorReadout
from inferential_studio.ui.widgets.data_table import DataTableView
from inferential_studio.ui.widgets.overlays import IntervalShade, mask_spans
from inferential_studio.ui.widgets.time_axis import to_epoch_seconds
from inferential_studio.ui.widgets.trend_view import TrendSeries, TrendView
from inferential_studio.ui.widgets.variable_list import VariableList

REGIME_COLOUR = "#7a7a7a"
VARIABLE_COLOUR = "#E69F00"
PENDING_COLOUR = "#D55E00"
MARKER_LIMIT = 5000
STATUS_LABELS = {
    EventStatus.PENDING: "Pending review",
    EventStatus.ACCEPTED: "Accepted — excluded",
    EventStatus.REJECTED: "Rejected — kept as valid",
    EventStatus.AUTO: "Auto-excluded",
    EventStatus.COVERED: "Inside an excluded period",
}


def _item(
    text: str, data=None, *, checkable: bool = False, checked: bool = False
) -> QTableWidgetItem:  # noqa: ANN001
    item = QTableWidgetItem(text)
    item.setFlags(
        Qt.ItemFlag.ItemIsSelectable
        | Qt.ItemFlag.ItemIsEnabled
        | (Qt.ItemFlag.ItemIsUserCheckable if checkable else Qt.ItemFlag.NoItemFlags)
    )
    if checkable:
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    if data is not None:
        item.setData(Qt.ItemDataRole.UserRole, data)
    return item


def _table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


class CleaningPage(StagePage):
    detection_requested = Signal()

    def __init__(
        self,
        stage: Stage,
        session: ProjectSession,
        undo_stack: QUndoStack,
        settings: AppSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(stage, parent)
        self._session = session
        self._undo = undo_stack
        self._settings = settings
        self._regions: list[tuple[pg.PlotItem, pg.LinearRegionItem]] = []
        self._overlay_items: list[tuple[pg.PlotItem, pg.GraphicsObject]] = []
        self._syncing = False
        self._selection: tuple[float, float] | None = None
        self._summary_dirty = True  # summary table is only computed when its tab is visible

        # -- toolbar ------------------------------------------------------------
        self.view_raw = QRadioButton("Raw")
        self.view_flagged = QRadioButton("Flagged")
        self.view_cleaned = QRadioButton("Cleaned")
        self.view_flagged.setChecked(True)
        self.view_raw.setToolTip("Original values, no overlays")
        self.view_flagged.setToolTip(
            "Original values; grey = excluded for all variables, orange = excluded for this variable, red dots = flagged, awaiting review"
        )
        self.view_cleaned.setToolTip("Values after exclusions (excluded points removed)")
        group = QButtonGroup(self)
        for b in (self.view_raw, self.view_flagged, self.view_cleaned):
            group.addButton(b)
            b.toggled.connect(lambda on: on and self._update_trends())
        self.select_button = QPushButton("Select range")
        self.select_button.setCheckable(True)
        self.select_button.setToolTip(
            "Show a draggable time range on the charts; drag its edges, then exclude it"
        )
        self.select_button.toggled.connect(self._toggle_selection)
        self.exclude_button = QPushButton("Exclude selection…")
        self.exclude_button.setEnabled(False)
        self.exclude_button.clicked.connect(self._exclude_selection)
        mark_primary(self.exclude_button)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("View:"))
        for b in (self.view_raw, self.view_flagged, self.view_cleaned):
            bar.addWidget(b)
        bar.addSpacing(16)
        bar.addWidget(self.select_button)
        bar.addWidget(self.exclude_button)
        bar.addStretch(1)

        # -- chart area -----------------------------------------------------------
        self.variables = VariableList()
        # Coalesce bursts of ticks: redraw once, a short pause after the last change.
        self._trend_timer = QTimer(self)
        self._trend_timer.setSingleShot(True)
        self._trend_timer.setInterval(150)
        self._trend_timer.timeout.connect(self._update_trends)
        self.variables.selection_changed.connect(self._trend_timer.start)
        self.trends = TrendView(session.project.display_timezone)
        self.trends.rebuilt.connect(self._add_overlays)
        self.cursor = CursorReadout()
        self.trends.hover.connect(self.cursor.show_hover)
        self.trends.point_selected.connect(self.cursor.show_point)
        self.trends.point_cleared.connect(self.cursor.unlock)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(self.variables, 1)  # the list scrolls when there are many tags
        left_layout.addWidget(self.cursor)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(left)
        top.addWidget(self.trends)
        top.setSizes([230, 870])

        # -- tabs -------------------------------------------------------------------
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_manual_tab(), "Manual exclusions")
        self.tabs.addTab(self._build_rules_tab(), "Rules")
        self.tabs.addTab(self._build_detection_tab(), "Anomaly detection")
        self._summary_tab = self._build_summary_tab()
        self.tabs.addTab(self._summary_tab, "Summary")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self.tabs)
        split.setSizes([560, 260])
        self.body.addLayout(bar)
        self.body.addWidget(split, 1)

        session.data_changed.connect(self._on_data_changed)
        session.cleaning_changed.connect(self.refresh)
        session.changed.connect(lambda: self.trends.set_timezone(session.project.display_timezone))
        self._on_data_changed()

    # ------------------------------------------------------------------ tab construction

    def _build_manual_tab(self) -> QWidget:
        self.manual_table = _table(
            ["On", "Start", "End", "Duration", "Reason", "Applies to", "Note", "Author"]
        )
        self.manual_table.itemChanged.connect(self._manual_toggled)
        self.manual_table.cellDoubleClicked.connect(lambda _r, _c: self._edit_interval())
        buttons = QHBoxLayout()
        for text, slot in (
            ("Edit…", self._edit_interval),
            ("Delete", self._delete_intervals),
            ("Zoom to", self._zoom_interval),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch(1)
        hint = QLabel(
            "To add one: Select range on the chart, drag its edges, then Exclude selection."
        )
        hint.setObjectName("placeholderNote")
        buttons.addWidget(hint)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addLayout(buttons)
        layout.addWidget(self.manual_table, 1)
        return w

    def _build_rules_tab(self) -> QWidget:
        self.rules_table = _table(
            ["On", "Name", "Condition", "Applies to", "Action", "Reason", "Rows matched"]
        )
        self.rules_table.itemChanged.connect(self._rule_toggled)
        self.rules_table.cellDoubleClicked.connect(lambda _r, _c: self._edit_rule())
        buttons = QHBoxLayout()
        for text, slot in (
            ("Add rule…", self._add_rule),
            ("Edit…", self._edit_rule),
            ("Duplicate", self._duplicate_rule),
            ("Delete", self._delete_rules),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch(1)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addLayout(buttons)
        layout.addWidget(self.rules_table, 1)
        return w

    def _build_detection_tab(self) -> QWidget:
        self.detector_table = _table(
            ["On", "Method", "Variables", "Settings", "Auto-exclude", "Events", "Pending"]
        )
        self.detector_table.itemChanged.connect(self._detector_toggled)
        self.detector_table.cellDoubleClicked.connect(lambda _r, _c: self._edit_detector())
        self.detector_table.itemSelectionChanged.connect(self._fill_events)
        det_buttons = QHBoxLayout()
        for text, slot in (
            ("Add…", self._add_detector),
            ("Add recommended", self._add_recommended),
            ("Edit…", self._edit_detector),
            ("Delete", self._delete_detectors),
            ("Run again", self.detection_requested.emit),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            det_buttons.addWidget(b)
        det_buttons.addStretch(1)

        self.event_filter = QComboBox()
        self.event_filter.addItem("All events", None)
        for status, label in STATUS_LABELS.items():
            self.event_filter.addItem(label, status)
        self.event_filter.setCurrentIndex(1)  # pending first: what needs attention
        self.event_filter.currentIndexChanged.connect(lambda _i: self._fill_events())
        self.events_table = _table(
            ["Variable", "Method", "Start", "End", "Points", "Size", "Status"]
        )
        self.events_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.events_table.cellDoubleClicked.connect(lambda _r, _c: self._zoom_event())
        ev_buttons = QHBoxLayout()
        ev_buttons.addWidget(self.event_filter)
        for text, slot, tip in (
            (
                "Accept (exclude)",
                lambda: self._decide(FlagDecisionKind.ACCEPTED),
                "Confirm the selected flags are bad data",
            ),
            (
                "Reject (keep)",
                lambda: self._decide(FlagDecisionKind.REJECTED),
                "The selected flags are valid process behaviour",
            ),
            ("Reset", lambda: self._decide(None), "Back to pending review"),
            ("Zoom to", self._zoom_event, "Show the selected event on the chart"),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            ev_buttons.addWidget(b)
        ev_buttons.addStretch(1)
        self.events_note = QLabel(
            "Flags never delete data: pending flags are only highlighted; accepted flags exclude "
            "that variable's values; rejected flags are kept as valid."
        )
        self.events_note.setObjectName("placeholderNote")
        self.events_note.setWordWrap(True)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addLayout(det_buttons)
        ll.addWidget(self.detector_table, 1)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addLayout(ev_buttons)
        rl.addWidget(self.events_table, 1)
        rl.addWidget(self.events_note)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([450, 550])
        w = QWidget()
        QVBoxLayout(w).addWidget(split)
        return w

    def _build_summary_tab(self) -> QWidget:
        self.summary_head = QLabel()
        self.summary_head.setWordWrap(True)
        self.summary_table = DataTableView()
        self.category_table = DataTableView()
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(self.summary_head)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.summary_table)
        split.addWidget(self.category_table)
        layout.addWidget(split, 1)
        note = QLabel(
            "Reasons can overlap, so their rows do not add up to the total. Durations are approximate "
            "(rows × typical sampling interval)."
        )
        note.setObjectName("placeholderNote")
        layout.addWidget(note)
        return w

    # ------------------------------------------------------------------ helpers

    @property
    def _tz(self) -> str:
        return self._session.project.display_timezone

    def _fmt(self, moment) -> str:  # noqa: ANN001
        return pd.Timestamp(moment).tz_convert(ZoneInfo(self._tz)).strftime("%Y-%m-%d %H:%M:%S")

    def _edit(self, text: str, **lists: list) -> None:
        try:
            edit_lists(self._undo, self._session, text, **lists)
        except ValueError as exc:
            self._session.messages.error(f"Cannot apply change: {exc}", "cleaning")

    def _all_variables(self) -> list[str]:
        frame, _m, target = self._session.frames()
        names = list(frame.columns)
        if target is not None:
            names.append(str(target.name))
        return names

    @staticmethod
    def _selected_data(table: QTableWidget) -> list:
        rows = sorted({i.row() for i in table.selectedIndexes()})
        return [table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows if table.item(r, 0)]

    # ------------------------------------------------------------------ data / refresh

    @Slot()
    def _on_data_changed(self) -> None:
        frame, _m, target = self._session.frames()
        project = self._session.project
        units = {v.name: v.units for v in project.variables}
        names = list(frame.columns) + ([str(target.name)] if target is not None else [])
        keep = set(self.variables.checked()) & set(names)
        if not keep:
            predictors = [v.name for v in project.predictors if v.name in frame.columns]
            keep = set((predictors or list(frame.columns))[:4])
            if target is not None:
                keep.add(str(target.name))
        self.variables.set_variables(names, units, keep)
        self.refresh(full=True)

    @Slot()
    def refresh(self, full: bool = False) -> None:
        self._fill_manual()
        self._fill_rules()
        self._fill_detectors()
        self._fill_events()
        if self.tabs.currentWidget() is self._summary_tab:
            self._fill_summary()
        else:
            self._summary_dirty = True  # the summary scans every row: only when its tab is open
        self._update_trends(keep_range=not full)

    @Slot(int)
    def _on_tab_changed(self, _index: int) -> None:
        if self.tabs.currentWidget() is self._summary_tab and self._summary_dirty:
            self._summary_dirty = False
            self._fill_summary()

    def _update_trends(self, keep_range: bool = True) -> None:
        frame, _mask, target = self._session.frames()
        state = self._session.cleaning
        if frame.empty:
            self.trends.set_series([])
            self._session.messages.set("cleaning.status", Level.INFO, "No process data loaded.")
            return
        cleaned_view = self.view_cleaned.isChecked() and state is not None
        x = to_epoch_seconds(frame.index)
        units = {v.name: v.units for v in self._session.project.variables}
        series = []
        for name in self.variables.checked():
            if name in frame.columns:
                y = frame[name].to_numpy(dtype=float)
                if cleaned_view:
                    y = np.where(state.excluded[name].to_numpy(), np.nan, y)
                series.append(TrendSeries(name, x, y, units.get(name)))
            elif target is not None and name == target.name:
                t = target.dropna()
                if cleaned_view and state.target is not None:
                    t = t[~state.target.excluded.reindex(t.index, fill_value=False)]
                series.append(
                    TrendSeries(
                        name,
                        to_epoch_seconds(t.index),
                        t.to_numpy(dtype=float),
                        units.get(name),
                        points=True,
                    )
                )
        self.trends.set_series(series, keep_range=keep_range)

    @Slot()
    def _add_overlays(self) -> None:
        frame, _mask, target = self._session.frames()
        state = self._session.cleaning
        # Plots now survive chart updates, so remove the previous pass's overlays first.
        for plot, item in self._overlay_items:
            if isinstance(item, IntervalShade):
                item.detach()
            if item.scene() is not None:
                plot.removeItem(item)
        self._overlay_items = []
        if state is None or frame.empty:
            return
        x = to_epoch_seconds(frame.index)
        show = not self.view_raw.isChecked()
        alpha = 35 if self.view_cleaned.isChecked() else 70
        regime_spans = mask_spans(state.regime, x)
        for plot, group in self.trends.groups:
            if show and regime_spans[0].size:
                shade = IntervalShade(*regime_spans, REGIME_COLOUR, alpha)
                shade.attach(plot)
                self._overlay_items.append((plot, shade))
            for s in group:
                if s.name in state.excluded.columns and show:
                    own = state.excluded[s.name].to_numpy() & ~state.regime
                    spans = mask_spans(own, x)
                    if spans[0].size:
                        shade = IntervalShade(*spans, VARIABLE_COLOUR, alpha)
                        shade.attach(plot)
                        self._overlay_items.append((plot, shade))
                    if self.view_flagged.isChecked():
                        pending = np.flatnonzero(state.pending[s.name].to_numpy())
                        self._markers(
                            plot, x[pending], frame[s.name].to_numpy()[pending], PENDING_COLOUR, "o"
                        )
                elif (
                    target is not None
                    and s.name == target.name
                    and state.target is not None
                    and show
                ):
                    if self.view_flagged.isChecked():
                        ex = state.target.excluded
                        pe = state.target.pending
                        t_ex = target[ex.reindex(target.index, fill_value=False)].dropna()
                        t_pe = target[pe.reindex(target.index, fill_value=False)].dropna()
                        self._markers(
                            plot,
                            to_epoch_seconds(t_ex.index),
                            t_ex.to_numpy(),
                            REGIME_COLOUR,
                            "x",
                            size=10,
                        )
                        self._markers(
                            plot,
                            to_epoch_seconds(t_pe.index),
                            t_pe.to_numpy(),
                            PENDING_COLOUR,
                            "o",
                            size=9,
                        )
        if self.select_button.isChecked():
            self._create_regions()
        self._update_status()

    def _markers(self, plot: pg.PlotItem, xs, ys, colour: str, symbol: str, size: int = 6) -> None:  # noqa: ANN001
        if len(xs) == 0:
            return
        if len(xs) > MARKER_LIMIT:
            keep = np.linspace(0, len(xs) - 1, MARKER_LIMIT).round().astype(int)
            xs, ys = np.asarray(xs)[keep], np.asarray(ys)[keep]
        item = pg.ScatterPlotItem(
            xs,
            ys,
            symbol=symbol,
            size=size,
            pen=pg.mkPen(colour, width=1.5),
            brush=QBrush(QColor(colour)),
        )
        plot.addItem(item, ignoreBounds=True)
        self._overlay_items.append((plot, item))

    def _update_status(self) -> None:
        state = self._session.cleaning
        msgs = self._session.messages
        if state is None:
            msgs.clear("cleaning.status")
            msgs.clear("cleaning.rules")
            return
        n = len(state.index)
        regime = int(state.regime.sum())
        pending_events = sum(e.status == EventStatus.PENDING for e in state.events)
        msgs.set(
            "cleaning.status",
            Level.INFO,
            f"{regime:,} of {n:,} rows excluded for all variables ({100 * regime / max(n, 1):.1f}%); "
            f"{pending_events} flag event(s) awaiting review.",
        )
        msgs.set(
            "cleaning.rules",
            Level.ERROR,
            f"{len(state.rule_errors)} rule(s) could not be evaluated: "
            + "; ".join(state.rule_errors.values())
            if state.rule_errors
            else "",
        )

    # ------------------------------------------------------------------ range selection

    @Slot(bool)
    def _toggle_selection(self, on: bool) -> None:
        self.exclude_button.setEnabled(on)
        if on:
            rng = self.trends.visible_x_range()
            if rng is None:
                self.select_button.setChecked(False)
                return
            lo, hi = rng
            span = hi - lo
            self._selection = (lo + 0.4 * span, lo + 0.6 * span)
            self._create_regions()
        else:
            for plot, region in self._regions:
                if region.scene() is not None:
                    plot.removeItem(region)
            self._regions = []
            self._selection = None

    def _create_regions(self) -> None:
        if self._selection is None:
            return
        for plot, region in self._regions:
            if region.scene() is not None:
                plot.removeItem(region)
        self._regions = []
        for plot in self.trends.plots:
            region = pg.LinearRegionItem(
                values=self._selection, brush=pg.mkBrush(0, 114, 178, 40), movable=True
            )
            region.setZValue(10)
            region.sigRegionChanged.connect(self._region_moved)
            plot.addItem(region, ignoreBounds=True)
            self._regions.append((plot, region))

    def _region_moved(self, region: pg.LinearRegionItem) -> None:
        if self._syncing:
            return
        self._syncing = True
        self._selection = tuple(region.getRegion())
        for _plot, other in self._regions:
            if other is not region:
                other.setRegion(self._selection)
        self._syncing = False

    def selection_utc(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if self._selection is None:
            return None
        lo, hi = sorted(self._selection)
        return pd.Timestamp(lo, unit="s", tz="UTC").round("s"), pd.Timestamp(
            hi, unit="s", tz="UTC"
        ).round("s")

    @Slot()
    def _exclude_selection(self) -> None:
        sel = self.selection_utc()
        if sel is None:
            return
        draft = ExclusionInterval(
            start=sel[0].to_pydatetime(),
            end=sel[1].to_pydatetime(),
            reason=ExclusionReason.SHUTDOWN,
            author=self._settings.author_name,
        )
        dialog = ExclusionDialog(
            draft, self._all_variables(), self._tz, self._settings.author_name, self
        )
        if dialog.exec():
            self.add_interval(dialog.result_interval())
            self.select_button.setChecked(False)

    def add_interval(self, interval: ExclusionInterval) -> None:
        self._edit("Exclude period", exclusions=[*self._session.project.exclusions, interval])

    # ------------------------------------------------------------------ manual table

    def _fill_manual(self) -> None:
        t = self.manual_table
        t.blockSignals(True)
        items = self._session.project.exclusions
        t.setRowCount(len(items))
        for r, iv in enumerate(items):
            duration = pd.Timestamp(iv.end) - pd.Timestamp(iv.start)
            cells = [
                _item("", iv.interval_id, checkable=True, checked=iv.enabled),
                _item(self._fmt(iv.start)),
                _item(self._fmt(iv.end)),
                _item(str(duration).replace("0 days ", "")),
                _item(REASON_LABELS[iv.reason]),
                _item("All variables" if iv.is_global else ", ".join(iv.signals)),
                _item(iv.note),
                _item(iv.author),
            ]
            for c, cell in enumerate(cells):
                t.setItem(r, c, cell)
        t.blockSignals(False)

    def _manual_toggled(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        iid = item.data(Qt.ItemDataRole.UserRole)
        on = item.checkState() == Qt.CheckState.Checked
        new = [
            replace(iv, enabled=on) if iv.interval_id == iid else iv
            for iv in self._session.project.exclusions
        ]
        self._edit("Enable exclusion" if on else "Disable exclusion", exclusions=new)

    def _edit_interval(self) -> None:
        ids = self._selected_data(self.manual_table)
        if not ids:
            return
        current = next(iv for iv in self._session.project.exclusions if iv.interval_id == ids[0])
        dialog = ExclusionDialog(
            current,
            self._all_variables(),
            self._tz,
            self._settings.author_name,
            self,
            title="Edit exclusion",
        )
        if dialog.exec():
            new_iv = dialog.result_interval()
            new = [
                new_iv if iv.interval_id == current.interval_id else iv
                for iv in self._session.project.exclusions
            ]
            self._edit("Edit exclusion", exclusions=new)

    def _delete_intervals(self) -> None:
        ids = set(self._selected_data(self.manual_table))
        if ids:
            new = [iv for iv in self._session.project.exclusions if iv.interval_id not in ids]
            self._edit(f"Delete {len(ids)} exclusion(s)", exclusions=new)

    def _zoom_interval(self) -> None:
        ids = self._selected_data(self.manual_table)
        if ids:
            iv = next(iv for iv in self._session.project.exclusions if iv.interval_id == ids[0])
            self.zoom_to(pd.Timestamp(iv.start), pd.Timestamp(iv.end))

    def zoom_to(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        pad = max((end - start) * 0.5, pd.Timedelta("30min"))
        lo, hi = (start - pad).timestamp(), (end + pad).timestamp()
        self.trends.set_x_range(lo, hi)

    # ------------------------------------------------------------------ rules

    def _rule_preview(self, rule: ExclusionRule) -> str:
        frame, mask, _t = self._session.frames()
        if frame.empty:
            return "No process data loaded to preview against."
        try:
            hit = evaluate_rule(rule, frame, mask)
        except (RuleError, ValueError) as exc:
            return tinted(str(exc))
        state = self._session.cleaning
        step_h = state.median_interval.total_seconds() / 3600 if state is not None else 0
        return f"Matches {int(hit.sum()):,} of {len(frame):,} rows (≈ {hit.sum() * step_h:.1f} h)."

    def _fill_rules(self) -> None:
        t = self.rules_table
        t.blockSignals(True)
        state = self._session.cleaning
        counts = {c.key: int(c.rows.sum()) for c in state.categories} if state else {}
        errors = state.rule_errors if state else {}
        rules = self._session.project.rules
        t.setRowCount(len(rules))
        for r, rule in enumerate(rules):
            matched = errors.get(rule.rule_id) or (
                f"{counts.get(f'rule:{rule.rule_id}', 0):,}" if rule.enabled else "—"
            )
            cells = [
                _item("", rule.rule_id, checkable=True, checked=rule.enabled),
                _item(rule.name),
                _item(rule.describe()),
                _item("Whole rows" if not rule.signals else ", ".join(rule.signals)),
                _item("Exclude" if rule.action is RuleAction.EXCLUDE else "Flag only"),
                _item(REASON_LABELS[rule.reason]),
                _item(matched),
            ]
            if rule.rule_id in errors:
                cells[-1].setForeground(theme_colour("error"))
            for c, cell in enumerate(cells):
                t.setItem(r, c, cell)
        t.blockSignals(False)

    def _rule_toggled(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        rid = item.data(Qt.ItemDataRole.UserRole)
        on = item.checkState() == Qt.CheckState.Checked
        new = [
            replace(r, enabled=on) if r.rule_id == rid else r for r in self._session.project.rules
        ]
        self._edit("Enable rule" if on else "Disable rule", rules=new)

    def _rule_variables(self) -> list[str]:
        frame, _m, _t = self._session.frames()
        return list(frame.columns)

    def _add_rule(self) -> None:
        if not self._rule_variables():
            self._session.messages.warning(
                "Rules: import process data first; rules are evaluated on it.", "cleaning"
            )
            return
        dialog = RuleDialog(
            None, self._rule_variables(), self._rule_preview, self._settings.author_name, self
        )
        if dialog.exec():
            self._edit("Add rule", rules=[*self._session.project.rules, dialog.result_rule()])

    def _edit_rule(self) -> None:
        ids = self._selected_data(self.rules_table)
        if not ids:
            return
        current = next(r for r in self._session.project.rules if r.rule_id == ids[0])
        dialog = RuleDialog(
            current, self._rule_variables(), self._rule_preview, self._settings.author_name, self
        )
        if dialog.exec():
            new_rule = dialog.result_rule()
            self._edit(
                "Edit rule",
                rules=[
                    new_rule if r.rule_id == current.rule_id else r
                    for r in self._session.project.rules
                ],
            )

    def _duplicate_rule(self) -> None:
        ids = self._selected_data(self.rules_table)
        if not ids:
            return
        current = next(r for r in self._session.project.rules if r.rule_id == ids[0])
        copy = ExclusionRule(
            name=f"{current.name} (copy)",
            condition=current.condition,
            action=current.action,
            reason=current.reason,
            signals=current.signals,
            min_duration_s=current.min_duration_s,
            enabled=False,
            note=current.note,
            author=self._settings.author_name,
        )
        self._edit("Duplicate rule", rules=[*self._session.project.rules, copy])

    def _delete_rules(self) -> None:
        ids = set(self._selected_data(self.rules_table))
        if ids:
            self._edit(
                f"Delete {len(ids)} rule(s)",
                rules=[r for r in self._session.project.rules if r.rule_id not in ids],
            )

    # ------------------------------------------------------------------ detectors

    def _fill_detectors(self) -> None:
        t = self.detector_table
        selected = set(self._selected_data(t))
        t.blockSignals(True)
        state = self._session.cleaning
        events = state.events if state else []
        detectors = self._session.project.detectors
        t.setRowCount(len(detectors))
        for r, det in enumerate(detectors):
            mine = [e for e in events if e.event.detector_id == det.detector_id]
            pending = sum(e.status == EventStatus.PENDING for e in mine)
            cells = [
                _item("", det.detector_id, checkable=True, checked=det.enabled),
                _item(det.kind.value),
                _item(", ".join(det.variables)),
                _item(det.describe()),
                _item("yes" if det.auto_exclude else "no"),
                _item(str(len(mine))),
                _item(str(pending)),
            ]
            for c, cell in enumerate(cells):
                t.setItem(r, c, cell)
        t.blockSignals(False)
        for r in range(t.rowCount()):
            if t.item(r, 0).data(Qt.ItemDataRole.UserRole) in selected:
                t.selectRow(r)

    def _detector_toggled(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        did = item.data(Qt.ItemDataRole.UserRole)
        on = item.checkState() == Qt.CheckState.Checked
        new = [
            replace(d, enabled=on) if d.detector_id == did else d
            for d in self._session.project.detectors
        ]
        self._edit("Enable detector" if on else "Disable detector", detectors=new)

    def _add_detector(self) -> None:
        names = self._all_variables()
        if not names:
            self._session.messages.warning("Anomaly detection: import data first.", "cleaning")
            return
        dialog = DetectorDialog(None, names, self._tz, self)
        if dialog.exec():
            self._edit(
                "Add detector",
                detectors=[*self._session.project.detectors, dialog.result_detector()],
            )

    def recommended_detectors(self) -> list[DetectorConfig]:
        project = self._session.project
        frame, mask, target = self._session.frames()
        predictors = [v.name for v in project.predictors if v.name in frame.columns] or list(
            frame.columns
        )
        out: list[DetectorConfig] = []
        if predictors:
            # Conservative defaults: isolated spikes well outside normal variation.
            out.append(
                DetectorConfig(DetectorKind.HAMPEL, tuple(predictors), threshold=6.0, window_s=3600)
            )
            out.append(DetectorConfig(DetectorKind.FLATLINE, tuple(predictors), window_s=3600))
        with_text = [c for c in mask.columns if mask[c].any()] if not mask.empty else []
        if with_text:
            out.append(DetectorConfig(DetectorKind.INVALID_TEXT, tuple(with_text)))
        limited = [
            v.name
            for v in project.variables
            if v.name in self._all_variables() and (v.eng_low is not None or v.eng_high is not None)
        ]
        if limited:
            out.append(DetectorConfig(DetectorKind.LIMITS, tuple(limited)))
        if target is not None:
            out.append(DetectorConfig(DetectorKind.MAD, (str(target.name),), threshold=5.0))
        return out

    def _add_recommended(self) -> None:
        new = self.recommended_detectors()
        if not new:
            self._session.messages.warning("Anomaly detection: import data first.", "cleaning")
            return
        self._edit("Add recommended detectors", detectors=[*self._session.project.detectors, *new])

    def _edit_detector(self) -> None:
        ids = self._selected_data(self.detector_table)
        if not ids:
            return
        current = next(d for d in self._session.project.detectors if d.detector_id == ids[0])
        dialog = DetectorDialog(current, self._all_variables(), self._tz, self)
        if not dialog.exec():
            return
        updated = dialog.result_detector()
        decisions = self._session.project.flag_decisions
        mine = [f for f in decisions if f.detector_id == current.detector_id]
        settings_changed = (
            replace(updated, auto_exclude=current.auto_exclude, enabled=current.enabled) != current
        )
        if mine and settings_changed:
            answer = QMessageBox.question(
                self,
                "Edit detector",
                f"Changing the settings may change the flagged events, so the {len(mine)} review decision(s) "
                "made for this detector will be cleared. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            decisions = [f for f in decisions if f.detector_id != current.detector_id]
        detectors = [
            updated if d.detector_id == current.detector_id else d
            for d in self._session.project.detectors
        ]
        self._edit("Edit detector", detectors=detectors, flag_decisions=decisions)

    def _delete_detectors(self) -> None:
        ids = set(self._selected_data(self.detector_table))
        if not ids:
            return
        project = self._session.project
        self._edit(
            f"Delete {len(ids)} detector(s)",
            detectors=[d for d in project.detectors if d.detector_id not in ids],
            flag_decisions=[f for f in project.flag_decisions if f.detector_id not in ids],
        )

    # ------------------------------------------------------------------ events and review

    def _visible_events(self):  # noqa: ANN202
        state = self._session.cleaning
        if state is None:
            return []
        wanted = self.event_filter.currentData()
        selected_dets = set(self._selected_data(self.detector_table))
        events = [e for e in state.events if wanted is None or e.status == wanted]
        if selected_dets:
            events = [e for e in events if e.event.detector_id in selected_dets]
        return sorted(events, key=lambda e: (e.event.start, e.event.variable))

    def _fill_events(self) -> None:
        t = self.events_table
        events = self._visible_events()
        kinds = {d.detector_id: d.kind.value for d in self._session.project.detectors}
        t.setRowCount(len(events))
        for r, ev in enumerate(events):
            e = ev.event
            cells = [
                _item(e.variable, e),
                _item(kinds.get(e.detector_id, "?")),
                _item(self._fmt(e.start)),
                _item(self._fmt(e.end)),
                _item(str(e.n_points)),
                _item(f"{e.score:.3g}"),
                _item(STATUS_LABELS[ev.status]),
            ]
            if ev.status == EventStatus.PENDING:
                cells[-1].setForeground(QColor(PENDING_COLOUR))
            for c, cell in enumerate(cells):
                t.setItem(r, c, cell)
        state = self._session.cleaning
        total = len(state.events) if state else 0
        if (
            self._session.project.detectors
            and self._session.detector_results_version != self._session.data_version
        ):
            self._session.messages.set("cleaning.detectors", Level.INFO, "Running detectors…")
        else:
            self._session.messages.set(
                "cleaning.detectors", Level.INFO,
                f"Flag events: showing {len(events)} of {total}." if total else "",
            )  # fmt: skip

    def _decide(self, decision: FlagDecisionKind | None) -> None:
        events = self._selected_data(self.events_table)
        if not events:
            return
        keys = {e.key for e in events}
        decisions = [
            f for f in self._session.project.flag_decisions
            if (f.detector_id, f.variable, pd.Timestamp(f.start), pd.Timestamp(f.end)) not in keys
        ]  # fmt: skip
        if decision is not None:
            now = utc_now()
            decisions += [
                FlagDecision(
                    e.detector_id,
                    e.variable,
                    e.start.to_pydatetime(),
                    e.end.to_pydatetime(),
                    decision,
                    author=self._settings.author_name,
                    decided_at=now,
                )  # fmt: skip
                for e in events
            ]
        label = {
            FlagDecisionKind.ACCEPTED: "Accept",
            FlagDecisionKind.REJECTED: "Reject",
            None: "Reset",
        }[decision]
        self._edit(f"{label} {len(events)} flag(s)", flag_decisions=decisions)

    def _zoom_event(self) -> None:
        events = self._selected_data(self.events_table)
        if not events:
            return
        e = events[0]
        if e.variable not in self.variables.checked():
            self.variables.set_checked(e.variable)
        self.zoom_to(e.start, e.end)

    # ------------------------------------------------------------------ summary

    def _fill_summary(self) -> None:
        state = self._session.cleaning
        frame, mask, _t = self._session.frames()
        if state is None:
            self.summary_head.setText("No data loaded.")
            self.summary_table.set_frame(pd.DataFrame())
            self.category_table.set_frame(pd.DataFrame())
            return
        n = len(state.index)
        regime = int(state.regime.sum())
        lines = [
            f"<b>{n:,}</b> process rows with valid timestamps · <b>{regime:,}</b> excluded for all variables "
            f"({100 * regime / max(n, 1):.1f}%) · <b>{n - regime:,}</b> remaining.",
        ]
        if state.target is not None:
            tn = len(state.target.excluded)
            lines.append(
                f"Target samples: {tn} · excluded {int(state.target.excluded.sum())} · awaiting review "
                f"{int(state.target.pending.sum())}. (Whether each sample has enough valid process history is "
                "decided during alignment in Signal Processing.)"
            )
        self.summary_head.setText("<br>".join(lines))
        self.summary_table.set_frame(
            summary_table(state, frame, mask),
            {
                "rows": "Rows",
                "valid_values": "Valid values",
                "invalid_text": "Text cells",
                "excluded": "Excluded",
                "pending_review": "Pending review",
                "usable": "Usable",
                "usable_pct": "Usable %",
                "excluded_duration_h": "Excluded (h)",
            },
        )
        self.category_table.set_frame(
            category_table(state).set_index("reason"),
            {"kind": "Type", "excludes": "Excludes", "rows": "Rows", "approx_hours": "≈ Hours"},
        )
