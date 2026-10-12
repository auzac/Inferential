"""Fixed-height message panel at the bottom of the main window (see ui/messages.py)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QSize, Qt, QTimer, Slot
from PySide6.QtGui import (
    QBrush,
    QColor,
    QGuiApplication,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.ui.messages import APP, Level, Message, MessageCenter
from inferential_studio.ui.stages import STAGES
from inferential_studio.ui.theme import TOKENS

LEVEL_TOKENS = {Level.INFO: "info", Level.WARNING: "warning", Level.ERROR: "error"}


def level_colour(level: Level) -> str:
    """The current theme's colour for a message level."""
    return TOKENS[LEVEL_TOKENS[level]]


LEVEL_NAMES = {Level.INFO: "Info", Level.WARNING: "Warning", Level.ERROR: "Error"}
STAGE_TITLES = {s.key: f"{s.number}. {s.title}" for s in STAGES} | {APP: "Application"}
ROWS_VISIBLE = 4
ROWS_COMPACT = 3  # short screens (laptops): one row less, more room for the page
ROW_HEIGHT = 22


def panel_height(rows: int = ROWS_VISIBLE) -> int:
    return 30 + rows * ROW_HEIGHT + 8


PANEL_HEIGHT = panel_height()


def level_icon(level: Level, size: int = 14) -> QIcon:
    """Small shape + colour per level (shape too, so it reads without colour vision)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(level_colour(level))))
    s = float(size)
    if level is Level.ERROR:  # octagon-like square
        p.drawRoundedRect(1, 1, s - 2, s - 2, 2, 2)
    elif level is Level.WARNING:  # triangle
        path = QPainterPath(QPointF(s / 2, 1))
        path.lineTo(s - 1, s - 1)
        path.lineTo(1, s - 1)
        path.closeSubpath()
        p.drawPath(path)
    else:  # circle
        p.drawEllipse(QPointF(s / 2, s / 2), s / 2 - 1.5, s / 2 - 1.5)
    p.setPen(QColor(TOKENS["on_accent"]))  # readable on both themes
    font = p.font()
    font.setBold(True)
    font.setPixelSize(int(s * 0.7))
    p.setFont(font)
    glyph = "i" if level is Level.INFO else "!" if level is Level.WARNING else "×"
    p.drawText(
        pm.rect().adjusted(0, 2 if level is Level.WARNING else 0, 0, 0),
        Qt.AlignmentFlag.AlignCenter,
        glyph,
    )
    p.end()
    return QIcon(pm)


def ok_icon(size: int = 14) -> QIcon:
    """Green circle with a check mark, for "OK" status cells (pairs with :func:`level_icon`)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(TOKENS["success"])))
    s = float(size)
    p.drawEllipse(QPointF(s / 2, s / 2), s / 2 - 1.5, s / 2 - 1.5)
    pen = QPen(QColor(TOKENS["on_accent"]))
    pen.setWidthF(max(1.5, s / 9))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    tick = QPainterPath(QPointF(s * 0.30, s * 0.52))
    tick.lineTo(s * 0.45, s * 0.66)
    tick.lineTo(s * 0.71, s * 0.36)
    p.drawPath(tick)
    p.end()
    return QIcon(pm)


class MessagePanel(QFrame):
    """Always-visible, fixed-height log: active conditions pinned on top, then events."""

    def __init__(
        self, center: MessageCenter, parent: QWidget | None = None, *, rows: int = ROWS_VISIBLE
    ) -> None:
        super().__init__(parent)
        self.setObjectName("messagePanel")
        self.rows = rows
        self.setFixedHeight(panel_height(rows))  # fixed for the session: pages never move
        self._center = center
        self._stage: str | None = None
        self._icons = {lvl: level_icon(lvl) for lvl in Level}

        title = QLabel("Messages")
        title.setObjectName("messageTitle")
        self.badges = {lvl: QLabel() for lvl in (Level.ERROR, Level.WARNING)}
        for lvl, badge in self.badges.items():
            badge.setObjectName(f"badge{LEVEL_NAMES[lvl]}")
        self.scope = QComboBox()
        self.scope.addItem("This stage", "stage")
        self.scope.addItem("All stages", "all")
        self.scope.currentIndexChanged.connect(self.refresh)
        self.copy_button = QToolButton()
        self.copy_button.setText("Copy")
        self.copy_button.setToolTip("Copy the visible messages to the clipboard")
        self.copy_button.clicked.connect(self._copy)
        self.clear_button = QToolButton()
        self.clear_button.setText("Clear")
        self.clear_button.setToolTip("Clear the message history (active conditions stay)")
        self.clear_button.clicked.connect(center.clear_events)
        head = QHBoxLayout()
        head.setContentsMargins(10, 4, 6, 0)
        head.setSpacing(8)
        head.addWidget(title)
        for badge in self.badges.values():
            head.addWidget(badge)
        head.addStretch(1)
        head.addWidget(self.scope)
        head.addWidget(self.copy_button)
        head.addWidget(self.clear_button)

        self.list = QTreeWidget()
        self.list.setObjectName("messageList")
        self.list.setColumnCount(4)
        self.list.setHeaderHidden(True)
        self.list.setRootIsDecorated(False)
        self.list.setUniformRowHeights(True)
        self.list.setIconSize(QSize(14, 14))
        self.list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.list.setVerticalScrollMode(QTreeWidget.ScrollMode.ScrollPerPixel)
        header = self.list.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.list.setColumnWidth(0, 92)
        self.list.setColumnWidth(1, 64)
        self.list.setColumnWidth(2, 190)
        self.list.itemDoubleClicked.connect(self._show_details)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(2)
        layout.addLayout(head)
        layout.addWidget(self.list, 1)

        self._flash = QTimer(self)
        self._flash.setSingleShot(True)
        self._flash.setInterval(1500)
        self._flash.timeout.connect(lambda: self._set_flash(False))
        center.changed.connect(self.refresh)
        center.posted.connect(self._on_posted)
        self.refresh()

    # ------------------------------------------------------------------ state

    def set_stage(self, key: str) -> None:
        self._stage = key
        self.refresh()

    def _source(self) -> str | None:
        return self._stage if self.scope.currentData() == "stage" else None

    def visible_messages(self) -> list[Message]:
        source = self._source()
        events = [m for m in self._center.events if m.visible_in(source)]
        return self._center.conditions(source) + events

    @Slot()
    def apply_theme(self) -> None:
        """Recolour the level icons; rebuild, keeping the selection and scroll position."""
        selected = self.list.indexOfTopLevelItem(self.list.currentItem())
        scroll = self.list.verticalScrollBar().value()
        self._icons = {lvl: level_icon(lvl) for lvl in Level}
        self.refresh()
        if 0 <= selected < self.list.topLevelItemCount():
            self.list.setCurrentItem(self.list.topLevelItem(selected))
        self.list.verticalScrollBar().setValue(scroll)

    def refresh(self) -> None:
        self.list.clear()
        for m in self.visible_messages():
            when = "active" if m.is_condition else m.time.strftime("%H:%M:%S")
            item = QTreeWidgetItem(["  " + LEVEL_NAMES[m.level], when, STAGE_TITLES.get(m.source, m.source),
                                    m.text.replace("\n", " ")])  # fmt: skip
            item.setIcon(0, self._icons[m.level])
            item.setForeground(0, QBrush(QColor(level_colour(m.level))))
            tip = m.text + (f"\n\n{m.details}" if m.details else "")
            for c in range(4):
                item.setToolTip(c, tip)
            if m.is_condition:
                font = item.font(3)
                font.setBold(m.level > Level.INFO)
                item.setFont(3, font)
            item.setData(0, Qt.ItemDataRole.UserRole, m)
            self.list.addTopLevelItem(item)
        if self.list.topLevelItemCount() == 0:
            empty = QTreeWidgetItem(["", "", "", "No messages."])
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            empty.setForeground(3, QBrush(QColor(TOKENS["text_faint"])))
            self.list.addTopLevelItem(empty)
        counts = self._center.counts(self._source())
        for lvl, badge in self.badges.items():
            n = counts[lvl]
            badge.setText(f"{n} {LEVEL_NAMES[lvl].lower()}{'s' if n != 1 else ''}")
            badge.setVisible(n > 0)

    def _on_posted(self, m: Message) -> None:
        if m.level >= Level.WARNING:
            self._set_flash(True)
            self._flash.start()

    def _set_flash(self, on: bool) -> None:
        self.setProperty("flash", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def _copy(self) -> None:
        lines = [
            f"[{LEVEL_NAMES[m.level]}] {STAGE_TITLES.get(m.source, m.source)}: {m.text}"
            for m in self.visible_messages()
        ]
        QGuiApplication.clipboard().setText("\n".join(lines))

    def _show_details(self, item: QTreeWidgetItem, _column: int) -> None:
        m: Message | None = item.data(0, Qt.ItemDataRole.UserRole)
        if m is None:
            return
        box = QMessageBox(self)
        box.setIcon({Level.INFO: QMessageBox.Icon.Information, Level.WARNING: QMessageBox.Icon.Warning,
                     Level.ERROR: QMessageBox.Icon.Critical}[m.level])  # fmt: skip
        box.setWindowTitle(f"{LEVEL_NAMES[m.level]} — {STAGE_TITLES.get(m.source, m.source)}")
        box.setText(m.text)
        if m.details:
            box.setDetailedText(m.details)
        box.exec()
