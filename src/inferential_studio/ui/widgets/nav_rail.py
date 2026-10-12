"""Navigation rail: a slim column of stage icons that slides out over the page to show names.

Collapsed, the rail is 48 px wide and shows one icon per workflow stage (the tooltip names it).
The toggle at the top slides the panel out to show the stage names. The panel lies *over* the
page instead of pushing it aside, so charts and tables never reflow or redraw while it moves:
only the rail itself repaints during the animation. It closes again when a stage is chosen,
on Esc, on the toggle, or on a click anywhere else.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QModelIndex,
    QObject,
    QRect,
    QRectF,
    QSize,
    Qt,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
    QWidget,
)

from inferential_studio.ui.icons import icon
from inferential_studio.ui.stages import Stage
from inferential_studio.ui.theme import TOKENS

RAIL_WIDTH = 48
EXPANDED_WIDTH = 240
ITEM_HEIGHT = 40
ICON_SIZE = 20
TOP = 48  # toggle row
SHADOW = 12  # soft edge drawn to the right of the open panel
ANIMATION_MS = 160


class _NavDelegate(QStyledItemDelegate):
    """Windows 11 style item: rounded hover/selection fill and a short accent pill."""

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # noqa: N802
        return QSize(EXPANDED_WIDTH, ITEM_HEIGHT)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        r = option.rect
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        fill = TOKENS["selected"] if selected else TOKENS["hover"] if hovered else None
        if fill:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(fill))
            painter.drawRoundedRect(QRectF(r).adjusted(4, 2, -4, -2), 4, 4)
        if selected:
            painter.setBrush(QColor(TOKENS["accent"]))
            pill = QRectF(r.left() + 4, r.center().y() - 8, 3, 16)
            painter.drawRoundedRect(pill, 1.5, 1.5)
        qicon: QIcon = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(qicon, QIcon):
            x = r.left() + (RAIL_WIDTH - ICON_SIZE) // 2
            y = r.top() + (r.height() - ICON_SIZE) // 2
            qicon.paint(painter, QRect(x, y, ICON_SIZE, ICON_SIZE))
        text_left = r.left() + RAIL_WIDTH + 2
        available = r.right() - text_left - 8
        if available > 12:  # names appear as the panel slides out
            painter.setPen(QColor(TOKENS["text"]))
            text = option.fontMetrics.elidedText(
                index.data(Qt.ItemDataRole.DisplayRole), Qt.TextElideMode.ElideRight, available
            )
            painter.drawText(
                QRect(text_left, r.top(), available, r.height()),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                text,
            )
        painter.restore()


class NavRail(QWidget):
    """The rail; ``list`` is the stage list (one row per stage, in workflow order)."""

    expanded_changed = Signal(bool)

    def __init__(self, stages: tuple[Stage, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("navRail")
        self._stages = stages
        self._panel = RAIL_WIDTH
        self._expanded = False

        self.toggle = QToolButton(self)
        self.toggle.setObjectName("navToggle")
        self.toggle.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.toggle.setAutoRaise(True)
        self.toggle.clicked.connect(lambda: self.set_expanded(not self._expanded))

        self.list = QListWidget(self)
        self.list.setObjectName("sidebar")
        self.list.setItemDelegate(_NavDelegate(self.list))
        self.list.setMouseTracking(True)  # hover highlight
        self.list.setUniformItemSizes(True)
        self.list.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.setFrameShape(QListWidget.Shape.NoFrame)
        for stage in stages:
            item = QListWidgetItem(stage.title)
            item.setData(Qt.ItemDataRole.UserRole, stage.key)
            item.setToolTip(f"<b>{stage.title}</b><br>{stage.goal}")
            item.setSizeHint(QSize(EXPANDED_WIDTH, ITEM_HEIGHT))
            self.list.addItem(item)
        self.list.itemClicked.connect(lambda _item: self.set_expanded(False))

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(ANIMATION_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(lambda v: self._set_panel_width(int(v)))
        self._anim.finished.connect(self._animation_done)
        self.apply_theme()
        self._set_panel_width(RAIL_WIDTH)

    # -- public ---------------------------------------------------------------------------

    @property
    def expanded(self) -> bool:
        return self._expanded

    @property
    def panel_width(self) -> int:
        return self._panel

    def set_expanded(self, expanded: bool, *, animate: bool = True) -> None:
        if expanded == self._expanded:
            return
        self._expanded = expanded
        app = QApplication.instance()
        if expanded:
            self.raise_()
            if app is not None:
                app.installEventFilter(self)  # click outside / Esc closes the panel
        elif app is not None:
            app.removeEventFilter(self)
        self._update_toggle()
        target = EXPANDED_WIDTH if expanded else RAIL_WIDTH
        self._anim.stop()
        if animate:
            self._anim.setStartValue(self._panel)
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self._set_panel_width(target)
        self.expanded_changed.emit(expanded)

    def apply_theme(self) -> None:
        for row, stage in enumerate(self._stages):
            self.list.item(row).setIcon(icon(stage.icon, "text"))
        self._update_toggle()
        self.update()

    # -- Qt overrides ---------------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().resizeEvent(event)
        self._layout()

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        p.fillRect(QRect(0, 0, self._panel, self.height()), QColor(TOKENS["base"]))
        if self._panel > RAIL_WIDTH:  # the open panel floats over the page: soft edge
            gradient = QLinearGradient(self._panel, 0, self._panel + SHADOW, 0)
            gradient.setColorAt(0.0, QColor(0, 0, 0, 46))
            gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.fillRect(QRect(self._panel, 0, SHADOW, self.height()), gradient)
            p.setPen(QColor(TOKENS["border"]))
            p.drawLine(self._panel - 1, 0, self._panel - 1, self.height())
        p.end()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        if not self._expanded:
            return False
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(watched, QWidget):
            where = event.globalPosition().toPoint()
            if QRect(0, 0, self._panel, self.height()).contains(self.mapFromGlobal(where)):
                return False
            self.set_expanded(False)
            # Like Windows' light dismiss: a click on the covered page only closes the panel
            # (no accidental action underneath); the title bar and other windows still work.
            shell = self.parentWidget()
            return (
                shell is not None
                and watched.window() is self.window()
                and shell.rect().contains(shell.mapFromGlobal(where))
            )
        elif event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            self.set_expanded(False)
            return True
        return False

    # -- internals ------------------------------------------------------------------------

    def _set_panel_width(self, width: int) -> None:
        self._panel = width
        extra = SHADOW if width > RAIL_WIDTH else 0
        self.resize(width + extra, self.height())
        self._layout()
        self.update()

    def _layout(self) -> None:
        self.toggle.setGeometry(4, 6, RAIL_WIDTH - 8, 36)
        self.list.setGeometry(0, TOP, self._panel, max(self.height() - TOP - 6, 0))

    def _animation_done(self) -> None:
        if not self._expanded:
            self._set_panel_width(RAIL_WIDTH)

    def _update_toggle(self) -> None:
        name = "panel_left_contract" if self._expanded else "panel_left_expand"
        self.toggle.setIcon(icon(name, "text"))
        self.toggle.setToolTip("Hide stage names" if self._expanded else "Show stage names")
