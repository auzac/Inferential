"""The window's title bar, merged with the command bar (Windows 11 style).

Left: the main menu (≡), the application icon and icon buttons for the frequent commands.
Centre: the project name. Right: the help panel toggle, the light/dark switch and, when the
window draws its own frame, the minimise / maximise / close buttons.

Dragging an empty part of the bar moves the window (the system handles snapping and moving
between screens); a double click maximises or restores it.
"""

from __future__ import annotations

from html import escape

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QIcon, QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from inferential_studio.ui.icons import icon
from inferential_studio.ui.theme import TOKENS
from inferential_studio.ui.widgets.theme_switch import ThemeSwitch

HEIGHT = 40
CAPTION_ICON = 16


def title_button(action: QAction | None = None, *, parent: QWidget | None = None) -> QToolButton:
    button = QToolButton(parent)
    button.setObjectName("titleButton")
    button.setAutoRaise(True)
    button.setIconSize(QSize(18, 18))
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
    button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    if action is not None:
        button.setDefaultAction(action)
    return button


class TitleBar(QWidget):
    def __init__(
        self,
        app_icon: QIcon,
        menu: QMenu,
        commands: list[QAction | None],
        help_action: QAction,
        *,
        frameless: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("titleBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(HEIGHT)
        self._frameless = frameless

        self.menu_button = title_button(parent=self)
        self.menu_button.setMenu(menu)
        self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_button.setToolTip("Menu (Alt+F)")
        self.menu_button.setAccessibleName("Main menu")
        self.app_icon = QLabel(self)
        self.app_icon.setPixmap(app_icon.pixmap(16, 16))
        self.app_icon.setFixedWidth(28)
        self.app_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.command_buttons: list[QToolButton] = []

        left = QHBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(2)
        left.addWidget(self.menu_button)
        left.addWidget(self.app_icon)
        for action in commands:
            if action is None:
                left.addWidget(self._separator())
                continue
            button = title_button(action, parent=self)
            self.command_buttons.append(button)
            left.addWidget(button)

        self.title = QLabel(self)  # centred over the whole bar in resizeEvent
        self.title.setObjectName("windowTitle")
        self.title.setTextFormat(Qt.TextFormat.RichText)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._title_html = ""
        self._title_text: tuple[str, str] = ("", "")

        self.help_button = title_button(help_action, parent=self)
        self.theme_switch = ThemeSwitch(self)
        right = QHBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(2)
        right.addWidget(self.help_button)
        right.addSpacing(6)
        right.addWidget(self.theme_switch)
        right.addSpacing(8)
        self.min_button = self._caption("subtract", "Minimise", self._minimise)
        self.max_button = self._caption("maximize", "Maximise", self._toggle_maximised)
        self.close_button = self._caption("dismiss", "Close", self._close, close=True)
        for b in (self.min_button, self.max_button, self.close_button):
            b.setVisible(frameless)
            right.addWidget(b)
        if not frameless:
            right.addSpacing(4)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(left)
        layout.addStretch(1)
        layout.addLayout(right)
        self._left, self._right = left, right
        self.apply_theme()

    # -- public ---------------------------------------------------------------------------

    def set_title(self, name: str, detail: str = "") -> None:
        """Centre text: the project name, then the application name in a muted colour."""
        self._title_text = (name, detail)
        self._render_title()

    def update_window_state(self) -> None:
        """Maximise ↔ restore icon and tooltip follow the window state."""
        maximised = self.window().isMaximized()
        self.max_button.setIcon(
            icon("square_multiple" if maximised else "maximize", size=CAPTION_ICON)
        )
        self.max_button.setToolTip("Restore down" if maximised else "Maximise")

    def apply_theme(self) -> None:
        self.menu_button.setIcon(icon("navigation"))
        self.min_button.setIcon(icon("subtract", size=CAPTION_ICON))
        self.close_button.setIcon(icon("dismiss", size=CAPTION_ICON, active="#ffffff"))
        self.update_window_state()
        if self._title_text[0]:
            self._render_title()

    # -- Qt overrides ---------------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().resizeEvent(event)
        self._place_title()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._frameless and event.button() == Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._frameless and event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximised()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # -- internals ------------------------------------------------------------------------

    def _separator(self) -> QFrame:
        line = QFrame(self)
        line.setObjectName("titleSeparator")
        line.setFixedWidth(9)  # 1 px line with 4 px margins (see theme)
        line.setFrameShape(QFrame.Shape.NoFrame)
        line.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        return line

    def _caption(self, name: str, tip: str, slot, *, close: bool = False) -> QToolButton:  # noqa: ANN001
        button = QToolButton(self)
        button.setObjectName("closeButton" if close else "captionButton")
        button.setAutoRaise(True)
        button.setIconSize(QSize(CAPTION_ICON, CAPTION_ICON))
        button.setFixedSize(46, HEIGHT)
        button.setToolTip(tip)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(slot)
        return button

    def _render_title(self) -> None:
        name, detail = self._title_text
        muted = TOKENS["text_muted"]
        text = escape(name)
        if detail:
            text += f"<span style='color:{muted}'>&nbsp;&nbsp;—&nbsp;&nbsp;{escape(detail)}</span>"
        self._title_html = text
        self.title.setText(text)
        self._place_title()

    def _place_title(self) -> None:
        """Centre the title on the window; never overlap the buttons on either side."""
        left_end = self._left.geometry().right() + 16
        right_start = self._right.geometry().left() - 16
        width = self.title.sizeHint().width()
        x = (self.width() - width) // 2
        x = max(x, left_end)
        available = max(right_start - x, 0)
        self.title.setGeometry(x, 0, min(width, available), self.height())
        self.title.setVisible(available > 40)

    def _minimise(self) -> None:
        self.window().showMinimized()

    def _toggle_maximised(self) -> None:
        window = self.window()
        if window.isMaximized():
            window.showNormal()
        else:
            window.showMaximized()

    def _close(self) -> None:
        self.window().close()
