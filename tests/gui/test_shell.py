"""The window shell: title bar, navigation rail, themes, frameless window."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QPushButton

from inferential_studio.infrastructure.settings import AppSettings
from inferential_studio.ui import theme
from inferential_studio.ui.icons import icon, icon_exists
from inferential_studio.ui.main_window import MainWindow
from inferential_studio.ui.stages import STAGES
from inferential_studio.ui.widgets.nav_rail import EXPANDED_WIDTH, RAIL_WIDTH
from inferential_studio.ui.widgets.plots import plot_widget

pytestmark = pytest.mark.gui


@pytest.fixture
def window(qtbot, monkeypatch):
    w = MainWindow(AppSettings())
    monkeypatch.setattr(w, "maybe_save", lambda: True)
    qtbot.addWidget(w)
    w.resize(1280, 800)
    w.show()
    qtbot.waitExposed(w)
    yield w
    theme.apply_theme(QApplication.instance(), theme.LIGHT)  # leave the light theme behind
    w.session._set_dirty(False)


def test_every_icon_in_use_is_bundled(window):
    names = {s.icon for s in STAGES} | set(window._action_icons.values())
    names |= {"navigation", "subtract", "maximize", "square_multiple", "dismiss", "history",
              "panel_left_expand", "panel_left_contract", "weather_sunny", "weather_moon"}  # fmt: skip
    missing = sorted(n for n in names if not icon_exists(n))
    assert missing == []
    assert not icon("save").isNull()
    assert icon("save") is icon("save")  # cached


def test_rail_lists_stages_in_order_without_numbers(window):
    rail = window.nav
    assert [rail.list.item(i).text() for i in range(rail.list.count())] == [s.title for s in STAGES]
    assert rail.panel_width == RAIL_WIDTH and not rail.expanded
    assert all(not rail.list.item(i).icon().isNull() for i in range(rail.list.count()))


def test_rail_slides_over_the_page_without_moving_it(qtbot, window):
    rail, layer = window.nav, window.content_layer
    before = layer.geometry()
    page_before = window.stack.currentWidget().geometry()
    rail.toggle.click()  # animated
    qtbot.waitUntil(lambda: rail.panel_width == EXPANDED_WIDTH, timeout=2000)
    assert rail.expanded and rail.width() > EXPANDED_WIDTH  # panel + soft edge
    # The page is covered, never resized or moved: nothing reflows or redraws its charts.
    assert layer.geometry() == before
    assert window.stack.currentWidget().geometry() == page_before
    # Choosing a stage navigates and closes the panel.
    item = rail.list.item(3)
    rect = rail.list.visualItemRect(item)
    qtbot.mouseClick(rail.list.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert window.stack.currentIndex() == 3
    qtbot.waitUntil(lambda: rail.panel_width == RAIL_WIDTH, timeout=2000)
    assert not rail.expanded and rail.width() == RAIL_WIDTH


def test_rail_closes_on_escape_and_click_outside(qtbot, window):
    rail = window.nav
    rail.set_expanded(True, animate=False)
    QApplication.sendEvent(
        window.stack,
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier),
    )
    assert not rail.expanded
    rail.set_expanded(True, animate=False)
    page = window.stack.currentWidget()
    qtbot.mouseClick(page, Qt.MouseButton.LeftButton, pos=QPoint(page.width() - 20, 40))
    assert not rail.expanded
    rail.set_expanded(True, animate=False)  # a click inside the panel keeps it open
    qtbot.mouseClick(rail, Qt.MouseButton.LeftButton, pos=QPoint(100, 20))
    assert rail.expanded
    # The click that closes the panel does nothing else on the page beneath it…
    clicks = []
    button = QPushButton("Probe", page)
    button.setGeometry(page.width() - 120, 60, 100, 30)
    button.show()
    button.clicked.connect(lambda: clicks.append(1))
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert not rail.expanded and clicks == []
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)  # …and the page works normally after
    assert clicks == [1]
    # A click in the soft edge beside the panel closes it too.
    rail.set_expanded(True, animate=False)
    qtbot.mouseClick(rail, Qt.MouseButton.LeftButton, pos=QPoint(rail.width() - 4, 200))
    assert not rail.expanded
    # The title bar keeps working while the panel is open (and closes it).
    rail.set_expanded(True, animate=False)
    shown = window.act_help_panel.isChecked()
    qtbot.mouseClick(window.title_bar.help_button, Qt.MouseButton.LeftButton)
    assert not rail.expanded and window.act_help_panel.isChecked() != shown


def test_theme_switch_recolours_the_whole_window(qtbot, window):
    app = QApplication.instance()
    theme.apply_theme(app, theme.LIGHT)
    window.set_theme(theme.LIGHT)
    chart = window.pages["validation"].avp
    assert chart.backgroundBrush().color().name() == theme.THEMES[theme.LIGHT]["chart_bg"]
    window.title_bar.theme_switch.click()  # → dark, once the knob has slid across
    qtbot.waitUntil(lambda: theme.current_theme() == theme.DARK, timeout=2000)
    assert window.settings.theme == theme.DARK
    assert "#272727" in window.styleSheet() and not QApplication.instance().styleSheet()
    assert theme.TOKENS["bg"] == theme.THEMES[theme.DARK]["bg"]
    assert (
        app.palette().color(app.palette().ColorRole.Window).name() == theme.THEMES[theme.DARK]["bg"]
    )
    assert chart.backgroundBrush().color().name() == theme.THEMES[theme.DARK]["chart_bg"]
    axis = chart.getPlotItem().getAxis("left")
    assert axis.textPen().color().name() == theme.THEMES[theme.DARK]["chart_fg"]
    # Charts made after the switch start in the new theme too.
    fresh = plot_widget("New chart")
    qtbot.addWidget(fresh)
    assert fresh.backgroundBrush().color().name() == theme.THEMES[theme.DARK]["chart_bg"]
    window.title_bar.theme_switch.click()  # → light again
    qtbot.waitUntil(lambda: theme.current_theme() == theme.LIGHT, timeout=2000)
    assert window.settings.theme == theme.LIGHT
    assert chart.backgroundBrush().color().name() == theme.THEMES[theme.LIGHT]["chart_bg"]


def test_resolve_theme_follows_setting_or_system():
    assert theme.resolve_theme("dark") == theme.DARK
    assert theme.resolve_theme("light") == theme.LIGHT
    assert theme.resolve_theme("auto") in (theme.LIGHT, theme.DARK)


def test_frameless_window_has_grips_and_caption_buttons(qtbot, window):
    assert window.frameless
    assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert window.windowFlags() & Qt.WindowType.WindowMinimizeButtonHint  # taskbar minimise
    bar = window.title_bar
    assert (
        bar.min_button.isVisible() and bar.max_button.isVisible() and bar.close_button.isVisible()
    )
    grips = window.grips.grips
    assert len(grips) == 8 and all(g.isVisible() for g in grips.values())
    assert grips["right"].geometry().right() == window.width() - 1
    assert grips["bottom_right"].cursor().shape() == Qt.CursorShape.SizeFDiagCursor
    window.showMaximized()
    qtbot.waitUntil(lambda: window.isMaximized(), timeout=2000)
    assert not any(g.isVisible() for g in grips.values())  # nothing to resize when maximised
    assert bar.max_button.toolTip() == "Restore down"
    window.showNormal()
    qtbot.waitUntil(lambda: not window.isMaximized(), timeout=2000)
    assert all(g.isVisible() for g in grips.values())


def test_system_title_bar_setting_keeps_the_native_frame(qtbot, monkeypatch):
    w = MainWindow(AppSettings(system_title_bar=True))
    monkeypatch.setattr(w, "maybe_save", lambda: True)
    qtbot.addWidget(w)
    w.show()
    assert not w.frameless and w.grips is None
    assert not (w.windowFlags() & Qt.WindowType.FramelessWindowHint)
    assert not w.title_bar.close_button.isVisible()
    assert w.title_bar.command_buttons  # the command bar is still there
    w.session._set_dirty(False)


def test_main_menu_opens_with_alt_f(window, monkeypatch):
    from PySide6.QtGui import QShortcut

    opened = []
    monkeypatch.setattr(window.title_bar.menu_button, "showMenu", lambda: opened.append(1))
    shortcuts = [s for s in window.findChildren(QShortcut) if s.key().toString() == "Alt+F"]
    assert len(shortcuts) == 1
    shortcuts[0].activated.emit()
    assert opened == [1]


def test_quick_double_toggle_applies_only_the_final_theme(qtbot, window):
    theme.apply_theme(QApplication.instance(), theme.LIGHT)
    window.set_theme(theme.LIGHT)
    applied = []
    original = window.set_theme
    window.set_theme = lambda name, **kw: (applied.append(name), original(name, **kw))
    switch = window.title_bar.theme_switch
    switch.click()
    switch.click()  # changed their mind before the window recoloured
    qtbot.waitUntil(lambda: not window._theme_timer.isActive(), timeout=2000)
    assert applied == [theme.LIGHT] and theme.current_theme() == theme.LIGHT
    assert not switch.isChecked()


def test_settings_dialog_sets_theme_and_title_bar_and_keeps_the_rest(qtbot, window, monkeypatch):
    from inferential_studio.ui import main_window as mw
    from inferential_studio.ui.settings_dialog import SettingsDialog

    window.settings.window_geometry = "abc"
    window.settings.recent_projects = ["C:/p.imsproj"]
    dialog = SettingsDialog(window.settings, window)
    qtbot.addWidget(dialog)
    assert dialog.theme.currentData() == window.settings.theme
    dialog.theme.setCurrentIndex(dialog.theme.findData("dark"))
    dialog.system_title_bar.setChecked(True)
    result = dialog.result_settings()
    assert (result.theme, result.system_title_bar) == ("dark", True)
    assert result.window_geometry == "abc" and result.recent_projects == ["C:/p.imsproj"]
    assert result.recent_projects is not window.settings.recent_projects

    saved = []
    monkeypatch.setattr(mw, "save_settings", lambda s: saved.append(s))
    monkeypatch.setattr(mw.SettingsDialog, "exec", lambda self: True)
    monkeypatch.setattr(mw.SettingsDialog, "result_settings", lambda self: result)
    window._settings_dialog()
    assert theme.current_theme() == theme.DARK and window.title_bar.theme_switch.isChecked()
    assert window.settings.theme == "dark" and window.settings.system_title_bar
    assert window.frameless  # the frame changes at the next start
    assert saved == [window.settings]


def test_auto_theme_follows_the_system(window, monkeypatch):
    window.set_theme(theme.LIGHT, setting="auto")
    from inferential_studio.ui import main_window as mw

    monkeypatch.setattr(mw, "resolve_theme", lambda s: theme.DARK)
    window._system_theme_changed()
    assert theme.current_theme() == theme.DARK and window.settings.theme == "auto"
    window.set_theme(theme.LIGHT)  # an explicit choice stops following the system
    window._system_theme_changed()
    assert theme.current_theme() == theme.LIGHT and window.settings.theme == theme.LIGHT


def test_rail_keyboard_choice_closes_the_panel(qtbot, window):
    rail = window.nav
    rail.set_expanded(True, animate=False)
    QApplication.sendEvent(
        rail.list,
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.NoModifier),
    )
    # The keyboard picks the stage (the page follows currentRowChanged) and the panel closes.
    assert window.stack.currentIndex() == 1
    assert not rail.expanded


def test_escape_reaches_an_open_dialog(qtbot, window):
    from PySide6.QtWidgets import QDialog

    rail = window.nav
    rail.set_expanded(True, animate=False)
    dialog = QDialog(window)
    qtbot.addWidget(dialog)
    dialog.open()  # modal
    qtbot.waitUntil(lambda: QApplication.activeModalWidget() is dialog, timeout=2000)
    QApplication.sendEvent(
        dialog,
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier),
    )
    assert not dialog.isVisible()  # the dialog took the Esc, the rail stayed out of the way
    assert rail.expanded
    QApplication.sendEvent(
        window.stack,
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier),
    )
    assert not rail.expanded


def test_long_project_names_are_elided(qtbot, window):
    window.resize(960, 640)
    window.session.project.name = "Counter-current distillation top temperature soft sensor "
    window._update_title()
    label = window.title_bar.title
    assert label.isVisible() and "…" in label.text()
    assert label.geometry().right() <= window.width()
    window.session.project.name = "Plant A"  # short names fit in full
    window._update_title()
    assert "…" not in window.title_bar.title.text()


def test_theme_choice_survives_a_quick_close(qtbot, window, monkeypatch):
    from inferential_studio.ui import main_window as mw

    saved = []
    monkeypatch.setattr(mw, "save_settings", lambda s: saved.append(s))
    theme.apply_theme(QApplication.instance(), theme.LIGHT)
    window.set_theme(theme.LIGHT)
    window.title_bar.theme_switch.click()  # → dark, once the knob animation has run
    assert window._theme_timer.isActive()
    window.close()  # before the timer fires: the choice must still be persisted
    assert window.settings.theme == theme.DARK
    assert saved == [window.settings]


def test_caption_buttons_sit_above_the_resize_grips(window):
    close = window.title_bar.close_button
    for offset in (QPoint(23, 2), QPoint(10, 38)):  # top edge and middle of the button
        assert window.childAt(close.mapTo(window, offset)) is close


def test_menu_highlight_differs_from_the_menu_background():
    for name in (theme.LIGHT, theme.DARK):
        t = theme.THEMES[name]
        assert t["menu_hover"] != t["surface"]  # the highlighted row must read on menus


def test_accessible_shell_names_states_and_keyboard(qtbot, window):
    from PySide6.QtGui import QAccessible

    bar = window.title_bar

    def accessible(widget):
        interface = QAccessible.queryAccessibleInterface(widget)
        assert interface is not None
        return interface

    switch = accessible(bar.theme_switch)
    assert switch.text(QAccessible.Text.Name) == "Dark theme"
    assert switch.role() == QAccessible.Role.CheckBox
    bar.theme_switch.set_dark(False)
    assert not switch.state().checked
    bar.theme_switch.setFocus(Qt.FocusReason.TabFocusReason)
    qtbot.keyClick(bar.theme_switch, Qt.Key.Key_Space)
    assert switch.state().checked

    toggle = accessible(window.nav.toggle)
    assert toggle.text(QAccessible.Text.Name) == "Show stage names"
    window.nav.toggle.setFocus(Qt.FocusReason.TabFocusReason)
    qtbot.keyClick(window.nav.toggle, Qt.Key.Key_Space)
    assert window.nav.expanded
    assert toggle.text(QAccessible.Text.Name) == "Hide stage names"

    for button, name in ((bar.min_button, "Minimise"), (bar.max_button, "Maximise"),
                         (bar.close_button, "Close")):
        interface = accessible(button)
        assert interface.text(QAccessible.Text.Name) == name
        assert interface.role() == QAccessible.Role.Button
        assert interface.state().focusable
        assert button.focusPolicy() == Qt.FocusPolicy.TabFocus

    bar.min_button.setFocus(Qt.FocusReason.TabFocusReason)
    qtbot.keyClick(bar.min_button, Qt.Key.Key_Tab)
    assert bar.max_button.hasFocus()
    qtbot.keyClick(bar.max_button, Qt.Key.Key_Tab)
    assert bar.close_button.hasFocus()
    for button in [bar.menu_button, bar.help_button, *bar.command_buttons]:
        assert accessible(button).text(QAccessible.Text.Name)

    window.showMaximized()
    qtbot.waitUntil(lambda: window.isMaximized())
    bar.update_window_state()
    assert accessible(bar.max_button).text(QAccessible.Text.Name) == "Restore down"
    bar.max_button.setFocus(Qt.FocusReason.TabFocusReason)
    qtbot.keyClick(bar.max_button, Qt.Key.Key_Space)
    qtbot.waitUntil(lambda: not window.isMaximized())
    assert accessible(bar.max_button).text(QAccessible.Text.Name) == "Maximise"


@pytest.mark.parametrize("name", [theme.LIGHT, theme.DARK])
def test_primary_button_enabled_text_contrast(name):
    def luminance(colour):
        channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                  for c in channels]
        return sum(c * weight for c, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

    palette = theme.THEMES[name]
    foreground = luminance(palette["on_accent"])
    for state in ("accent", "accent_hover", "accent_pressed"):
        background = luminance(palette[state])
        ratio = (max(foreground, background) + 0.05) / (min(foreground, background) + 0.05)
        assert ratio >= 4.5, (name, state, ratio)
