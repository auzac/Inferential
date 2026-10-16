"""Main application window: one seamless surface in the Windows 11 style.

Title bar (main menu, command icons, project name, help toggle, theme switch, window buttons),
a navigation rail that slides out over the pages, the page stack with an optional help pane,
the message panel and the status bar. The window draws its own frame unless the user chose the
system title bar in Settings (see ``widgets/window_frame.py``).
"""

from __future__ import annotations

import logging
import platform
from dataclasses import replace
from pathlib import Path

import PySide6
from PySide6.QtCore import QByteArray, QEvent, QSize, Qt, QTimer, QUrl, Slot
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QPainter,
    QPaintEvent,
    QShortcut,
    QUndoStack,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from inferential_studio import APP_NAME, __version__
from inferential_studio.core import synthetic
from inferential_studio.core.dataset import ProjectData, Severity, SourceData
from inferential_studio.core.importers import SourceChangedError, import_source, load_project_data
from inferential_studio.core.models import train_models
from inferential_studio.core.persistence import PROJECT_EXTENSION, ProjectFileError
from inferential_studio.core.quality import run_detectors
from inferential_studio.core.schemas import Project, SchemaError, SourceKind
from inferential_studio.infrastructure.logging_config import LOG_FILE_NAME
from inferential_studio.infrastructure.paths import cache_dir, log_dir, resource_path
from inferential_studio.infrastructure.recovery import (
    RecoveryWriter,
    discard_copy,
    find_recovery_copies,
)
from inferential_studio.infrastructure.settings import AppSettings, save_settings
from inferential_studio.infrastructure.workers import TaskManager
from inferential_studio.ui.help import GuideDialog, HelpPanel
from inferential_studio.ui.icons import icon
from inferential_studio.ui.import_wizard import ImportPlan, ImportWizard
from inferential_studio.ui.messages import APP
from inferential_studio.ui.pages.cleaning_page import CleaningPage
from inferential_studio.ui.pages.eda_page import EdaPage
from inferential_studio.ui.pages.export_page import ExportPage
from inferential_studio.ui.pages.import_page import ImportPage
from inferential_studio.ui.pages.model_setup_page import ModelSetupPage
from inferential_studio.ui.pages.processing_page import ProcessingPage
from inferential_studio.ui.pages.stage_page import PlaceholderPage, StagePage
from inferential_studio.ui.pages.training_page import TrainingPage
from inferential_studio.ui.pages.validation_page import ValidationPage
from inferential_studio.ui.pages.variables_page import VariablesPage
from inferential_studio.ui.project_dialog import ProjectSettingsDialog
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.settings_dialog import SettingsDialog
from inferential_studio.ui.stages import STAGES, stage_states
from inferential_studio.ui.theme import (
    DARK,
    LIGHT,
    TOKENS,
    apply_theme,
    current_theme,
    refresh_widgets,
    resolve_theme,
    stylesheet,
)
from inferential_studio.ui.widgets.message_panel import ROWS_COMPACT, ROWS_VISIBLE, MessagePanel
from inferential_studio.ui.widgets.nav_rail import RAIL_WIDTH, NavRail
from inferential_studio.ui.widgets.task_progress import TaskProgressWidget
from inferential_studio.ui.widgets.title_bar import TitleBar
from inferential_studio.ui.widgets.window_frame import (
    EdgeGrips,
    round_windows_corners,
    set_windows_border,
)

log = logging.getLogger(__name__)
PROJECT_FILTER = f"Inferential Model Studio project (*{PROJECT_EXTENSION})"
COMPACT_SCREEN_HEIGHT = 900  # px available; below this the message panel shows one row less
WIDE_SCREEN_WIDTH = 1600  # px available; narrower screens start with the help panel closed
THEME_SWITCH_DELAY_MS = 150  # the switch's knob animation runs first


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.grips = None  # EdgeGrips after the central widgets; early events must see it
        self.title_bar = None  # built by _build_title_bar; _update_frame guards for early events
        self.settings = settings
        self.session = ProjectSession(self)
        self.tasks = TaskManager(self)
        self.session.attach_tasks(self.tasks)  # exclusion masks recompute in the background
        self.undo_stack = QUndoStack(self)
        self.setMinimumSize(QSize(960, 640))
        # One seamless surface: the window draws its own title bar (unless the user prefers the
        # system's); minimise/maximise hints keep taskbar minimise and snapping on Windows.
        self.frameless = not settings.system_title_bar
        if self.frameless:
            self.setWindowFlags(
                Qt.WindowType.Window
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowSystemMenuHint
                | Qt.WindowType.WindowMinMaxButtonsHint
            )
        self._system_border = False  # Windows 11 draws the border and rounds the corners
        self.setStyleSheet(stylesheet())  # before the widgets exist: each is styled once
        self._build_actions()
        self._build_title_bar()
        self._build_central()
        self._build_status_bar()
        self.grips = EdgeGrips(self) if self.frameless else None
        self._restore_window()
        self._update_frame()
        self.session.changed.connect(self._refresh_status)
        self.session.data_changed.connect(self._refresh_status)
        self.session.cleaning_changed.connect(self._refresh_status)
        self.session.detectors_changed.connect(self.run_detection)
        self.session.data_changed.connect(self._data_changed_detection)
        self._load_generation = 0
        self.session.dirty_changed.connect(lambda _d: self._update_title())
        self.session.path_changed.connect(lambda _p: self._update_title())
        self.tasks.busy_changed.connect(self._on_busy_changed)
        self._reproduce_pending = False  # reopened project with saved results: re-train once
        self.session.features_changed.connect(self._maybe_reproduce)
        # Crash recovery: a copy of unsaved work every minute.
        self.recovery = RecoveryWriter()
        self._recovery_timer = QTimer(self)
        self._recovery_timer.setInterval(60_000)
        self._recovery_timer.timeout.connect(self._write_recovery)
        self._recovery_timer.start()
        self.session.new(Project(display_timezone=settings.display_timezone))

    # ------------------------------------------------------------------ construction

    def _build_actions(self) -> None:
        def action(text: str, slot, shortcut=None, tip: str = "") -> QAction:  # type: ignore[no-untyped-def]
            act = QAction(text, self)
            if shortcut is not None:
                act.setShortcut(shortcut)
            if tip:
                act.setToolTip(tip)
                act.setStatusTip(tip)
            act.triggered.connect(slot)
            return act

        self.act_new = action(
            "&New project",
            self.new_project,
            QKeySequence.StandardKey.New,
            "Start a new empty project",
        )
        self.act_open = action(
            "&Open…", self._open_dialog, QKeySequence.StandardKey.Open, "Open a saved project"
        )
        self.act_save = action(
            "&Save", self._save, QKeySequence.StandardKey.Save, "Save the project"
        )
        self.act_save_as = action(
            "Save &as…",
            self._save_as_dialog,
            QKeySequence.StandardKey.SaveAs,
            "Save the project under a new name",
        )
        self.act_import = action(
            "&Import data…",
            lambda: self.import_dialog(),
            QKeySequence("Ctrl+I"),
            "Import data from CSV or Excel — process, lab, or both in one file",
        )
        self.act_export = action(
            "&Export…",
            lambda: self.sidebar.setCurrentRow(self._stage_row("export")),
            QKeySequence("Ctrl+E"),
            "Export predictions, the model card and the model file",
        )
        self.act_save_portable = action(
            "Save &portable copy…",
            self._save_portable_dialog,
            None,
            "Save a project file that contains the data files too (for sharing or archiving)",
        )
        self.act_exit = action("E&xit", self.close, QKeySequence.StandardKey.Quit)
        self.act_project = action(
            "&Project settings…",
            self._project_dialog,
            None,
            "Project name, display timezone and notes",
        )
        self.act_settings = action(
            "&Settings…",
            self._settings_dialog,
            QKeySequence.StandardKey.Preferences,
            "Application settings",
        )
        self.act_logs = action(
            "Open &log folder", self._open_log_folder, None, "Show the application log files"
        )
        self.act_about = action(f"&About {APP_NAME}", self._about)
        self.act_example = action(
            "Generate &example dataset…",
            lambda: self._generate_example_dialog(),
            None,
            "Create an artificial distillation dataset and a matching sample project",
        )
        self.act_guide = action(
            "&User guide",
            self.show_guide,
            QKeySequence.StandardKey.HelpContents,
            "Open the user guide",
        )
        self.act_undo = self.undo_stack.createUndoAction(self, "&Undo")
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_redo = self.undo_stack.createRedoAction(self, "&Redo")
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.act_help_panel = action(
            "Help &panel",
            lambda checked: self._show_help_pane(checked),
            QKeySequence("Shift+F1"),
            "Show or hide the stage help on the right",
        )
        self.act_help_panel.setCheckable(True)
        # Icons (recoloured on a theme switch, see apply_theme) and tooltips with shortcuts.
        self._action_icons = {
            self.act_new: "document_add",
            self.act_open: "folder_open",
            self.act_save: "save",
            self.act_save_as: "save_edit",
            self.act_save_portable: "save_copy",
            self.act_import: "arrow_import",
            self.act_export: "share",
            self.act_undo: "arrow_undo",
            self.act_redo: "arrow_redo",
            self.act_project: "document_settings",
            self.act_settings: "settings",
            self.act_logs: "folder",
            self.act_guide: "question_circle",
            self.act_example: "beaker",
            self.act_about: "info",
            self.act_exit: "arrow_exit",
            self.act_help_panel: "panel_right",
        }
        for act in self._action_icons:
            name = act.text().replace("&", "").rstrip("…")
            if act in (self.act_undo, self.act_redo):
                name = act.text().replace("&", "")  # "Undo Exclude interval"
            shortcut = act.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
            act.setToolTip(f"{name} ({shortcut})" if shortcut else name)
        # Undo/redo tooltips name the command ("Undo Exclude interval (Ctrl+Z)").
        self.undo_stack.undoTextChanged.connect(self._undo_tooltips)
        self.undo_stack.redoTextChanged.connect(self._undo_tooltips)
        self._apply_action_icons()
        # The window has no menu bar: register the shortcuts on the window itself.
        self.addActions(list(self._action_icons))

    def _apply_action_icons(self) -> None:
        for act, name in self._action_icons.items():
            act.setIcon(icon(name))
        # The sub-menus keep the icon they were created with; colour them like their actions.
        if hasattr(self, "recent_menu"):
            self.recent_menu.setIcon(icon("history"))
        if hasattr(self, "help_menu"):
            self.help_menu.setIcon(icon("question_circle"))

    def _undo_tooltips(self, _text: str = "") -> None:
        try:
            for act in (self.act_undo, self.act_redo):
                shortcut = act.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
                text = act.text().replace("&", "")
                act.setToolTip(f"{text} ({shortcut})" if shortcut else text)
        except RuntimeError:  # the window is being destroyed
            pass

    def _build_title_bar(self) -> None:
        """Main menu (≡) with every command, plus icon buttons for the frequent ones."""
        menu = QMenu(self)
        menu.setObjectName("mainMenu")
        menu.addActions([self.act_new, self.act_open])
        self.recent_menu = menu.addMenu(icon("history"), "Open &recent")
        self.recent_menu.aboutToShow.connect(self._populate_recent)
        menu.addSeparator()
        menu.addActions([self.act_save, self.act_save_as, self.act_save_portable])
        menu.addSeparator()
        menu.addActions([self.act_import, self.act_export])
        menu.addSeparator()
        menu.addActions([self.act_undo, self.act_redo])
        menu.addSeparator()
        menu.addActions([self.act_project, self.act_settings])
        help_menu = menu.addMenu(icon("question_circle"), "&Help")
        help_menu.addActions([self.act_guide, self.act_help_panel, self.act_example])
        help_menu.addSeparator()
        help_menu.addActions([self.act_logs, self.act_about])
        menu.addSeparator()
        menu.addAction(self.act_exit)
        self.main_menu, self.help_menu = menu, help_menu
        app_icon = QIcon(str(resource_path("assets/icons/app.png")))
        commands = [
            self.act_open,
            self.act_save,
            self.act_save_as,
            self.act_import,
            None,
            self.act_undo,
            self.act_redo,
        ]
        self.title_bar = TitleBar(
            app_icon, menu, commands, self.act_help_panel, frameless=self.frameless, parent=self
        )
        self.title_bar.theme_switch.set_dark(current_theme() == DARK)
        # Let the knob finish sliding before the (brief) recolouring of the window; quick
        # repeated clicks restart the wait, so only the final state is applied. (toggled
        # carries the check state: without the lambda it would become the timer's interval.)
        self._theme_timer = QTimer(self, singleShot=True, interval=THEME_SWITCH_DELAY_MS)
        self._theme_timer.timeout.connect(self._apply_switch_theme)
        self.title_bar.theme_switch.toggled.connect(
            lambda _checked: self._theme_timer.start(THEME_SWITCH_DELAY_MS)
        )
        app = QApplication.instance()
        if app is not None:  # "auto" follows Windows when its light/dark mode changes
            app.styleHints().colorSchemeChanged.connect(self._system_theme_changed)
        self.setMenuWidget(self.title_bar)
        self._apply_action_icons()  # sub-menu icons were not built yet at the first pass
        QShortcut(QKeySequence("Alt+F"), self, activated=self._open_main_menu)

    def _open_main_menu(self) -> None:
        self.title_bar.menu_button.showMenu()

    def _build_central(self) -> None:
        self.stack = QStackedWidget()
        self.pages: dict[str, StagePage] = {}
        for stage in STAGES:
            if stage.key == "import":
                page: StagePage = ImportPage(stage, self.session)
                page.generate_example_requested.connect(self._generate_example_dialog)
                page.open_project_requested.connect(self._open_dialog)
                page.import_requested.connect(self.import_dialog)
                page.remove_requested.connect(self._remove_source)
            elif stage.key == "initial_eda":
                page = EdaPage(stage, self.session, self.tasks)
            elif stage.key == "cleaning":
                page = CleaningPage(stage, self.session, self.undo_stack, self.settings)
                page.detection_requested.connect(self.run_detection)
            elif stage.key == "variables":
                page = VariablesPage(stage, self.session, self.tasks, self.undo_stack)
            elif stage.key == "processing":
                page = ProcessingPage(stage, self.session, self.tasks, self.undo_stack)
            elif stage.key == "model_setup":
                page = ModelSetupPage(stage, self.session, self.undo_stack)
                page.train_requested.connect(self.run_training)
            elif stage.key == "training":
                page = TrainingPage(stage, self.session)
                page.train_requested.connect(self.run_training)
            elif stage.key == "validation":
                page = ValidationPage(stage, self.session)
            elif stage.key == "export":
                page = ExportPage(stage, self.session, self.tasks, self.undo_stack, self.settings)
            else:
                page = PlaceholderPage(stage)
            self.pages[stage.key] = page
            self.stack.addWidget(page)

        # Help pane (right) and message panel (bottom) share the content layer with the pages.
        self.help_panel = HelpPanel()
        pane = QWidget()
        pane.setObjectName("helpPane")
        pane.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        pane.setMinimumWidth(220)
        head = QHBoxLayout()
        head.setContentsMargins(12, 6, 4, 0)
        title = QLabel("Help")
        title.setObjectName("helpPaneTitle")
        self.help_close = QToolButton()
        self.help_close.setObjectName("titleButton")
        self.help_close.setAutoRaise(True)
        self.help_close.setToolTip("Hide the help panel (Shift+F1)")
        self.help_close.clicked.connect(lambda: self._show_help_pane(False))
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.help_close)
        pl = QVBoxLayout(pane)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(0)
        pl.addLayout(head)
        pl.addWidget(self.help_panel, 1)
        self.properties_dock = pane  # kept name: the help pane

        self.content_split = QSplitter(Qt.Orientation.Horizontal)
        self.content_split.addWidget(self.stack)
        self.content_split.addWidget(pane)
        self.content_split.setStretchFactor(0, 1)
        self.content_split.setCollapsible(0, False)
        self.content_split.setCollapsible(1, False)
        self.content_split.setSizes([1000, 270])

        screen = QGuiApplication.primaryScreen()
        short = screen is not None and screen.availableGeometry().height() < COMPACT_SCREEN_HEIGHT
        self.message_panel = MessagePanel(
            self.session.messages, rows=ROWS_COMPACT if short else ROWS_VISIBLE
        )

        layer = QFrame()
        layer.setObjectName("contentLayer")
        ll = QVBoxLayout(layer)
        ll.setContentsMargins(1, 1, 0, 0)  # keep the rounded corner and edge visible
        ll.setSpacing(0)
        ll.addWidget(self.content_split, 1)
        ll.addWidget(self.message_panel)
        self.content_layer = layer

        shell = _Shell(layer)
        self.nav = NavRail(STAGES, shell)
        shell.set_rail(self.nav)
        self.sidebar = self.nav.list  # stage list (one row per stage)
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.sidebar.currentRowChanged.connect(
            lambda row: self.help_panel.show_stage(STAGES[row].key) if row >= 0 else None
        )
        self.sidebar.currentRowChanged.connect(
            lambda row: self.message_panel.set_stage(STAGES[row].key) if row >= 0 else None
        )
        self.sidebar.currentRowChanged.connect(self._stage_visited)
        s = self.session
        for signal in (s.changed, s.data_changed, s.cleaning_changed, s.inputs_changed,
                       s.modelling_changed, s.features_changed, s.training_changed,
                       s.progress_changed):  # fmt: skip
            signal.connect(self._update_progress)
        for page in self.pages.values():
            page.progress.stage_clicked.connect(self.sidebar.setCurrentRow)
        self.sidebar.setCurrentRow(0)
        self._update_progress()
        self.help_panel.show_stage(STAGES[0].key)
        self.message_panel.set_stage(STAGES[0].key)
        self.help_close.setIcon(icon("dismiss", size=16))
        self.setCentralWidget(shell)

    def _show_help_pane(self, shown: bool) -> None:
        self.properties_dock.setVisible(shown)
        if self.act_help_panel.isChecked() != shown:
            self.act_help_panel.setChecked(shown)

    def notify(self, text: str, source: str = APP) -> None:
        """An informational event (formerly a transient status-bar message)."""
        self.session.messages.info(text, source)

    def _build_status_bar(self) -> None:
        status = self.statusBar()
        # Frameless: every edge of the window resizes it. System frame: keep the OS size grip.
        status.setSizeGripEnabled(not self.frameless)
        self.lbl_records = QLabel()
        self.lbl_samples = QLabel()
        self.lbl_exclusions = QLabel()
        for label in (self.lbl_records, self.lbl_samples, self.lbl_exclusions):
            status.addWidget(label)
        self.task_widget = TaskProgressWidget(self.tasks)
        status.addPermanentWidget(self.task_widget)

    # ------------------------------------------------------------------ state display

    def _update_title(self) -> None:
        self.setWindowTitle(f"{self.session.display_name}[*] — {APP_NAME}")
        self.setWindowModified(self.session.dirty)
        name = self.session.project.name or "Untitled project"
        self.title_bar.set_title(
            name + (" (unsaved changes)" if self.session.dirty else ""), APP_NAME
        )
        path = self.session.path
        # The label is transparent to mouse events (dragging works everywhere), so the
        # tooltip sits on the bar: hovering the title shows the full project path.
        self.title_bar.setToolTip(str(path) if path else "Not saved yet")

    @Slot()
    def _refresh_status(self) -> None:
        project, data = self.session.project, self.session.data
        loaded = bool(data.sources)
        self.lbl_records.setText(
            f"Process records: {data.n_process_rows:,}" if loaded else "Process records: —"
        )
        self.lbl_samples.setText(
            f"Lab samples: {data.n_lab_samples:,}" if loaded else "Lab samples: —"
        )
        state = self.session.cleaning
        if state is None:
            self.lbl_exclusions.setText(f"Manual exclusions: {len(project.exclusions)}")
        else:
            n, regime = len(state.index), int(state.regime.sum())
            self.lbl_exclusions.setText(
                f"Excluded rows: {regime:,} ({100 * regime / max(n, 1):.1f}%)"
            )
        self._update_title()

    @Slot(bool)
    def _on_busy_changed(self, busy: bool) -> None:
        # Project-changing actions are disabled while a background task runs.
        for act in (self.act_new, self.act_open, self.act_save, self.act_save_as, self.act_import):
            act.setEnabled(not busy)

    def show_error(self, title: str, message: str, details: str = "") -> None:
        """Report a problem in the message panel (double-click it for the details)."""
        stage = STAGES[max(self.sidebar.currentRow(), 0)].key
        self.session.messages.error(f"{title}: {message}", stage, details)

    # ------------------------------------------------------------------ project commands

    def maybe_save(self) -> bool:
        """Ask about unsaved changes. Returns True if it is OK to discard the current project."""
        if not self.session.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            f"Save changes to '{self.session.display_name}'?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self._save()
        return answer == QMessageBox.StandardButton.Discard

    @Slot()
    def new_project(self) -> None:
        if not self.maybe_save():
            return
        self.session.new(Project(display_timezone=self.settings.display_timezone))
        self.undo_stack.clear()
        self.recovery.discard()

    def open_project_file(self, path: str | Path) -> bool:
        try:
            self.session.open(path, cache_dir=cache_dir())
        except (ProjectFileError, OSError) as exc:
            log.warning("Could not open project %s: %s", path, exc)
            self.show_error("Cannot open project", str(exc))
            return False
        self.undo_stack.clear()
        self._remember(path)
        log.info("Opened project %s (%s)", path, self.session.storage)
        self.notify(
            f"Opened {Path(path).name}"
            + (" (portable)" if self.session.storage == "portable" else "")
        )
        self._reproduce_pending = self.session.saved_results is not None
        self.recovery.discard()
        self.load_data()
        return True

    # ------------------------------------------------------------------ data loading and import

    def load_data(self, *, verify_checksum: bool = True, only: list[str] | None = None) -> None:
        """Read the project's source files in the background (reference mode)."""
        project = self.session.project
        if not project.sources:
            return
        self._load_generation += 1
        generation = self._load_generation
        target = project
        project_path = self.session.path
        if self.session.embedded_paths:  # portable project: read the verified extracted copies
            target = replace(
                project,
                sources=[
                    replace(s, path=str(self.session.embedded_paths.get(s.source_id, s.path)))
                    for s in project.sources
                ],
            )
        if only is not None:
            target = Project(
                name=project.name,
                display_timezone=project.display_timezone,
                sources=[s for s in target.sources if s.source_id in only],
                variables=[v for v in project.variables if v.source_id in only],
            )
        self.tasks.submit(
            "Loading project data",
            load_project_data,
            target,
            project_path,
            verify_checksum=verify_checksum,
            on_result=lambda data: self._data_loaded(data, generation, partial=only is not None),
            on_error=lambda msg, details: self.show_error("Cannot load data", msg, details),
        )

    def _data_loaded(self, data: ProjectData, generation: int, *, partial: bool) -> None:
        if generation != self._load_generation:
            return  # a different project was opened meanwhile
        if partial:
            for sd in data.sources.values():
                self.session.accept_changed_source(sd)
            for sid, exc in data.failures.items():
                self.session.data.failures[sid] = exc
            if data.failures:
                self.session.data_changed.emit()
        else:
            self.session.set_data(data)
        if not data.failures:
            self.notify(f"Loaded {len(data.sources)} data source(s).")
            return
        lines = [f"• {sid}: {exc}" for sid, exc in data.failures.items()]
        self.show_error(
            "Some source files could not be loaded",
            "The project is open, but these sources need attention:\n\n" + "\n".join(lines),
        )
        changed = [sid for sid, exc in data.failures.items() if isinstance(exc, SourceChangedError)]
        if changed and not partial:
            answer = QMessageBox.question(
                self,
                "Source files changed",
                "Some files changed since they were imported. Load the new contents and record their "
                "new checksums? Results computed earlier may differ.",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.load_data(verify_checksum=False, only=changed)

    def import_dialog(self, initial: ImportPlan | None = None) -> None:
        source_id = (
            initial.source.source_id
            if initial
            # Provisional id: the kind is resolved from the wizard's choices (see _fit_plan_id).
            else self.session.new_source_id(SourceKind.PROCESS)
        )
        wizard = ImportWizard(
            self.session.project,
            self.session.project.display_timezone,
            source_id=source_id,
            initial=initial,
            parent=self,
        )
        if wizard.exec():
            self.run_import(self._fit_plan_id(wizard.plan()))

    def _fit_plan_id(self, plan: ImportPlan) -> ImportPlan:
        """The wizard proposed an id before the kind was known: make it kind-consistent and free."""
        sid = plan.source.source_id
        used = {s.source_id for s in self.session.project.sources}
        if sid in used or sid.split("_", 1)[0] != plan.source.kind.value:
            new_id = self.session.new_source_id(plan.source.kind)
            return ImportPlan(
                replace(plan.source, source_id=new_id),
                [replace(v, source_id=new_id) for v in plan.variables],
            )
        return plan

    def run_import(self, plan: ImportPlan) -> None:
        self.tasks.submit(
            f"Importing {Path(plan.source.path).name}",
            import_source,
            plan.source,
            None,
            value_columns=[v.source_column for v in plan.variables],
            verify_checksum=False,
            on_result=lambda data: self._import_done(plan, data),
            on_error=lambda msg, details: self._import_failed(plan, msg, details),
        )

    def _import_failed(self, plan: ImportPlan, message: str, details: str) -> None:
        box = QMessageBox(QMessageBox.Icon.Warning, "Import failed", message, parent=self)
        box.setDetailedText(details)
        change = box.addButton("Change settings…", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is change:
            self.import_dialog(initial=plan)

    def _import_done(self, plan: ImportPlan, data: SourceData) -> None:
        if data.report.has_errors:
            errors = [i.message for i in data.report.issues if i.severity is Severity.ERROR]
            self._import_failed(
                plan, "The file was read but cannot be used yet:\n\n" + "\n".join(errors), ""
            )
            return
        variables = [replace(v, source_id=data.source_id) for v in plan.variables]
        try:
            self.session.add_source(data, variables)
        except SchemaError as exc:
            self.show_error("Cannot add data", str(exc))
            return
        self.sidebar.setCurrentRow(0)
        warnings = sum(i.severity is Severity.WARNING for i in data.report.issues)
        self.notify(
            f"Imported {data.report.n_rows:,} rows from {Path(data.source.path).name}"
            + (f" — {warnings} warning(s), see the validation report" if warnings else "")
        )

    # ------------------------------------------------------------------ anomaly detection

    @Slot()
    def _data_changed_detection(self) -> None:
        if self.session.project.detectors and self.session.data.sources:
            self.run_detection()

    @Slot()
    def run_detection(self) -> None:
        """Run the project's detectors in the background on the current data."""
        session = self.session
        frame, mask, target = session.frames()
        version = session.data_version
        detectors = list(session.project.detectors)
        if frame.empty or not detectors:
            session.set_detector_results({}, version)
            return
        series = {c: frame[c] for c in frame.columns}
        if target is not None:
            series[str(target.name)] = target
        text_masks = {c: mask[c] for c in mask.columns}
        metas = {v.name: v for v in session.project.variables}

        def done(results) -> None:  # noqa: ANN001
            if session.project.detectors == detectors:  # ignore results for edited detectors
                session.set_detector_results(results, version)

        self.tasks.submit(
            "Detecting anomalies",
            run_detectors,
            detectors,
            series,
            text_masks,
            metas,
            on_result=done,
            on_error=lambda msg, details: self.show_error("Anomaly detection failed", msg, details),
        )

    # ------------------------------------------------------------------ training

    @Slot()
    def run_training(self, *, reproduce: bool = False) -> None:
        """Train every enabled model in the background on the current aligned features."""
        session = self.session
        fm = session.features
        if fm is None:
            self.show_error(
                "Cannot train", session.features_error or "The aligned features are not ready yet."
            )
            return
        config = session.project.modelling

        def done(run) -> None:  # noqa: ANN001
            session.set_training(run, reproduce=reproduce)
            if session.features is not fm or session.project.modelling != config:
                session._invalidate_training()  # inputs changed while training
            if not reproduce:
                self.sidebar.setCurrentRow(self._stage_row("training"))

        self.tasks.submit(
            "Reproducing saved results" if reproduce else "Training models",
            train_models,
            fm,
            config,
            on_result=done,
            on_error=lambda msg, details: self.show_error("Training failed", msg, details),
        )

    @Slot()
    def _maybe_reproduce(self) -> None:
        """A reopened project had saved results: re-train once and compare (deterministic)."""
        if self._reproduce_pending and self.session.features is not None:
            self._reproduce_pending = False
            self.run_training(reproduce=True)

    def _stage_visited(self, row: int) -> None:
        # Export counts once something is exported (the page records it), not on a visit.
        if row >= 0 and STAGES[row].key != "export":
            self.session.mark_progress(STAGES[row].key)

    def _update_progress(self) -> None:
        states = stage_states(self.session)
        for page in self.pages.values():
            page.set_progress(states)

    def _stage_row(self, key: str) -> int:
        return [s.key for s in STAGES].index(key)

    @Slot(str)
    def _remove_source(self, source_id: str) -> None:
        answer = QMessageBox.question(
            self,
            "Remove source",
            f"Remove '{source_id}' and its variables from the project? The file itself is not changed.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.session.remove_source(source_id)

    def save_project_to(self, path: str | Path, *, portable: bool | None = None) -> bool:
        path = Path(path)
        if path.suffix.lower() != PROJECT_EXTENSION:
            path = path.with_suffix(PROJECT_EXTENSION)
        try:
            self.session.save(path, portable=portable)
        except (ProjectFileError, ValueError) as exc:
            self.show_error("Cannot save project", str(exc))
            return False
        self._remember(path)
        self.recovery.discard()
        kind = "portable project (data included)" if self.session.storage == "portable" else ""
        self.notify(f"Saved {path.name} {kind}".strip())
        return True

    @Slot()
    def _save_portable_dialog(self) -> bool:
        start = str(
            Path(self.settings.last_directory or Path.home())
            / f"{self.session.project.name} (portable){PROJECT_EXTENSION}"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save portable project (data included)", start, PROJECT_FILTER
        )
        return self.save_project_to(path, portable=True) if path else False

    # ------------------------------------------------------------------ crash recovery

    @Slot()
    def _write_recovery(self) -> None:
        if self.session.dirty and (self.session.project.sources or self.session.project.notes):
            self.recovery.write(self.session.project, self.session.path)

    def offer_recovery(self) -> None:
        """Called once at start-up: offer copies left behind by a crash or power loss."""
        copies = find_recovery_copies()
        for copy in sorted(copies, key=lambda c: c.saved_at, reverse=True):
            where = str(copy.original_path) if copy.original_path else "a project never saved"
            answer = QMessageBox.question(
                self,
                "Recover unsaved work?",
                f"{APP_NAME} did not close normally. Unsaved changes to '{copy.name}' ({where}) "
                f"from {copy.saved_at:%Y-%m-%d %H:%M} UTC were kept.\n\nRecover them now? "
                "Choosing No deletes this recovery copy.",
            )
            if answer == QMessageBox.StandardButton.Yes:
                try:
                    self.session.open_recovered(copy.project_file, copy.original_path)
                except ProjectFileError as exc:
                    self.show_error("Cannot recover", str(exc))
                    discard_copy(copy)
                    continue
                self.undo_stack.clear()
                discard_copy(copy)
                self.load_data()
                break
            discard_copy(copy)

    @Slot()
    def _open_dialog(self) -> None:
        if not self.maybe_save():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", self.settings.last_directory, PROJECT_FILTER
        )
        if path:
            self.open_project_file(path)

    @Slot()
    def _save(self) -> bool:
        if self.session.path is None:
            return self._save_as_dialog()
        return self.save_project_to(self.session.path)

    @Slot()
    def _save_as_dialog(self) -> bool:
        start = str(
            Path(self.settings.last_directory or Path.home())
            / f"{self.session.project.name}{PROJECT_EXTENSION}"
        )
        path, _ = QFileDialog.getSaveFileName(self, "Save project as", start, PROJECT_FILTER)
        return self.save_project_to(path) if path else False

    def _remember(self, path: str | Path) -> None:
        self.settings.add_recent(path)
        self.settings.last_directory = str(Path(path).parent)
        save_settings(self.settings)

    @Slot()
    def _populate_recent(self) -> None:
        self.recent_menu.clear()
        if not self.settings.recent_projects:
            empty = self.recent_menu.addAction("(none)")
            empty.setEnabled(False)
            return
        for path in self.settings.recent_projects:
            act = self.recent_menu.addAction(path)
            act.triggered.connect(
                lambda _checked=False, p=path: self.maybe_save() and self.open_project_file(p)
            )

    # ------------------------------------------------------------------ example data

    def generate_example_to(self, folder: str | Path) -> None:
        """Generate example files in a background task and offer to open the sample project."""
        folder = Path(folder)
        self.tasks.submit(
            "Generating example dataset",
            synthetic.create_example,
            folder,
            on_result=self._example_ready,
            on_error=lambda msg, details: self.show_error(
                "Example generation failed", msg, details
            ),
            on_cancelled=lambda: self.notify("Example generation cancelled"),
        )

    @Slot()
    def _generate_example_dialog(self) -> None:
        default = (
            Path(self.settings.last_directory or Path.home()) / "InferentialModelStudio Example"
        )
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a folder for the example files", str(default.parent)
        )
        if folder:
            self.generate_example_to(Path(folder) / default.name)

    def _example_ready(self, paths: dict[str, Path]) -> None:
        self.notify(f"Example written to {paths['project'].parent}")
        answer = QMessageBox.question(
            self,
            "Example ready",
            f"Example files were written to:\n{paths['project'].parent}\n\nOpen the sample project now?",
        )
        if answer == QMessageBox.StandardButton.Yes and self.maybe_save():
            self.open_project_file(paths["project"])

    # ------------------------------------------------------------------ tools / help

    @Slot()
    def _settings_dialog(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            new = dialog.result_settings()
            self.settings.display_timezone = new.display_timezone
            self.settings.author_name = new.author_name
            if new.log_level != self.settings.log_level:
                logging.getLogger().setLevel(new.log_level)
            self.settings.log_level = new.log_level
            self.set_theme(resolve_theme(new.theme), setting=new.theme)
            self.settings.system_title_bar = new.system_title_bar  # used at the next start
            save_settings(self.settings)

    @Slot()
    def _project_dialog(self) -> None:
        dialog = ProjectSettingsDialog(self.session.project, self)
        if dialog.exec() and dialog.apply_to(self.session.project):
            self.session.mark_dirty()

    @Slot()
    def _open_log_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir())))

    @Slot()
    def show_guide(self) -> None:
        GuideDialog(self).exec()

    @Slot()
    def _about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<h3>{APP_NAME} {__version__}</h3>"
            "<p>Offline workbench for building industrial inferential models.</p>"
            f"<p>Python {platform.python_version()} · Qt for Python (PySide6) {PySide6.__version__}</p>"
            "<p>PySide6/Qt are used under the GNU LGPL v3. Third-party notices are listed in "
            "LICENSES.md, installed with the application.</p>"
            f"<p>Log file: {log_dir() / LOG_FILE_NAME}</p>"
            "<p>Models produced by this tool require independent engineering validation before "
            "any online or closed-loop use.</p>",
        )

    # ------------------------------------------------------------------ shutdown

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        if not self.maybe_save():
            event.ignore()
            return
        if self.tasks.busy:
            self.tasks.cancel_all()
            if not self.tasks.wait_for_done(5000):
                log.warning("Background tasks did not stop within 5 s at shutdown")
        if self._theme_timer.isActive():  # a switch clicked just before closing: keep the choice
            self._theme_timer.stop()
            self.settings.theme = DARK if self.title_bar.theme_switch.isChecked() else LIGHT
        self._remember_window()
        save_settings(self.settings)
        self.recovery.discard()  # closed normally: nothing to recover
        event.accept()

    # ------------------------------------------------------------------ window layout

    def _help_panel_default(self) -> bool:
        """The help panel starts open only where it does not crowd the pages (wide screens)."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        return screen is not None and screen.availableGeometry().width() >= WIDE_SCREEN_WIDTH

    def _restore_window(self) -> None:
        """Last session's size and position; on first run, a size that suits the screen."""
        saved = self.settings.window_geometry
        if not (saved and self.restoreGeometry(QByteArray.fromBase64(saved.encode("ascii")))):
            screen = QGuiApplication.primaryScreen()
            if screen is None:
                self.resize(1280, 800)
            else:
                area = screen.availableGeometry()
                width = max(min(1600, int(area.width() * 0.92)), self.minimumWidth())
                height = max(min(1000, int(area.height() * 0.9)), self.minimumHeight())
                self.resize(width, height)
                self.move(area.center() - self.rect().center())
        shown = self.settings.help_panel
        self._show_help_pane(self._help_panel_default() if shown is None else shown)

    def _remember_window(self) -> None:
        self.settings.window_geometry = bytes(self.saveGeometry().toBase64().data()).decode("ascii")
        shown = self.act_help_panel.isChecked()
        # Only a choice that differs from the automatic default is kept, so the panel keeps
        # adapting to the screen until the user decides otherwise.
        self.settings.help_panel = None if shown == self._help_panel_default() else shown

    # ------------------------------------------------------------------ theme and frame

    def set_theme(self, name: str, *, setting: str | None = None) -> None:
        """Show the light or dark theme; ``setting`` (default ``name``) is remembered."""
        self.settings.theme = setting or name
        switch = self.title_bar.theme_switch
        if switch.isChecked() != (name == DARK):
            switch.set_dark(name == DARK)
        app = QApplication.instance()
        if app is None or name == current_theme():
            return
        apply_theme(app, name)
        self.setStyleSheet(stylesheet())
        self._apply_action_icons()
        self.help_close.setIcon(icon("dismiss", size=16))
        refresh_widgets(self)
        if self._system_border:
            set_windows_border(self, TOKENS["border_strong"])
        self.update()

    def _apply_switch_theme(self) -> None:
        self.set_theme(DARK if self.title_bar.theme_switch.isChecked() else LIGHT)

    def _system_theme_changed(self, _scheme: object = None) -> None:
        if self.settings.theme == "auto":
            self.set_theme(resolve_theme("auto"), setting="auto")

    def _update_frame(self) -> None:
        """Grips and the 1 px window border only for a normal (not maximised) frameless window."""
        if not self.frameless:
            return
        normal = not (self.isMaximized() or self.isFullScreen())
        own_border = normal and not self._system_border
        m = 1 if own_border else 0
        self.setContentsMargins(m, m, m, m)
        if self.grips is not None:
            self.grips.set_enabled(normal)
        if self.title_bar is not None:
            self.title_bar.update_window_state()
        self.update()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._update_frame()

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().resizeEvent(event)
        if self.grips is not None:
            self.grips.update_geometry()

    def showEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().showEvent(event)
        if self.frameless and not self._system_border:
            self._system_border = round_windows_corners(self, TOKENS["border_strong"])
            if self._system_border:
                self._update_frame()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        if self.contentsMargins().left() > 0:  # own 1 px border (frameless, not maximised)
            p = QPainter(self)
            p.setPen(QColor(TOKENS["border_strong"]))
            p.drawRect(self.rect().adjusted(0, 0, -1, -1))
            p.end()


class _Shell(QWidget):
    """Central area: the content layer, inset by the rail width, with the rail floating over it.

    The rail is not in the layout: when it slides out it covers the page instead of resizing it.
    """

    def __init__(self, layer: QWidget) -> None:
        super().__init__()
        self._rail: NavRail | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(RAIL_WIDTH, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(layer)

    def set_rail(self, rail: NavRail) -> None:
        self._rail = rail
        rail.raise_()
        self._place_rail()

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().resizeEvent(event)
        self._place_rail()

    def _place_rail(self) -> None:
        if self._rail is not None:
            self._rail.setGeometry(0, 0, self._rail.width(), self.height())
