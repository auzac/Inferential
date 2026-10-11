"""Main application window: sidebar stage navigation, page stack, properties panel, status bar."""

from __future__ import annotations

import logging
import platform
from dataclasses import replace
from pathlib import Path

import PySide6
from PySide6.QtCore import QSize, Qt, QUrl, Slot
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QKeySequence, QUndoStack
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QWidget,
)

from inferential_studio import APP_NAME, __version__
from inferential_studio.core import synthetic
from inferential_studio.core.dataset import ProjectData, Severity, SourceData
from inferential_studio.core.importers import SourceChangedError, import_source, load_project_data
from inferential_studio.core.persistence import PROJECT_EXTENSION, ProjectFileError
from inferential_studio.core.quality import run_detectors
from inferential_studio.core.schemas import Project, SchemaError, SourceKind
from inferential_studio.infrastructure.logging_config import LOG_FILE_NAME
from inferential_studio.infrastructure.paths import log_dir
from inferential_studio.infrastructure.settings import AppSettings, save_settings
from inferential_studio.infrastructure.workers import TaskManager
from inferential_studio.ui.import_wizard import ImportPlan, ImportWizard
from inferential_studio.ui.pages.cleaning_page import CleaningPage
from inferential_studio.ui.pages.eda_page import EdaPage
from inferential_studio.ui.pages.import_page import ImportPage
from inferential_studio.ui.pages.stage_page import PlaceholderPage, StagePage
from inferential_studio.ui.project_dialog import ProjectSettingsDialog
from inferential_studio.ui.session import ProjectSession
from inferential_studio.ui.settings_dialog import SettingsDialog
from inferential_studio.ui.stages import STAGES
from inferential_studio.ui.widgets.task_progress import TaskProgressWidget

log = logging.getLogger(__name__)
PROJECT_FILTER = f"Inferential Model Studio project (*{PROJECT_EXTENSION})"


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.session = ProjectSession(self)
        self.tasks = TaskManager(self)
        self.undo_stack = QUndoStack(self)
        self.setMinimumSize(QSize(960, 640))
        self.resize(1280, 800)
        self._build_actions()
        self._build_menus_and_toolbar()
        self._build_central()
        self._build_properties_panel()
        self._build_status_bar()
        self.session.changed.connect(self._refresh_status)
        self.session.data_changed.connect(self._refresh_status)
        self.session.cleaning_changed.connect(self._refresh_status)
        self.session.detectors_changed.connect(self.run_detection)
        self.session.data_changed.connect(self._data_changed_detection)
        self._load_generation = 0
        self.session.dirty_changed.connect(self.setWindowModified)
        self.session.path_changed.connect(lambda _p: self._update_title())
        self.tasks.busy_changed.connect(self._on_busy_changed)
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
            "&New Project",
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
            "Save &As…",
            self._save_as_dialog,
            QKeySequence.StandardKey.SaveAs,
            "Save the project under a new name",
        )
        self.act_import = action(
            "&Import Data…",
            lambda: self.import_dialog(),
            QKeySequence("Ctrl+I"),
            "Import process (historian) or lab data from CSV or Excel",
        )
        self.act_export = action(
            "&Export…", lambda: None, None, "Export predictions, report and model (Milestone 5)"
        )
        self.act_export.setEnabled(False)
        self.act_exit = action("E&xit", self.close, QKeySequence.StandardKey.Quit)
        self.act_project = action(
            "&Project Settings…",
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
            "Open &Log Folder", self._open_log_folder, None, "Show the application log files"
        )
        self.act_about = action(f"&About {APP_NAME}", self._about)
        self.act_undo = self.undo_stack.createUndoAction(self, "&Undo")
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_redo = self.undo_stack.createRedoAction(self, "&Redo")
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)

    def _build_menus_and_toolbar(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        file_menu.addActions([self.act_new, self.act_open])
        self.recent_menu = file_menu.addMenu("Open &Recent")
        self.recent_menu.aboutToShow.connect(self._populate_recent)
        file_menu.addSeparator()
        file_menu.addActions([self.act_save, self.act_save_as])
        file_menu.addSeparator()
        file_menu.addActions([self.act_import, self.act_export])
        file_menu.addSeparator()
        file_menu.addAction(self.act_project)
        file_menu.addSeparator()
        file_menu.addAction(self.act_exit)
        edit_menu = bar.addMenu("&Edit")
        edit_menu.addActions([self.act_undo, self.act_redo])
        tools_menu = bar.addMenu("&Tools")
        tools_menu.addActions([self.act_settings, self.act_logs])
        help_menu = bar.addMenu("&Help")
        help_menu.addAction(self.act_about)
        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        toolbar.addActions([self.act_new, self.act_open, self.act_save, self.act_save_as])
        toolbar.addSeparator()
        toolbar.addActions([self.act_import, self.act_export])
        toolbar.addSeparator()
        toolbar.addActions([self.act_undo, self.act_redo])
        toolbar.addSeparator()
        toolbar.addAction(self.act_settings)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

    def _build_central(self) -> None:
        self.sidebar = QListWidget(self)
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(230)
        self.sidebar.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Stable item geometry prevents text clipping when Windows DPI scaling
        # or a Qt stylesheet changes the rendered font/padding.
        self.sidebar.setUniformItemSizes(True)
        self.sidebar.setSpacing(2)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.sidebar.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.stack = QStackedWidget()
        self.pages: dict[str, StagePage] = {}
        for stage in STAGES:
            if stage.key == "import":
                page: StagePage = ImportPage(stage, self.session)
                page.generate_example_requested.connect(self._generate_example_dialog)
                page.open_project_requested.connect(self._open_dialog)
                page.import_requested.connect(lambda kind: self.import_dialog(kind))
                page.remove_requested.connect(self._remove_source)
            elif stage.key == "initial_eda":
                page = EdaPage(stage, self.session, self.tasks)
            elif stage.key == "cleaning":
                page = CleaningPage(stage, self.session, self.undo_stack, self.settings)
                page.detection_requested.connect(self.run_detection)
            else:
                page = PlaceholderPage(stage)
            self.pages[stage.key] = page
            self.stack.addWidget(page)
            item = QListWidgetItem(f"{stage.number}.  {stage.title}")
            # Use an explicit row height rather than relying on QSS padding
            # to expand QListWidget's cached size hints.
            item.setSizeHint(QSize(0, 40))
            item.setData(Qt.ItemDataRole.UserRole, stage.key)
            item.setToolTip(f"{stage.goal}\n(Delivered in Milestone {stage.milestone})")
            self.sidebar.addItem(item)
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.sidebar.setCurrentRow(0)
        self.sidebar.scrollToTop()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        self.setCentralWidget(splitter)

    def _build_properties_panel(self) -> None:
        dock = QDockWidget("Properties", self)
        dock.setObjectName("propertiesDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        placeholder = QLabel("Select a variable, rule or model to see its settings here.")
        placeholder.setWordWrap(True)
        placeholder.setAlignment(Qt.AlignmentFlag.AlignTop)
        placeholder.setMargin(12)
        dock.setWidget(placeholder)
        dock.setMinimumWidth(220)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.properties_dock = dock

    def _build_status_bar(self) -> None:
        status = self.statusBar()
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
        box = QMessageBox(
            QMessageBox.Icon.Warning, title, message, QMessageBox.StandardButton.Ok, self
        )
        if details:
            box.setDetailedText(details)
        box.exec()

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

    def open_project_file(self, path: str | Path) -> bool:
        try:
            self.session.open(path)
        except ProjectFileError as exc:
            log.warning("Could not open project %s: %s", path, exc)
            self.show_error("Cannot open project", str(exc))
            return False
        self.undo_stack.clear()
        self._remember(path)
        log.info("Opened project %s", path)
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
        if only is not None:
            target = Project(
                name=project.name,
                display_timezone=project.display_timezone,
                sources=[s for s in project.sources if s.source_id in only],
                variables=[v for v in project.variables if v.source_id in only],
            )
        self.tasks.submit(
            "Loading project data",
            load_project_data,
            target,
            self.session.path,
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

    def import_dialog(
        self, kind: SourceKind | None = None, initial: ImportPlan | None = None
    ) -> None:
        kind = kind or (initial.source.kind if initial else None)
        source_id = (
            initial.source.source_id
            if initial
            else self.session.new_source_id(kind or SourceKind.PROCESS)
        )
        wizard = ImportWizard(
            self.session.project,
            self.session.project.display_timezone,
            kind=kind,
            source_id=source_id,
            initial=initial,
            parent=self,
        )
        if wizard.exec():
            plan = wizard.plan()
            if plan.source.source_id in {s.source_id for s in self.session.project.sources}:
                plan = ImportPlan(
                    replace(plan.source, source_id=self.session.new_source_id(plan.source.kind)),
                    plan.variables,
                )
                plan = ImportPlan(
                    plan.source,
                    [replace(v, source_id=plan.source.source_id) for v in plan.variables],
                )
            self.run_import(plan)

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
        self.statusBar().showMessage(
            f"Imported {data.report.n_rows:,} rows from {Path(data.source.path).name}"
            + (f" — {warnings} warning(s), see the validation report" if warnings else ""),
            8000,
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

    @Slot(str)
    def _remove_source(self, source_id: str) -> None:
        answer = QMessageBox.question(
            self,
            "Remove source",
            f"Remove '{source_id}' and its variables from the project? The file itself is not changed.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.session.remove_source(source_id)

    def save_project_to(self, path: str | Path) -> bool:
        path = Path(path)
        if path.suffix.lower() != PROJECT_EXTENSION:
            path = path.with_suffix(PROJECT_EXTENSION)
        try:
            self.session.save(path)
        except (ProjectFileError, ValueError) as exc:
            self.show_error("Cannot save project", str(exc))
            return False
        self._remember(path)
        self.statusBar().showMessage(f"Saved {path.name}", 4000)
        return True

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
            on_cancelled=lambda: self.statusBar().showMessage("Example generation cancelled", 4000),
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
        self.statusBar().showMessage(f"Example written to {paths['project'].parent}", 6000)
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
        save_settings(self.settings)
        event.accept()
