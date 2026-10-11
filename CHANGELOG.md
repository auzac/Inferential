# Changelog

Versioning: `MAJOR.MINOR.PATCH`. Development milestones are tagged `v0.<milestone>.x`; the first
production release is `v1.0.0`.

## v0.0.1 — Milestone 0: engineering foundation (2026-10-11)

### Added
- PySide6 application shell: nine-stage sidebar navigation, toolbar and menus
  (New/Open/Open Recent/Save/Save As, Undo/Redo, Settings, About, Open Log Folder),
  properties panel, status bar with background-task progress and Cancel.
- Unsaved-changes tracking with save prompts; window title marker.
- Project schema v1 (`core/schemas.py`): data sources with timestamp specifications
  (including lab result time), variable metadata and roles, per-predictor feature settings with
  a fixed window convention and causality check, manual exclusion intervals, strict JSON
  conversion.
- `.imsproj` project file: atomic save, safe load (no extraction, entry and size checks,
  no pickle), schema-version check and migration hook, source-file checksums.
- Background `TaskManager` with progress throttling, cooperative cancellation and
  user-facing error messages.
- Rotating application log and JSON user settings in `%LOCALAPPDATA%\InferentialModelStudio`.
- Synthetic distillation dataset generator with recorded ground truth, CLI
  `inferential-studio-synth`, committed example data and sample project.
- `--smoke-test` and `--version` command-line options.
- Test suite (unit, GUI via pytest-qt, integration), ruff lint configuration.
- PyInstaller one-folder spec, Windows build script, license inventory script, LGPL/GPL texts.
