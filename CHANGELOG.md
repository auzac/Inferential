# Changelog

Versioning: `MAJOR.MINOR.PATCH`. Development milestones are tagged `v0.<milestone>.x`; the first
production release is `v1.0.0`.

## v0.1.0 — Milestone 1: import and raw exploration (2026-10-11)

### Added
- Import wizard for CSV/TSV/TXT and Excel: file type, read options with live preview
  (delimiter, encoding incl. UTF-8 BOM/UTF-16/cp1252, decimal comma, header row, worksheet),
  timestamp column/format/source timezone with parse preview, lab result time and sample ID,
  column selection with names/roles/units, duplicate-timestamp policy. Drag and drop.
- Timestamp detection and parsing: ISO 8601 with offsets, day-first/month-first/year-first,
  month names, AM/PM, Excel serial (1900/1904) and date cells, Unix seconds/milliseconds.
  Ambiguous day/month order blocks the wizard until the user chooses; the whole column is
  searched for a value that settles the order. DST gap/overlap times are flagged, not guessed.
- Raw and normalized data layers with per-cell text-token tracking, file line numbers and
  SHA-256; validation report (invalid/blank/DST times, duplicates, ordering, gaps, irregular
  sampling, text values, missing targets, result-before-sample, lab outside process period).
- Background loading of project data on open; changed source files reported with the option
  to accept new contents; per-source failures do not block other sources.
- Data Import page: source table, validation report, remove source.
- Initial EDA page: synchronized trends with timezone-correct axis, crosshair readout and
  peak downsampling; statistics table; Pearson/Spearman heatmap with clustering and |r|;
  pair scatter; scatter matrix; histograms; coverage view; preliminary causal X–Y screening.
- Project Settings dialog (name, display timezone, notes).
- Save As rebases relative source paths.
- Docs: user guide, benchmarks; methodology and file-format updates.

### Changed
- Project schema 2 (automatic upgrade from 1): native date cells, duplicate policy, sample ID.
- Schema enums accept their string values (protects against Qt widgets returning plain strings).

### Fixed
- Exact, faster Spearman (grouped by missing-value pattern) and faster moment statistics.

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
