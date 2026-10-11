# Changelog

Versioning: `MAJOR.MINOR.PATCH`. Development milestones are tagged `v0.<milestone>.x`; the first
production release is `v1.0.0`.

## v0.4.0 — Milestone 4: modelling, interpretation and validation (2026-10-11)

### Added
- Model Setup page: MLR, PLS, PCR and Ridge candidates; components / alpha tuned by
  time-aware cross-validation inside the training period (or fixed); standard, robust or no
  scaling, fitted on training samples only; optional one-standard-error rule.
- Chronological train / validation / final-test split by share of eligible samples or by dates,
  with an automatic (or manual) purge gap so no process data is shared across a boundary;
  expanding-window or blocked CV folds with the same purge. Live split chart and a list of
  problems (too few samples, constant variables, strong autocorrelation with effective sample
  size, impossible folds, non-causal features).
- Training & Interpretation page: coefficients in original units and standardised,
  cross-validation curve with standard errors, PLS VIP, X weights and X loadings, PCR
  loadings, scores plots coloured by period and explained-variance charts (X and Y kept
  separate). Training runs in the background; results are marked out of date after edits.
- Validation page: comparison table on identical samples for every model (ranked by
  validation RMSE), per-period metrics (R², adjusted R², RMSE, MAE, bias, max |error|, RMSE as
  % of training SD), actual-vs-predicted, trend, residuals vs time / prediction / any
  predictor, residual histogram and autocorrelation, applicability ranges. Final-test results
  stay hidden until explicitly revealed, with a warning about exploratory reuse.
- Core: `splits.py`, `models.py`; schema 4 stores the modelling configuration (v3 files open
  with defaults). Methodology, user guide and benchmarks updated.
- `--smoke-test` also aligns and trains all four model types on a small synthetic dataset, so
  a packaged build with missing numerical libraries fails its build check.

## v0.3.0 — Milestone 3: variable diagnostics and time processing (2026-10-11)

### Added
- Variable Selection page: target and predictor choice (undoable), ranking by aligned Pearson /
  Spearman correlation, VIF with constant/singular handling, adjustable `|r|` warning, aligned
  correlation heatmap, per-predictor sample impact, raw-vs-cleaned statistics and screening
  correlations.
- Signal Processing page: per-predictor deadtime, signed shift, window, lookback/centred
  placement, aggregation (mean, median, min, max, SD, last), trailing filter (rolling
  mean/median, EWMA), causal resampling, minimum coverage and maximum gap; batch apply;
  causal/non-causal warning; option to reject windows overlapping operating-regime exclusions;
  preview of cleaned vs processed signal with the moving window of any lab sample; lag
  sensitivity on the training period only; coverage report and full feature matrix with the
  reason for every unusable sample.
- Core: `features.py` (one row per lab sample, never forward-filled Y, exclusions respected,
  deterministic), `diagnostics.py`. Aligned features are recomputed in the background after
  every relevant edit.
- Tests: hand-calculated windows, causality under future perturbation, synthetic 20-minute
  deadtime recovery, determinism, end-to-end save/reopen reproducibility.

### Changed
- Tables keep the order of their data (e.g. rankings) until a column header is clicked, and
  long tables fill much faster (cells read from column arrays; column widths measured on the
  first 200 rows).
- Timestamp conversion to nanoseconds is much faster on large indexes (benefits rules,
  exclusions and alignment).

## v0.2.2 — Trend selection interactions (2026-10-11)

### Added
- Clicking a trend pins the values at that point in time: the readout retitles to "Selected
  point" and stays locked until the next click (a new drag also releases it).
- Dragging left-to-right on a trend draws a full-height highlight band across every chart,
  the same look as the cleaning page's "Select range" regions (bands stay in sync and can be
  adjusted by dragging; clicking anywhere else removes the highlight).

### Changed
- Panning is no longer the default mouse action on trends: the mouse wheel zooms, middle-drag
  pans and right-drag zooms to a box.
- The "At cursor" readout is taller and ignores hover while a point is pinned.
- User guide: trends section updated for the new mouse behaviour.

### Fixed
- Right-clicking a trend opens the plot menu (View All, axes, Mouse Mode, Plot Options,
  Export...) again: the menu path crashed under pyqtgraph 0.14 because the view box passed an
  argument the click handler does not take.

## v0.2.1 — Performance: large-data responsiveness (2026-10-11)

### Added
- "At cursor" readout on the trends pages, below the variable selection: the values of every
  visible series at the crosshair (fixed height, scrolls when there are many variables).
- Trends cap at 12 charts at once; more show a note telling the user to untick variables.
- User guide: trends section updated for the "At cursor" readout.

### Changed
- Variable selection lists scroll; the trend columns are fixed-width, so long values no
  longer resize or move the charts.
- Ticking or unticking a variable updates only the charts it affects instead of rebuilding
  all charts (and keeps the zoom level).
- Zoomed-out trends draw from a precomputed 2,048-bin min/max envelope; zooming in switches
  to the raw samples of the visible window, so panning and zooming stay fast on large data.
- EDA statistics, correlation and matrix tabs recompute only when visible; selection and
  checkbox bursts are debounced (150 ms).
- Exclusion recomputation runs in the background and is coalesced: a burst of edits produces
  one computation of the latest state, and results from stale edits are discarded.
- The cleaning summary table is computed only when its tab is open.

## v0.2.0 — Milestone 2: cleaning and exclusions (2026-10-11)

### Added
- Cleaning & Exclusions page with Raw / Flagged / Cleaned views over synchronized trends.
- Manual exclusions: draggable range selection synchronized across charts, reason, scope
  (all variables or selected ones), note, author; editable to the second; enable/disable.
- Visual rule builder: whitelisted operators (`<`, `<=`, `>`, `>=`, `==`, `!=`, between,
  outside, is missing, is invalid), AND/OR groups, minimum continuous duration with gap
  handling, exclude rows / selected variables / flag only, live match count. No `eval`.
- Anomaly detectors: z-score, robust MAD z-score, IQR (optional fitting period), Hampel,
  rate of change, flatline, engineering limits, status text; plain-language help;
  conservative "Add recommended" set; background execution.
- Flag review: accept / reject / reset per event, opt-in auto-exclude, events inside already
  excluded periods reported separately; decisions persisted.
- Unified exclusion masks with per-reason categories, target-sample masks, summary tables.
- Undo/redo for every cleaning edit (Edit menu, Ctrl+Z / Ctrl+Y).
- Efficient exclusion shading (one item per chart regardless of interval count).

### Changed
- Project schema 3 (automatic upgrade): rules, detectors, flag decisions.
- Removing a data source also removes rules, detector variables and decisions that used it.

### Fixed
- Time arithmetic is done in nanoseconds regardless of index resolution (pandas 3 may use µs).
- Charts keep their zoom when exclusions change.
- Sidebar readability on Windows dark mode (contributed in 8e1e84b).

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
