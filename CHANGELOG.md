# Changelog

Versioning: `MAJOR.MINOR.PATCH`. Development milestones are tagged `v0.<milestone>.x`; the first
production release is `v1.0.0`.

## Unreleased

### Changed
- Tables fill the window. When the view is wider than the table, the columns grow in proportion
  to their content instead of leaving an empty band beside the last column; when it is narrower,
  the table keeps its own widths and scrolls rather than squeezing the text. Column widths you
  drag become the proportions from then on. This covers every table — the stage pages, Cleaning &
  Exclusions, Data Import and the import wizard.
- Every header sits over its own values: left over text, right over numbers. Qt centres them by
  default, which drifts away from the column's values as the column widens.
- Cells in the Cleaning tables carry their full text as a tooltip, since a long note now elides
  at the column width cap instead of stretching its column.
- Clicking a variable name toggles its checkbox in EDA, cleaning, predictor selection and
  cleaning dialogs. Checkbox clicks and keyboard selection remain available.

### Fixed
- **Hampel spike filter** now follows Hampel's definition: the MAD is taken about the window's
  own median. It previously used each neighbour's own local median, which underestimated the
  spread and roughly tripled false flags on noisy signals (same spikes caught).
- **Flatline** detection with a tolerance now requires the values to stay inside a band of that
  width. Before, small steps could chain, so a slow steady drift was reported as a stuck value.
  (With the default tolerance 0 nothing changes.)

### Added
- Optional input-subset model search for up to five selected predictors. Candidates use the same
  eligible rows, chronological split and training-period CV folds; the leaderboard ranks by CV
  RMSE and shows the best candidate at each input count. Validation results remain visible for
  review, and the final test is evaluated only for the selected candidate when revealed.
- The selected candidate and subset-search setting are saved with the project. Model cards and
  prediction exports include only the candidate's inputs.
- **What-if** tab on Validation: a **step test** on the chosen model. Inputs start at steady
  state (training means); at the step time the inputs set with their sliders (± their training
  range) change and stay there. They pass through the model's own features, so the prediction
  responds as the soft sensor would online: flat for the deadtime, a ramp over the window, an
  EWMA lag, settling at coefficient × step. Input and prediction trends, final change, first
  response and time to 95 %, and an extrapolation warning. Calculation in `core/what_if.py`.
- Reference checks: every statistical method is validated against an independent textbook
  implementation (`tests/unit/test_reference_methods.py`, 31 checks) — MLR, PLS (incl. VIP), PCR,
  Ridge, the one-standard-error rule, splits and CV folds (no shared process data), descriptive
  statistics, correlations, VIF, window features, trailing filters and resampling.
- A warning when cross-validation picks a value at the edge of the range tried (smallest or
  largest Ridge alpha, or the maximum number of PLS/PCR components allowed).
- Methodology: Hampel and flatline definitions, and a table of conventions that make numbers
  differ from Minitab / JMP / SIMCA (S vs RMSE, quartile method, ridge α scale …).

### Note for existing projects
Hampel and flatline (tolerance > 0) events can change. Accepted or rejected decisions on events
whose time span changed no longer match and the events return to *pending*; reopening such a
project reports that saved results differ. Re-review the events and retrain.

## v0.5.7 — Lag sensitivity correctness (2026-10-13)

### Changed
- **Lag sensitivity** works from the training period and nothing else. With no valid split it
  declines to run and says so, instead of falling back to the whole history, and it waits for
  cleaning and feature alignment to finish rather than reading a half-built feature matrix.
  Candidate windows must be causal: a centred window needs enough delay to keep every candidate
  in the past, and the message says which. The project's regime-overlap policy is applied to
  every candidate delay. Pair counts can vary by delay — this is a single-predictor diagnostic,
  not a multivariable identification test.
- A scan is shown only for the inputs that produced it. Changing the predictor, its settings,
  the split, the cleaning, the lag range or the step clears the plot and disables *Use best
  delay as deadtime*; a background scan that finishes after any of those moved is discarded
  rather than shown or applied, and a late-finishing older scan can no longer replace a newer
  one.

### Fixed
- Predictor preprocessing during a scan is now bounded by the training cutoff, so missing data
  after the cutoff can no longer change the sampling interval behind the coverage check — and
  with it the correlation. Exclusions are copied before use, so a scan cannot write back into
  the cleaning state.

## v0.5.6 — Scatter matrix alignment and speed (2026-10-13)

### Changed
- **Scatter matrix**: the rows all start at the same place, so the six charts form a real grid.
  Each y-axis used to size itself from whichever tick labels that one row happened to draw, so
  the rows sat 12–36 px apart and their spines did not line up. The left column now shares one
  width, measured from the labels every row will actually draw.
- **Scatter matrix, faster**: ticking a variable redraws the grid instead of rebuilding it. The
  plots are made once and refilled, and a tick that changes nothing is not drawn at all: about
  0.6 s for the first draw (was 1.3–1.7 s), 0.04 s for a repeat. Each cell draws at most 1,000
  evenly spaced points — a cell is roughly 90 × 50 px, so more of them were invisible.

## v0.5.5 — New window shell and themes (2026-10-12)

### Changed
- **One seamless window**: the application draws its own title bar, so title bar, commands and
  page read as one surface (Windows 11 style, rounded corners and snap kept — moving and
  resizing still go through Windows). The title bar holds:
  - **≡ menu** (Alt+F) with every command — new, open, open recent, save, save as, save
    portable copy, import, export, undo, redo, project settings, settings, Help ▸ (user guide,
    Help panel, example dataset, log folder, about) and exit. It replaces the menu bar.
  - **Icon buttons** for open, save, save as, import, undo and redo (tooltips give the name,
    the shortcut and, for undo/redo, the step). They replace the toolbar.
  - The project name (and "unsaved changes") in the centre, the Help panel toggle, the
    **light/dark switch**, and minimise / maximise / close.
- **Stage rail**: the sidebar is a 48 px column of stage icons in workflow order (no numbers).
  The button at its top slides the names out *over* the page (160 ms) — the page never moves,
  so charts are not redrawn. Choosing a stage, Esc or a click elsewhere closes it; a click on
  the page only closes it.
- **Themes**: Windows dark mode colours or a warm light theme. *Settings → Theme* adds *Follow
  Windows*, which changes along with Windows. Charts, tables, messages and dialogs follow;
  data colours are mid-tones that read on both backgrounds. A switch takes a fraction of a
  second: the stylesheet sits on the window, not the application, so Qt re-styles far less.
- Fluent UI System Icons (MIT) throughout; check boxes, radio buttons and list check marks
  restyled to match the theme.
- Menu items in sentence case, like the rest of the interface.
- Faster start: the main window builds in less than half the time (same reason).

### Added
- *Settings → Use the system title bar* (after a restart): the standard Windows frame above the
  application's title bar, as a fallback for screen readers or window-management tools.

### Fixed
- Short trend charts no longer draw y-axis labels on top of each other (e.g. "200" over "100").
- Review fixes: the menu highlight reads on the dark theme; the caption buttons are clickable
  to the window edge; Esc reaches an open dialog before closing the stage rail; keyboard stage
  choices close the rail; a theme change made just before closing is kept; the title bar shows
  the project path on hover and elides long names; error cells, sub-menu icons and the message
  panel follow a theme switch (without losing the selection); the exported model card uses the
  same series colours as the app; "not eligible" points sit further back; the app still opens
  when the temporary theme assets cannot be written.

## v0.5.4 — Interface finish (2026-10-12)

### Changed
- **Laptop screens**: the Help panel starts closed below 1600 px of screen width (open on wide
  screens) and is one click away as **Help panel** on the menu bar (Shift+F1). Button rows wrap
  instead of clipping, the message panel and the *At cursor* box give up height on short
  screens, and pages have lower minimum sizes.
- **The window remembers** its size, position and the Help panel choice between sessions; the
  first start sizes it to the screen.
- **Charts**: series names sit on the chart's title line with a colour mark (●, ━, ┄) instead
  of a legend box over the data; titles are left-aligned in one style. Empty charts show a
  message saying what is missing (e.g. *Train models on Model Setup…*, or why loadings do not
  apply to MLR) instead of bare 0–1 axes, and long messages wrap. Trend lanes get the same
  one-line header (units stay on the y-axis).
- **Validation**: the model comparison table is exactly as tall as its rows; *Diagnostics for*
  and its metrics now sit with the diagnostic charts they describe.
- Tables no longer show a grey band under the last row.
- Calmer window chrome: *Stage 2 of 9*, *Step 3 of 5* and *Messages* in sentence case; the
  open project's name (and "unsaved changes") under the application name on the left.
- Import wizard: step counter on every page, no row numbers in previews. Rule dialog: a visible
  ✕ to remove a condition, guidance instead of a red error while a rule is incomplete.
- Smaller fixes: the coefficient chart lost its axes after a re-train; the anomaly review row
  no longer hides a button under the table (double-click an event to zoom to it); batch
  prediction options have short names.

## v0.5.3 — Exploration and polish (2026-10-12)

### Changed
- Import wizard: **one timestamp per file**. The separate *result available time* is no longer
  asked for; lab files use the sample time. The optional **Sample ID column** now sits next to
  the source timezone and still marks a file as lab data. (Projects that already record a result
  time keep it, for audit only.)
- Initial EDA: the **correlation matrix follows the ticked variables** and updates as you tick or
  untick (the *Only checked variables* box is gone). **All / None** buttons and a ticked count sit
  above the variable list. The **Recalculate** button is removed — statistics recompute on their
  own when data changes.
- Trends (Initial EDA, Cleaning): **drag across a chart to zoom to that period**; a new **Back**
  button steps out and **Fit all** shows everything. Navigation is along time only and each
  y-axis fits the data on screen.
- **One right-click menu for every chart** (trends, EDA, modelling, validation, export):
  **Copy image** and **Save image…**, acting on the whole chart as shown. PyQtGraph's own menu
  (*View All*, axis and mouse-mode options, export dialog) and its small "A" auto-range button
  are gone everywhere.
- Tables show numbers to five significant figures, floating-point residue (e.g. an OLS training
  bias of 1e-17) as 0, counts with thousands separators and Yes/No for flags.
- Data Import: the source details are a labelled summary; OK sources and reports show a green
  check. The example-data box only appears for an empty project; **Help → Generate Example
  Dataset…** is always available.
- Signal Processing: the settings form runs the full height on the right, giving the preview
  charts most of the page. Cleaning gives the trend charts more room.
- Coverage chart: calmer green (colours were being stretched), time axes of both charts line up.
- Empty charts (EDA scatter, batch prediction) show a short message instead of bare axes.
- Inline validation text uses the theme's error / warning colours throughout.
- Example dataset: the lab file has a single timestamp (`Sample_Timestamp`) and the sample
  project no longer records a result time. Process data, lab values and ground truth are
  unchanged.

## v0.5.2 — Single data import flow (2026-10-12)

### Changed
- Data Import: one **Import data…** button replaces the separate process and lab buttons. The
  wizard no longer asks for the source type on its first page — it is resolved from your choices:
  a result-available time or sample ID, or a target with no predictors, means lab / analyzer
  data; anything else (including one CSV holding both the target and predictors) is process data.
- The result-available time and sample-ID fields always appear on the Timestamps page, grouped as
  optional lab / analyzer extras; picking a column there marks the source as lab data.
- The wizard blocks lab sources with predictor columns, since predictors must come from process
  data.

## v0.5.1 — Interface refresh and message panel (2026-10-12)

### Added
- **Messages panel**: a fixed-height log at the bottom of the window collects every status
  line, warning and error. Active conditions are pinned on top until their cause is fixed;
  events (saved, imported, failed) are kept with a time stamp. Filter by current stage or all
  stages, warning/error badges, double-click for details, copy, clear. Pages no longer insert
  banners or status lines, so their charts and tables never move.
- Errors that used to open a dialog (load, save, training failures …) are reported in the panel
  (questions that need a decision, and the import "Change settings…" dialog, stay dialogs).

### Changed
- New light, high-contrast industrial theme: dark slate navigation rail and status bar, white
  surfaces on a cool grey work area, one steel-blue accent for selection and the main action of
  each page, flat tabs, consistent inputs, tables and scroll bars. Colours live in one token set
  (`ui/theme.py`).
- Page header with stage number, title and goal; navigation rail with application title.
- Train, Apply, Exclude selection, Import process data and Save model card are styled as the
  primary action of their page; the Train button on Model Setup no longer scrolls away.
- Validation: per-period metrics always show all three rows; shorter captions.

## v0.5.0 — Milestone 5: persistence, reporting and release preparation (2026-10-11)

Release candidate for v1.0.0 (pending the installer test on a clean Windows PC).

### Added
- **Export & Report page**: HTML model card (self-contained, inline SVG charts, printable to
  PDF) with summary, data sources and checksums, every exclusion / rule / detector review,
  feature definitions, equation, coefficients and VIP, metrics and plots per period, training
  ranges, limitations and provenance; predictions and feature matrix to Excel/CSV; model-card
  details stored in the project.
- **Model files (`.imsmodel`)**: plain JSON (no pickle) with feature definitions, coefficients
  in feature units, training ranges, metrics and provenance; safe loading with format/size
  checks.
- **Batch prediction**: apply a saved model to the project's data at the lab sample times
  (with RMSE/bias against the lab) or on a regular grid; missing/insufficient inputs give no
  prediction with a reason; extrapolation beyond the training range is flagged; export.
- **Portable projects** (*File → Save Portable Copy…*): data files embedded and verified by
  SHA-256 on opening. Project schema 5.
- **Saved results are reproduced**: projects store a summary of the last training run; on
  reopening the models are retrained and the pages report whether the results were reproduced.
  Whether the final test was revealed is remembered.
- **Crash recovery**: a recovery copy of unsaved work every minute, offered at the next start.
- **Contextual help**: the right-hand panel shows a short guide for each stage; F1 opens the
  user guide, methodology and file formats inside the application.
- **Packaging**: application icon, Inno Setup per-user installer script (Start menu, optional
  desktop shortcut and `.imsproj` association, uninstaller), installer smoke test script,
  build-script integration with checksums; `--smoke-test <project>` runs the whole example
  workflow (load, align, train, write the model card) and fails with a non-zero exit code.
- Tests: artifacts, exports, report, portable/results/recovery persistence, full GUI workflow,
  subprocess smoke tests.

### Changed
- Median and standard-deviation windows are vectorised (about 3× faster alignment, much faster
  batch prediction on dense grids).
- The Properties dock became the Help panel; *File → Export…* (Ctrl+E) opens the Export page.
- pytest dumps all thread stacks if a test hangs for 5 minutes (`faulthandler_timeout`).

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
