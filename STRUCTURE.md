# Repository structure

A map of what each file does and where to look when something needs changing. For how to use
the app see [docs/user-guide.md](docs/user-guide.md); for the maths and conventions see
[docs/methodology.md](docs/methodology.md).

## The one rule

```text
ui/  ──calls──▶  core/        core never imports Qt (works in Jupyter / Seeq Data Lab)
 │
 └──runs long core functions through──▶ infrastructure/workers.py (background threads)
```

- **Numbers, data and file formats** live in `core/`. Fix a calculation there, not in a page.
- **Pages and widgets** in `ui/` only display state and call `core/`.
- `tests/unit/test_core_isolation.py` fails if anything in `core/` imports Qt.

## Top level

```text
README.md              Overview, quick start, repository layout
STRUCTURE.md           This file
CHANGELOG.md           What changed in each version
LICENSES.md            Third-party notices (LGPL obligations for Qt / PySide6)
pyproject.toml         Package metadata, dependencies, entry points, ruff and pytest settings
requirements-lock.txt  Exact dependency versions used for release builds
assets/icons/          Application icon (app.ico, app.png); fluent/ = interface icons (MIT)
docs/                  User guide, methodology, file formats, packaging, benchmarks
examples/              Synthetic distillation dataset + sample project (artificial data)
licenses/              LGPL / GPL v3 texts shipped with the app
packaging/             Windows build, installer and smoke-test scripts
src/inferential_studio/  The application (below)
tests/                 unit / integration / gui tests (below)
```

## `src/inferential_studio/`

```text
__init__.py      Version number (__version__), app name/id
__main__.py      `python -m inferential_studio`
app.py           Entry point: logging, Qt setup, icon, main window, crash-recovery prompt,
                 --smoke-test (with a project: open → align → train → model card → exit)
```

### `core/` — data, statistics, modelling (no Qt)

Listed in workflow order.

| File | What it does |
|---|---|
| `schemas.py` | Every project setting as a typed dataclass (sources, variables, feature windows, exclusions, rules, detectors, split/model config, report info) + strict JSON conversion. **Add a project setting here** (then bump `SCHEMA_VERSION` and add a migration in `persistence.py`). |
| `persistence.py` | `.imsproj` save/load: atomic writes, safe ZIP reading, schema migrations, portable (embedded data) mode, saved results, source checksums and path rebasing. |
| `timestamps.py` | Detects and parses timestamp formats to UTC (day/month ambiguity, Excel serials, Unix, DST). |
| `importers.py` | Reads CSV/Excel, builds the validation report, loads all sources of a project. |
| `dataset.py` | Raw and normalized data layers (`SourceData`, `ProjectData`), process frame and sparse target. |
| `statistics.py` | Raw EDA: summary stats, Pearson/Spearman, coverage, preliminary X–Y screening. |
| `rules.py` | Evaluates exclusion rules (whitelisted operators, AND/OR, minimum duration). Also `ns()` time helper. |
| `quality.py` | Anomaly detectors (z-score, MAD, IQR, Hampel, rate of change, flatline, limits, text). |
| `exclusions.py` | Combines manual intervals, rules and reviewed flags into one exclusion mask per variable. |
| `features.py` | **Time alignment**: per-predictor resample → trailing filter → window aggregation at each lab sample; coverage/gap checks; eligibility; lag scan. |
| `diagnostics.py` | Raw-vs-cleaned stats, aligned correlations, VIF. |
| `what_if.py` | What-if: the selected model's prediction with inputs shifted or held (exact, affine). |
| `splits.py` | Chronological train/validation/test split, purge gap, time-aware CV folds. |
| `models.py` | MLR/PLS/PCR/Ridge pipelines, CV tuning, coefficients, PLS/PCR details, VIP, metrics, feasibility checks. |
| `artifacts.py` | `.imsmodel` JSON model files, batch prediction with applicability checks, saved-results summary/compare. |
| `exports.py` | Predictions / feature matrix / batch results to CSV or Excel. |
| `report.py`, `svg.py` | HTML model card and its dependency-free SVG charts. |
| `synthetic.py` | Synthetic distillation dataset with known truth (tests, example, `inferential-studio-synth`). |
| `tasks.py` | Progress callback and cancel token used by long core functions. |

### `infrastructure/` — app plumbing

| File | What it does |
|---|---|
| `workers.py` | `TaskManager`: runs core functions on a thread pool, returns results/errors/progress to the GUI thread. |
| `settings.py` | Per-user settings (timezone, author, recent projects…). |
| `paths.py` | Per-user folders (config, logs, recovery, cache) and bundled resource lookup. Override with `INFERENTIAL_STUDIO_HOME`. |
| `logging_config.py` | Log files and the uncaught-exception hook. |
| `recovery.py` | Crash-recovery copies of unsaved projects. |

### `ui/` — windows, pages, widgets

| File | What it does |
|---|---|
| `main_window.py` | Actions, title bar and main menu, stage rail, page stack, Help panel, status bar, theme switching, frameless frame; open/save/import, detection, training, recovery. **Start here to follow any user action.** |
| `session.py` | The open project + loaded data + derived state (cleaning masks, aligned features, training run); background recomputation and signals that pages listen to. |
| `stages.py` | The nine workflow stages (titles, goals, rail icons), in workflow order. |
| `commands.py` | Undo/redo: edits replace whole project lists/configs. |
| `messages.py` | `MessageCenter` (on `session.messages`): every status line, warning and error. Pages call `messages.set(key, level, text)` for a state and `messages.info/warning/error(...)` for an event — never a label in the page. |
| `help.py` | Per-stage help text (Help panel) and the in-app document viewer (F1). |
| `theme.py` | Light and dark colour tokens, the stylesheet (set on the main window, not the application: much faster to switch), palette, `refresh_widgets()` for a theme switch, `mark_primary()` for a page's main button. **Change the look here.** |
| `icons.py` | `icon(name, token)`: Fluent SVG icons from `assets/icons/fluent`, recoloured to a theme token and cached. |
| `import_wizard.py` | The five-step import wizard. |
| `cleaning_dialogs.py` | Dialogs for exclusions, rules and detectors. |
| `project_dialog.py`, `settings_dialog.py` | Project settings and application settings dialogs. |

`ui/pages/` — one file per workflow stage:

| Stage | File |
|---|---|
| (shared header) | `stage_page.py` |
| 1 Data Import | `import_page.py` |
| 2 Initial EDA | `eda_page.py` |
| 3 Cleaning & Exclusions | `cleaning_page.py` |
| 4 Variable Selection | `variables_page.py` |
| 5 Signal Processing | `processing_page.py` |
| 6 Model Setup | `model_setup_page.py` |
| 7 Training & Interpretation | `training_page.py` |
| 8 Validation | `validation_page.py` |
| 9 Export & Report | `export_page.py` |

`ui/widgets/` — reusable pieces: `title_bar.py` (title bar = ≡ menu, command icons, project
name, Help panel toggle, `theme_switch.py`, caption buttons), `nav_rail.py` (stage rail that
slides out over the page), `window_frame.py` (resize grips and Windows 11 corners for the
frameless window), `message_panel.py` (fixed message log at the bottom),
`chart_menu.py` (the one right-click menu every chart uses: Copy / Save image; create charts with
`plots.plot_widget` or `chart_menu.add_plot` so they get it),
`trend_view.py` (synchronized trend charts: drag-to-zoom, Back, Fit all), `what_if.py`
(Validation's What-if tab: input sliders over the prediction trend), `cursor_readout.py`
(values at cursor), `overlays.py` (exclusion shading), `time_axis.py` (timezone-aware axis),
`plots.py` (`plot_widget` / `ChartPlotItem`: empty-state message and series key in the chart
header; series colours; `ValueAxis` without overlapping labels), `heatmap.py`, `data_table.py` (pandas table view; number formatting in `format_value`,
`fit_to_rows` for short summary tables), `flow_layout.py` (button rows that wrap on narrow
windows), `variable_list.py`, `timezone_combo.py`, `task_progress.py` (status-bar
progress/cancel).

## `tests/`

```text
conftest.py      Offscreen Qt, isolated settings folder per test, shared synthetic dataset
unit/            One file per core module (test_<module>.py) + core/Qt isolation check
integration/     Whole pipelines: import → EDA, align → train → save → reopen, app smoke tests
gui/             pytest-qt tests per page, main window, widgets, workers, full export workflow
```

Run everything: `pytest` (about 3 minutes). Lint: `ruff check src tests` and `ruff format src tests`.

## `packaging/`

| File | What it does |
|---|---|
| `pyinstaller.spec` | One-folder Windows build (what goes in the bundle). |
| `launcher.py` | PyInstaller entry script. |
| `windows_build.ps1` | Full release build: tests → bundle → smoke test → zip → installer → installer smoke test. |
| `installer/InferentialModelStudio.iss` | Inno Setup per-user installer. |
| `installer_smoke_test.ps1` | Silent install → smoke tests → uninstall. |
| `make_icon.py` | Regenerates `assets/icons/`. |
| `license_inventory.py` | Regenerates the dependency table in `LICENSES.md`. |

## Where to find…

| Task | Look in |
|---|---|
| A new project setting | `core/schemas.py` (+ migration in `core/persistence.py`) |
| How windows / deadtime are applied | `core/features.py`, documented in `docs/methodology.md` |
| A new model type | `core/models.py` (`make_pipeline`, `_train_one`), then `ui/pages/model_setup_page.py` |
| A new metric | `core/models.py` (`regression_metrics`, `METRIC_COLUMNS`) |
| A new anomaly detector | `core/quality.py` + `DetectorKind` in `core/schemas.py` + `ui/cleaning_dialogs.py` |
| Model card content | `core/report.py` |
| Project or model file format | `core/persistence.py`, `core/artifacts.py`, `docs/file-formats.md` |
| Something is slow in the GUI | the page's update method; long work should go through `TaskManager` |
| Help text on a page | `ui/help.py` |
| A warning or status message | `session.messages` (`ui/messages.py`), shown by `ui/widgets/message_panel.py` |
| Colours, fonts, spacing | `ui/theme.py` (`TOKENS`); a widget that paints itself implements `apply_theme()` |
| An icon | `ui/icons.py`; add the Fluent SVG to `assets/icons/fluent/` |
| Chart right-click menu | `ui/widgets/chart_menu.py` |
| Logs and settings on a PC | `%LOCALAPPDATA%\InferentialModelStudio\` (menu ≡ → Help → Open log folder) |
| Regenerate the example | `inferential-studio-synth --out examples/synthetic_distillation` |
