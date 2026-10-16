# Inferential Model Studio

Offline Windows desktop workbench for building, validating and documenting industrial
inferential models (soft sensors), starting with distillation columns.

Python · PySide6 (Qt 6) · PyQtGraph · pandas · scikit-learn. No cloud services, no telemetry.

> **Status: v0.5.7 — Milestone 5 + new window shell (release candidate for v1.0.0).** All nine stages work:
> import, raw EDA, cleaning with reviewable flags, variable selection, per-predictor alignment to
> lab samples, MLR / PLS / PCR / Ridge with chronological splits and time-aware CV, validation,
> and export (HTML model card, predictions, JSON model file, batch prediction). Projects can be
> portable (data embedded), reproduce their saved results on reopening and recover after a crash.
> One seamless window (title bar with menu and commands, sliding stage rail) in a light or dark
> theme. Remaining gate for v1.0.0: build and test the installer on a clean Windows PC, including
> the frameless window (snap, multi-monitor, scaling) ([packaging](docs/packaging.md)). See the
> [user guide](docs/user-guide.md).

## Quick start (Windows, PowerShell)

Requires Python 3.12 (64-bit) from python.org and Git.

```powershell
git clone https://github.com/auzac/Inferential.git
cd Inferential
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

Run the application:

```powershell
python -m inferential_studio
python -m inferential_studio examples\synthetic_distillation\sample_project.imsproj
```

Run the tests and lint:

```powershell
pytest
ruff check src tests
```

GUI tests run with Qt's offscreen platform by default, so no windows appear. Set
`$env:QT_QPA_PLATFORM="windows"` before `pytest` to watch them.

Build the Windows application folder + zip (see [docs/packaging.md](docs/packaging.md)):

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows_build.ps1
```

Regenerate the synthetic example data:

```powershell
inferential-studio-synth --out examples\synthetic_distillation
```

## Repository layout

```text
src/inferential_studio/
  app.py                  entry point (also: --smoke-test, --version)
  core/                   pure Python, never imports Qt (usable from Jupyter / Seeq Data Lab)
    schemas.py            typed project configuration + strict JSON conversion
    persistence.py        .imsproj read/write: atomic saves, safe loading, migrations
    timestamps.py         timestamp format detection and parsing to UTC (no guessing)
    importers.py          CSV/Excel reading, validation report, project data loading
    dataset.py            raw and normalized data layers
    statistics.py         descriptive statistics, correlations, coverage, X–Y screening
    rules.py              rule evaluation (typed conditions, AND/OR, minimum duration)
    quality.py            anomaly detectors producing reviewable flag events
    exclusions.py         unified exclusion masks with per-reason categories
    features.py           alignment to lab samples: resample, filter, windows, coverage, lag scan
    diagnostics.py        raw-vs-cleaned statistics, aligned correlations, VIF
    what_if.py            what-if step test run through the model's own features
    splits.py             chronological train/validation/test split, purge gap, time-aware CV
    models.py             MLR/PLS/PCR/Ridge pipelines, tuning, interpretation, metrics
    artifacts.py          JSON model files (.imsmodel), batch prediction, saved-results checks
    exports.py            predictions / feature matrix / batch results to CSV or Excel
    report.py, svg.py     self-contained HTML model card with inline SVG charts
    colours.py            data series colours shared by the on-screen charts and model-card SVG
    synthetic.py          synthetic distillation dataset with ground truth
    tasks.py              progress/cancellation primitives for long-running functions
  infrastructure/         logging, per-user settings, paths, background TaskManager (Qt)
  ui/                     main window, stage pages, dialogs, widgets (no algorithms)
tests/
  unit/                   core numerics: timestamps, import, rules, detectors, alignment, models …
  gui/                    pytest-qt: every working page, background workers, undo
  integration/            import → clean → align → train → save → reload with identical results
examples/synthetic_distillation/   artificial data + sample project
packaging/                PyInstaller spec, Windows build + installer (Inno Setup) scripts,
                          installer smoke test, icon generator, license inventory
docs/                     methodology conventions, file formats, packaging
licenses/                 LGPL/GPL v3 texts shipped with the application
```

## How the GUI is kept separate from the numerics

- `core/` contains every data, statistics and modelling function. It imports no Qt module; a
  test (`tests/unit/test_core_isolation.py`) imports all of `core` in a fresh interpreter and
  fails if PySide6, shiboken6 or pyqtgraph were loaded.
- Long-running core functions accept optional `progress` and `cancel` arguments (plain Python
  callables and a thread-safe flag from `core/tasks.py`), so they run the same way in a
  notebook.
- The GUI runs them through `infrastructure/workers.TaskManager`, which executes them on a
  thread pool and delivers progress, results, errors and cancellation back to the GUI thread
  via Qt signals. Workers never touch widgets.
- Widgets only display state and call core functions.

## Documentation

- [STRUCTURE.md](STRUCTURE.md) — what each file does and where to find things
- [docs/user-guide.md](docs/user-guide.md) — how to use the application
- [docs/methodology.md](docs/methodology.md) — time, import, exclusion, alignment, split, model and metric definitions
- [docs/benchmarks.md](docs/benchmarks.md) — measured import, statistics, alignment and training performance
- [docs/file-formats.md](docs/file-formats.md) — project, model file, export and example data formats
- [docs/packaging.md](docs/packaging.md) — Windows build, installer, smoke tests, release checklist, licensing
- [LICENSES.md](LICENSES.md) — third-party notices
- [CHANGELOG.md](CHANGELOG.md)

Example data in this repository is synthetic. Models built with this tool need independent
engineering validation before any online or closed-loop use.
