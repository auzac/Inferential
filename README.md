# Inferential Model Studio

Offline Windows desktop workbench for building, validating and documenting industrial
inferential models (soft sensors), starting with distillation columns.

Python · PySide6 (Qt 6) · PyQtGraph · pandas · scikit-learn. No cloud services, no telemetry.

> **Status: v0.4.0 — Milestones 3 and 4 (alignment, modelling and validation).** Import with
> a timestamp wizard, raw EDA, cleaning with reviewable flags, variable selection with VIF,
> per-predictor deadtime/window/filter alignment to lab samples, MLR / PLS / PCR / Ridge with
> chronological splits and time-aware CV, interpretation and validation plots. Export, model
> artifacts and the report arrive in Milestone 5. See the [user guide](docs/user-guide.md).

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
    splits.py             chronological train/validation/test split, purge gap, time-aware CV
    models.py             MLR/PLS/PCR/Ridge pipelines, tuning, interpretation, metrics
    synthetic.py          synthetic distillation dataset with ground truth
    tasks.py              progress/cancellation primitives for long-running functions
  infrastructure/         logging, per-user settings, paths, background TaskManager (Qt)
  ui/                     main window, stage pages, dialogs, widgets (no algorithms)
tests/
  unit/                   core numerics: timestamps, import, rules, detectors, alignment, models …
  gui/                    pytest-qt: every working page, background workers, undo
  integration/            import → clean → align → train → save → reload with identical results
examples/synthetic_distillation/   artificial data + sample project
packaging/                PyInstaller spec, Windows build script, license inventory
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

- [docs/user-guide.md](docs/user-guide.md) — how to use the application
- [docs/methodology.md](docs/methodology.md) — time, import, exclusion, alignment, split, model and metric definitions
- [docs/benchmarks.md](docs/benchmarks.md) — measured import, statistics, alignment and training performance
- [docs/file-formats.md](docs/file-formats.md) — project file and example data formats
- [docs/packaging.md](docs/packaging.md) — Windows build, licensing obligations
- [LICENSES.md](LICENSES.md) — third-party notices
- [CHANGELOG.md](CHANGELOG.md)

Example data in this repository is synthetic. Models built with this tool need independent
engineering validation before any online or closed-loop use.
