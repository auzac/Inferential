# File formats

## Project file (`.imsproj`), schema version 5

A ZIP archive with these entries (anything else is rejected when opening):

| Entry | Content |
|---|---|
| `manifest.json` | `format` = `"inferential-model-studio-project"`, `schema_version`, `app_version`, `saved_at` (UTC ISO 8601), `libraries` (Python, NumPy, pandas, SciPy, scikit-learn versions), `storage` (`"reference"` or `"portable"`), `embedded` (list of `{source_id, entry, sha256, size}`) |
| `project.json` | The project configuration (see `core/schemas.py`): name, display timezone, data sources, variables, per-predictor feature settings, manual exclusions, rules, detectors, flag decisions, alignment options, modelling configuration (scaling, split, CV, candidate models), model-card details (`report`), notes, created/modified times |
| `results.json` | Optional: summary of the last (up-to-date) training run — per model hyperparameters, intercept, coefficients and metrics (final-test metrics only if revealed), split counts, whether the final test was revealed, versions. Used to check that reopening reproduces the results. |
| `data/<source id>/<file name>` | Portable projects only: byte-exact copies of the source files |

Rules:

- All timestamps are ISO 8601 with an explicit offset and are stored in UTC.
- Durations are seconds.
- **Reference mode** (default): source paths may be **relative to the project file's folder**
  (used by the example) or absolute. Each source records the SHA-256 of the file at import; a
  mismatch on opening is reported and the user decides whether to accept the new contents.
  *Save As* to another folder rewrites relative source paths so they still point at the same
  files.
- **Portable mode** (*File → Save Portable Copy…*): the source files are embedded under `data/`.
  Saving refuses a file whose checksum differs from the one recorded at import. Opening copies
  the embedded files to the per-user cache (`%LOCALAPPDATA%\InferentialModelStudio\cache`),
  writing only manifest-listed entries with validated names, enforcing sizes while copying and
  verifying every SHA-256.
- Security: entries are read into memory or copied as above; names with `..`, absolute paths,
  drive letters or backslashes are rejected; JSON entries are size-checked before reading; no
  pickle/joblib content is ever read.
- Trained scikit-learn objects are not stored. Models are recomputed deterministically from the
  saved settings; `results.json` lets the application confirm the result is the same.

Schema history: **1** (v0.0.1) initial; **2** (v0.1.0) adds `TimestampKind.native` (date/time
cells), `DataSource.duplicate_policy` and `DataSource.sample_id_column`; **3** (v0.2.0) adds `rules`
(typed condition trees), `detectors` and `flag_decisions`; **4** (v0.4.0) adds `alignment` and
`modelling`; **5** (v0.5.0) adds the `report` details, optional `results.json`, embedded data
and the `storage`/`embedded` manifest fields. Older files are upgraded automatically on opening
(defaults are used for new sections).

## Model file (`.imsmodel`), artifact version 1

A ZIP archive with exactly `manifest.json` (`format` = `"inferential-model-studio-model"`,
`artifact_version`, `app_version`) and `model.json`. `model.json` is plain JSON:

| Key | Content |
|---|---|
| `target` | name, units, source id and column |
| `predictors` | in model order: name, units, source column, the full `feature` definition (deadtime, shift, window, placement, aggregation, filter, resample, coverage, maximum gap), a plain-language description, causal flag, data span, training range (min, max, mean, SD), coefficient, standardised coefficient |
| `alignment` | regime-overlap option used to build the features |
| `model` | kind, label, hyperparameters, scaling, intercept, coefficients, CV curve, PLS (VIP, X weights, X loadings, Y loadings, explained variance) or PCR details, warnings |
| `split`, `metrics` | split configuration, purge gap, periods (start/end/samples), metrics per period (final test only when revealed), `test_revealed` |
| provenance | `created_at`, `app_version`, `libraries`, `model_version`, `owner`, `purpose`, project name/file/notes, data sources with SHA-256, `display_timezone`, `equation` and `equation_caveat` |

Every model of v1 is an affine function of its features, so the intercept and coefficients (in
original feature units, scaling folded in) reproduce the trained pipeline exactly. Loading
validates the format, sizes and that the predictor list matches the coefficients.

## Exports

| Export | Format |
|---|---|
| Model card | One HTML file, inline CSS and SVG charts, no scripts or external references; print to PDF from a browser |
| Predictions | `.xlsx` (sheets `Predictions`, `Notes`) or `.csv` (UTF-8 with BOM): `sample_time_utc`, `sample_time_local (<zone>)`, `sample_id`, `result_time_*` (only for projects that record a result time), `period`, actual, predicted, `residual (actual - predicted)`, `f[<tag>]` features |
| Feature matrix | Every lab sample: times, ids, target, features, `eligible`, `reason` |
| Batch prediction | `time_utc`, `time_local`, `prediction`, `status`, `reason`, `n_outside_training_range`, `outside_training_range`, `f[<tag>]` |

## Supported input files

| Type | Notes |
|---|---|
| CSV / TSV / TXT | Delimiter `,` `;` tab `|`; encodings UTF-8 (with/without BOM), cp1252, latin-1, UTF-16; decimal point or comma (no thousands separators); header on any line (lines above are skipped) |
| Excel `.xlsx` / `.xlsm` | Any worksheet and header row; date cells, Excel serial numbers (1900 or 1904 system) or text timestamps |
| Legacy `.xls` | Not supported — save as `.xlsx` |

Timestamp formats: ISO 8601 (with or without `T`, fractional seconds, `Z`/offsets);
`dd/mm/yyyy`, `mm/dd/yyyy`, `yyyy/mm/dd` with `/`, `-` or `.` separators and 2- or 4-digit
years; month names (`11-Oct-2026`, `Oct 11, 2026`); 24-hour or AM/PM times with optional
seconds and fractions; Excel serial dates; Unix seconds or milliseconds.

Reading is strict and safe:

- Unknown entries, unsafe entry names, unknown JSON fields and oversized entries are rejected.
- Nothing is extracted to disk and no pickle/joblib content is ever loaded from a project file.
- A file with a newer schema version than the application supports is refused with a message
  asking the user to upgrade; older versions are upgraded by explicit migration functions.

Saving is atomic (temporary file in the same folder, then rename), so an interrupted save never
damages the previous version.

## Example data (`examples/synthetic_distillation/`)

| File | Content |
|---|---|
| `tower_process.csv` | 1-minute process data, plant local time (Asia/Singapore), **day-first** `dd/mm/yyyy HH:MM` |
| `lab_results.xlsx` | Sheet `Lab`: `Sample_Timestamp` (`yyyy-mm-dd HH:MM`, local), `Product_Composition`, `Sample_ID` |
| `ground_truth.json` | Generator configuration, target dynamics and every injected event (UTC) |
| `sample_project.imsproj` | Project referencing the two data files (reference mode) |

Regenerate with `inferential-studio-synth --out examples/synthetic_distillation`.
