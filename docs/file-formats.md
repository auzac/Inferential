# File formats

## Project file (`.imsproj`), schema version 2

A ZIP archive with exactly these entries:

| Entry | Content |
|---|---|
| `manifest.json` | `format` = `"inferential-model-studio-project"`, `schema_version`, `app_version`, `saved_at` (UTC ISO 8601), `libraries` (Python, NumPy, pandas, SciPy, scikit-learn versions) |
| `project.json` | The project configuration (see `core/schemas.py`): name, display timezone, data sources, variables, per-predictor feature settings, manual exclusions, notes, created/modified times |

Rules:

- All timestamps are ISO 8601 with an explicit offset and are stored in UTC.
- Durations are seconds.
- Data source paths may be **relative to the project file's folder** (used by the example) or
  absolute. Each source records the SHA-256 of the file at import; a mismatch on opening is
  reported to the user.
- The project references the original data files (reference mode). On opening, the files are
  read again in the background; a changed checksum is reported and the user decides whether to
  accept the new contents. Embedded ("portable") data and fitted model artifacts are added in
  later milestones with an explicit schema migration.
- *Save As* to another folder rewrites relative source paths so they still point at the same
  files (relative if the data sits inside the new project folder, absolute otherwise).

Schema history: **1** (v0.0.1) initial; **2** (v0.1.0) adds `TimestampKind.native` (date/time
cells), `DataSource.duplicate_policy` and `DataSource.sample_id_column`. Version 1 files are
upgraded automatically on opening (no data changes needed).

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
| `lab_results.xlsx` | Sheet `Lab`: `Sample_Timestamp`, `Result_Timestamp` (`yyyy-mm-dd HH:MM`, local), `Product_Composition`, `Sample_ID` |
| `ground_truth.json` | Generator configuration, target dynamics and every injected event (UTC) |
| `sample_project.imsproj` | Project referencing the two data files |

Regenerate with `inferential-studio-synth --out examples/synthetic_distillation`.
