# File formats

## Project file (`.imsproj`), schema version 1

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
- In schema version 1 the project references the original data files (reference mode). Embedded
  ("portable") data and fitted model artifacts are added in later milestones with an explicit
  schema migration.

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
