# Methodology and conventions

This document fixes the conventions every part of the application relies on. Code that
deviates from them is a bug. Later milestones extend this document; they do not change it
silently.

## Time

- Every instant inside the application is **timezone-aware UTC**. Source files are parsed with
  an explicit source timezone (applied only to values without an offset) and converted to UTC.
- The **display timezone** is a project setting used only when showing times to the user.
- Durations are stored as **float seconds**.
- Day/month order is never guessed. Format detection tests every candidate format against a
  sample spread over the whole column; if both day-first and month-first still fit, the full
  column is searched for a value that only one order can read (e.g. day 25). Only if none
  exists is the column **ambiguous**, and the import wizard will not continue until the user
  chooses. The decision is stored with the source (`format`, `dayfirst`).
- Values with an explicit UTC offset (`Z`, `+08:00`) are converted with that offset. A column
  that mixes values with and without offsets is rejected.
- Local times inside a daylight-saving change are **not** resolved by guessing: a time in the
  skipped hour (does not exist) or the repeated hour (occurs twice) becomes invalid, is excluded
  from the normalized data and is counted in the validation report.
- Unix epoch values are absolute; the source timezone does not apply. Excel serial dates are
  local times and require the workbook's 1900/1904 date system (read from the workbook).

## Import and data layers

- **Raw layer**: one entry per file row in file order: original timestamp text, parsed UTC time
  (NaT if invalid), numeric values, and a mask of cells that held text instead of a number.
  The file itself, identified by its SHA-256, is the ultimate raw record and is never modified.
- **Normalized layer**: rows with a valid time, sorted by UTC time, duplicate timestamps
  resolved by the source's explicit policy. Every row keeps its original file line number.
- Empty cells and the tokens `nan, NaN, NA, N/A, #N/A, null, NULL, None, -` are *missing*.
  Any other non-numeric text (e.g. `Bad`, `I/O Timeout`, `Shutdown`) is *invalid*: it becomes
  NaN in the numeric layer, is flagged in the text mask, and is reported per column with its
  most frequent tokens. Nothing is imputed.
- Duplicate timestamps: process data defaults to **error** (the user must choose *keep first*
  or *average*); lab data defaults to **keep all** (replicate samples stay separate rows).
- Gaps are reported when consecutive valid times are more than 5× the median interval (at least
  1 minute) apart. Sampling is reported irregular when more than 5 % of intervals differ from
  the median by over 10 %.
- Several process files are combined by an **outer join on exact timestamps** without filling,
  so cross-file statistics use only matching times. Resampling to a common grid is an explicit
  step in Signal Processing (Milestone 3).
- Re-importing the same file with the same settings yields identical timestamps (tested).

## Raw exploratory statistics

See the docstring of `core/statistics.py`; in short: std/variance with n−1; bias-corrected
skewness and excess kurtosis; CV only when all values share a sign; pairwise-complete Pearson
and exact Spearman (minimum 30 pairs). The **preliminary X–Y screen** pairs each lab sample with
the last process value at or before its sample time (within 2× the process interval), never a
later value. It is labelled preliminary in the UI because it ignores deadtime and averaging.

## Observations

- One supervised observation = **one valid lab/analyzer sample**. Sparse targets are never
  forward-filled onto the process grid.
- Lab samples carry a **sample (collection) time** and optionally a **result (availability)
  time**. Feature alignment uses the sample time. The result time is recorded for audit and for
  deployment-delay analysis; it is never mistaken for the sample time.

## Feature windows

For a target sample at time `t` and one predictor:

| Symbol | Meaning | Constraint |
|---|---|---|
| `d` | process deadtime (`deadtime_s`) | `d >= 0` |
| `L` | additional signed shift (`shift_s`) | positive = further into the past |
| `D = d + L` | effective delay (`effective_delay_s`), applied **once** | |
| `W` | window length (`window_s`) | `W > 0` |

Windows are **left-open, right-closed** so consecutive windows never double-count a sample:

- Lookback (default): `(t − D − W, t − D]`
- Centered: `(t − D − W/2, t − D + W/2]`

A feature is **causal** when its window ends at or before `t` (`window end offset <= 0`) and
its filter is trailing. Causality is computed (`FeatureSpec.is_causal`), shown in the UI and
recorded with every model. Non-causal features are allowed only for retrospective analysis and
are flagged.

## Filters

All offered filters (rolling mean, rolling median, EWMA) are **trailing** and time-based.
Centered smoothing is not offered for deployable pipelines.

## Exclusions and cleaning

- Raw imported data is never modified or deleted. Exclusions are masks over the normalized
  data, recomputed from definitions stored in the project, so they are reproducible and
  reversible (every edit is undoable).
- **Manual intervals** are **closed** `[start, end]` in UTC. With no listed variables they are
  operating-regime exclusions (whole rows, every variable, including target samples); with
  listed variables they affect only those variables.
- **Rules** are typed condition trees (variable, whitelisted operator, value(s)) combined with
  AND/OR, evaluated with NumPy; no text is ever evaluated as code. They run on the process
  data's own timestamps; nothing is interpolated. Comparisons are False where a value is
  missing; `is missing` = no valid number; `is invalid` = the cell held status text.
  *Minimum duration*: consecutive rows where the condition holds form a run; a time gap longer
  than 2× the typical interval ends a run; a run lasts `last − first + typical interval`.
  A rule either **excludes** (whole rows or listed variables) or only **flags** for review.
- **Detectors** (z-score, robust MAD z-score, IQR, Hampel, rate of change, flatline,
  engineering limits, status text) produce *events* (contiguous flagged samples of one
  variable). Definitions are in `core/quality.py`. Defaults are deliberately conservative.
  Statistics for z/MAD/IQR can be fitted on a chosen period so that, once a test period exists,
  it does not influence thresholds.
- **Review**: an event is *pending* until the user accepts it (excluded for that variable only)
  or rejects it (kept as valid — e.g. a genuine process upset). Pending flags are highlighted
  but **never exclude data**. A detector can opt in to *auto-exclude*, still reviewable. A
  pending event lying entirely inside data already excluded for another reason is reported as
  "inside an excluded period" and not counted as awaiting review. Decisions are stored by
  detector, variable and event time span; editing a detector's settings clears its decisions
  (after confirmation) because its events may change.
- One authoritative mask per variable = regime exclusions ∪ variable-specific intervals ∪
  variable-scoped rules ∪ accepted/auto flags. Each reason stays a separate category for the
  summary ("why is this excluded?"). A flag on one variable never invalidates another
  variable's value at the same time.
- Lab target samples are excluded by regime intervals, intervals naming the target, and
  accepted target flags. Whether a sample has enough valid process history behind it (rules,
  coverage, feature windows) is decided during alignment in Milestone 3.

## Synthetic example data

`inferential_studio.core.synthetic` produces an artificial distillation dataset with recorded
ground truth: target = first-order lag (τ = 15 min) of a linear combination of tray-15
temperature, pressure and feed temperature, followed by a 20-minute deadtime. Tests use this to
verify alignment and modelling against known answers.
