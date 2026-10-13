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
- Every source has **one timestamp column**; for lab data it is the **sample (collection)
  time**, and feature alignment uses it. A separate result-availability time is not imported
  by the wizard (projects that already record one keep it for audit only; it is never mistaken
  for the sample time).

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
  **Hampel identifier** (Pearson 2002): for each value, the centred time window
  `[t − W/2, t + W/2]` gives the local median `m` and `MAD = median |xⱼ − m|` over that window;
  the value is flagged when `|x − m| > max(k · 1.4826 · MAD, minimum deviation)` (at least three
  values in the window). **Flatline**: values that stay inside a band of width *tolerance*
  (max − min of the run) for at least the given duration; a gap longer than 2× the typical
  interval ends the run, and a slow drift made of small steps is not a flatline.
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
  coverage, feature windows) is decided during alignment (below).

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
recorded with every training run. Non-causal features are allowed only for retrospective
analysis and are flagged on the Signal Processing page and in every training run.

## Feature processing order (`core/features.py`)

For each predictor, in this order:

1. **Clean** — the normalized process values minus every excluded cell (manual intervals,
   rules, accepted or auto-excluded flags) and minus blank/text cells. Nothing is interpolated
   or carried forward.
2. **Resample** (optional, interval `R`) — valid values are averaged in bins `(kR − R, kR]`,
   labelled at the bin **end** `kR` (bins aligned to the UTC epoch). A resampled value never
   uses data recorded after its label time. Bins without valid data stay empty.
3. **Filter** (optional, always trailing, time-based) — rolling mean or median over
   `(τ − F, τ]`, or an EWMA with time-based half-life (pandas `ewm(halflife, times)`), evaluated
   only at valid points and only from valid points. The start of the record and the edges of
   excluded periods simply have fewer points in the filter window. Centred smoothing is not
   offered.
4. **Aggregate** each lab sample's window: mean, median, minimum, maximum, standard deviation
   (`n − 1`, needs two values) or last valid value.

Quality checks per window (a failure leaves the feature empty for that sample):

| Check | Definition |
|---|---|
| Coverage | (valid values in the window × sampling interval) ÷ `W`, capped at 1. The sampling interval is the variable's median spacing, or `R` when resampled. Must be ≥ `min_coverage` (default 80 %). |
| Largest gap | Longest time inside the window without a valid value; the window edges count as boundaries, so regular sampling gives one sampling interval. Must be ≤ `max_gap_s` when set. |
| Too few values | Standard deviation needs at least two values. |
| Regime overlap | With *Reject windows that overlap operating-regime exclusions* (on by default), a window touching any row excluded for all signals (shutdown, start-up …) fails even if coverage passes. |

A feature's **data history** is its window plus the filter's memory (rolling window, or five
EWMA half-lives) plus one resample interval. The longest history sets the automatic purge gap.

Not yet offered (planned): bounded interpolation / carry-forward of process X with explicit
consent, and composite lab samples with a collection interval.

## Lab alignment and eligibility

- One feature row per lab sample, at the **sample (collection) time**. Replicates (same time)
  stay separate rows and share their windows; result (availability) times and sample ids are
  carried along for audit.
- Y is never forward-filled or interpolated; process rows without a lab sample produce no
  supervised row.
- A row is **model-eligible** only when Y is a valid number, Y is not excluded, and every
  selected predictor passes all window checks. Unreviewed (pending) flags on Y do not exclude
  it; they are counted on the Signal Processing page.
- The coverage report counts, per predictor and among samples with a usable Y, how many fail
  each check; the feature matrix lists the reason for every ineligible row.

## Variable diagnostics (`core/diagnostics.py`)

- **Raw vs cleaned**: summary statistics of the process data before and after exclusions, plus
  the screening X–Y correlation (last value at or before each lab sample, no deadtime) on both.
- **Aligned correlations**: Pearson and Spearman between each aligned feature and Y, and among
  features, on model-eligible rows only.
- **VIF** of predictor j = 1 / (1 − R²ⱼ) from an OLS regression with intercept of xⱼ on the
  other predictors (complete rows). Constant predictors have no VIF; an exact linear combination
  has VIF = ∞. `|r| > 0.9` (adjustable) and `VIF > 10` are warnings, never automatic deletions.
- **Sample impact**: for each predictor, how many more samples would be eligible without it.
- **Lag sensitivity**: correlation of one aligned predictor with Y over candidate effective
  delays. A valid training cutoff is required; there is no all-history fallback. Valid,
  non-excluded targets at or before the last eligible training timestamp are considered.
  All candidate windows must be causal. Predictor exclusions, coverage, gap limits and the
  project's regime-overlap policy are applied for each candidate. Pair counts can vary by
  delay; eligibility of other predictors is not required. Results are discarded if scan
  inputs change, and obsolete results cannot be applied. It is exploratory (autocorrelation
  and common trends also correlate); choose a physically plausible delay.

## Splits and cross-validation (`core/splits.py`)

- **Chronological** train → validation → final test, by share of eligible samples (default
  60 / 20 / 20) or by explicit boundary instants (train `t < train_end`, validation
  `train_end ≤ t < test_start`, test `t ≥ test_start`). Rows are never shuffled; replicate
  samples share a time and therefore a split.
- **Purge gap**: two samples whose feature histories overlap share process data. Before each
  boundary the **earlier** period drops samples closer than the purge gap to the first sample
  of the next period; the final test period is never trimmed. Default gap = the longest feature
  history; it can be set longer (e.g. for slow process dynamics).
- **Cross-validation** runs only inside the training period, on contiguous blocks of sample
  times: *expanding window* (rolling origin, default) trains on all earlier blocks and
  validates on the next one; *blocked k-fold* holds out each block in turn. The purge gap
  applies at every fold boundary (both sides for blocked folds). Impossible fold layouts are
  reported.

## Models (`core/models.py`)

Each candidate is a scikit-learn pipeline `scaler → estimator`, fitted on training samples only:

| Model | Estimator | Hyperparameter |
|---|---|---|
| MLR | `LinearRegression` | — |
| PLS | `PLSRegression(scale=False)` after the chosen scaler | components 1…max (CV) or fixed |
| PCR | `PCA(svd_solver="full")` → `LinearRegression` | components 1…max (CV) or fixed |
| Ridge | `Ridge` | alpha on a log grid (CV) or fixed |

Scaling: standardise (mean/SD), robust (median/IQR) or none — always fitted on training samples
only. Tuning minimises the mean CV RMSE; with the *one-standard-error* option (default) the
simplest setting (fewest components, largest alpha) whose mean CV RMSE is within one standard
error of the best is chosen. Component counts are capped at the number of predictors and at the
smallest CV training fold − 1. When the best value lies at the edge of the range tried (smallest
or largest alpha, or the maximum number of components allowed) the model carries a warning,
because the optimum may lie outside the range. The validation period ranks the models; the final
test period is evaluated with the same fitted pipelines, but the application hides its metrics
until the user explicitly reveals them. All models of one run share the same split.

Invalid situations are reported, not hidden: fewer than 10 training samples, constant target or
predictor in the training period, MLR with too few samples for its predictors (or linearly
dependent predictors — the minimum-norm solution is shown with a warning), more components than
possible, impossible CV folds, and strongly autocorrelated targets (effective sample size
`n(1 − ρ₁)/(1 + ρ₁)` from the lag-1 autocorrelation of consecutive lab samples).

## Interpretation

- **Coefficients in original units**: every pipeline is affine, so `ŷ = b₀ + Σ bⱼxⱼ` is
  recovered exactly from predictions at the origin and at unit vectors. This is the model as it
  runs on the aligned, filtered **features** — not on raw tag values.
- **Standardised coefficients**: `bⱼ · sd(xⱼ) / sd(y)` with training standard deviations.
- **PLS**: X weights `W`, X loadings `P`, Y loadings `q` and scores `T` are distinct objects and
  are labelled as such. **VIP** (Wold): `VIPⱼ = sqrt(p · Σₐ SSₐ (wⱼₐ/‖wₐ‖)² / Σₐ SSₐ)` with
  `SSₐ = qₐ² · tₐᵀtₐ` on training data (so `Σ VIP² = p`). X variance per component =
  `‖tₐpₐᵀ‖² / ‖X₀‖²`; Y variance per component = `SSₐ / ‖y₀‖²`.
- **PCR**: PCA loadings (eigenvectors), scores and explained X-variance ratio — which is not
  the share of Y explained.
- No coefficient confidence intervals or p-values are shown: lab samples are autocorrelated
  and predictors correlated, so textbook OLS standard errors would overstate certainty.

## Metrics

Per split, with `e = predicted − actual`: `n`; `R² = 1 − Σe² / Σ(y − ȳ_split)²` (empty when the
split's Y is constant or `n < 2`); adjusted R² (training split only; `p` = predictors for
MLR/Ridge, components for PLS/PCR); `RMSE = sqrt(mean e²)`; `MAE = mean |e|`;
**bias = mean(predicted − actual)**; max |e|; RMSE as a percentage of the training standard
deviation of Y. Residual plots use `actual − predicted`. Residual autocorrelation is over
consecutive lab samples with ±1.96/√n reference lines. Applicability: training minimum,
maximum, mean and SD of every feature, and how many validation/test samples fall outside the
training range.

**What-if step test** (`core/what_if.py`): synthetic inputs on a regular grid (1 min, finer for
windows under 10 min) sit at their training means, preceded by a steady-state lead-in long
enough for every window and filter (20 half-lives for an EWMA, whose weights are normalised over
the available history). At the step time the chosen inputs change by their steps. Features are
built with the model's stored definitions by the same code as training and batch prediction,
and `ŷ = b₀ + Σ bⱼ fⱼ` is evaluated at every grid time. The final change is exactly `Σ bⱼ·Δⱼ`;
the first response is the first time the prediction moves (the deadtime for lookback windows,
before the step for centred ones) and *settled* is when it stays at or beyond 95 % of the final
change. Tested against the discrete deadtime + moving-average response and the first-order
EWMA response `1 − 0.5^(Δt/h)`.

## Comparing results with other statistics packages

The estimators are the standard ones and are checked against independent textbook
implementations in `tests/unit/test_reference_methods.py` (OLS, NIPALS PLS1 and VIP, PCA + OLS,
closed-form ridge, the one-standard-error rule, Hampel, window features, trailing filters).
Numbers can still differ from Minitab, JMP or SIMCA because of conventions:

| Quantity | Here | Typical elsewhere |
|---|---|---|
| Training error | `RMSE = √(SSE/n)` | Minitab *S*, JMP *Root Mean Square Error*: `√(SSE/(n−p−1))` (slightly larger) |
| Quartiles, IQR | linear interpolation (`(n−1)p`, Excel/R type 7) | Minitab, JMP: `(n+1)p` |
| Ridge penalty | `‖y − Xb‖² + α‖b‖²` on the scaled predictors (scikit-learn; not divided by n) | other tools scale λ differently: compare fits, not α values |
| PLS / PCR components | sign of a component is arbitrary | may be flipped; predictions, coefficients and VIP are unaffected |
| Scaling | population SD (`n`) | often `n − 1`; predictions of MLR, PLS and PCR are unaffected |
| R² of validation / test | uses that period's own mean | some tools' "predicted R²/Q²" use the training mean |
| Collinear MLR | minimum-norm solution with a warning | redundant terms are dropped |
| Coefficient SE, p-values | not shown (autocorrelated samples) | shown |

## Model files and batch prediction (`core/artifacts.py`)

- A saved model is the list of feature definitions plus `ŷ = b₀ + Σ bⱼ fⱼ` — exact for every
  v1 model type because each pipeline is affine in its features. No Python objects are stored.
- **Prediction** rebuilds each feature with the stored definition (cleaning masks of the data
  it is applied to, resample, trailing filter, window, coverage and gap checks, regime-overlap
  rule). A time whose inputs fail any check gets **no** prediction and a reason — never a value
  computed from partial or stale data. Missing tags (or units that differ where both are known)
  stop the prediction with an error.
- **Extrapolation**: for each prediction, the features outside the training range
  (min–max) are listed and counted.
- Applying a model to the samples it was trained on reproduces its predictions to floating-point
  precision (tested).

## Reproducibility of saved projects

Alignment, splitting, CV and fitting are deterministic (no random seeds involved: PCA uses the
full SVD, PLS NIPALS is deterministic). A project saved after training stores a summary of the
results; on reopening, the models are trained again and compared (coefficients and metrics,
relative tolerance 10⁻⁶). Differences — e.g. changed data files or library versions — are
reported, not hidden.

## Synthetic example data

`inferential_studio.core.synthetic` produces an artificial distillation dataset with recorded
ground truth: target = first-order lag (τ = 15 min) of a linear combination of tray-15
temperature, pressure and feed temperature, followed by a 20-minute deadtime. Tests use this to
verify alignment and modelling against known answers.
