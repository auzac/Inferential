# User guide

This guide covers **v0.5.1**: all nine stages, from data import to the model card. The **Help**
panel on the right gives a short guide to the stage on screen; **F1** opens this guide inside
the application (offline).

**Messages.** Every status line, warning and error appears in the fixed **Messages** panel at
the bottom of the window — never inside the page, so charts and tables stay where they are.
Messages marked *active* describe the current state (e.g. "2 features read future data") and
disappear when the cause is fixed; timed entries are events (saved, imported, failed). The panel
shows the current stage by default (*All stages* shows everything); badges count warnings and
errors; double-click a message for details; *Copy* puts the list on the clipboard; *Clear*
removes the history.

## 1. Start with the example

1. **Data Import → Generate example dataset…** and choose a folder. The app writes an artificial
   distillation dataset (`tower_process.csv`, `lab_results.xlsx`), its ground truth and
   `sample_project.imsproj`, then offers to open the project.
2. Or open the copy shipped with the application: `examples\synthetic_distillation\sample_project.imsproj`.

Opening a project reads its data files in the background; the status bar shows progress and a
**Cancel** button. When loading finishes, the record counts appear at the bottom left.

## 2. Import your own data

**File → Import Data…** (Ctrl+I) or the buttons on the Data Import page start a five-step wizard:

1. **File** — process data (historian export: one timestamp column, many tags) or lab/analyzer
   results (sparse samples of the target). Drag a file onto the window or browse. CSV, TSV, TXT
   and Excel `.xlsx` are supported. The file is never modified.
2. **Read options** — delimiter, encoding and decimal separator are proposed from the file;
   check the preview and change them if columns look wrong. Set the line/row holding the
   column names if the export has extra lines at the top.
3. **Timestamps** — pick the timestamp column, its format and the **source timezone** (the
   plant's local time zone, used for values without an offset). The preview shows the value in
   the file, the local interpretation, the stored UTC time and the project display time.
   - If dates could be read either day-first or month-first (for example only days 1–12
     appear), the wizard says so and **Next stays disabled until you choose**. It never guesses.
   - For lab data you can also choose the **result available** time (when the lab reported the
     value) and a **sample ID** column. Models are aligned to the sample time.
4. **Columns** — choose columns to import, their variable names, role (candidate predictor,
   target, not used) and units. Units are never guessed; type them if you know them. A column
   with little numeric content in the preview is shown in red.
5. **Duplicates** — what to do with rows sharing a timestamp. For process data the default
   stops and reports them so you can decide; lab replicates are kept separately by default.

After **Finish** the file is read in the background. If it cannot be used (for example
duplicate timestamps with the "stop" policy) a message offers **Change settings…**.

## 3. Check the validation report

Select a source on the Data Import page. The report lists, with severity:

- timestamps that do not match the format, fall in a daylight-saving gap/overlap, or are blank
  (with file line numbers) — these rows are excluded from analysis but kept in the raw data;
- duplicate and out-of-order timestamps, gaps and irregular sampling;
- per column: cells holding text such as `Bad` or `I/O Timeout` (treated as invalid, not as
  numbers) and empty columns;
- for lab data: samples without a value, result times earlier than sample times, and samples
  outside the period covered by process data (a common sign of a timezone mistake).

## 4. Explore the raw data (Initial EDA)

Statistics are calculated in the background when you open the page. Tick variables in the list
(use the filter box when there are many tags).

- **Trends** — one plot per signal or grouped by units, on a shared time axis in the project's
  display timezone. The mouse wheel zooms, middle-drag pans, right-drag zooms to a box and
  **Fit all** resets. Drag left-to-right to highlight a time range across every chart (click
  anywhere else to remove it); click a point to pin its values — the readout below the
  variable list switches to *Selected point* and stays there until you click again. While
  unpinned, hovering shows the values at the
  cursor in the *At cursor* box, one line per signal (the last sample at or before it; nothing
  is shown inside gaps). Lab samples appear as markers. Right-click a plot → *Export* to save
  it as an image.
- **Statistics** — count, valid/missing/invalid %, quartiles, mean, std, variance, CV,
  skewness, excess kurtosis. Sort by clicking a column header.
- **Correlation** — Pearson or Spearman heatmap, optional |r| and grouping of similar
  variables. Hover for r and the number of pairs; click a cell for the scatter plot.
- **Scatter matrix** — for 2–6 ticked variables.
- **Distribution** — histogram of any variable or the target.
- **Coverage** — data availability over time per variable, and rows per time bin.
- **X–Y screening** — a preliminary ranking of predictors against the target using the last
  process value at or before each lab sample. It ignores process deadtime and averaging; final
  correlations come after alignment in Signal Processing.

Raw correlations are often dominated by shutdowns and load changes. They show association, not
causation — clean the data first before choosing predictors.

## 5. Clean the data (Cleaning & Exclusions)

Nothing you do here deletes data: exclusions are stored as definitions and can be switched
off, edited or undone (**Edit → Undo**, Ctrl+Z).

- **View**: *Raw* (original values), *Flagged* (original values; grey bands = excluded for all
  variables, orange = excluded for that variable only, red dots = flagged and awaiting your
  review), *Cleaned* (excluded points removed).
- **Exclude a period by hand**: **Select range**, drag the blue band's edges on any chart (all
  charts follow), then **Exclude selection…**. Choose a reason (shutdown, startup, transition,
  instrument/analyzer fault, maintenance, operator judgement, other), whether it applies to all
  variables or only some, and add a note. Start/end can be typed to the second. The *Manual
  exclusions* tab lists them; untick to restore, double-click to edit, *Zoom to* to inspect.
- **Rules** — *Add rule…* opens a visual builder: pick variable, operator (`<`, `<=`, `>`,
  `>=`, `==`, `!=`, between, outside, is missing, is invalid) and value(s). Use groups for
  combinations such as `(Feed_Flow < 50) AND (Pressure > 3)` or `(A < 1 OR B > 2) AND C < 3`.
  Optionally require the condition to hold continuously for a minimum time. The dialog shows
  how many rows match before you save. Rules can exclude whole rows (operating regime), only
  selected variables, or just flag for review. Untick a rule to see the data without it.
- **Anomaly detection** — *Add recommended* sets up a conservative spike filter (Hampel) and a
  stuck-value check (flatline) on candidate predictors, a status-text check where text cells
  exist, engineering limits where defined, and a robust z-score on the target. Each method's
  dialog explains in plain words what it does. Detection runs in the background. Then review the
  events: **Accept** (bad data — excluded for that variable), **Reject** (valid — e.g. a real
  feed disturbance), **Reset**. *Zoom to* (or double-click) shows an event on the chart.
  Real disturbances and turndowns are often valid data: an outlier is not automatically bad.
- **Summary** — per variable: valid, excluded, pending review and usable values; per reason:
  rows and approximate hours. The status bar shows the share of rows excluded for all
  variables.

## 6. Choose the target and predictors (Variable Selection)

- **Target (Y)** — usually a lab variable. Changing it refreshes every statistic.
- **Candidate predictors (X)** — tick process variables (search box for many tags; *All* /
  *None*). Every tick is one undoable step.
- **Ranking & VIF** — correlation of each *aligned* feature with Y on model-eligible lab
  samples (after the deadtime, window and filters of Signal Processing), the VIF of each
  predictor, and the simple screening correlation on raw and cleaned data for comparison. High
  VIF or `|r|` above the threshold you set are warnings: PLS, PCR and Ridge cope with correlated
  predictors; for MLR keep one variable of each correlated group.
- **Correlation (aligned)** — heatmap of the aligned features with Y in the first row.
- **Sample impact** — how many more samples would be usable without each predictor (a
  predictor with poor coverage can cost many samples).
- **Raw vs cleaned** — statistics before and after your exclusions.

## 7. Align predictors to the lab samples (Signal Processing)

Each lab sample becomes one row; every predictor contributes one value computed from a time
window before the sample time. Y is never filled in between samples.

- Select one or more predictors in the table, edit the settings on the right and press
  **Apply to selected** (or **Apply to all**):
  - **Deadtime** — how long a change in X takes to show in Y. The window ends this long before
    the sample time. **Shift** is an extra signed offset; the table's *Uses data from* column
    shows the combined effect, e.g. `(t − 50 min, t − 20 min]`.
  - **Window** and **Aggregation** (mean, median, min, max, standard deviation, last value).
  - **Placement** — *lookback* (default) or *centred*. A centred window, or a negative total
    delay, reads data recorded after the sample: the page shows a warning, because an online
    model cannot see the future.
  - **Filter** (trailing rolling mean/median or EWMA) and **Resample** (averages ending at each
    interval) are applied before the window.
  - **Min coverage** and **Max gap** — a window with too little data leaves that sample unusable
    instead of averaging a few points.
  - *Reject windows that overlap operating-regime exclusions* keeps shutdowns and start-ups out
    of every window.
- **Preview** — the cleaned signal (grey), the processed signal (blue) and, for the chosen lab
  sample (◀ ▶), its window (orange band) and the resulting feature value. The lab samples are
  shown below (grey = not usable). The status line explains the sample's result.
- **Lag sensitivity** — correlation with Y for a range of delays, using only the training
  period of the current split. *Use best delay as deadtime* copies the peak, but prefer a delay
  that is physically plausible.
- **Coverage report** — per predictor, how many samples fail each check. **Feature matrix** —
  every lab sample with its feature values, eligibility and the reason when it is not usable.

## 8. Configure the models (Model Setup)

- **Models to compare** — MLR, PLS, PCR and Ridge. For PLS/PCR the number of components, and
  for Ridge the alpha, is either tuned by cross-validation (search range) or fixed.
- **Scaling** — standardise (default), robust or none; always learned from training samples.
- **Split** — by share of eligible samples (default 60 % training, 20 % validation, 20 % final
  test, in time order) or by dates. The chart shows which samples fall where.
- **Purge gap** — samples just before a boundary whose windows would share process data with
  the next period are left out. Automatic = the longest feature history.
- **Cross-validation** — inside the training period only: expanding window (default) or blocked
  folds. *Prefer the simplest setting within one standard error* avoids over-fitted choices.
- Problems (too few samples, constant variables, strong autocorrelation …) are listed under the
  chart. **Train models** runs in the background and opens the next page.

## 9. Interpret the models (Training & Interpretation)

Choose a model at the top. **Coefficients** shows standardised coefficients (comparable across
predictors) and a table with coefficients in original units. **Cross-validation** shows the CV
error against components or alpha, the best and the chosen setting. For PLS and PCR:
**Loadings & VIP** (PLS: VIP, X weights, X loadings; PCR: PCA loadings), **Scores** and
**Explained variance**. Coefficients and VIP describe how the model uses its inputs, not physical
cause and effect. If you change settings after training, a banner says the results are out of
date until you train again.

## 10. Compare and validate (Validation)

- The **comparison table** ranks the models by validation RMSE (or CV RMSE without a validation
  period) and shows R², RMSE, MAE and bias (mean of predicted − actual) for training and
  validation. All models use exactly the same samples.
- **Show final test results…** reveals the untouched final test period. Look at it once, after
  choosing a model: changing models afterwards makes the test result exploratory.
- For the chosen model: metrics per period, **Actual vs predicted**, **Trend** at the lab sample
  times, residuals against time, prediction and any predictor, residual histogram and
  autocorrelation, and **Applicability** — the training range of every predictor and how many
  later samples fall outside it.
- Prefer a model that is accurate *and* simple, stable and physically sensible. Good fit
  statistics alone do not make a model ready for online use.

## 11. Export and report (Export & Report)

Train first; if settings changed since training, the page asks you to train again so the exports
match the project.

- **Model card** — choose the model (best ranked first), give a title, model version, owner,
  purpose and known limitations (kept in the project), then **Save model card (HTML)…**. The
  card is one self-contained file (no internet, no scripts): summary, data sources with
  checksums, every exclusion, rule and detector review, the feature definition of every
  predictor, the equation, coefficients (and VIP for PLS), validation metrics and plots,
  training ranges, limitations and versions. Open it in any browser; print it to PDF from there.
  Final-test results appear only after you revealed them on the Validation page and keep
  *Include final-test results* ticked.
- **Export predictions…** — one row per evaluated lab sample: sample time in UTC and local
  time, sample id, period, actual, predicted, residual and the feature values (Excel or CSV).
- **Export feature matrix…** — every lab sample with its features, eligibility and the reason
  when it is not usable (for audit).
- **Save model file (.imsmodel)…** — the model as plain JSON: feature definitions (deadtime,
  window, filter, coverage rules), intercept and coefficients, training ranges, metrics, data
  checksums and versions. It contains no executable code, so it is safe to e-mail and open.
- **Copy equation** — the model as `Y = b0 + b1 × f[X1] + …` on the aligned features, with the
  caveat that it is exact only when every feature is computed exactly as defined.

### Apply a saved model to data (batch prediction)

**Load model file…**, then **Apply model**. The model's tags must exist in this project (same
names; units are checked where both sides have them). Choose *at the lab sample times* (compares
with the lab: RMSE and bias are shown) or *at regular intervals*. Features are rebuilt with the
model's own definitions and coverage rules: a time whose inputs are missing or excluded gets no
prediction and the reason is listed; predictions that use inputs outside the training range are
marked orange (extrapolation). **Export results…** writes the table. To check a model on new
plant data, import the new files into a project (the same tag names) and apply the model there.

## 12. Projects