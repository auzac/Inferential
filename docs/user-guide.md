# User guide

This guide grows with each release. It covers what works in **v0.2.0**: projects, data import,
raw exploration and cleaning (stages 1–3). Stages 4–9 show which milestone delivers them.

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
  display timezone. Mouse wheel zooms, drag pans, **Fit all** resets. Hovering shows the values
  at the cursor in the *At cursor* box below the variable list, one line per signal (the last
  sample at or before it; nothing is shown inside gaps). Lab samples appear as markers.
  Right-click a plot → *Export* to save it as an image.
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
causation — clean the data first (Milestone 2) before choosing predictors.

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

## 6. Projects

- **Save / Save As** write a `.imsproj` file holding all settings and references to your data
  files (with checksums). Saving is atomic, so an interrupted save cannot damage the previous
  file. Saving to another folder keeps the data references valid.
- If a data file has changed since import, opening the project tells you which one and asks
  whether to accept the new contents.
- **File → Project Settings…** sets the project name, display timezone and notes.
- **Tools → Settings…** sets the default display timezone for new projects, your name and the
  log level. **Tools → Open Log Folder** shows the application logs (no process data values are
  logged).
