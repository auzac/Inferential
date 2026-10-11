# User guide

This guide grows with each release. It covers what works in **v0.1.0**: projects, data import
and raw exploration (stages 1–2). Stages 3–9 show a note saying which milestone delivers them.

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
  display timezone. Mouse wheel zooms, drag pans, **Fit all** resets. The readout shows the
  values at the cursor (the last sample at or before it; nothing is shown inside gaps). Lab
  samples appear as markers. Right-click a plot → *Export* to save it as an image.
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

## 5. Projects

- **Save / Save As** write a `.imsproj` file holding all settings and references to your data
  files (with checksums). Saving is atomic, so an interrupted save cannot damage the previous
  file. Saving to another folder keeps the data references valid.
- If a data file has changed since import, opening the project tells you which one and asks
  whether to accept the new contents.
- **File → Project Settings…** sets the project name, display timezone and notes.
- **Tools → Settings…** sets the default display timezone for new projects, your name and the
  log level. **Tools → Open Log Folder** shows the application logs (no process data values are
  logged).
