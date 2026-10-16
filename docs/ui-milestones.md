# UI improvement milestones

Scope agreed 17 October 2026. Audit baseline: `00b6703`; implementation starts at
`4a31aea` and preserves the subsequent static-chart navigation fix.

## Milestone 1 — Responsive table sorting (audit finding 1)

- [x] Replace repeated per-cell proxy comparisons with a vectorized row order.
- [x] Run large sorts in a worker; reject results after data or sort requests change.
- [x] Preserve selected records, row labels, row colours and missing-value ordering.
- [x] Verify ascending/descending, replacement data and responsiveness on 50,000 rows.
- [x] Record before/after timings and run the relevant checks.

## Milestone 2 — Background table exports (audit finding 2)

- [x] Run prediction, feature-matrix and batch-result preparation/writing in workers.
- [x] Capture export inputs before dispatch and report completion only after writing succeeds.
- [x] Show progress and support cancellation through the existing task display.
- [x] Write atomically; preserve existing files and remove temporary files on failure/cancellation.
- [x] Verify file contents, cancellation, failure handling and GUI responsiveness.
- [x] Record before/after timings and run the relevant checks.

## Milestone 3 — Usable small-window layouts (audit finding 3)

- [x] Keep Validation diagnostics usable at 960×640 without overlapping controls.
- [x] Keep Cleaning signal plots readable by scrolling instead of squeezing chart heights.
- [x] Reclaim comparison space on compact Validation layouts without removing access to metrics.
- [x] Check both minimum and desktop sizes and light/dark themes.
- [x] Add layout regression checks and save before/after screenshots.

## Milestone 4 — What-if calculation correctness (audit finding 4)

**Reassessed: Claude's fix passes.** Reviewed `5f38dc8`, included in main through PR #18.

- [x] Reassess Claude's fix when available.
- [x] Verify standard-deviation features have the correct initial/final prediction and transient response.
- [x] Verify the live UI's current/settling summary agrees with the simulation.

The original reproduction (intercept 1, coefficient 2, constant input 10, step +3,
10-minute standard-deviation window) now starts and settles at **1**, with a genuine
transient and **zero final change**. The old reported baseline 21 and final change +6
are gone. The live UI no longer offers the old first-response/95% timing summary and
omits the misleading coefficient-times-raw-step label for standard-deviation inputs.
Added regression checks cover both numerical output and the displayed summary.

## Milestone 5 — Expensive and hidden simulations (audit finding 5)

**Completed locally after reassessment.** Claude stopped the hidden clock and made theme
changes redraw-only. Remaining initialization, slider and live calculations still ran
on the GUI thread, and mixed short/long feature windows could create huge histories.

- [x] Reassess against Claude's updated What-if implementation.
- [x] Defer numerical initialization until the tab is visible; cancel obsolete hidden work.
- [x] Run calculation in a single background worker; coalesce slider edits and reject stale results.
- [x] Reuse steady-state results between live ticks and redraw cached values on theme changes.
- [x] Bound the visible history to about 2,000 samples while preserving the feature calculation grid.
- [x] Skip invisible catch-up history after a delayed worker; explain display sampling of long trends.
- [x] Recheck long-run slider, model-change and theme-switch responsiveness.

A model with 6-second and 7-day feature windows retains a **1-second calculation grid**,
but initializes **1,999 display samples** in **1.03 s** in the background. Hidden model
setup took 0.9 ms; slider dispatch took 0.13 ms and its result arrived in 147 ms. A 10 ms
GUI timer fired 51 times during initialization; its largest measured gap was 0.43 s.
These are macOS observations, not a Windows performance guarantee. Raw feature-history
construction still uses memory and can cause some Python/GIL-related jitter. Long-trend
sampling may miss short transients; the UI now explicitly says so.

Tests verify no hidden numerical work, no theme-triggered recalculation, a live event
loop during worker execution, coalescing to the latest slider values, cancellation,
bounded history and correct standard-deviation summaries.

## Milestone 6 — Ranking metric visibility (audit finding 6)

**Completed locally.**

- [x] Reassess comparison-column order after the compact-layout change.
- [x] Put the active ranking metric first, immediately beside the model's row label.
- [x] Label the active metric “ranking”.
- [x] Bound long model row labels so they cannot push the metric out of view; keep full labels in tooltips.
- [x] Put CV RMSE immediately after the candidate name in “Best by input count”.
- [x] Preserve the existing ranking calculations, detailed columns and final-test gating.
- [x] Verify ordinary comparisons, no-validation CV fallback and subset searches at 960×640 and 1440×900.

The model-comparison section remains collapsible on short screens (milestone 3). When
opened, the ranking metric is visible at the initial horizontal scroll position.

## Milestone 7 — Accessible control names (audit finding 7)

**Deferred for reassessment.**

- [ ] Recheck the theme switch, navigation toggle and caption controls with Qt accessibility queries.
- [ ] Add missing names and visible keyboard focus where needed.
- [ ] Verify names, roles and states with a screen reader on the target platform.

## Milestone 8 — Primary-button contrast (audit finding 8)

**Deferred for reassessment.**

- [ ] Recalculate light-theme primary-button text contrast after any palette changes.
- [ ] Adjust the accent if it still falls below the chosen 4.5:1 normal-text benchmark.
- [ ] Verify default, hover, pressed and focus states visually.

## Local verification for milestones 1–3

- Implementation commits: milestone 1 `b57ed6d`, milestone 2 `c11f53e`, milestone 3 `e3fd57c`.
- Local branch: `fix/ui-milestones-1-3`.
- Large numeric sort, 50,000 rows: **8.35 s before → 0.014 s after**; dispatch returned in 0.3 ms.
- XLSX export, 50,000 × 20: writing still takes about **7.3 s**, but runs in a worker.
  A 10 ms GUI timer fired 430 times during writing; the longest observed interval was 0.51 s.
  This removes the seven-second synchronous stall, not all Python/GIL-related jitter.
- Cancellation/failure tests preserve the previous destination and remove temporary files.
- Record selection, row labels/colours, nullable numbers, full integer precision, stale results
  and repeated sort requests have regression coverage.
- Compact Validation and scrolling trend layouts have regression coverage. Light/dark screenshots
  were reviewed at 960×640 and 1440×900.
- Lint passes. The broad test run excluding the macOS offscreen import-wizard crash produced
  **397 passes and one existing platform-specific shortcut assertion failure** (`Ctrl+S` vs `⌘S`).
  The additional atomic-commit worker behavior and export checks pass in an **18-test focused run**.

## Follow-up verification for milestones 4–6

- Reviewed against main `bea65ff`; no newer remote change was present at the start.
- **410 tests passed, one deselected**, with the import-wizard file excluded for the known
  macOS offscreen abort and the existing shortcut assertion deselected for its Ctrl/Command mismatch.
- Lint passes. Normal/subset ranking and live-view layouts were rendered for visual review.
- Milestones 7–8 remain deferred. No accessibility-name or palette changes are included.

### Remaining platform checks

- [ ] Validate milestones 1–6 on Windows with the pinned release dependencies.
- [ ] Repeat layout checks at Windows 125–200% scaling.
- [ ] Rerun the import-wizard GUI tests on a supported display; macOS offscreen aborted at wizard.show().
- [ ] Reassess the existing shortcut assertion on macOS separately from these UI changes.
