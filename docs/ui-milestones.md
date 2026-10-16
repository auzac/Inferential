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

**Excluded from this implementation. Claude is working on it.**

- [ ] Reassess Claude's fix when available.
- [ ] Verify standard-deviation features have the correct initial/final prediction and transient response.
- [ ] Verify the displayed response/settling summary agrees with the simulation.

This work changes layout containers around What-if but leaves its calculation and control files untouched.

## Milestone 5 — Expensive and hidden simulations (audit finding 5)

**Deferred for reassessment.**

- [ ] Reassess against Claude's updated What-if implementation.
- [ ] Decide whether visible-tab computation, caching or background execution is still needed.
- [ ] Recheck long-run slider, model-change and theme-switch responsiveness.

## Milestone 6 — Ranking metric visibility (audit finding 6)

**Deferred for reassessment.**

- [ ] Reassess comparison-column order after the compact-layout change.
- [ ] Keep the active ranking metric visible without horizontal scrolling if still required.
- [ ] Recheck ordinary comparisons and subset searches at both window sizes.

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

### Remaining platform checks

- [ ] Validate milestones 1–3 on Windows with the pinned release dependencies.
- [ ] Repeat layout checks at Windows 125–200% scaling.
- [ ] Rerun the import-wizard GUI tests on a supported display; macOS offscreen aborted at wizard.show().
- [ ] Reassess the existing shortcut assertion on macOS separately from these UI changes.
