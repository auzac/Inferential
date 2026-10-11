# Methodology and conventions

This document fixes the conventions every part of the application relies on. Code that
deviates from them is a bug. Later milestones extend this document; they do not change it
silently.

## Time

- Every instant inside the application is **timezone-aware UTC**. Source files are parsed with
  an explicit source timezone (applied only to values without an offset) and converted to UTC.
- The **display timezone** is a project setting used only when showing times to the user.
- Durations are stored as **float seconds**.
- Day/month order is never guessed. If a file's dates are ambiguous the import cannot proceed
  until the user chooses (Milestone 1).

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

## Exclusions

- Raw imported data is never modified or deleted.
- Manual exclusion intervals are **closed** intervals `[start, end]` in UTC.
- An interval with no listed signals is an operating-regime (global) exclusion; with listed
  signals it affects only those variables. A sample becomes ineligible for modelling if any
  variable it requires is excluded over its feature window (Milestone 2–3).

## Synthetic example data

`inferential_studio.core.synthetic` produces an artificial distillation dataset with recorded
ground truth: target = first-order lag (τ = 15 min) of a linear combination of tray-15
temperature, pressure and feed temperature, followed by a 20-minute deadtime. Tests use this to
verify alignment and modelling against known answers.
