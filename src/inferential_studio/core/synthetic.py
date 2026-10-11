"""Synthetic distillation-column dataset for tests, demos and the first-run example.

All values are artificial. The generator is deterministic for a given configuration and
records its ground truth (deadtime, dynamics, coefficients, injected events) so later
milestones can test alignment, cleaning and modelling against known answers.

Process data: 1-minute historian-style wide table with realistic artefacts
    * two shutdowns, a low-load period, a genuine feed upset (valid data!)
    * isolated pressure-transmitter spikes, a stuck (flatlined) top temperature,
      a communication gap in feed temperature
Lab data: product composition sampled about every 2 hours (one timestamp per sample: the
collection time), with a sample ID, measurement noise and one gross analyzer error.

Ground-truth dynamics of the target (top product purity, mole fraction):

    u(t)   = c0 + c_T15 * (Tray_15_Temp - 145) + c_P * (Column_Pressure - 3.15)
                + c_FT * (Feed_Temp - 85)
    y_f    = first-order lag of u with time constant ``tau``
    y(t)   = y_f(t - deadtime)

Command line: ``inferential-studio-synth --out <folder>`` (or ``python -m
inferential_studio.core.synthetic``) writes the example files and a sample project.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from inferential_studio.core.schemas import (
    Aggregation,
    DataSource,
    ExclusionInterval,
    ExclusionReason,
    FeatureSpec,
    Project,
    SourceKind,
    TimestampKind,
    TimestampSpec,
    VariableMeta,
    VariableRole,
)
from inferential_studio.core.tasks import CancelToken, ProgressCallback, check, report

PROCESS_FILE = "tower_process.csv"
LAB_FILE = "lab_results.xlsx"
TRUTH_FILE = "ground_truth.json"
PROJECT_FILE = "sample_project.imsproj"

PROCESS_TIME_FORMAT = "%d/%m/%Y %H:%M"  # deliberately day-first, as many plant historians export
LAB_TIME_FORMAT = "%Y-%m-%d %H:%M"

UNITS = {
    "Feed_Flow": "t/h",
    "Feed_Temp": "degC",
    "Reflux_Flow": "t/h",
    "Reboiler_Duty": "kW",
    "Column_Pressure": "barg",
    "Tray_15_Temp": "degC",
    "Tray_20_Temp": "degC",
    "Top_Temp": "degC",
    "Product_Composition": "mol frac",
}


@dataclass(frozen=True)
class SyntheticConfig:
    days: int = 28
    seed: int = 20261011
    start_local: str = "2026-03-02 00:00"
    timezone: str = "Asia/Singapore"  # plant local time (no DST)
    lab_interval_h: float = 2.0
    deadtime_min: int = 20
    time_constant_min: float = 15.0
    coefficients: dict[str, float] = field(
        default_factory=lambda: {
            "intercept": 0.982,
            "Tray_15_Temp": -0.0035,
            "Column_Pressure": 0.080,
            "Feed_Temp": -0.0005,
        }
    )
    lab_noise_sd: float = 0.0004


@dataclass
class SyntheticDataset:
    process: pd.DataFrame  # UTC DatetimeIndex named "Timestamp", one column per tag
    lab: pd.DataFrame  # columns Sample_Timestamp (UTC), Product_Composition, Sample_ID
    target_truth: pd.Series  # noiseless target on the process grid (UTC index)
    truth: dict[str, Any]  # JSON-serialisable description of everything injected


# --------------------------------------------------------------------------- signal helpers


def _ar1(rng: np.random.Generator, n: int, sd: float, tau_samples: float) -> np.ndarray:
    """Stationary AR(1) noise with standard deviation ``sd`` and correlation time ``tau``."""
    phi = float(np.exp(-1.0 / tau_samples))
    e = rng.standard_normal(n) * sd * np.sqrt(1.0 - phi**2)
    out = lfilter([1.0], [1.0, -phi], e)
    out[0] = rng.standard_normal() * sd
    return out


def _first_order_lag(u: np.ndarray, tau_samples: float) -> np.ndarray:
    a = float(np.exp(-1.0 / tau_samples))
    y, _ = lfilter([1.0 - a], [1.0, -a], u, zi=[a * u[0]])
    return y


def _smooth_window(n: int, start: int, end: int, ramp: int) -> np.ndarray:
    """0..1 profile that ramps up over ``ramp`` samples at ``start`` and down at ``end``."""
    idx = np.arange(n)
    up = np.clip((idx - start) / max(ramp, 1), 0, 1)
    down = np.clip((end - idx) / max(ramp, 1), 0, 1)
    w = np.minimum(up, down)
    return np.clip(w, 0, 1)


# --------------------------------------------------------------------------- generator


def generate_distillation_dataset(
    config: SyntheticConfig | None = None,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> SyntheticDataset:
    cfg = config or SyntheticConfig()
    if cfg.days < 3:
        raise ValueError("The synthetic dataset needs at least 3 days.")
    rng = np.random.default_rng(cfg.seed)
    n = cfg.days * 24 * 60
    minutes_per_day = 24 * 60

    start = pd.Timestamp(cfg.start_local, tz=cfg.timezone).tz_convert("UTC")
    index = pd.date_range(start, periods=n, freq="1min", name="Timestamp")
    report(progress, 0.05, "Generating process disturbances")
    check(cancel)

    def day(d: float) -> int:
        return int(round(d * minutes_per_day))

    # --- operating events (positions as fractions of the dataset, so short datasets still work)
    span = cfg.days
    shutdowns = [
        (day(span * 0.21), day(span * 0.21 + 0.75)),
        (day(span * 0.71), day(span * 0.71 + 0.5)),
    ]
    low_load = (day(span * 0.50), day(span * 0.50) + 10 * 60)
    upset = (day(span * 0.32), day(span * 0.32) + 3 * 60)

    # --- independent drivers
    feed = 100.0 + _ar1(rng, n, 4.0, 120)
    feed += 15.0 * _smooth_window(n, *upset, ramp=10)  # genuine process upset: valid data
    feed -= 55.0 * _smooth_window(n, *low_load, ramp=30)  # turndown below 50 t/h
    running = np.ones(n)
    for s, e in shutdowns:
        running -= _smooth_window(n, s, e, ramp=60)
    running = np.clip(running, 0.0, 1.0)
    feed = feed * running + rng.normal(0, 0.2, n) * (1 - running)

    feed_temp = 85.0 + _ar1(rng, n, 1.5, 240)
    feed_temp = feed_temp * running + 30.0 * (1 - running)
    pressure = 3.15 + _ar1(rng, n, 0.03, 60)
    pressure = pressure * running + 0.05 * (1 - running)
    load = np.clip(feed / 100.0, 0.0, None)  # controller noise scales with throughput
    reflux = 0.9 * feed + _ar1(rng, n, 1.5, 90) * load
    duty = 2.5 * feed + _ar1(rng, n, 5.0, 90) * load
    report(progress, 0.30, "Simulating column temperatures")
    check(cancel)

    # --- column response: separation index q (positive = hotter, heavier top)
    safe_feed = np.maximum(feed, 1.0)
    q_raw = ((duty / safe_feed) - 2.5) / 0.05 - ((reflux / safe_feed) - 0.9) / 0.03
    q_raw = np.where(running > 0.5, q_raw, 0.0)
    q = _first_order_lag(q_raw * 0.2, 10.0) + 0.10 * (feed_temp - 85.0) * running

    tray15 = 145.0 + 20.0 * (pressure - 3.15) + 3.0 * q + rng.normal(0, 0.10, n)
    tray20 = (
        151.6 + 22.0 * (pressure - 3.15) + 3.3 * q + _ar1(rng, n, 0.35, 45) + rng.normal(0, 0.12, n)
    )
    top_temp = (
        120.0 + 15.0 * (pressure - 3.15) + 1.5 * _first_order_lag(q, 20.0) + rng.normal(0, 0.15, n)
    )
    cool = 1 - running  # temperatures fall towards ambient during shutdowns
    tray15 = tray15 * running + 35.0 * cool
    tray20 = tray20 * running + 35.0 * cool
    top_temp = top_temp * running + 33.0 * cool

    # --- target (computed from the clean signals, before instrument artefacts are added)
    c = cfg.coefficients
    u = (
        c["intercept"]
        + c["Tray_15_Temp"] * (tray15 - 145.0)
        + c["Column_Pressure"] * (pressure - 3.15)
        + c["Feed_Temp"] * (feed_temp - 85.0)
    )
    y_lag = _first_order_lag(u, cfg.time_constant_min)
    d = cfg.deadtime_min
    y_true = np.concatenate([np.full(d, y_lag[0]), y_lag[:-d]]) if d > 0 else y_lag
    report(progress, 0.55, "Injecting instrument artefacts")
    check(cancel)

    # --- instrument artefacts (affect measurements only, not the true process)
    pressure_meas = pressure.copy()
    candidates = np.flatnonzero(running > 0.99)
    spike_idx = np.sort(rng.choice(candidates, size=min(20, candidates.size), replace=False))
    pressure_meas[spike_idx] += rng.choice([-1, 1], size=spike_idx.size) * rng.uniform(
        0.3, 0.5, spike_idx.size
    )

    flat = (day(span * 0.60), day(span * 0.60) + 6 * 60)
    top_meas = top_temp.copy()
    top_meas[flat[0] : flat[1]] = top_meas[flat[0]]

    gap = (day(span * 0.40), day(span * 0.40) + 90)
    feed_temp_meas = feed_temp.copy()
    feed_temp_meas[gap[0] : gap[1]] = np.nan

    process = pd.DataFrame(
        {
            "Feed_Flow": np.round(feed, 2),
            "Feed_Temp": np.round(feed_temp_meas, 2),
            "Reflux_Flow": np.round(reflux, 2),
            "Reboiler_Duty": np.round(duty, 1),
            "Column_Pressure": np.round(pressure_meas, 3),
            "Tray_15_Temp": np.round(tray15, 2),
            "Tray_20_Temp": np.round(tray20, 2),
            "Top_Temp": np.round(top_meas, 2),
        },
        index=index,
    )
    target_truth = pd.Series(y_true, index=index, name="Product_Composition")
    report(progress, 0.75, "Sampling laboratory results")
    check(cancel)

    # --- lab samples: nominal every lab_interval_h, collection time jittered +-10 min
    step = int(round(cfg.lab_interval_h * 60))
    nominal = np.arange(step, n - 15, step)
    jitter = rng.integers(-10, 11, size=nominal.size)
    sample_idx = np.clip(nominal + jitter, 0, n - 1)
    # Nobody samples a column that is shut down or has run normally for less than ~3 hours.
    settled = pd.Series(running > 0.99).rolling(180, min_periods=1).min().to_numpy() > 0.5
    sample_idx = sample_idx[settled[sample_idx]]
    # Formerly the result-reporting delay. Still drawn so the random stream, and with it the
    # lab values below, stay exactly as in earlier examples.
    rng.integers(45, 121, size=sample_idx.size)
    values = y_true[sample_idx] + rng.normal(0, cfg.lab_noise_sd, sample_idx.size)
    bad_pos = int(sample_idx.size * 0.8)
    values[bad_pos] = 0.900  # gross analyzer / transcription error
    sample_ids = [f"L{i + 1:04d}" for i in range(sample_idx.size)]

    lab = pd.DataFrame(
        {
            "Sample_Timestamp": index[sample_idx],
            "Product_Composition": np.round(values, 4),
            "Sample_ID": sample_ids,
        }
    )

    def iso(i: int) -> str:
        return index[min(i, n - 1)].isoformat()

    truth = {
        "description": "Synthetic distillation dataset. All values are artificial.",
        "config": asdict(cfg),
        "process_rows": n,
        "lab_samples": int(sample_idx.size),
        "target": "Product_Composition",
        "target_model": {
            "deadtime_min": cfg.deadtime_min,
            "time_constant_min": cfg.time_constant_min,
            "coefficients": dict(c),
            "reference_point": {"Tray_15_Temp": 145.0, "Column_Pressure": 3.15, "Feed_Temp": 85.0},
        },
        "events_utc": {
            "shutdowns": [[iso(s), iso(e)] for s, e in shutdowns],
            "low_load": [iso(low_load[0]), iso(low_load[1])],
            "genuine_feed_upset": [iso(upset[0]), iso(upset[1])],
            "pressure_spikes": [iso(int(i)) for i in spike_idx],
            "top_temp_flatline": [iso(flat[0]), iso(flat[1])],
            "feed_temp_gap": [iso(gap[0]), iso(gap[1])],
        },
        "bad_lab_sample_ids": [sample_ids[bad_pos]],
    }
    report(progress, 1.0, "Done")
    return SyntheticDataset(process=process, lab=lab, target_truth=target_truth, truth=truth)


# --------------------------------------------------------------------------- writing example files


def write_example_files(
    dataset: SyntheticDataset,
    out_dir: str | Path,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> dict[str, Path]:
    """Write process CSV, lab Excel, ground truth and a sample project into ``out_dir``."""
    from inferential_studio.core.persistence import save_project, sha256_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tz = dataset.truth["config"]["timezone"]

    report(progress, 0.1, "Writing process CSV")
    check(cancel)
    process = dataset.process.copy()
    process.index = process.index.tz_convert(tz).strftime(PROCESS_TIME_FORMAT)
    process.index.name = "Timestamp"
    process_path = out / PROCESS_FILE
    process.to_csv(process_path, lineterminator="\n", float_format="%.6g")

    report(progress, 0.5, "Writing lab Excel workbook")
    check(cancel)
    lab = dataset.lab.copy()
    lab["Sample_Timestamp"] = lab["Sample_Timestamp"].dt.tz_convert(tz).dt.strftime(LAB_TIME_FORMAT)
    lab_path = out / LAB_FILE
    lab.to_excel(lab_path, sheet_name="Lab", index=False)

    truth_path = out / TRUTH_FILE
    truth_path.write_text(json.dumps(dataset.truth, indent=2), encoding="utf-8", newline="\n")

    report(progress, 0.8, "Writing sample project")
    check(cancel)
    project = build_sample_project(
        dataset,
        process_sha256=sha256_file(process_path),
        lab_sha256=sha256_file(lab_path),
    )
    project_path = out / PROJECT_FILE
    save_project(project, project_path)
    report(progress, 1.0, "Done")
    return {"process": process_path, "lab": lab_path, "truth": truth_path, "project": project_path}


def build_sample_project(
    dataset: SyntheticDataset, *, process_sha256: str | None = None, lab_sha256: str | None = None
) -> Project:
    """Project configuration matching the files written by :func:`write_example_files`."""
    tz = dataset.truth["config"]["timezone"]
    deadtime_s = float(dataset.truth["target_model"]["deadtime_min"] * 60)
    sources = [
        DataSource(
            source_id="process",
            kind=SourceKind.PROCESS,
            path=PROCESS_FILE,
            sha256=process_sha256,
            delimiter=",",
            timestamp=TimestampSpec(
                column="Timestamp",
                kind=TimestampKind.TEXT,
                format=PROCESS_TIME_FORMAT,
                dayfirst=True,
                source_timezone=tz,
            ),
        ),
        DataSource(
            source_id="lab",
            kind=SourceKind.LAB,
            path=LAB_FILE,
            sha256=lab_sha256,
            sheet="Lab",
            timestamp=TimestampSpec(
                column="Sample_Timestamp", format=LAB_TIME_FORMAT, source_timezone=tz
            ),
            sample_id_column="Sample_ID",
        ),
    ]
    predictors = ["Tray_15_Temp", "Tray_20_Temp", "Column_Pressure", "Reboiler_Duty", "Reflux_Flow"]
    variables = [
        VariableMeta(
            name=name,
            source_id="process",
            source_column=name,
            role=VariableRole.PREDICTOR if name in predictors else VariableRole.UNUSED,
            units=UNITS[name],
        )
        for name in dataset.process.columns
    ]
    variables.append(
        VariableMeta(
            name="Product_Composition",
            source_id="lab",
            source_column="Product_Composition",
            role=VariableRole.TARGET,
            units=UNITS["Product_Composition"],
            description="Top product purity (synthetic lab result)",
            eng_low=0.0,
            eng_high=1.0,
        )
    )
    features = [
        FeatureSpec(
            variable=name, window_s=1800.0, deadtime_s=deadtime_s, aggregation=Aggregation.MEAN
        )
        for name in predictors
    ]
    exclusions = [
        ExclusionInterval(
            start=pd.Timestamp(s).to_pydatetime(),
            end=pd.Timestamp(e).to_pydatetime(),
            reason=ExclusionReason.SHUTDOWN,
            note="Synthetic shutdown",
            author="synthetic generator",
        )
        for s, e in dataset.truth["events_utc"]["shutdowns"]
    ]
    return Project(
        name="Synthetic distillation example",
        display_timezone=tz,
        sources=sources,
        variables=variables,
        features=features,
        exclusions=exclusions,
        notes="Artificial data generated by inferential_studio.core.synthetic. Not plant data.",
    )


def create_example(
    out_dir: str | Path,
    config: SyntheticConfig | None = None,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> dict[str, Path]:
    """Generate the dataset and write all example files (progress spans both steps)."""

    def gen_progress(f: float, m: str) -> None:
        report(progress, 0.5 * f, m)

    def write_progress(f: float, m: str) -> None:
        report(progress, 0.5 + 0.5 * f, m)

    dataset = generate_distillation_dataset(config, progress=gen_progress, cancel=cancel)
    return write_example_files(dataset, out_dir, progress=write_progress, cancel=cancel)


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic distillation example dataset."
    )
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--days", type=int, default=SyntheticConfig.days)
    parser.add_argument("--seed", type=int, default=SyntheticConfig.seed)
    args = parser.parse_args(argv)
    paths = create_example(args.out, SyntheticConfig(days=args.days, seed=args.seed))
    for label, path in paths.items():
        print(f"{label:8s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
