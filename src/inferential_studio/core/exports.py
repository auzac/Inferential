"""Tabular exports: per-sample predictions, feature matrices and batch predictions (CSV / Excel).

Timestamps are written twice: in UTC (ISO 8601, the reference) and in the display timezone
(local wall-clock, column name states the zone). Final-test rows are included only when the
caller says the final test has been revealed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from inferential_studio.core.features import FeatureMatrix
from inferential_studio.core.models import TrainingRun
from inferential_studio.core.splits import TEST


class ExportError(Exception):
    """An export could not be written. The message is shown to the user."""


def _time_columns(
    times: pd.Series | pd.DatetimeIndex, tz: str, prefix: str
) -> dict[str, pd.Series]:
    t = pd.DatetimeIndex(times)
    return {
        f"{prefix}_utc": pd.Series(t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")),
        f"{prefix}_local ({tz})": pd.Series(t.tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S")),
    }


def predictions_table(
    run: TrainingRun,
    model_name: str,
    fm: FeatureMatrix | None = None,
    *,
    timezone: str = "UTC",
    include_test: bool = False,
) -> pd.DataFrame:
    """One row per evaluated lab sample: times, id, period, actual, predicted, residual, features."""
    m = next((x for x in run.models if x.name == model_name and x.ok), None)
    if m is None:
        raise ExportError(f"'{model_name}' has no predictions to export.")
    pred = m.predictions
    if not include_test:
        pred = pred[pred["split"] != TEST]
    pred = pred.reset_index(drop=True)
    out = pd.DataFrame(_time_columns(pred["sample_time"], timezone, "sample_time"))
    rows = pred["row"].to_numpy()
    same = fm is not None and fm.sample_time.equals(run.sample_time)  # the run's own matrix
    if same and fm.sample_id is not None:
        out["sample_id"] = np.asarray(fm.sample_id)[rows]
    if same and fm.result_time is not None:
        for k, v in _time_columns(fm.result_time[rows], timezone, "result_time").items():
            out[k] = v.to_numpy()
    out["period"] = pred["split"].to_numpy()
    out[f"actual {run.target}"] = pred["actual"].to_numpy()
    out[f"predicted {run.target}"] = pred["predicted"].to_numpy()
    out["residual (actual - predicted)"] = pred["residual"].to_numpy()
    for name in run.features:
        out[f"f[{name}]"] = run.feature_values[name].to_numpy()[rows]
    return out


def feature_matrix_table(fm: FeatureMatrix, timezone: str = "UTC") -> pd.DataFrame:
    """Every lab sample with its aligned features, eligibility and reason (for audit)."""
    table = fm.table()
    times = _time_columns(table.pop("sample_time"), timezone, "sample_time")
    if "result_time" in table:
        rt = table.pop("result_time")
        times.update(_time_columns(rt, timezone, "result_time"))
    front = pd.DataFrame({k: v.to_numpy() for k, v in times.items()})
    return pd.concat([front, table.reset_index(drop=True)], axis=1)


def batch_prediction_table(table: pd.DataFrame, timezone: str = "UTC") -> pd.DataFrame:
    """A :class:`~inferential_studio.core.artifacts.Prediction` table ready for writing."""
    out = pd.DataFrame(_time_columns(table["time"], timezone, "time"))
    rest = table.drop(columns=["time"]).reset_index(drop=True)
    return pd.concat([out, rest], axis=1)


def write_table(frame: pd.DataFrame, path: str | os.PathLike[str], *, sheet: str = "Data",
                notes: list[str] | None = None) -> Path:  # fmt: skip
    """Write CSV (UTF-8 with BOM so Excel opens it correctly) or .xlsx, chosen by extension."""
    target = Path(path)
    suffix = target.suffix.lower()
    try:
        if suffix == ".csv":
            frame.to_csv(target, index=False, encoding="utf-8-sig", lineterminator="\r\n")
        elif suffix == ".xlsx":
            with pd.ExcelWriter(target, engine="openpyxl") as writer:
                frame.to_excel(writer, sheet_name=sheet[:31], index=False)
                if notes:
                    pd.DataFrame({"Notes": notes}).to_excel(writer, sheet_name="Notes", index=False)
        else:
            raise ExportError("Choose a .csv or .xlsx file name.")
    except OSError as exc:
        raise ExportError(f"Could not write '{target}': {exc.strerror or exc}.") from exc
    return target
