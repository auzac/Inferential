# Benchmarks

Measured numbers only. Release thresholds will be set after baselining on the team's own
Windows reference PC (spec §12). Re-run with the commands below on that PC and add a row.

## Environment of the recorded runs

| Item | Value |
|---|---|
| Machine | Cloud Linux container, 2 vCPU Intel Xeon @ 2.80 GHz, 7 GB RAM |
| Python / pandas / NumPy | 3.12.3 / 3.0.6 / 2.5.4 |
| Dataset | 1,000,000 rows × 20 numeric tags, 1-minute timestamps `dd/mm/yyyy HH:MM`, CSV 165 MB |

## Results (2026-10-11)

| Step | Time | Peak memory (process) |
|---|---|---|
| Import (read, parse timestamps, numeric checks, report) | 9.0 s | ≈ 700 MB |
| Normalize (sort, duplicate policy) | 2.1 s | |
| EDA statistics (summary, Pearson, exact Spearman, coverage) | 7.5 s | ≈ 820 MB |
| Synthetic example (40,320 rows × 8): full project load | 0.7 s | |
| Detectors: Hampel + flatline on 20 tags (v0.2.0) | 23.5 s | |
| Exclusion masks: intervals, a 2-condition rule with minimum duration, detector events (v0.2.0) | 0.44 s | |
| Feature alignment, 20 predictors × 8,332 lab samples, 30-min mean windows (v0.4.0) | 0.66 s | ≈ 380 MB |
| Same with median windows (per-window loop) (v0.4.0) | 3.3 s | |
| Training MLR + PLS + PCR + Ridge with 5-fold expanding CV tuning, 8,332 × 20 (v0.4.0) | 1.0 s | |

All of these steps run in background tasks (exclusion masks and feature alignment are
recomputed automatically and coalesced after edits since v0.2.1 / v0.4.0). Trend charts draw with
peak-preserving downsampling, so plotting time does not grow with the number of rows.

## How to reproduce

```python
import time, numpy as np, pandas as pd
from inferential_studio.core.importers import import_source
from inferential_studio.core.schemas import DataSource, SourceKind, TimestampSpec
from inferential_studio.core.statistics import compute_eda

n = 1_000_000
df = pd.DataFrame(np.random.default_rng(0).normal(100, 5, (n, 20)).round(3),
                  columns=[f"TAG_{i:02d}" for i in range(20)])
df.insert(0, "Time", pd.date_range("2024-01-01", periods=n, freq="1min").strftime("%d/%m/%Y %H:%M"))
df.to_csv("big.csv", index=False)

src = DataSource(source_id="p", kind=SourceKind.PROCESS, path="big.csv", delimiter=",",
                 timestamp=TimestampSpec(column="Time", format="%d/%m/%Y %H:%M", source_timezone="UTC"))
t = time.time(); data = import_source(src); print("import", time.time() - t)
t = time.time(); norm = data.normalized; print("normalize", time.time() - t)
t = time.time(); compute_eda(norm.values, norm.text_mask); print("eda", time.time() - t)
```

Feature alignment and training (v0.4.0), continuing from the frame above:

```python
from inferential_studio.core.features import TargetSamples, build_feature_matrix
from inferential_studio.core.models import train_models
from inferential_studio.core.schemas import FeatureSpec, ModellingConfig

frame = norm.values
times = frame.index[200::120]  # a lab sample every 2 h
rng = np.random.default_rng(1)
target = TargetSamples("y", times, rng.normal(size=len(times)),
                       np.zeros(len(times), bool), np.zeros(len(times), bool))
specs = [FeatureSpec(c, window_s=1800, deadtime_s=1200) for c in frame.columns]
t = time.time(); fm = build_feature_matrix(target, frame, specs); print("align", time.time() - t)
fm.y[:] = fm.X.sum(axis=1).to_numpy() + rng.normal(size=len(fm.y))
t = time.time(); train_models(fm, ModellingConfig()); print("train", time.time() - t)
```
