# Benchmarks

Measured numbers only. Release thresholds will be set after baselining on the team's own
Windows reference PC (spec §12). Re-run with the commands below on that PC and add a row.

## Environment of the recorded runs

| Item | Value |
|---|---|
| Machine | Cloud Linux container, 2 vCPU Intel Xeon @ 2.80 GHz, 7 GB RAM |
| Python / pandas / NumPy | 3.12.3 / 3.0.6 / 2.5.4 |
| Dataset | 1,000,000 rows × 20 numeric tags, 1-minute timestamps `dd/mm/yyyy HH:MM`, CSV 165 MB |

## Results (v0.1.0, 2026-10-11)

| Step | Time | Peak memory (process) |
|---|---|---|
| Import (read, parse timestamps, numeric checks, report) | 9.0 s | ≈ 700 MB |
| Normalize (sort, duplicate policy) | 2.1 s | |
| EDA statistics (summary, Pearson, exact Spearman, coverage) | 7.5 s | ≈ 820 MB |
| Synthetic example (40,320 rows × 8): full project load | 0.7 s | |

All of these run in a background task; the window stays responsive. Trend charts draw with
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
