"""Milestone 1 integration: two files with different timestamp conventions → EDA → save → reload."""

from __future__ import annotations

import numpy as np
import pandas as pd

from inferential_studio.core.importers import import_source, load_project_data
from inferential_studio.core.persistence import load_project, save_project
from inferential_studio.core.schemas import (
    DataSource,
    Project,
    SourceKind,
    TimestampSpec,
    VariableMeta,
    VariableRole,
)
from inferential_studio.core.statistics import compute_eda
from inferential_studio.core.timestamps import detect_format


def test_mixed_conventions_end_to_end(tmp_path):
    # Process: day-first local Singapore time, semicolons and decimal commas.
    utc_grid = pd.date_range("2026-03-01 00:00", periods=600, freq="1min", tz="UTC")
    local = utc_grid.tz_convert("Asia/Singapore")
    rng = np.random.default_rng(0)
    x1 = 100 + rng.normal(size=600).cumsum() * 0.1
    x2 = 3 + 0.05 * x1 + rng.normal(scale=0.01, size=600)
    lines = ["Zeit;X1;X2"] + [
        f"{t:%d/%m/%Y %H:%M};{a:.3f};{b:.4f}".replace(".", ",")
        for t, a, b in zip(local, x1, x2, strict=True)
    ]
    (tmp_path / "process.csv").write_text("\n".join(lines) + "\n", encoding="cp1252")

    # Lab: ISO 8601 with explicit UTC offsets from a system running in New York.
    sample_idx = np.arange(30, 600, 60)
    sample_utc = utc_grid[sample_idx]
    y = 0.9 + 0.001 * x1[sample_idx]
    lab_lines = ["sample_time,result_time,purity"] + [
        f"{t.tz_convert('America/New_York').isoformat()},"
        f"{(t + pd.Timedelta('50min')).tz_convert('America/New_York').isoformat()},{v:.5f}"
        for t, v in zip(sample_utc, y, strict=True)
    ]
    (tmp_path / "lab.csv").write_text("\n".join(lab_lines) + "\n", encoding="utf-8")

    # The format detectors agree with what we wrote.
    proc_raw = pd.read_csv(tmp_path / "process.csv", sep=";", dtype=str, encoding="cp1252")
    detected = detect_format(proc_raw["Zeit"])
    assert detected.ambiguous  # 01/03 and 02/03 only: the user must say which order is used
    assert "%d/%m/%Y %H:%M" in {c.format for c in detected.candidates}
    lab_raw = pd.read_csv(tmp_path / "lab.csv", dtype=str)
    assert detect_format(lab_raw["sample_time"]).has_offsets

    project = Project(
        name="mixed",
        display_timezone="Asia/Singapore",
        sources=[
            DataSource(
                source_id="process",
                kind=SourceKind.PROCESS,
                path="process.csv",
                delimiter=";",
                decimal=",",
                encoding="cp1252",
                timestamp=TimestampSpec(
                    column="Zeit",
                    format="%d/%m/%Y %H:%M",
                    dayfirst=True,
                    source_timezone="Asia/Singapore",
                ),
            ),
            DataSource(
                source_id="lab",
                kind=SourceKind.LAB,
                path="lab.csv",
                delimiter=",",
                timestamp=TimestampSpec(column="sample_time", source_timezone="UTC"),
                result_timestamp=TimestampSpec(column="result_time", source_timezone="UTC"),
            ),
        ],
        variables=[
            VariableMeta("X1", "process", "X1", VariableRole.PREDICTOR),
            VariableMeta("X2", "process", "X2", VariableRole.PREDICTOR),
            VariableMeta("Purity", "lab", "purity", VariableRole.TARGET),
        ],
    )
    project_path = tmp_path / "mixed.imsproj"
    data = load_project_data(project, project_path)
    assert not data.failures

    proc = data.sources["process"]
    lab = data.sources["lab"]
    assert (pd.DatetimeIndex(proc.time) == utc_grid).all()  # local day-first → exact UTC instants
    assert (pd.DatetimeIndex(lab.time) == sample_utc).all()  # offsets honoured, source tz ignored
    assert ((lab.result_time - lab.time) == pd.Timedelta("50min")).all()
    np.testing.assert_allclose(proc.values["X1"], x1.round(3))

    frame, mask = data.process_frame(project)
    target = data.target_series(project)
    assert len(target) == len(sample_idx)  # never forward-filled onto the process grid
    eda = compute_eda(frame, mask, target)
    assert eda.pearson.loc["X1", "X2"] > 0.5
    assert eda.xy.table.loc["X1", "pearson"] > 0.99  # last value at or before each sample
    assert eda.xy.table.loc["X1", "n_pairs"] == len(sample_idx)

    # Record checksums from the import, save, reopen, reload: identical timestamps and values.
    project.sources = [data.sources[s.source_id].source for s in project.sources]
    save_project(project, project_path)
    reopened = load_project(project_path)
    again = load_project_data(reopened, project_path)
    assert not again.failures
    pd.testing.assert_series_equal(again.sources["process"].time, proc.time)
    pd.testing.assert_frame_equal(again.sources["process"].values, proc.values)
    pd.testing.assert_series_equal(again.sources["lab"].time, lab.time)


def test_multiple_process_sources_are_joined_without_filling(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("t,A\n2026-01-01 00:00,1\n2026-01-01 00:01,2\n2026-01-01 00:02,3\n")
    b.write_text("t,B\n2026-01-01 00:01,20\n2026-01-01 00:01:30,25\n")
    spec = TimestampSpec(column="t")
    sources = [
        DataSource(source_id=n, kind=SourceKind.PROCESS, path=str(p), delimiter=",", timestamp=spec)
        for n, p in (("a", a), ("b", b))
    ]
    project = Project(
        sources=sources,
        variables=[VariableMeta("A", "a", "A"), VariableMeta("B", "b", "B")],
    )
    data = load_project_data(project, None)
    frame, _ = data.process_frame(project)
    assert len(frame) == 4
    assert frame["B"].notna().sum() == 2  # no interpolation or forward fill across sources
    assert import_source(sources[0]).report.n_rows == 3
