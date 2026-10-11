from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inferential_studio.core.dataset import DataImportError, Severity
from inferential_studio.core.importers import (
    ReadOptions,
    SourceChangedError,
    excel_date_system,
    file_kind,
    import_source,
    lab_coverage_issue,
    list_sheets,
    read_header,
    read_preview,
    sniff_csv,
)
from inferential_studio.core.schemas import (
    DataSource,
    DuplicatePolicy,
    SourceKind,
    TimestampKind,
    TimestampSpec,
)


def write(path, text: str, encoding: str = "utf-8"):
    path.write_bytes(text.encode(encoding))
    return path


def process_source(path, **kw) -> DataSource:
    ts = kw.pop(
        "timestamp", TimestampSpec(column="Time", format="%d/%m/%Y %H:%M", source_timezone="UTC")
    )
    return DataSource(source_id="p", kind=SourceKind.PROCESS, path=str(path), timestamp=ts, **kw)


def codes(data) -> dict[str, Severity]:
    return {i.code: i.severity for i in data.report.issues}


# ----------------------------------------------------------------------------- inspection


def test_file_kind():
    assert file_kind("a.CSV") == "csv" and file_kind("b.xlsx") == "excel"
    with pytest.raises(DataImportError, match="save it as .xlsx"):
        file_kind("old.xls")
    with pytest.raises(DataImportError, match="Unsupported"):
        file_kind("x.json")


def test_sniff_european_csv(tmp_path):
    path = write(
        tmp_path / "eu.csv",
        "Zeit;Temp;Druck\n01.02.2026 00:00;145,2;3,15\n01.02.2026 00:01;145,4;3,16\n",
        "cp1252",
    )
    opts = sniff_csv(path)
    assert opts.delimiter == ";" and opts.decimal == ","


def test_sniff_detects_non_utf8_encoding(tmp_path):
    path = write(tmp_path / "ansi.csv", "Time,Temp °C\n01/02/2026 00:00,1\n", "cp1252")
    assert sniff_csv(path).encoding == "cp1252"
    assert read_header(path, sniff_csv(path)) == ["Time", "Temp °C"]


def test_sniff_utf8_bom(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbfTime,A\n2026-01-01 00:00,1\n")
    opts = sniff_csv(path)
    assert opts.encoding == "utf-8-sig"
    assert read_header(path, opts) == ["Time", "A"]  # no BOM glued to the first name


def test_duplicate_and_blank_headers_rejected(tmp_path):
    dup = write(tmp_path / "d.csv", "Time,A,A\n1,2,3\n")
    with pytest.raises(DataImportError, match="Duplicate column name"):
        read_header(dup, ReadOptions(delimiter=","))
    blank = write(tmp_path / "b.csv", "Time,,B\n1,2,3\n")
    with pytest.raises(DataImportError, match="no name"):
        read_header(blank, ReadOptions(delimiter=","))


def test_header_row_offset_and_preview(tmp_path):
    path = write(
        tmp_path / "h.csv", "Exported from historian\nunits,degC\nTime,T\n01/02/2026 00:00,1.5\n"
    )
    opts = ReadOptions(delimiter=",", header_row=2)
    assert read_header(path, opts) == ["Time", "T"]
    preview = read_preview(path, opts)
    assert preview.iloc[0].tolist() == ["01/02/2026 00:00", "1.5"]


# ----------------------------------------------------------------------------- CSV import


def test_import_csv_values_text_tokens_and_lines(tmp_path):
    path = write(
        tmp_path / "p.csv",
        "Time,A,B\n"
        "01/02/2026 00:00,1.0,10\n"
        "01/02/2026 00:01,Bad,11\n"
        "01/02/2026 00:02,,I/O Timeout\n"
        "01/02/2026 00:03,NaN,13\n"
        "not a time,5,14\n",
    )
    data = import_source(process_source(path, delimiter=","))
    a, b = data.values["A"].to_numpy(), data.values["B"].to_numpy()
    np.testing.assert_array_equal(np.isnan(a), [False, True, True, True, False])
    np.testing.assert_array_equal(np.isnan(b), [False, False, True, False, False])
    assert data.text_mask["A"].tolist() == [False, True, False, False, False]  # NaN text = missing
    assert data.text_mask["B"].tolist() == [False, False, True, False, False]
    qa = data.report.column("A")
    assert (qa.n_numeric, qa.n_blank, qa.n_text) == (2, 2, 1)
    assert qa.text_examples == {"Bad": 1}
    assert data.report.n_unparseable_times == 1
    assert data.report.invalid_time_examples == [(6, "not a time")]  # line 6 of the file
    assert codes(data)["unparseable_times"] is Severity.WARNING
    assert codes(data)["text_values"] is Severity.WARNING
    assert len(data.normalized.values) == 4  # invalid-time row excluded from the normalized layer
    assert len(data.values) == 5  # ...but kept in the raw layer
    assert data.raw_time_text.iloc[4] == "not a time"


def test_decimal_comma_and_semicolon(tmp_path):
    path = write(tmp_path / "eu.csv", "Time;T\n01.02.2026 00:00;145,25\n01.02.2026 00:01;-0,5\n")
    src = process_source(
        path,
        delimiter=";",
        decimal=",",
        timestamp=TimestampSpec(column="Time", format="%d.%m.%Y %H:%M"),
    )
    assert import_source(src).values["T"].tolist() == [145.25, -0.5]


def test_blank_lines_keep_line_numbers_aligned(tmp_path):
    path = write(
        tmp_path / "p.csv", "Time,A\n01/02/2026 00:00,1\n\n01/02/2026 00:02,bad-time-below\nxx,3\n"
    )
    data = import_source(process_source(path, delimiter=","))
    assert data.report.n_blank_times == 1
    assert data.report.invalid_time_examples == [(5, "xx")]


def test_out_of_order_rows_sorted(tmp_path):
    path = write(
        tmp_path / "p.csv", "Time,A\n01/02/2026 00:02,3\n01/02/2026 00:00,1\n01/02/2026 00:01,2\n"
    )
    data = import_source(process_source(path, delimiter=","))
    assert data.report.n_out_of_order == 1
    assert data.normalized.values["A"].tolist() == [1, 2, 3]
    assert data.normalized.file_line.tolist() == [3, 4, 2]


@pytest.mark.parametrize(
    "policy, expected",
    [(DuplicatePolicy.KEEP_FIRST, [1.0, 5.0]), (DuplicatePolicy.MEAN, [2.0, 5.0])],
)
def test_duplicate_policies(tmp_path, policy, expected):
    path = write(
        tmp_path / "p.csv", "Time,A\n01/02/2026 00:00,1\n01/02/2026 00:00,3\n01/02/2026 00:01,5\n"
    )
    data = import_source(process_source(path, delimiter=",", duplicate_policy=policy))
    assert data.normalized.values["A"].tolist() == expected
    assert codes(data)["duplicate_times"] is Severity.INFO


def test_duplicates_without_policy_are_an_error(tmp_path):
    path = write(tmp_path / "p.csv", "Time,A\n01/02/2026 00:00,1\n01/02/2026 00:00,3\n")
    data = import_source(process_source(path, delimiter=","))
    assert data.report.has_errors
    with pytest.raises(DataImportError, match="repeat an earlier timestamp"):
        _ = data.normalized


def test_gaps_and_irregular_sampling_reported(tmp_path):
    times = pd.date_range("2026-02-01", periods=100, freq="1min").append(
        pd.date_range("2026-02-01 05:00", periods=100, freq="1min")
    )
    body = "".join(f"{t:%d/%m/%Y %H:%M},{i}\n" for i, t in enumerate(times))
    data = import_source(
        process_source(write(tmp_path / "g.csv", "Time,A\n" + body), delimiter=",")
    )
    rep = data.report
    assert rep.median_interval == pd.Timedelta("1min")
    assert rep.n_gaps == 1 and rep.gaps[0].duration == pd.Timedelta("3h21min")
    assert codes(data)["gaps"] is Severity.WARNING


def test_dst_times_flagged_in_report(tmp_path):
    path = write(
        tmp_path / "ny.csv", "Time,A\n2026-03-08 01:30,1\n2026-03-08 02:30,2\n2026-03-08 03:30,3\n"
    )
    src = process_source(
        path,
        delimiter=",",
        timestamp=TimestampSpec(
            column="Time", format="%Y-%m-%d %H:%M", source_timezone="America/New_York"
        ),
    )
    data = import_source(src)
    assert data.report.n_dst_nonexistent == 1
    assert "dst_nonexistent" in codes(data)
    assert len(data.normalized.values) == 2


def test_missing_columns_and_wrong_encoding(tmp_path):
    path = write(tmp_path / "p.csv", "Time,A\n01/02/2026 00:00,1\n")
    with pytest.raises(DataImportError, match="no column"):
        import_source(process_source(path, delimiter=","), value_columns=["Nope"])
    bad = write(tmp_path / "ansi.csv", "Time,Temp\n01/02/2026 00:00,1°\n", "cp1252")
    with pytest.raises(DataImportError, match="not valid utf-8"):
        import_source(process_source(bad, delimiter=","))


def test_checksum_recorded_and_verified(tmp_path):
    path = write(tmp_path / "p.csv", "Time,A\n01/02/2026 00:00,1\n")
    data = import_source(process_source(path, delimiter=","))
    assert data.source.sha256 == data.sha256
    path.write_text("Time,A\n01/02/2026 00:00,2\n")
    with pytest.raises(SourceChangedError):
        import_source(data.source)
    assert import_source(data.source, verify_checksum=False).values["A"].tolist() == [2.0]


def test_relative_path_resolved_against_project(tmp_path):
    write(tmp_path / "p.csv", "Time,A\n01/02/2026 00:00,1\n")
    src = process_source("p.csv", delimiter=",")
    assert import_source(src, tmp_path / "proj.imsproj").values["A"].tolist() == [1.0]


def test_reimport_gives_identical_timestamps(tmp_path):
    path = write(tmp_path / "p.csv", "Time,A\n01/02/2026 00:00,1\n02/02/2026 13:45,2\n")
    src = process_source(
        path,
        delimiter=",",
        timestamp=TimestampSpec(
            column="Time", format="%d/%m/%Y %H:%M", source_timezone="Europe/London"
        ),
    )
    pd.testing.assert_series_equal(import_source(src).time, import_source(src).time)


# ----------------------------------------------------------------------------- Excel and lab


def test_excel_native_dates_and_lab_result_time(tmp_path):
    lab = pd.DataFrame(
        {
            "Sample time": pd.to_datetime(
                ["2026-02-01 08:00", "2026-02-01 10:00", "2026-02-01 10:00"]
            ),
            "Result time": pd.to_datetime(
                ["2026-02-01 09:00", "2026-02-01 09:30", "2026-02-01 11:00"]
            ),
            "Purity": [0.985, "pending", 0.987],
            "ID": ["L1", "L2", "L3"],
        }
    )
    path = tmp_path / "lab.xlsx"
    lab.to_excel(path, sheet_name="Results", index=False)
    assert list_sheets(path) == ["Results"]
    tz = "Asia/Singapore"
    src = DataSource(
        source_id="lab",
        kind=SourceKind.LAB,
        path=str(path),
        sheet="Results",
        timestamp=TimestampSpec(
            column="Sample time", kind=TimestampKind.NATIVE, source_timezone=tz
        ),
        result_timestamp=TimestampSpec(
            column="Result time", kind=TimestampKind.NATIVE, source_timezone=tz
        ),
        sample_id_column="ID",
    )
    data = import_source(src)
    assert data.columns == ["Purity"]
    assert data.time.iloc[0] == pd.Timestamp("2026-02-01 00:00", tz="UTC")
    c = codes(data)
    assert c["result_before_sample"] is Severity.WARNING  # L2 result before its sample time
    assert c["missing_target"] is Severity.WARNING  # 'pending'
    assert c["duplicate_times"] is Severity.INFO  # lab replicates kept by default
    norm = data.normalized
    assert len(norm.values) == 3 and norm.sample_id.tolist() == ["L1", "L2", "L3"]


def test_excel_1904_serial_dates(tmp_path):
    from openpyxl import Workbook
    from openpyxl.utils.datetime import CALENDAR_MAC_1904

    wb = Workbook()
    wb.epoch = CALENDAR_MAC_1904
    ws = wb.active
    ws.append(["Time", "A"])
    ws.append([44478.5, 1.0])  # 2025-10-10 12:00 in the 1904 system
    path = tmp_path / "mac.xlsx"
    wb.save(path)
    assert excel_date_system(path) == 1904
    src = process_source(
        path,
        timestamp=TimestampSpec(
            column="Time", kind=TimestampKind.EXCEL_SERIAL, excel_date_system=1904
        ),
    )
    assert import_source(src).time.iloc[0] == pd.Timestamp("2025-10-10 12:00", tz="UTC")


def test_lab_outside_process_period(tmp_path, small_dataset):
    proc = write(tmp_path / "p.csv", "Time,A\n01/02/2026 00:00,1\n01/02/2026 01:00,2\n")
    labf = write(tmp_path / "l.csv", "Time,Y\n01/02/2026 00:30,1\n05/02/2026 00:30,2\n")
    p = import_source(process_source(proc, delimiter=","))
    lab_src = DataSource(
        source_id="l", kind=SourceKind.LAB, path=str(labf), delimiter=",",
        timestamp=TimestampSpec(column="Time", format="%d/%m/%Y %H:%M"),
    )  # fmt: skip
    issue = lab_coverage_issue(p, import_source(lab_src))
    assert issue is not None and "1 of 2" in issue.message


def test_cancellation_between_chunks(tmp_path, monkeypatch):
    from inferential_studio.core import importers
    from inferential_studio.core.tasks import CancelToken, TaskCancelled

    monkeypatch.setattr(importers, "CSV_CHUNK_ROWS", 10)
    body = "".join(f"01/02/2026 {i // 60:02d}:{i % 60:02d},{i}\n" for i in range(100))
    path = write(tmp_path / "p.csv", "Time,A\n" + body)
    token = CancelToken()
    seen = []

    def progress(f, m):
        seen.append(f)
        if len(seen) == 3:
            token.cancel()

    with pytest.raises(TaskCancelled):
        import_source(process_source(path, delimiter=","), progress=progress, cancel=token)


def test_utf16_with_bom(tmp_path):
    path = tmp_path / "u16.csv"
    path.write_bytes("Time,A\n2026-01-01 00:00,1.5\n".encode("utf-16"))
    opts = sniff_csv(path)
    assert opts.encoding == "utf-16" and opts.delimiter == ","
    src = process_source(
        path, delimiter=",", encoding="utf-16", timestamp=TimestampSpec(column="Time")
    )
    assert import_source(src).values["A"].tolist() == [1.5]
