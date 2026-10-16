from __future__ import annotations

import pandas as pd
import pytest

from inferential_studio.core.exports import ExportError, write_table
from inferential_studio.core.tasks import CancelToken, TaskCancelled


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_chunked_export_keeps_all_rows_and_headers(tmp_path, extension):
    frame = pd.DataFrame({"label": ["a,b", "é", "x\ny"] * 1500, "x": range(4500)})
    path = tmp_path / f"data.{extension}"
    write_table(frame, path, notes=["audit note"])
    back = pd.read_csv(path, encoding="utf-8-sig") if extension == "csv" else pd.read_excel(path)
    pd.testing.assert_frame_equal(back, frame)
    if extension == "xlsx":
        assert pd.read_excel(path, sheet_name="Notes").iloc[0, 0] == "audit note"


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_cancelled_export_preserves_destination_and_cleans_temp_file(tmp_path, extension):
    path = tmp_path / f"data.{extension}"
    path.write_bytes(b"original file")
    token = CancelToken()
    with pytest.raises(TaskCancelled):
        write_table(pd.DataFrame({"x": range(4500)}), path, cancel=token,
                    progress=lambda *_args: token.cancel())
    assert path.read_bytes() == b"original file"
    assert list(tmp_path.iterdir()) == [path]


def test_failed_export_preserves_destination_and_cleans_temp_file(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    path.write_bytes(b"original file")

    def fail(*_args):
        raise PermissionError("file is open elsewhere")

    monkeypatch.setattr("inferential_studio.core.exports.os.replace", fail)
    with pytest.raises(ExportError):
        write_table(pd.DataFrame({"x": [1]}), path)
    assert path.read_bytes() == b"original file"
    assert list(tmp_path.iterdir()) == [path]
