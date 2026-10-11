from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from inferential_studio.core import persistence
from inferential_studio.core.persistence import (
    ProjectFileError,
    load_project,
    resolve_source_path,
    save_project,
    sha256_file,
    verify_source,
)
from inferential_studio.core.schemas import (
    SCHEMA_VERSION,
    DataSource,
    Project,
    SourceKind,
    TimestampSpec,
    to_jsonable,
)
from inferential_studio.core.synthetic import build_sample_project


@pytest.fixture
def project(small_dataset) -> Project:
    return build_sample_project(small_dataset)


def _write_zip(path: Path, entries: dict[str, bytes | str]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return path


def _manifest(version: int = SCHEMA_VERSION) -> str:
    return json.dumps({"format": persistence.FORMAT_ID, "schema_version": version})


def test_roundtrip(tmp_path, project):
    path = tmp_path / "p.imsproj"
    saved = save_project(project, path)
    loaded = load_project(path)
    assert loaded == saved
    assert to_jsonable(loaded)["features"] == to_jsonable(project)["features"]
    assert saved.modified_at >= project.modified_at


def test_save_leaves_no_temp_files_and_records_versions(tmp_path, project):
    path = tmp_path / "p.imsproj"
    save_project(project, path)
    assert [p.name for p in tmp_path.iterdir()] == ["p.imsproj"]
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert {"python", "numpy", "pandas", "scikit-learn"} <= set(manifest["libraries"])


def test_failed_save_keeps_previous_file_intact(tmp_path, project, monkeypatch):
    path = tmp_path / "p.imsproj"
    save_project(project, path)
    before = path.read_bytes()

    def boom(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(zipfile.ZipFile, "writestr", boom)
    with pytest.raises(ProjectFileError, match="No space left"):
        save_project(project, path)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["p.imsproj"]


def test_wrong_extension_rejected(tmp_path, project):
    with pytest.raises(ProjectFileError, match="extension"):
        save_project(project, tmp_path / "p.zip")


def test_missing_and_non_zip_files(tmp_path):
    with pytest.raises(ProjectFileError, match="does not exist"):
        load_project(tmp_path / "nope.imsproj")
    bad = tmp_path / "bad.imsproj"
    bad.write_text("hello")
    with pytest.raises(ProjectFileError, match="not a valid project"):
        load_project(bad)


@pytest.mark.parametrize("name", ["../evil.json", "/abs.json", "dir\\x.json", "C:evil.json"])
def test_unsafe_entry_names_rejected(tmp_path, name):
    path = _write_zip(tmp_path / "x.imsproj", {"manifest.json": _manifest(), name: "{}"})
    with pytest.raises(ProjectFileError, match="unsafe entry"):
        load_project(path)


def test_unexpected_entries_rejected(tmp_path):
    path = _write_zip(
        tmp_path / "x.imsproj", {"manifest.json": _manifest(), "model.pkl": b"\x80\x04"}
    )
    with pytest.raises(ProjectFileError, match="unexpected entry"):
        load_project(path)


def test_newer_schema_version_refused(tmp_path, project):
    path = _write_zip(
        tmp_path / "x.imsproj",
        {
            "manifest.json": _manifest(SCHEMA_VERSION + 1),
            "project.json": json.dumps(to_jsonable(project)),
        },
    )
    with pytest.raises(ProjectFileError, match="newer version"):
        load_project(path)


def test_foreign_zip_and_missing_parts(tmp_path):
    path = _write_zip(
        tmp_path / "x.imsproj",
        {"manifest.json": json.dumps({"format": "other"}), "project.json": "{}"},
    )
    with pytest.raises(ProjectFileError, match="not an Inferential Model Studio project"):
        load_project(path)
    path = _write_zip(tmp_path / "y.imsproj", {"manifest.json": _manifest()})
    with pytest.raises(ProjectFileError, match="missing 'project.json'"):
        load_project(path)


def test_invalid_json_and_invalid_content(tmp_path):
    path = _write_zip(
        tmp_path / "x.imsproj", {"manifest.json": _manifest(), "project.json": "{not json"}
    )
    with pytest.raises(ProjectFileError, match="not valid JSON"):
        load_project(path)
    path = _write_zip(
        tmp_path / "y.imsproj",
        {"manifest.json": _manifest(), "project.json": json.dumps({"name": 3})},
    )
    with pytest.raises(ProjectFileError, match="invalid project data"):
        load_project(path)


def test_oversized_entries_rejected_before_reading(tmp_path, project, monkeypatch):
    path = tmp_path / "p.imsproj"
    save_project(project, path)
    monkeypatch.setattr(persistence, "MAX_JSON_BYTES", 100)
    with pytest.raises(ProjectFileError, match="unreasonably large"):
        load_project(path)


def test_source_resolution_and_verification(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n")
    project_file = tmp_path / "p.imsproj"
    src = DataSource(
        source_id="p",
        kind=SourceKind.PROCESS,
        path="data.csv",
        sha256=sha256_file(data),
        timestamp=TimestampSpec(column="a"),
    )
    assert resolve_source_path(src, project_file) == data
    assert verify_source(src, project_file) is None
    data.write_text("a,b\n1,3\n")
    assert "checksum mismatch" in verify_source(src, project_file)
    data.unlink()
    assert "not found" in verify_source(src, project_file)


def test_rebase_source_paths_on_save_as(tmp_path):
    from inferential_studio.core.persistence import rebase_source_paths

    old_dir, new_dir, other = tmp_path / "a", tmp_path / "a" / "sub" / "b", tmp_path / "c"
    old_dir.mkdir()
    src_in_old = DataSource(
        source_id="p", kind=SourceKind.PROCESS, path="data.csv", timestamp=TimestampSpec(column="t")
    )
    project = Project(sources=[src_in_old])
    rebased = rebase_source_paths(project, old_dir / "p.imsproj", other / "q.imsproj")
    assert Path(rebased.sources[0].path) == (old_dir / "data.csv").resolve()  # outside -> absolute
    inside = rebase_source_paths(project, old_dir / "p.imsproj", old_dir / "q.imsproj")
    assert inside.sources[0].path == "data.csv"  # same folder -> stays relative
    deeper = rebase_source_paths(
        Project(sources=[replace(src_in_old, path=str(new_dir / "x.csv"))]),
        None,
        new_dir.parent / "r.imsproj",
    )
    assert deeper.sources[0].path == "b/x.csv"


def test_schema_v1_file_is_migrated(tmp_path, project):
    """Files saved by v0.0.1 (schema 1) open in later versions."""
    data = to_jsonable(project)
    for src in data["sources"]:  # fields that did not exist in schema 1
        src.pop("duplicate_policy")
        src.pop("sample_id_column")
    path = _write_zip(
        tmp_path / "old.imsproj", {"manifest.json": _manifest(1), "project.json": json.dumps(data)}
    )
    loaded = load_project(path)
    assert loaded.source("lab").sample_id_column is None
    assert loaded.source("process").effective_duplicate_policy.value == "error"
