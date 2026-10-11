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
def test_unsafe_entry_names_rejected(tmp_path, name, monkeypatch):
    # zipfile's own sanitizer stores every entry name with "/" separators, so a raw
    # backslash name cannot reach the reader through the public API. Disable the
    # sanitizer here to simulate a naive zip implementation handing the application
    # a hostile, spec-violating entry name — the case the check exists for.
    monkeypatch.setattr(zipfile, "_sanitize_filename", lambda f: f)
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


def test_schema_v3_file_gets_default_modelling_settings(tmp_path, project):
    """Files saved by v0.2.x (schema 3) have no alignment or modelling sections."""
    data = to_jsonable(project)
    data.pop("alignment")
    data.pop("modelling")
    path = _write_zip(
        tmp_path / "v3.imsproj", {"manifest.json": _manifest(3), "project.json": json.dumps(data)}
    )
    loaded = load_project(path)
    assert loaded.alignment.reject_regime_overlap
    assert [m.kind.value for m in loaded.modelling.models] == ["mlr", "pls", "pcr", "ridge"]
    assert loaded.modelling.split.train_fraction == 0.6


def test_modelling_settings_roundtrip(tmp_path, project):
    from dataclasses import replace
    from datetime import UTC, datetime

    from inferential_studio.core.schemas import (
        CvKind,
        ModelKind,
        ModellingConfig,
        ModelSpec,
        Scaling,
        SplitConfig,
        SplitMode,
    )

    cfg = ModellingConfig(
        scaling=Scaling.ROBUST,
        split=SplitConfig(
            mode=SplitMode.DATES,
            train_end=datetime(2026, 3, 10, tzinfo=UTC),
            validation_end=datetime(2026, 3, 12, tzinfo=UTC),
            purge_s=3600.0,
            cv_kind=CvKind.BLOCKED,
            cv_folds=3,
        ),
        models=(ModelSpec(ModelKind.PLS, n_components=3), ModelSpec(ModelKind.RIDGE, alpha=0.5)),
        one_standard_error=False,
    )
    project = replace(project, modelling=cfg)
    save_project(project, tmp_path / "m.imsproj")
    assert load_project(tmp_path / "m.imsproj").modelling == cfg


# --------------------------------------------------------------------------- schema 5


def _portable(tmp_path, project):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    proc = data_dir / "tower_process.csv"
    proc.write_text("Timestamp,x\n01/01/2026 00:00,1\n", encoding="utf-8")
    lab = data_dir / "lab_results.xlsx"
    lab.write_bytes(b"not really excel")
    sources = [replace(s, sha256=sha256_file(proc if s.kind is SourceKind.PROCESS else lab))
               for s in project.sources]  # fmt: skip
    project = replace(project, sources=sources)
    path = tmp_path / "out" / "portable.imsproj"
    save_project(project, path, embed={"process": proc, "lab": lab}, results={"models": []})
    return project, path, proc, lab


def test_portable_project_embeds_and_extracts_verified_copies(tmp_path, project):
    project, path, proc, lab = _portable(tmp_path, project)
    manifest = persistence.read_manifest(path)
    assert manifest["storage"] == "portable"
    assert {e["entry"] for e in manifest["embedded"]} == {
        "data/process/tower_process.csv",
        "data/lab/lab_results.xlsx",
    }
    assert load_project(path) == load_project(path)  # project JSON still loads normally
    assert persistence.load_results(path) == {"models": []}
    out = persistence.extract_embedded(path, tmp_path / "cache")
    assert out["process"].read_bytes() == proc.read_bytes()
    assert out["lab"].read_bytes() == lab.read_bytes()
    assert persistence.extract_embedded(path, tmp_path / "cache") == out  # idempotent


def test_portable_refuses_changed_source_and_tampered_archive(tmp_path, project):
    project, path, proc, _lab = _portable(tmp_path, project)
    proc.write_text("changed", encoding="utf-8")
    with pytest.raises(ProjectFileError, match="changed since it was imported"):
        save_project(project, tmp_path / "again.imsproj", embed={"process": proc})
    tampered = tmp_path / "tampered.imsproj"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tampered, "w") as dst:
        for info in src.infolist():
            data = src.read(info)
            if info.filename == "data/process/tower_process.csv":
                data = b"evil"
            dst.writestr(info.filename, data)
    with pytest.raises(ProjectFileError, match="checksum"):
        persistence.extract_embedded(tampered, tmp_path / "cache2")


@pytest.mark.parametrize(
    "name", ["data/../x.csv", "data/a/../../x", "data/a/b/c.csv", "data/.hidden/x", "data/a/.."]
)
def test_unsafe_data_entries_rejected(tmp_path, name):
    path = _write_zip(
        tmp_path / "x.imsproj",
        {"manifest.json": _manifest(), "project.json": "{}", name: b"x"},
    )
    with pytest.raises(ProjectFileError):
        load_project(path)


def test_reference_projects_have_no_results_by_default(tmp_path, project):
    path = save_project(project, tmp_path / "r.imsproj") and tmp_path / "r.imsproj"
    assert persistence.read_manifest(path)["storage"] == "reference"
    assert persistence.load_results(path) is None


def test_recovery_copy_roundtrip(tmp_path, project):
    from inferential_studio.infrastructure.recovery import (
        RecoveryWriter,
        discard_copy,
        find_recovery_copies,
    )

    original = tmp_path / "proj" / "p.imsproj"
    writer = RecoveryWriter()
    assert writer.write(project, original)
    copies = find_recovery_copies()
    assert len(copies) == 1 and copies[0].original_path == original
    recovered = load_project(copies[0].project_file)
    # Source paths were rebased so the data files still resolve from the recovery folder.
    assert resolve_source_path(recovered.sources[0], copies[0].project_file).resolve() == (
        resolve_source_path(project.sources[0], original).resolve()
    )
    discard_copy(copies[0])
    assert find_recovery_copies() == []
