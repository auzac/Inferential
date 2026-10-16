"""Reading and writing ``.imsproj`` project files.

Format (documented in docs/file-formats.md): a ZIP archive containing

    manifest.json   format id, schema version, app and library versions, save time, storage
                    mode and the list of embedded data files with their SHA-256
    project.json    the :class:`~inferential_studio.core.schemas.Project` as JSON
    results.json    optional: summary of the last training run (to check reproduction)
    data/<source id>/<file name>
                    optional, portable projects only: byte-exact copies of the source files

Security rules (project files are data, never code):

* Entries are read into memory; nothing is extracted to disk, so path traversal is impossible.
  Entry names are still checked and unexpected entries are rejected.
* Entry sizes are checked *before* decompression to stop zip bombs.
* No pickle/joblib is ever read from a project file.

Saves are atomic: the archive is written to a temporary file in the same folder and then
moved over the target with :func:`os.replace`, so a crash never leaves a half-written project.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from inferential_studio import __version__
from inferential_studio.core.schemas import (
    SCHEMA_VERSION,
    DataSource,
    Project,
    SchemaError,
    from_jsonable,
    to_jsonable,
    utc_now,
)

PROJECT_EXTENSION = ".imsproj"
FORMAT_ID = "inferential-model-studio-project"

MANIFEST_NAME = "manifest.json"
PROJECT_NAME = "project.json"
RESULTS_NAME = "results.json"
ALLOWED_ENTRIES = frozenset({MANIFEST_NAME, PROJECT_NAME, RESULTS_NAME})
DATA_ENTRY = re.compile(r"^data/[A-Za-z0-9_][A-Za-z0-9_.-]*/[^/\\:*?\"<>|.][^/\\:*?\"<>|]*$")
SAFE_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
MAX_EMBEDDED_BYTES = 4 * 1024 * 1024 * 1024  # per embedded data file (uncompressed)
REFERENCE = "reference"
PORTABLE = "portable"

MAX_JSON_BYTES = 64 * 1024 * 1024  # uncompressed limit per JSON entry
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024  # refuse absurd files before opening


class ProjectFileError(Exception):
    """A project file could not be read or written. The message is shown to the user."""


# --------------------------------------------------------------------------- save


def library_versions() -> dict[str, str]:
    """Versions recorded in every saved file, for reproducibility audits."""
    import numpy
    import pandas
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def save_project(
    project: Project,
    path: str | os.PathLike[str],
    *,
    embed: dict[str, str | os.PathLike[str]] | None = None,
    results: dict[str, Any] | None = None,
) -> Project:
    """Atomically save ``project`` to ``path``. Returns the saved copy (with updated timestamp).

    ``embed`` (portable projects): source id -> file to copy into the archive. Each file must
    match the checksum recorded at import. ``results``: JSON summary of a training run.
    """
    target = Path(path)
    if target.suffix.lower() != PROJECT_EXTENSION:
        raise ProjectFileError(f"Project files must use the '{PROJECT_EXTENSION}' extension.")

    try:
        project.validate()
    except SchemaError as exc:
        raise ProjectFileError(
            f"The project cannot be saved because it is inconsistent: {exc}"
        ) from exc

    embedded = []
    for source_id, file in (embed or {}).items():
        file = Path(file)
        source = next((s for s in project.sources if s.source_id == source_id), None)
        if source is None or not SAFE_ID.match(source_id):
            raise ProjectFileError(f"Cannot embed data for unknown source '{source_id}'.")
        if not file.is_file():
            raise ProjectFileError(f"Cannot embed '{file}': the file was not found.")
        digest = sha256_file(file)
        if source.sha256 is not None and digest != source.sha256:
            raise ProjectFileError(
                f"'{file.name}' changed since it was imported; reload it before saving a "
                "portable project."
            )
        name = re.sub(r'[/\\:*?"<>|]', "_", Path(source.path).name) or "data"
        embedded.append(
            {"source_id": source_id, "entry": f"data/{source_id}/{name}", "sha256": digest,
             "size": file.stat().st_size, "file": file}
        )  # fmt: skip

    saved = replace(project, modified_at=utc_now())
    manifest = {
        "format": FORMAT_ID,
        "schema_version": SCHEMA_VERSION,
        "app_version": __version__,
        "saved_at": saved.modified_at.isoformat(),
        "libraries": library_versions(),
        "storage": PORTABLE if embedded else REFERENCE,
        "embedded": [{k: v for k, v in e.items() if k != "file"} for e in embedded],
    }
    project_json = json.dumps(to_jsonable(saved), indent=2, ensure_ascii=False)

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".imsproj-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
                archive.writestr(PROJECT_NAME, project_json)
                if results is not None:
                    archive.writestr(RESULTS_NAME, json.dumps(results, indent=2, allow_nan=False))
                for e in embedded:
                    archive.write(e["file"], e["entry"])
            handle.flush()
            os.fsync(handle.fileno())
        # mkstemp creates owner-only files; give the project ordinary document permissions
        # (or keep those of the file being replaced). No effect on Windows ACLs.
        os.chmod(tmp_name, target.stat().st_mode & 0o777 if target.exists() else 0o644)
        os.replace(tmp_name, target)
    except OSError as exc:
        _silent_unlink(tmp_name)
        raise ProjectFileError(
            f"Could not save project to '{target}': {exc.strerror or exc}."
        ) from exc
    except BaseException:
        _silent_unlink(tmp_name)
        raise
    return saved


def _silent_unlink(name: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(name)


# --------------------------------------------------------------------------- load


def _open_checked(path: str | os.PathLike[str]) -> Path:
    source = Path(path)
    if not source.is_file():
        raise ProjectFileError(f"Project file '{source}' does not exist.")
    if source.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ProjectFileError(f"'{source.name}' is too large to be a project file.")
    if not zipfile.is_zipfile(source):
        raise ProjectFileError(f"'{source.name}' is not a valid project file (not a ZIP archive).")
    return source


def load_project(path: str | os.PathLike[str]) -> Project:
    """Load and validate a project file. Raises ProjectFileError with a user-facing message."""
    source = _open_checked(path)
    try:
        with zipfile.ZipFile(source) as archive:
            entries = _checked_entries(archive)
            manifest = _read_json(archive, entries, MANIFEST_NAME)
            project_data = _read_json(archive, entries, PROJECT_NAME)
    except zipfile.BadZipFile as exc:
        raise ProjectFileError(f"'{source.name}' is damaged and cannot be opened: {exc}.") from exc

    _check_manifest(manifest, source.name)
    project_data = migrate(project_data, int(manifest["schema_version"]))
    try:
        return from_jsonable(Project, project_data)
    except SchemaError as exc:
        raise ProjectFileError(f"'{source.name}' contains invalid project data: {exc}") from exc


def read_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The validated manifest (storage mode, embedded files, versions) of a project file."""
    source = _open_checked(path)
    try:
        with zipfile.ZipFile(source) as archive:
            entries = _checked_entries(archive)
            manifest = _read_json(archive, entries, MANIFEST_NAME)
    except zipfile.BadZipFile as exc:
        raise ProjectFileError(f"'{source.name}' is damaged and cannot be opened: {exc}.") from exc
    _check_manifest(manifest, source.name)
    embedded = manifest.get("embedded", [])
    if not isinstance(embedded, list):
        raise ProjectFileError(f"'{source.name}' has an invalid list of embedded files.")
    for e in embedded:
        if (
            not isinstance(e, dict)
            or not isinstance(e.get("entry"), str)
            or not DATA_ENTRY.match(e["entry"])
            or e["entry"] not in entries
            or not SAFE_ID.match(str(e.get("source_id", "")))
            or not isinstance(e.get("sha256"), str)
        ):
            raise ProjectFileError(f"'{source.name}' lists an invalid embedded data file.")
    return manifest


def load_results(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The saved training-results summary, or None when the project has none."""
    source = _open_checked(path)
    with zipfile.ZipFile(source) as archive:
        entries = _checked_entries(archive)
        if RESULTS_NAME not in entries:
            return None
        data = _read_json(archive, entries, RESULTS_NAME)
    return data if isinstance(data, dict) else None


def extract_embedded(
    path: str | os.PathLike[str], dest_dir: str | os.PathLike[str]
) -> dict[str, Path]:
    """Copy a portable project's data files to ``dest_dir`` (verified). Source id -> file path.

    Only entries listed in the manifest are written, under names built from validated parts;
    sizes are enforced while copying and every file must match its recorded SHA-256.
    """
    manifest = read_manifest(path)
    out: dict[str, Path] = {}
    dest_root = Path(dest_dir).resolve()
    with zipfile.ZipFile(path) as archive:
        for e in manifest.get("embedded", []):
            info = archive.getinfo(e["entry"])
            if info.file_size > MAX_EMBEDDED_BYTES:
                raise ProjectFileError(f"Embedded file '{e['entry']}' is too large.")
            _, source_id, name = e["entry"].split("/", 2)
            folder = dest_root / source_id
            target = (folder / name).resolve()
            if dest_root not in target.parents:
                raise ProjectFileError(f"Embedded file '{e['entry']}' has an unsafe name.")
            if target.is_file() and sha256_file(target) == e["sha256"]:
                out[e["source_id"]] = target  # already extracted earlier
                continue
            folder.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            digest = hashlib.sha256()
            written = 0
            with archive.open(info) as src, open(tmp, "wb") as dst:
                while chunk := src.read(1024 * 1024):
                    written += len(chunk)
                    if written > info.file_size:
                        dst.close()
                        tmp.unlink(missing_ok=True)
                        raise ProjectFileError(f"Embedded file '{e['entry']}' is corrupt.")
                    digest.update(chunk)
                    dst.write(chunk)
            if digest.hexdigest() != e["sha256"]:
                tmp.unlink(missing_ok=True)
                raise ProjectFileError(
                    f"Embedded file '{e['entry']}' does not match its checksum; the project is damaged."
                )
            shutil.move(tmp, target)
            out[e["source_id"]] = target
    return out


def _checked_entries(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    entries: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        name = info.filename
        if name.startswith(("/", "\\")) or ".." in name.split("/") or "\\" in name or ":" in name:
            raise ProjectFileError(f"Project file contains an unsafe entry name '{name}'.")
        if name not in ALLOWED_ENTRIES and not DATA_ENTRY.match(name):
            raise ProjectFileError(f"Project file contains an unexpected entry '{name}'.")
        if name in entries:
            raise ProjectFileError(f"Project file contains duplicate entry '{name}'.")
        entries[name] = info
    return entries


def _read_json(archive: zipfile.ZipFile, entries: dict[str, zipfile.ZipInfo], name: str) -> Any:
    info = entries.get(name)
    if info is None:
        raise ProjectFileError(f"Project file is missing '{name}'.")
    if info.file_size > MAX_JSON_BYTES:
        raise ProjectFileError(f"'{name}' inside the project file is unreasonably large.")
    with archive.open(info) as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:  # header lied about the size
        raise ProjectFileError(f"'{name}' inside the project file is unreasonably large.")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectFileError(
            f"'{name}' inside the project file is not valid JSON: {exc}."
        ) from exc


def _check_manifest(manifest: Any, filename: str) -> None:
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT_ID:
        raise ProjectFileError(f"'{filename}' is not an Inferential Model Studio project.")
    version = manifest.get("schema_version")
    if not isinstance(version, int) or version < 1:
        raise ProjectFileError(f"'{filename}' has an invalid schema version.")
    if version > SCHEMA_VERSION:
        raise ProjectFileError(
            f"'{filename}' was saved by a newer version of the application "
            f"(project schema {version}; this version supports up to {SCHEMA_VERSION}). "
            "Please upgrade Inferential Model Studio to open it."
        )


# --------------------------------------------------------------------------- migrations


def _v1_to_v2(project_data: Any) -> Any:
    """v2 only added optional fields with defaults; v1 data is valid as-is."""
    return project_data


def _v2_to_v3(project_data: Any) -> Any:
    """v3 added rules, detectors and flag decisions (empty lists by default)."""
    return project_data


def _v3_to_v4(project_data: Any) -> Any:
    """v4 added alignment options and modelling configuration (defaults when absent)."""
    return project_data


def _v4_to_v5(project_data: Any) -> Any:
    """v5 added optional archive entries (embedded data, results); project JSON unchanged."""
    return project_data


def _v5_to_v6(project_data: Any) -> Any:
    """v6 added opt-in predictor subset search; old projects keep it disabled."""
    return project_data


def _v6_to_v7(project_data: Any) -> Any:
    """v7 added workflow progress (stages visited); old projects start with none."""
    return project_data


# Map "from version" -> function converting project JSON to version + 1.
_MIGRATIONS: dict[int, Any] = {
    1: _v1_to_v2, 2: _v2_to_v3, 3: _v3_to_v4, 4: _v4_to_v5, 5: _v5_to_v6, 6: _v6_to_v7
}


def migrate(project_data: Any, from_version: int) -> Any:
    """Apply explicit schema migrations up to :data:`SCHEMA_VERSION`."""
    version = from_version
    while version < SCHEMA_VERSION:
        step = _MIGRATIONS.get(version)
        if step is None:
            raise ProjectFileError(f"No migration is available from project schema {version}.")
        project_data = step(project_data)
        version += 1
    return project_data


# --------------------------------------------------------------------------- source files


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source_path(source: DataSource, project_path: str | os.PathLike[str] | None) -> Path:
    """Resolve a source path; relative paths are relative to the project file's folder."""
    path = Path(source.path)
    if path.is_absolute() or project_path is None:
        return path
    return Path(project_path).parent / path


def rebase_source_paths(
    project: Project,
    old_project_path: str | os.PathLike[str] | None,
    new_project_path: str | os.PathLike[str],
) -> Project:
    """Return a copy whose source paths still point at the same files after "Save As".

    Files inside the new project folder (or below it) are stored relative to it, so the folder
    can be moved as a whole; anything else is stored as an absolute path.
    """
    new_dir = Path(new_project_path).resolve().parent
    sources = []
    for source in project.sources:
        absolute = resolve_source_path(source, old_project_path).resolve()
        try:
            path = absolute.relative_to(new_dir).as_posix()
        except ValueError:
            path = str(absolute)
        sources.append(replace(source, path=path))
    return replace(project, sources=sources)


def verify_source(source: DataSource, project_path: str | os.PathLike[str] | None) -> str | None:
    """Return a user-facing problem description, or None if the source file is present and unchanged."""
    path = resolve_source_path(source, project_path)
    if not path.is_file():
        return f"Source file '{source.path}' was not found."
    if source.sha256 is not None and sha256_file(path) != source.sha256:
        return f"Source file '{source.path}' has changed since it was imported (checksum mismatch)."
    return None
