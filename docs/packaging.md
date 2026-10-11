# Packaging and release

## Build (Windows)

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows_build.ps1
```

The script creates `.venv-build` from `requirements-lock.txt`, runs the test suite, builds the
one-folder application with PyInstaller (`packaging/pyinstaller.spec`), launches the exe with
`--smoke-test`, and writes `packaging/output/InferentialModelStudio-<version>-win64.zip` plus a
`.sha256` checksum.

Status at v0.1.0: the spec has been built and smoke-tested **on Linux** (bundle ≈ 349 MB,
launches and opens the sample project). The PowerShell script and the Windows build have
**not yet been run on Windows**.

## Why one-folder, no UPX

- Qt libraries stay as separate DLLs that a user can replace — part of meeting LGPL v3.
- Faster start than one-file builds (no unpacking to a temp folder).
- Fewer antivirus false positives; UPX compression is disabled for the same reason.

## Installer

A conventional per-user installer (Inno Setup) wrapping the one-folder build is added in
Milestone 5, together with the application icon, Start-menu entry and uninstaller.

## Licensing release gate (PySide6 / Qt under LGPL v3)

The application is distributed closed-source to a small internal team. That is permitted under
the LGPL v3 provided we:

1. Ship the LGPL v3 and GPL v3 license texts and the Qt/PySide6 notices (LICENSES.md, installed
   with the app).
2. Link Qt dynamically and allow replacement of the Qt/PySide6 libraries (one-folder build).
3. Do not modify Qt/PySide6 sources (we do not). If we ever did, we would have to provide the
   modified sources.
4. Do not impose terms that forbid reverse engineering of the Qt-using parts for debugging
   modifications of those libraries.
5. State where the Qt/PySide6 source can be obtained (https://download.qt.io/official_releases/).

Regenerate the dependency table in LICENSES.md for every release:

```powershell
.venv-build\Scripts\python packaging\license_inventory.py
```

## Code signing

Not configured. Unsigned executables may trigger Windows SmartScreen warnings. Add signing in
the build script when a certificate is available.
