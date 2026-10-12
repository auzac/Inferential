# Packaging and release

## Build (Windows)

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows_build.ps1
```

The script creates `.venv-build` from `requirements-lock.txt`, runs the test suite, builds the
one-folder application with PyInstaller (`packaging/pyinstaller.spec`), launches the exe with
`--smoke-test` (creates the main window and aligns + trains all four model types on a small
synthetic dataset, so missing bundled numerics fail the build), and writes `packaging/output/InferentialModelStudio-<version>-win64.zip` plus a
`.sha256` checksum.

Status at v0.5.0: the one-folder bundle has been built and smoke-tested **on Linux** (≈ 349 MB;
both smoke tests pass inside the frozen bundle). The PowerShell scripts were syntax-checked with
PowerShell 7 on Linux. The Windows build, the Inno Setup installer and the installer smoke test
have **not yet been run on Windows** — that is the remaining release gate for v1.0.0. The PowerShell script and the Windows build have
**not yet been run on Windows**.

## Why one-folder, no UPX

- Qt libraries stay as separate DLLs that a user can replace — part of meeting LGPL v3.
- Faster start than one-file builds (no unpacking to a temp folder).
- Fewer antivirus false positives; UPX compression is disabled for the same reason.

## Installer

`packaging/installer/InferentialModelStudio.iss` (Inno Setup 6) wraps the one-folder build in a
conventional **per-user** installer — no administrator rights needed:

- installs to `%LOCALAPPDATA%\Programs\InferentialModelStudio`; Start-menu entries for the
  application and the bundled example project; optional desktop shortcut;
- optional association of `.imsproj` files (per user, HKCU);
- uninstaller registered under *Apps & features*; uninstalling removes the program and the
  cache of extracted portable-project data, and keeps settings, logs and recovery copies;
- application icon `assets/icons/app.ico` (drawn by `packaging/make_icon.py`), version
  resource and Windows file details.

`windows_build.ps1` builds the installer when Inno Setup 6 is installed (`ISCC.exe` on PATH or in
its default folder, or `-Iscc <path>`) and then runs `packaging/installer_smoke_test.ps1`:
silent per-user install into a temporary folder → `--smoke-test` → the full workflow on the
bundled example (load data, align, train all models, write a model card) → silent uninstall →
check that the program folder is gone. Outputs: `InferentialModelStudio-<version>-win64.zip`
(portable folder) and `InferentialModelStudio-<version>-setup-win64.exe`, each with a `.sha256`.

## Smoke tests

- `InferentialModelStudio.exe --smoke-test` — create the main window and train MLR, PLS, PCR
  and Ridge on a small synthetic dataset; exit code 0 = pass.
- `InferentialModelStudio.exe <project.imsproj> --smoke-test` — open the project, wait for its
  data and aligned features, train, write `smoke-model-card.html` to the settings folder, exit
  0; any failure (missing data, no trainable model, timeout after 3 minutes) exits non-zero.
  Both run in the automated tests from source (`tests/integration/test_app_smoke.py`).

## Release checklist (manual, on a clean Windows 10/11 x64 VM without Python)

1. Run `windows_build.ps1` on the build PC (tests, bundle, installer, installer smoke test).
2. On the clean VM: install with the setup exe (no admin prompt expected), start from the Start
   menu, open the example from the Start menu entry.
3. Walk the sample journey: import the example files by hand, clean, select variables, align,
   train, validate, reveal the test, export the model card, predictions and model file, save,
   close, reopen (results reproduced), save a portable copy and open it elsewhere.
4. Check display scaling at 100 %, 150 % and 200 %.
5. Window frame (Windows 11 and 10): drag the title bar to move, snap to both halves and to the
   top (Win+arrows and Snap layouts), resize from every edge and corner, double-click to
   maximise, minimise from the taskbar, move between monitors with different scaling. Windows 11
   shows rounded corners and a thin border; Windows 10 a square 1 px border. Switch light/dark,
   and with *Follow Windows* change the Windows mode. If anything misbehaves, *Settings → Use
   the system title bar* is the fallback; record it as a bug.
6. Launch and close several times; check `%LOCALAPPDATA%\InferentialModelStudio\logs`.
7. Uninstall; confirm the program folder and Start-menu entries are gone.

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
