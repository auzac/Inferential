# PyInstaller build specification: one-folder ("onedir") windowed build.
#
# One-folder keeps Qt as separate, replaceable DLLs (LGPL requirement), starts faster than
# one-file builds and triggers fewer antivirus false positives.
#
# Run from the repository root:  pyinstaller --noconfirm --clean packaging/pyinstaller.spec
# (packaging/windows_build.ps1 does this and runs a smoke test.)

import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH is defined by PyInstaller
sys.path.insert(0, str(ROOT / "src"))
from inferential_studio import __version__  # noqa: E402

APP_EXE_NAME = "InferentialModelStudio"

datas = [
    (str(ROOT / "LICENSES.md"), "."),
    (str(ROOT / "licenses"), "licenses"),  # LGPL/GPL v3 texts (PySide6 wheels do not ship them)
    (str(ROOT / "README.md"), "."),
    (str(ROOT / "CHANGELOG.md"), "."),
    (str(ROOT / "docs"), "docs"),
    (str(ROOT / "examples" / "synthetic_distillation"), "examples/synthetic_distillation"),
]

# Ship each runtime dependency's package metadata, including its license files, so the
# third-party notices travel with the binaries.
from PyInstaller.utils.hooks import copy_metadata  # noqa: E402

RUNTIME_DISTRIBUTIONS = [
    "PySide6", "PySide6_Essentials", "PySide6_Addons", "shiboken6", "pyqtgraph", "numpy",
    "pandas", "scipy", "scikit-learn", "openpyxl", "et_xmlfile", "joblib", "threadpoolctl",
    "python-dateutil", "six", "tzdata",
]  # fmt: skip
for _dist in RUNTIME_DISTRIBUTIONS:
    datas += copy_metadata(_dist)

# Large optional Qt modules the application does not use.
excludes = [
    "tkinter",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.QtQuick3D",
    "PySide6.QtMultimedia",
    "PySide6.QtPdf",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtSerialPort",
    "PySide6.QtSensors",
    "PySide6.QtWebSockets",
    "PySide6.QtNetworkAuth",
    "matplotlib",
    "IPython",
    "pytest",
]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    hiddenimports=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

icon = ROOT / "assets" / "icons" / "app.ico"

# Windows version resource (shown in Explorer > Properties > Details).
_numbers = [int(p) for p in __version__.split(".")[:3]] + [0]
_tuple = tuple(_numbers[:4])
version_file = ROOT / "packaging" / "output" / "version_info.txt"
version_file.parent.mkdir(parents=True, exist_ok=True)
version_file.write_text(
    f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={_tuple}, prodvers={_tuple}, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('ProductName', 'Inferential Model Studio'),
      StringStruct('FileDescription', 'Inferential Model Studio'),
      StringStruct('ProductVersion', '{__version__}'),
      StringStruct('FileVersion', '{__version__}'),
      StringStruct('InternalName', '{APP_EXE_NAME}'),
      StringStruct('OriginalFilename', '{APP_EXE_NAME}.exe'),
      StringStruct('LegalCopyright', 'Proprietary. Third-party notices in LICENSES.md.')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""",
    encoding="utf-8",
)

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_EXE_NAME,
    console=False,  # windowed application; logs go to %LOCALAPPDATA%\InferentialModelStudio\logs
    icon=str(icon) if icon.exists() else None,
    version=str(version_file) if version_file.exists() else None,
    upx=False,  # UPX-compressed binaries are a common antivirus false-positive trigger
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name=APP_EXE_NAME,
)
