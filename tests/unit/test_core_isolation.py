"""Architecture guard: the core package must be importable without Qt (Jupyter / Seeq Data Lab)."""

from __future__ import annotations

import pkgutil
import subprocess
import sys

import inferential_studio.core as core


def test_core_modules_do_not_import_qt():
    modules = [f"inferential_studio.core.{m.name}" for m in pkgutil.iter_modules(core.__path__)]
    assert modules, "no core modules found"
    code = (
        "import importlib, sys\n"
        f"for name in {modules!r}: importlib.import_module(name)\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('PySide6', 'pyqtgraph', 'shiboken6'))\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", f"core imported GUI modules: {result.stdout}"
