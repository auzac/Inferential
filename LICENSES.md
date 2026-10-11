# Licenses and third-party notices

Inferential Model Studio is proprietary software for internal use. It bundles the open-source
components listed below, each under its own license.

## Qt for Python (PySide6, shiboken6) and Qt

PySide6, shiboken6 and the Qt 6 libraries are used under the **GNU Lesser General Public
License v3** (LGPL-3.0-only). They are dynamically linked and shipped as separate, replaceable
files in the application folder. They have not been modified. Source code is available from
https://download.qt.io/official_releases/ (Qt and Qt for Python). The full license texts are
shipped with the application in `licenses/LGPL-3.0.txt` and `licenses/GPL-3.0.txt` (the LGPL
v3 incorporates the GPL v3 by reference).

Qt itself includes third-party components with their own notices; see
https://doc.qt.io/qt-6/licenses-used-in-qt.html.

## Runtime dependency inventory (v0.0.1)

Generated with `packaging/license_inventory.py` from the release environment.

| Package | Version | License | Home page |
|---|---|---|---|
| cloudpickle | 3.1.2 | BSD License | https://github.com/cloudpipe/cloudpickle |
| colorama | 0.4.6 | BSD License | https://github.com/tartley/colorama |
| et_xmlfile | 2.0.0 | MIT License | https://foss.heptapod.net/openpyxl/et_xmlfile |
| joblib | 1.6.0 | BSD-3-Clause | https://joblib.readthedocs.io |
| narwhals | 2.27.1 | MIT | https://github.com/narwhals-dev/narwhals |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org |
| openpyxl | 3.1.5 | MIT License | https://openpyxl.readthedocs.io |
| pandas | 3.0.6 | BSD License | https://pandas.pydata.org |
| pyqtgraph | 0.14.0 | MIT | http://www.pyqtgraph.org |
| PySide6 | 6.9.3 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (used under LGPL-3.0) | https://pyside.org |
| PySide6_Addons | 6.9.3 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (used under LGPL-3.0) | https://pyside.org |
| PySide6_Essentials | 6.9.3 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (used under LGPL-3.0) | https://pyside.org |
| python-dateutil | 2.9.0.post0 | BSD License; Apache Software License | https://github.com/dateutil/dateutil |
| scikit-learn | 1.9.1 | BSD-3-Clause | https://scikit-learn.org |
| scipy | 1.18.1 | BSD License | https://scipy.org/ |
| shiboken6 | 6.9.3 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (used under LGPL-3.0) | https://pyside.org |
| six | 1.17.0 | MIT License | https://github.com/benjaminp/six |
| threadpoolctl | 3.7.0 | BSD-3-Clause | https://github.com/joblib/threadpoolctl |
| tzdata | 2026.5 | Apache-2.0 | https://github.com/python/tzdata |

Each package's own license files are shipped in its `*.dist-info` folder inside the
application's `_internal` folder. NumPy and SciPy wheels also bundle compiled libraries
(OpenBLAS and, for SciPy, Fortran runtime libraries); their notices are part of those packages'
license files.

The Python runtime is distributed under the PSF License Agreement
(https://docs.python.org/3/license.html).

## Development-only tools (not shipped)

pytest (MIT), pytest-qt (MIT), ruff (MIT), PyInstaller (GPL-2.0 with a bootloader exception
that permits distributing applications built with it under any license).
