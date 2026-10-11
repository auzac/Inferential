"""Inferential Model Studio.

Offline desktop workbench for developing industrial process inferential models.

Package layout:
    core            Pure-Python data, statistics and modelling logic. Never imports Qt,
                    so it can be used from Jupyter or Seeq Data Lab.
    infrastructure  Logging, settings and background-task plumbing used by the app.
    ui              PySide6 widgets. Calls into ``core``; never implements algorithms.
"""

__version__ = "0.5.4"
APP_NAME = "Inferential Model Studio"
APP_ID = "InferentialModelStudio"  # filesystem-safe identifier for config/log directories
