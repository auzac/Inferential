"""Data series colours shared by the on-screen charts and the exported model-card SVGs.

Mid-tones that keep contrast on both the light and the dark chart background (the UI chrome
is themed; the data colours are fixed so a theme switch never recolours the data). The UI
charts (``ui/widgets/plots.py``) and the SVG charts (``report.py``, ``svg.py``) read these,
so the exported model card shows the same splits in the same colours as the app.
"""

TRAIN = "#2F8FD8"
VALIDATION = "#E69F00"
TEST = "#E0632B"
POSITIVE = TRAIN  # contribution bars, matching training
NEGATIVE = TEST  # contribution bars, matching final test
