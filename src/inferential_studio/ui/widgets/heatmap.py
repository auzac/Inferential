"""Correlation heatmap: diverging colours for r in [-1, 1], sequential for |r|; clickable cells."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from inferential_studio.ui.widgets.chart_menu import add_plot

DIVERGING = pg.ColorMap(
    [0.0, 0.5, 1.0], [(33, 102, 172), (247, 247, 247), (178, 24, 43)]
)  # blue-white-red
SEQUENTIAL = pg.ColorMap([0.0, 1.0], [(247, 247, 247), (8, 69, 148)])
ANNOTATE_MAX = 10  # write r in each cell up to this many variables


class _ClickableImage(pg.ImageItem):
    clicked = Signal(int, int)

    def mouseClickEvent(self, ev):  # noqa: N802 - pyqtgraph override
        if ev.button() == Qt.MouseButton.LeftButton:
            pos = ev.pos()
            self.clicked.emit(int(pos.y()), int(pos.x()))
            ev.accept()


class CorrelationHeatmap(QWidget):
    cell_clicked = Signal(str, str)  # row variable, column variable

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._corr = pd.DataFrame()
        self._pairs: pd.DataFrame | None = None
        self._absolute = False

        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setBackground("w")
        self.canvas.setProperty("chartName", "correlation")
        self.plot = add_plot(self.canvas)
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.image = _ClickableImage()
        self.image.clicked.connect(self._on_click)
        self.plot.addItem(self.image)
        self.colorbar = pg.ColorBarItem(
            values=(-1, 1), colorMap=DIVERGING, interactive=False, width=12
        )
        self.colorbar.setImageItem(self.image, insert_in=self.plot)
        self._texts: list[pg.TextItem] = []
        self._cell_font = QFont()
        self._cell_font.setPointSize(7)
        self._placeholder = pg.TextItem("", color="#596675", anchor=(0.5, 0.5))
        self._placeholder.setPos(0.5, 0.5)
        self._placeholder.setVisible(False)
        self.plot.addItem(self._placeholder)

        self.hover = QLabel(" ")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.hover)
        self._proxy = pg.SignalProxy(
            self.canvas.scene().sigMouseMoved, rateLimit=30, slot=self._on_hover
        )

    def set_placeholder(self, text: str) -> None:
        """Show a message instead of a matrix (e.g. fewer than two variables chosen)."""
        self._corr = pd.DataFrame()
        self._pairs = None
        for t in self._texts:
            self.plot.removeItem(t)
        self._texts = []
        self.image.clear()
        self.plot.getAxis("left").setTicks([[]])
        self.plot.getAxis("bottom").setTicks([[]])
        self.colorbar.setVisible(False)
        self._placeholder.setText(text)
        self._placeholder.setVisible(True)
        self.plot.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
        self.hover.setText(" ")

    def set_matrix(
        self, corr: pd.DataFrame, pairs: pd.DataFrame | None = None, *, absolute: bool = False
    ) -> None:
        if len(corr) == 0:
            self.set_placeholder("No variables to correlate.")
            return
        self._placeholder.setVisible(False)
        self.colorbar.setVisible(True)
        self._corr = corr
        self._pairs = pairs
        self._absolute = absolute
        n = len(corr)
        values = corr.to_numpy(dtype=float)
        if absolute:
            values = np.abs(values)
        cmap, lo = (SEQUENTIAL, 0.0) if absolute else (DIVERGING, -1.0)
        # Float image coloured by the colour bar's map; NaN cells (too few pairs) stay blank.
        self.colorbar.setColorMap(cmap)
        self.colorbar.setLevels((lo, 1.0))
        self.image.setImage(values.T, autoLevels=False, levels=(lo, 1.0))  # ImageItem is [x, y]

        names = [str(c) for c in corr.columns]
        self.plot.getAxis("left").setTicks(
            [[(i + 0.5, f"{i + 1}. {name}") for i, name in enumerate(names)]]
        )
        self.plot.getAxis("bottom").setTicks([[(i + 0.5, str(i + 1)) for i in range(n)]])
        for t in self._texts:
            self.plot.removeItem(t)
        self._texts = []
        if 0 < n <= ANNOTATE_MAX:
            for i in range(n):
                for j in range(n):
                    v = values[i, j]
                    if np.isnan(v):
                        continue
                    colour = "w" if abs(v) > 0.6 else "k"
                    text = pg.TextItem(f"{v:.2f}", color=colour, anchor=(0.5, 0.5))
                    text.setFont(self._cell_font)
                    text.setPos(j + 0.5, i + 0.5)
                    self.plot.addItem(text)
                    self._texts.append(text)
        self.plot.setRange(xRange=(0, max(n, 1)), yRange=(0, max(n, 1)), padding=0.02)

    def _cell_text(self, i: int, j: int) -> str:
        names = list(self._corr.columns)
        r = self._corr.iat[i, j]
        n_pairs = f", n = {int(self._pairs.iat[i, j]):,}" if self._pairs is not None else ""
        r_text = "not enough data" if pd.isna(r) else f"r = {r:.3f}"
        return f"{names[i]}  ×  {names[j]}:  {r_text}{n_pairs}"

    def _on_click(self, row: int, col: int) -> None:
        n = len(self._corr)
        if 0 <= row < n and 0 <= col < n:
            self.cell_clicked.emit(str(self._corr.index[row]), str(self._corr.columns[col]))

    def _on_hover(self, event) -> None:  # noqa: ANN001
        if self._corr.empty:
            return
        pos = self.plot.vb.mapSceneToView(event[0])
        i, j = int(np.floor(pos.y())), int(np.floor(pos.x()))
        n = len(self._corr)
        self.hover.setText(self._cell_text(i, j) if 0 <= i < n and 0 <= j < n else " ")
