"""One right-click menu for every chart: **Copy image** and **Save image…**.

PyQtGraph's own menus (view range, axis options, mouse mode, transforms, export dialog) and its
small "A" auto-range button are switched off everywhere: view commands live on the page (for
example **Back** / **Fit all** on trends), and the raw settings could leave a chart in a state
the page does not expect (log axes, frozen ranges).

Every chart's view box is a :class:`ChartViewBox`, whose mouse navigation is disabled (no pan,
rubber-band zoom or wheel zoom), so a chart can never be left in a state the page does not
expect. A right *click* (not a right-drag) opens the standard menu, which acts on the whole
chart widget — all stacked trends, the full scatter matrix — exactly as shown on screen.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFileDialog, QGraphicsView, QMenu, QMessageBox, QWidget

COPY_TEXT = "Copy image"
SAVE_TEXT = "Save image…"


class ChartViewBox(pg.ViewBox):
    """View box for the static charts: right-click opens the standard chart menu, and mouse
    navigation is off so a chart can never be dragged into a state the page cannot undo.

    ``trend_view._TrendViewBox`` subclasses this and re-enables the axes it needs for its
    deliberate drag-to-zoom / pan navigation.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003 - pyqtgraph signature
        kwargs["enableMenu"] = False
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(False, False)  # no pan, no rubber-band zoom, no wheel zoom
        self.empty = False  # set by plots.ChartPlotItem while it shows its "no data" message

    def mouseClickEvent(self, ev) -> None:  # noqa: ANN001, N802 - pyqtgraph override
        if ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            view = chart_view(self)
            if view is not None and not self.empty:
                show_chart_menu(view, ev.screenPos().toPoint())
            return
        super().mouseClickEvent(ev)


def tidy_plot(plot: pg.PlotItem) -> pg.PlotItem:
    """Switch off PyQtGraph's plot menu and "A" button (the view box handles right-click)."""
    plot.setMenuEnabled(False)
    plot.hideButtons()
    return plot


def add_plot(layout: pg.GraphicsLayoutWidget, **kwargs) -> pg.PlotItem:  # noqa: ANN003
    """``layout.addPlot`` with the standard view box and menu (use instead of ``addPlot``)."""
    kwargs.setdefault("viewBox", ChartViewBox())
    return tidy_plot(layout.addPlot(**kwargs))


def chart_view(item: pg.GraphicsObject) -> QGraphicsView | None:
    """The widget showing ``item`` (a PlotWidget or GraphicsLayoutWidget)."""
    scene = item.scene()
    views = scene.views() if scene is not None else []
    return views[0] if views else None


def chart_name(view: QWidget) -> str:
    """File-name stem for a chart: its ``chartName`` property, else its first title, else "chart"."""
    name = view.property("chartName")
    if not name and isinstance(view, pg.GraphicsView):
        for it in view.scene().items():
            if isinstance(it, pg.PlotItem) and it.titleLabel.text:
                name = re.sub(r"<[^>]+>", "", it.titleLabel.text)
                break
    stem = re.sub(r"[^\w\-]+", "_", str(name or "chart")).strip("_").lower()
    return stem or "chart"


def copy_chart(view: QWidget) -> None:
    QGuiApplication.clipboard().setPixmap(view.grab())


def save_chart(view: QWidget, path: str | Path) -> bool:
    """Write the chart as shown to ``path`` (format from the suffix: .png, .jpg, .bmp)."""
    return view.grab().save(str(path))


def show_chart_menu(view: QWidget, global_pos: QPoint) -> QMenu:
    menu = QMenu(view)
    copy = menu.addAction(COPY_TEXT)
    copy.triggered.connect(lambda: copy_chart(view))
    save = menu.addAction(SAVE_TEXT)
    save.triggered.connect(lambda: _save_dialog(view))
    menu.popup(global_pos)
    return menu


def _save_dialog(view: QWidget) -> None:
    path, _ = QFileDialog.getSaveFileName(
        view, "Save chart image", f"{chart_name(view)}.png", "PNG image (*.png);;JPEG image (*.jpg)"
    )
    if path and not save_chart(view, path):
        QMessageBox.warning(view, "Save chart image", f"Could not write {path}.")
