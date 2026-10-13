"""What-if view: move the selected model's inputs and watch its prediction over time change.

One slider per input. *Shift* adds an offset to the input (± its training range); *Hold* fixes
it at a value inside its training range, which removes its own variation from the line. The
original prediction stays on the chart for comparison. The maths is in ``core/what_if.py``.
"""

from __future__ import annotations

from html import escape

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from inferential_studio.core.splits import SPLITS
from inferential_studio.core.what_if import HOLD, SHIFT, WhatIfBase, WhatIfInput, evaluate
from inferential_studio.ui.theme import TOKENS
from inferential_studio.ui.widgets.plots import (
    BLUE,
    GREY,
    ORANGE,
    SPLIT_NAMES,
    legend,
    plot_widget,
    scatter,
)
from inferential_studio.ui.widgets.time_axis import to_epoch_seconds

STEPS = 1000  # slider resolution
PANEL_WIDTH = 340


def _fmt(value: float, units: str | None = None, *, signed: bool = False) -> str:
    text = f"{value:+.4g}" if signed else f"{value:.4g}"
    text = text.replace("-", "−")
    return f"{text} {units}" if units else text


class _InputControl(QFrame):
    """Name, mode, slider and the input's effect on the prediction."""

    changed = Signal()

    def __init__(self, inp: WhatIfInput, target_units: str | None) -> None:
        super().__init__()
        self.setObjectName("whatIfInput")
        self.inp = inp
        self._target_units = target_units
        self.name = QLabel(
            f"<b>{escape(inp.name)}</b>"
            + (f"&nbsp;&nbsp;<span>[{escape(inp.units)}]</span>" if inp.units else "")
        )
        self.name.setTextFormat(Qt.TextFormat.RichText)
        self.mode = QComboBox()
        self.mode.addItem("Shift", SHIFT)
        self.mode.addItem("Hold", HOLD)
        self.mode.setToolTip(
            "Shift: add an offset to every value of this input.\n"
            "Hold: keep this input at one value (removes its own variation from the line)."
        )
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, STEPS)
        self.slider.setEnabled(inp.span > 0)
        self.value = QLabel()
        self.value.setFixedWidth(96)
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.effect = QLabel()
        self.effect.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Two lines: name … effect on Y / mode, slider, value.
        grid = QGridLayout(self)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        top = QHBoxLayout()
        top.addWidget(self.name)
        top.addStretch(1)
        top.addWidget(self.effect)
        grid.addLayout(top, 0, 0, 1, 3)
        grid.addWidget(self.mode, 1, 0)
        grid.addWidget(self.slider, 1, 1)
        grid.addWidget(self.value, 1, 2)
        grid.setColumnStretch(1, 1)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.slider.valueChanged.connect(self._moved)
        self.reset()

    # -- state ------------------------------------------------------------------------------

    def change(self) -> tuple[str, float]:
        frac = self.slider.value() / STEPS
        if self.mode.currentData() == HOLD:
            return HOLD, self.inp.low + frac * self.inp.span
        return SHIFT, (2 * frac - 1) * self.inp.span

    def reset(self) -> None:
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(0)
        self.mode.blockSignals(False)
        self._place_slider()

    def set_effect(self, impact: float | None) -> None:
        if impact is None or abs(impact) == 0:
            self.effect.setText("no change")
            self.effect.setObjectName("placeholderNote")
        else:
            label = "mean Δŷ" if self.mode.currentData() == HOLD else "Δŷ"  # change of Y
            self.effect.setText(f"{label} {_fmt(impact, self._target_units, signed=True)}")
            self.effect.setObjectName("")
        self.effect.style().unpolish(self.effect)
        self.effect.style().polish(self.effect)

    # -- internals --------------------------------------------------------------------------

    def _place_slider(self) -> None:
        inp = self.inp
        self.slider.blockSignals(True)
        if self.mode.currentData() == HOLD and inp.span > 0:
            self.slider.setValue(round((inp.mean - inp.low) / inp.span * STEPS))
            tip = (
                f"Hold anywhere in the training range {_fmt(inp.low)} – "
                f"{_fmt(inp.high, inp.units)}; starts at the training mean {_fmt(inp.mean)}."
            )
        else:
            self.slider.setValue(STEPS // 2)
            tip = f"Shift by up to ± {_fmt(inp.span, inp.units)} (the width of the training range)."
        tip += f"\nCoefficient: {_fmt(inp.coefficient)} {self._target_units or ''} per {inp.units or 'unit'}."
        self.slider.setToolTip(tip)
        self.name.setToolTip(tip)
        self.slider.blockSignals(False)
        self._show_value()

    def _mode_changed(self) -> None:
        self._place_slider()
        self.changed.emit()

    def _moved(self) -> None:
        self._show_value()
        self.changed.emit()

    def _show_value(self) -> None:
        mode, amount = self.change()
        if mode == HOLD:
            self.value.setText(_fmt(amount, self.inp.units))
        else:
            self.value.setText(_fmt(amount, self.inp.units, signed=True))


class WhatIfView(QWidget):
    """Prediction over time for the selected model, with one control per input."""

    def __init__(self, empty_text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._base: WhatIfBase | None = None
        self._controls: list[_InputControl] = []
        self._tz = "UTC"
        self._target_units: str | None = None

        self.chart = plot_widget(
            "What-if prediction", name="what_if", time_axis="UTC", empty_text=empty_text
        )
        legend(self.chart)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.reset_button = QPushButton("Reset all")
        self.reset_button.clicked.connect(self.reset)

        self._list = QWidget()
        self._list_layout = QVBoxLayout(self._list)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(6)
        self._list_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._list)

        panel = QWidget()
        panel.setMinimumWidth(260)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        title = QLabel("Inputs")
        title.setObjectName("sectionTitle")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.reset_button)
        pl.addLayout(head)
        pl.addWidget(self.summary)
        pl.addWidget(scroll, 1)
        title.setToolTip(
            "Inputs are the model's features (windowed, filtered values in their own units).\n"
            "Shift adds an offset to every value; Hold keeps the input at one value, which\n"
            "removes its own variation from the line. One input moves while the others keep\n"
            "their recorded values — in the plant, correlated inputs usually move together.\n"
            "Outside the training range the model extrapolates."
        )
        title.setText("Inputs ⓘ")

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.chart)
        split.addWidget(panel)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([900, PANEL_WIDTH])
        split.setChildrenCollapsible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)

        self._original: pg.PlotDataItem | None = None
        self._what_if: pg.PlotDataItem | None = None
        self._set_enabled(False)

    # -- public -----------------------------------------------------------------------------

    def set_base(
        self, base: WhatIfBase | None, *, timezone: str = "UTC", target_units: str | None = None
    ) -> None:
        """Show ``base`` (None clears the view). Controls start with no change."""
        keep = self._current_changes() if self._same_inputs(base) else {}
        self._base, self._target_units = base, target_units
        if timezone != self._tz:
            self._tz = timezone
            self.chart.getPlotItem().getAxis("bottom").set_timezone(timezone)
        self._rebuild_controls(keep)
        self.chart.clear()
        if base is None or not len(base.times):
            self._set_enabled(False)
            self.summary.setText("")
            self.chart.setTitle("What-if prediction")
            self._what_if = None
            return
        x = to_epoch_seconds(base.times)
        for split in SPLITS:
            m = base.splits == split
            if m.any():
                scatter(self.chart, x[m], base.actual[m], GREY, f"Lab — {SPLIT_NAMES[split]}",
                        size=5)  # fmt: skip
        self._original = self.chart.plot(x, base.predicted, pen=pg.mkPen(BLUE, width=1.5),
                                         name="Prediction")  # fmt: skip
        self._what_if = self.chart.plot(
            x, base.predicted, pen=pg.mkPen(ORANGE, width=2, style=Qt.PenStyle.DashLine),
            name="What-if",
        )  # fmt: skip
        self.chart.setTitle(f"{base.target} [{target_units}]" if target_units else base.target)
        self._set_enabled(True)
        self._update()

    def reset(self) -> None:
        for c in self._controls:
            c.reset()
        self._update()

    def changes(self) -> dict[str, tuple[str, float]]:
        return self._current_changes()

    def apply_theme(self) -> None:
        self._update()

    # -- internals --------------------------------------------------------------------------

    def _same_inputs(self, base: WhatIfBase | None) -> bool:
        if base is None or self._base is None:
            return False
        return [i.name for i in base.inputs] == [i.name for i in self._base.inputs]

    def _current_changes(self) -> dict[str, tuple[str, float]]:
        return {c.inp.name: c.change() for c in self._controls}

    def _rebuild_controls(self, keep: dict[str, tuple[str, float]]) -> None:
        for c in self._controls:
            c.setParent(None)
            c.deleteLater()
        self._controls = []
        if self._base is None:
            return
        # Largest effect over the training range first: the inputs that matter most on top.
        order = sorted(self._base.inputs, key=lambda i: -abs(i.coefficient) * i.span)
        for inp in order:
            control = _InputControl(inp, self._target_units)
            if inp.name in keep:
                mode, amount = keep[inp.name]
                control.mode.blockSignals(True)
                control.mode.setCurrentIndex(control.mode.findData(mode))
                control.mode.blockSignals(False)
                control._place_slider()
                frac = (
                    (amount - inp.low) / inp.span
                    if mode == HOLD
                    else (amount / inp.span + 1) / 2
                    if inp.span
                    else 0.5
                )
                control.slider.setValue(round(min(max(frac, 0.0), 1.0) * STEPS))
            control.changed.connect(self._update)
            self._list_layout.insertWidget(self._list_layout.count() - 1, control)
            self._controls.append(control)

    def _set_enabled(self, enabled: bool) -> None:
        self.reset_button.setEnabled(enabled)

    def _update(self) -> None:
        base = self._base
        if base is None or not len(base.times) or self._what_if is None:
            return
        res = evaluate(base, self._current_changes())
        self._what_if.setData(to_epoch_seconds(base.times), res.predicted)
        for c in self._controls:
            c.set_effect(res.impact.get(c.inp.name))
        units = self._target_units
        if np.allclose(res.delta, 0.0):
            text = "Move a slider to see how the prediction responds."
        elif np.ptp(res.delta) <= 1e-12 * max(1.0, float(np.max(np.abs(res.delta)))):
            text = f"What-if moves the whole prediction by <b>{_fmt(float(res.delta[0]), units, signed=True)}</b>."
        else:
            text = (
                f"Prediction moves by <b>{_fmt(float(np.mean(res.delta)), units, signed=True)}</b> "
                f"on average ({_fmt(float(res.delta.min()), signed=True)} to "
                f"{_fmt(float(res.delta.max()), signed=True)})."
            )
        if res.outside_inputs:
            n = int(res.outside.sum())
            text += (
                f"<br><span style='color:{TOKENS['warning']}'>"
                f"{n} of {len(res.outside)} samples outside the training range "
                f"({escape(', '.join(res.outside_inputs))}) — extrapolation.</span>"
            )
        self.summary.setText(text)
