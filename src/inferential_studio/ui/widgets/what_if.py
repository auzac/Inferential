"""What-if step test: bump the selected model's inputs at steady state and watch it respond.

Top chart: the inputs (change as % of their training range, so different units share an axis).
Bottom chart: the inferential's prediction over time, with the steady-state baseline. One
slider per input sets its step size (± its training range); the step time and run length are
shared. The simulation is in ``core/what_if.py``.
"""

from __future__ import annotations

from html import escape

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
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

from inferential_studio.core.models import ModelResult, TrainingRun
from inferential_studio.core.what_if import (
    StepInput,
    StepModel,
    default_timing,
    simulate,
    step_model,
)
from inferential_studio.ui.theme import TOKENS
from inferential_studio.ui.widgets.plots import (
    BLUE,
    GREY,
    SERIES,
    clear_placeholder,
    legend,
    plot_widget,
    show_placeholder,
)

STEPS = 1000  # slider resolution (0 … STEPS = −span … +span)
PANEL_WIDTH = 340
UPDATE_MS = 30  # coalesce slider drags into one simulation
COMPACT_HEIGHT = 320  # below this the inputs chart hides; the prediction gets the room


def _fmt(value: float, units: str | None = None, *, signed: bool = False) -> str:
    text = (f"{value:+.4g}" if signed else f"{value:.4g}").replace("-", "−")
    return f"{text} {units}" if units else text


def _minutes(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    m = seconds / 60.0
    return f"{m:.0f} min" if abs(m - round(m)) < 1e-6 else f"{m:.1f} min"


class _InputControl(QFrame):
    """Name and effect on Y / step slider and step size."""

    changed = Signal()

    def __init__(self, inp: StepInput, colour: str, target_units: str | None) -> None:
        super().__init__()
        self.inp = inp
        self.colour = colour
        self._target_units = target_units
        swatch = f"<span style='color:{colour}'>━</span>&nbsp;"
        units = f"&nbsp;&nbsp;[{escape(inp.units)}]" if inp.units else ""
        self.name = QLabel(f"{swatch}<b>{escape(inp.name)}</b>{units}")
        self.name.setTextFormat(Qt.TextFormat.RichText)
        self.effect = QLabel()
        self.effect.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, STEPS)
        self.slider.setEnabled(inp.span > 0)
        self.value = QLabel()
        self.value.setFixedWidth(104)
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        tip = (
            f"Steady state {_fmt(inp.baseline, inp.units)} (training mean); training range "
            f"{_fmt(inp.low)} – {_fmt(inp.high, inp.units)}.\nStep up to ± "
            f"{_fmt(inp.span, inp.units)}. Coefficient {_fmt(inp.coefficient)} "
            f"{target_units or ''} per {inp.units or 'unit'}."
        )
        self.name.setToolTip(tip)
        self.slider.setToolTip(tip)
        grid = QGridLayout(self)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        top = QHBoxLayout()
        top.addWidget(self.name)
        top.addStretch(1)
        top.addWidget(self.effect)
        grid.addLayout(top, 0, 0, 1, 2)
        grid.addWidget(self.slider, 1, 0)
        grid.addWidget(self.value, 1, 1)
        grid.setColumnStretch(0, 1)
        self.slider.valueChanged.connect(self._moved)
        self.reset()

    def step(self) -> float:
        return (2 * self.slider.value() / STEPS - 1) * self.inp.span

    def set_step(self, step: float) -> None:
        frac = (step / self.inp.span + 1) / 2 if self.inp.span else 0.5
        self.slider.setValue(round(min(max(frac, 0.0), 1.0) * STEPS))

    def reset(self) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(STEPS // 2)
        self.slider.blockSignals(False)
        self._show()

    def _moved(self) -> None:
        self._show()
        self.changed.emit()

    def _show(self) -> None:
        step = self.step()
        self.value.setText(_fmt(step, self.inp.units, signed=True) if step else "no step")
        if step:
            effect = self.inp.coefficient * step
            self.effect.setText(f"Δŷ {_fmt(effect, self._target_units, signed=True)}")
        else:
            self.effect.setText("")


class WhatIfView(QWidget):
    """Step test on the selected model, starting from steady state."""

    def __init__(self, empty_text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model: StepModel | None = None
        self._controls: list[_InputControl] = []
        self._target_units: str | None = None
        self._timer = QTimer(self, singleShot=True, interval=UPDATE_MS)
        self._timer.timeout.connect(self._update)

        self.inputs_chart = plot_widget("Inputs", name="step_inputs", empty_text=empty_text)
        self.output_chart = plot_widget("Prediction", name="step_response", empty_text=empty_text)
        legend(self.inputs_chart)
        legend(self.output_chart)
        self.inputs_chart.setXLink(self.output_chart)
        self.inputs_chart.setLabel("left", "% of range")
        self.output_chart.setLabel("bottom", "Minutes from start")
        charts = QSplitter(Qt.Orientation.Vertical)
        charts.addWidget(self.inputs_chart)
        charts.addWidget(self.output_chart)
        charts.setSizes([1, 2])
        charts.setChildrenCollapsible(False)

        self.step_at = QDoubleSpinBox()
        self.step_at.setRange(0, 7 * 24 * 60)
        self.step_at.setSuffix(" min")
        self.step_at.setDecimals(0)
        self.step_at.setSingleStep(5)
        self.length = QDoubleSpinBox()
        self.length.setRange(0.5, 7 * 24)
        self.length.setSuffix(" h")
        self.length.setDecimals(1)
        self.length.setSingleStep(0.5)
        for spin in (self.step_at, self.length):
            spin.valueChanged.connect(self._schedule)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)

        self._list = QWidget()
        self._list_layout = QVBoxLayout(self._list)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(4)
        self._list_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._list)

        title = QLabel("Step test ⓘ")
        title.setObjectName("sectionTitle")
        title.setToolTip(
            "Every input starts at steady state (its training mean). At the step time the\n"
            "inputs you set change by their step and stay there. They pass through the model's\n"
            "own features — deadtime, filter, averaging window — so the prediction responds as\n"
            "the soft sensor would online: flat during the deadtime, then a ramp over the window\n"
            "(an EWMA adds a lag), settling at coefficient × step.\n\n"
            "The shape comes from the feature design, not from the real process dynamics."
        )
        head = QHBoxLayout()
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.reset_button)
        timing = QHBoxLayout()
        timing.setContentsMargins(0, 0, 0, 0)
        timing.addWidget(QLabel("Step at"))
        timing.addWidget(self.step_at, 1)
        timing.addSpacing(6)
        timing.addWidget(QLabel("Run"))
        timing.addWidget(self.length, 1)
        panel = QWidget()
        panel.setMinimumWidth(260)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addLayout(head)
        pl.addLayout(timing)
        pl.addWidget(self.summary)
        pl.addWidget(scroll, 1)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(charts)
        split.addWidget(panel)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([900, PANEL_WIDTH])
        split.setChildrenCollapsible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)
        self._enable(False)

    # -- public -----------------------------------------------------------------------------

    def set_model(
        self,
        run: TrainingRun | None,
        model: ModelResult | None,
        *,
        units: dict[str, str | None] | None = None,
    ) -> None:
        """Show ``model`` (None clears the view). Step sizes and timing are kept when the next
        model has the same inputs (switching between models of one run)."""
        new = step_model(run, model, units) if run is not None and model and model.ok else None
        same = (
            new is not None
            and self._model is not None
            and [i.name for i in new.inputs] == [i.name for i in self._model.inputs]
        )
        keep = {c.inp.name: c.step() for c in self._controls} if same else {}
        self._model = new
        self._target_units = (units or {}).get(new.target) if new else None
        self._rebuild_controls(keep)
        if new is None:
            self._enable(False)
            self.summary.setText("")
            for chart in (self.inputs_chart, self.output_chart):
                chart.clear()
            return
        if not same:
            step_at, length = default_timing(new)
            for spin, value in ((self.step_at, step_at / 60), (self.length, length / 3600)):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self._enable(True)
        self._update()

    def reset(self) -> None:
        for c in self._controls:
            c.reset()
        self._update()

    def steps(self) -> dict[str, float]:
        return {c.inp.name: c.step() for c in self._controls if c.step()}

    def apply_theme(self) -> None:
        self._update()

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().resizeEvent(event)
        # On short screens two stacked charts are unreadable; the step sizes stay in the panel.
        self.inputs_chart.setVisible(self.height() >= COMPACT_HEIGHT)

    # -- internals --------------------------------------------------------------------------

    def _enable(self, enabled: bool) -> None:
        for w in (self.step_at, self.length, self.reset_button):
            w.setEnabled(enabled)

    def _rebuild_controls(self, keep: dict[str, float]) -> None:
        for c in self._controls:
            c.setParent(None)
            c.deleteLater()
        self._controls = []
        if self._model is None:
            return
        # Largest possible effect (|b| × training range) first: the inputs that matter most.
        order = sorted(self._model.inputs, key=lambda i: -abs(i.coefficient) * i.span)
        colour = {inp.name: SERIES[k % len(SERIES)] for k, inp in enumerate(order)}
        for inp in order:
            control = _InputControl(inp, colour[inp.name], self._target_units)
            if keep.get(inp.name):
                control.set_step(keep[inp.name])
            control.changed.connect(self._schedule)
            self._list_layout.insertWidget(self._list_layout.count() - 1, control)
            self._controls.append(control)

    def _schedule(self) -> None:
        self._timer.start()

    def _update(self) -> None:
        model = self._model
        if model is None:
            return
        length = self.length.value() * 3600
        step_at = min(self.step_at.value() * 60, length)
        steps = self.steps()
        res = simulate(model, steps, step_at, length)
        t = res.t_s / 60.0

        self.inputs_chart.clear()
        if steps:
            clear_placeholder(self.inputs_chart)
            ts = step_at / 60
            for c in self._controls:
                if c.inp.name in steps:  # an exact step: vertical at the step time
                    pct = 100.0 * steps[c.inp.name] / c.inp.span
                    self.inputs_chart.plot([t[0], ts, ts, t[-1]], [0.0, 0.0, pct, pct],
                                           pen=pg.mkPen(c.colour, width=2), name=c.inp.name)  # fmt: skip
        else:
            show_placeholder(self.inputs_chart, "Move a slider to step an input.")
        self.output_chart.clear()
        self.output_chart.plot(
            [t[0], t[-1]], [res.baseline, res.baseline],
            pen=pg.mkPen(GREY, style=Qt.PenStyle.DashLine), name="Steady state",
        )  # fmt: skip
        self.output_chart.plot(t, res.predicted, pen=pg.mkPen(BLUE, width=2), name="Prediction")
        for chart in (self.inputs_chart, self.output_chart):
            chart.addItem(
                pg.InfiniteLine(step_at / 60, angle=90,
                                pen=pg.mkPen(GREY, style=Qt.PenStyle.DotLine)),
                ignoreBounds=True,
            )  # fmt: skip
        units = self._target_units
        target = f"{model.target} [{units}]" if units else model.target
        self.output_chart.setTitle(f"Prediction — {target}")
        self._summary(res, units)

    def _summary(self, res, units: str | None) -> None:  # noqa: ANN001
        if not res.final_change:
            self.summary.setText(
                f"Steady state: <b>{_fmt(res.baseline, units)}</b>. Move a slider to step an input."
            )
            return
        parts = [
            f"Output settles <b>{_fmt(res.final_change, units, signed=True)}</b> from "
            f"{_fmt(res.baseline, units)}.",
            f"Responds after {_minutes(res.first_response_s)}; 95 % at "
            f"{_minutes(res.settled_s) if res.settled_s is not None else 'beyond the run'}.",
        ]
        text = " ".join(parts)
        warn = []
        if res.outside:
            warn.append(
                f"Stepped outside the training range ({', '.join(res.outside)}): extrapolation."
            )
        warn.extend(res.notes)
        if warn:
            colour = TOKENS["warning"]
            text += "".join(f"<br><span style='color:{colour}'>{escape(w)}</span>" for w in warn)
        self.summary.setText(text)
        self.summary.setToolTip("")
        if np.isnan(res.predicted).any():
            self.summary.setToolTip("Gaps in the line: a feature failed its coverage checks.")
