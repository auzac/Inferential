"""What-if: a live trend of the selected model, like a soft sensor running online.

A clock runs (sped up) from steady state. Moving an input's slider changes that input *now*;
the prediction then plays out through the model's own features: flat during the deadtime, a
ramp over the window, then settling. Top chart: the inputs (change as % of their training
range, so different units share an axis). Bottom chart: the prediction, its starting steady
state and where it is heading. The calculation is in ``core/what_if.py``.
"""

from __future__ import annotations

from html import escape

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
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
from inferential_studio.core.schemas import Aggregation
from inferential_studio.core.what_if import (
    StepInput,
    StepModel,
    default_timing,
    grid_step,
    input_values,
    notes,
    respond,
    steady_state,
    step_model,
)
from inferential_studio.ui.theme import TOKENS
from inferential_studio.ui.widgets.plots import (
    BLUE,
    GREY,
    SERIES,
    legend,
    plot_widget,
)

STEPS = 1000  # slider resolution (0 … STEPS = −span … +span)
PANEL_WIDTH = 340
TICK_MS = 50  # clock tick: the trend scrolls smoothly; new points come once per grid step
SCROLL_S = 30.0  # real seconds for the clock to cross the whole trend
COMPACT_HEIGHT = 320  # below this the inputs chart hides; the prediction gets the room


def _fmt(value: float, units: str | None = None, *, signed: bool = False) -> str:
    text = (f"{value:+.4g}" if signed else f"{value:.4g}").replace("-", "−")
    return f"{text} {units}" if units else text


def _minutes(seconds: float) -> str:
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
        std = self.inp.spec is not None and self.inp.spec.aggregation is Aggregation.STD
        if step and not std:  # a std feature returns to 0 once the step has passed
            effect = self.inp.coefficient * step
            self.effect.setText(f"Δŷ {_fmt(effect, self._target_units, signed=True)}")
        else:
            self.effect.setText("")


class WhatIfView(QWidget):
    """Live trend of the selected model: move a slider and watch the prediction respond."""

    def __init__(self, empty_text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model: StepModel | None = None
        self._controls: list[_InputControl] = []
        self._target_units: str | None = None
        self._dt = 60.0  # grid step of the model (s)
        self._view = 7200.0  # trend width (s)
        self._clock = 0.0  # simulated seconds since the start
        self._schedule: list[tuple[float, dict[str, float]]] = []
        self._t = np.empty(0)  # trend times (s) and predictions
        self._y = np.empty(0)
        self._baseline = self._target = float("nan")
        self._timer = QTimer(self, interval=TICK_MS)
        self._timer.timeout.connect(self._tick)

        self.inputs_chart = plot_widget("Inputs", name="step_inputs", empty_text=empty_text)
        self.output_chart = plot_widget("Prediction", name="step_response", empty_text=empty_text)
        legend(self.inputs_chart)
        legend(self.output_chart)
        self.inputs_chart.setXLink(self.output_chart)
        self.inputs_chart.setLabel("left", "% of range")
        self.output_chart.setLabel("bottom", "Minutes")
        charts = QSplitter(Qt.Orientation.Vertical)
        charts.addWidget(self.inputs_chart)
        charts.addWidget(self.output_chart)
        charts.setSizes([1, 2])
        charts.setChildrenCollapsible(False)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._pause)
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

        title = QLabel("Live trend ⓘ")
        title.setObjectName("sectionTitle")
        title.setToolTip(
            "The model runs live (sped up) from steady state: every input at its training mean.\n"
            "Move a slider and that input changes now. It passes through the model's own\n"
            "features — deadtime, filter, averaging window — so the prediction responds as the\n"
            "soft sensor would online: flat during the deadtime, then a ramp over the window\n"
            "(an EWMA adds a lag), settling at coefficient × change.\n\n"
            "The shape comes from the feature design, not from the real process dynamics."
        )
        head = QHBoxLayout()
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.pause_button)
        head.addWidget(self.reset_button)
        panel = QWidget()
        panel.setMinimumWidth(260)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addLayout(head)
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
        """Show ``model`` (None clears the view). The input changes are kept when the next model
        has the same inputs (switching between models of one run); it starts settled at them."""
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
            self._timer.stop()
            self._enable(False)
            self.summary.setText("")
            for chart in (self.inputs_chart, self.output_chart):
                chart.clear()
            return
        self._dt = grid_step(new)
        self._view = default_timing(new)[1]
        self._enable(True)
        self._restart(self.steps())

    def reset(self) -> None:
        for c in self._controls:
            c.reset()
        if self._model is not None:
            self._restart({})

    def steps(self) -> dict[str, float]:
        return {c.inp.name: c.step() for c in self._controls if c.step()}

    def apply_theme(self) -> None:
        self._draw()

    def showEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().showEvent(event)
        self._pause(self.pause_button.isChecked())

    def hideEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().hideEvent(event)
        self._timer.stop()  # no clock while nobody watches

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().resizeEvent(event)
        # On short screens two stacked charts are unreadable; the step sizes stay in the panel.
        self.inputs_chart.setVisible(self.height() >= COMPACT_HEIGHT)

    # -- internals --------------------------------------------------------------------------

    def _enable(self, enabled: bool) -> None:
        for w in (self.pause_button, self.reset_button):
            w.setEnabled(enabled)

    def _pause(self, paused: bool) -> None:
        self.pause_button.setText("Run" if paused else "Pause")
        if paused or self._model is None or not self.isVisible():
            self._timer.stop()
        else:
            self._timer.start()

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
            control.changed.connect(self._moved)
            self._list_layout.insertWidget(self._list_layout.count() - 1, control)
            self._controls.append(control)

    def _restart(self, steps: dict[str, float]) -> None:
        """Clock to 0, settled at ``steps`` (a trend's worth of steady history on screen)."""
        model = self._model
        self._clock = 0.0
        self._schedule = [(-np.inf, steps)]
        self._t = np.arange(-self._view, self._dt / 2, self._dt)
        self._y = respond(model, self._schedule, self._t, self._dt)
        self._baseline = steady_state(model, {}, self._dt)
        self._target = steady_state(model, steps, self._dt)
        self._draw()
        self._pause(self.pause_button.isChecked())

    def _moved(self) -> None:
        """A slider moved: the inputs change at the next grid time and stay there."""
        if self._model is None:
            return
        steps = self.steps()
        at = self._t[-1] + self._dt
        if self._schedule[-1][0] == at:  # still dragging within one grid step
            self._schedule[-1] = (at, steps)
        else:
            # shortcut: the schedule keeps every change (≤ one per grid step while dragging);
            # trim entries older than the trend's lookback if sessions ever run for hours.
            self._schedule.append((at, steps))
        self._target = steady_state(self._model, steps, self._dt)
        self._draw()

    def _tick(self) -> None:
        self._clock += self._view / SCROLL_S * TICK_MS / 1000.0
        new = np.arange(self._t[-1] + self._dt, self._clock + self._dt / 2, self._dt)
        if new.size:
            y = respond(self._model, self._schedule, new, self._dt)
            keep = self._t > self._clock - self._view - self._dt
            self._t = np.concatenate([self._t[keep], new])
            self._y = np.concatenate([self._y[keep], y])
            self._draw()
        self.output_chart.setXRange((self._clock - self._view) / 60, self._clock / 60, padding=0)

    def _draw(self) -> None:
        model = self._model
        if model is None:
            return
        t = self._t / 60.0
        values = input_values(model, self._schedule, self._t)
        self.inputs_chart.clear()
        for c in self._controls:
            pct = 100.0 * (values[c.inp.name] - c.inp.baseline) / (c.inp.span or 1.0)
            if pct.any():  # only the inputs that moved within the trend
                self.inputs_chart.plot(t, pct, pen=pg.mkPen(c.colour, width=2), name=c.inp.name)
        self.output_chart.clear()
        span = [t[0], t[-1] + self._view / 60]  # reference lines reach past the live edge
        self.output_chart.plot(
            span, [self._baseline] * 2,
            pen=pg.mkPen(GREY, style=Qt.PenStyle.DashLine), name="Steady state",
        )  # fmt: skip
        if self.steps():
            self.output_chart.plot(
                span, [self._target] * 2,
                pen=pg.mkPen(BLUE, style=Qt.PenStyle.DotLine), name="Settling at",
            )  # fmt: skip
        self.output_chart.plot(t, self._y, pen=pg.mkPen(BLUE, width=2), name="Prediction")
        self.output_chart.setXRange((self._clock - self._view) / 60, self._clock / 60, padding=0)
        units = self._target_units
        target = f"{model.target} [{units}]" if units else model.target
        self.output_chart.setTitle(f"Prediction — {target}")
        self._summary(units)

    def _summary(self, units: str | None) -> None:
        now = self._y[-1]
        text = f"Now <b>{_fmt(now, units)}</b>" if np.isfinite(now) else "Now <b>—</b>"
        steps = self.steps()
        if not steps:
            text += (
                f"; steady state {_fmt(self._baseline, units)}. Move a slider to change an input."
            )
        else:
            change = self._target - self._baseline
            text += (
                f", settling at <b>{_fmt(self._target, units)}</b> "
                f"({_fmt(change, units, signed=True)} from steady state "
                f"{_fmt(self._baseline, units)})."
            )
        warn = [
            f"{i.name} is outside its training range: extrapolation."
            for i in self._model.inputs
            if steps.get(i.name) and not i.low <= i.baseline + steps[i.name] <= i.high
        ]
        warn.extend(notes(self._model))
        if not np.isfinite(self._y).all():
            warn.append("Gaps in the line: a feature failed its coverage checks.")
        colour = TOKENS["warning"]
        text += "".join(f"<br><span style='color:{colour}'>{escape(w)}</span>" for w in warn)
        self.summary.setText(text)
