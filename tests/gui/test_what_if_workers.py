from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QTimer

from inferential_studio.core.schemas import Aggregation, FeatureSpec
from inferential_studio.ui.widgets import what_if
from inferential_studio.ui.widgets.what_if import DISPLAY_POINTS, WhatIfView

pytestmark = pytest.mark.gui


def make_view(qtbot, specs=None):
    specs = specs or [FeatureSpec("x", window_s=600)]
    names = [s.variable for s in specs]
    run = SimpleNamespace(
        target="y", feature_specs={s.variable: s for s in specs},
        train_ranges=pd.DataFrame({"min": 0., "max": 20., "mean": 10.}, index=names),
    )
    model = SimpleNamespace(ok=True, intercept=1., coefficients=pd.Series(2., index=names))
    view = WhatIfView()
    qtbot.addWidget(view)
    view.set_model(run, model)
    view.pause_button.setChecked(True)
    return view, run, model


def ready(qtbot, view):
    qtbot.waitUntil(lambda: not view._initializing and view._pending is None, timeout=10000)


def test_hidden_view_defers_computation_and_theme_only_redraws(qtbot, monkeypatch):
    calls = []
    original = what_if._response_task

    def measured(*args, **kwargs):
        calls.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(what_if, "_response_task", measured)
    view, run, model = make_view(qtbot)
    view.set_model(run, model)
    view.apply_theme()
    qtbot.wait(30)
    assert calls == [] and view._pending is None
    view.show()
    ready(qtbot, view)
    assert calls and all(t != threading.get_ident() for t in calls)
    count = len(calls)
    view.apply_theme()
    assert len(calls) == count
    view.pause_button.setChecked(False)
    view.hide()
    assert not view._timer.isActive()
    qtbot.wait(100)
    assert len(calls) == count


def test_slider_changes_coalesce_and_event_loop_stays_live(qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    original = what_if._response_task

    def held(*args, **kwargs):
        calls.append(threading.get_ident())
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(what_if, "_response_task", held)
    view, _run, _model = make_view(qtbot)
    heartbeat = []
    try:
        view.show()
        qtbot.waitUntil(entered.is_set)
        for value in range(701, 751):
            view._controls[0].slider.setValue(value)
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        qtbot.waitUntil(lambda: bool(heartbeat))
        assert len(calls) == 1 and len(view._tasks.active) == 1
    finally:
        release.set()
    ready(qtbot, view)
    assert len(calls) == 2
    assert view._target == pytest.approx(41.)  # latest +10 input change, not a stale result
    assert calls[0] != threading.get_ident()


def test_long_trend_keeps_display_bounded_without_coarsening_features(qtbot, monkeypatch):
    grids = []

    def response(model, schedule, times, dt, **kwargs):
        grids.append((len(times), dt))
        return np.full(len(times), 41.)

    monkeypatch.setattr(what_if, "respond", response)
    monkeypatch.setattr(what_if, "steady_state", lambda *_args, **_kwargs: 41.)
    view, _run, _model = make_view(qtbot, [FeatureSpec("x", window_s=6),
                                         FeatureSpec("z", window_s=604800)])
    view.show()
    ready(qtbot, view)
    assert len(view._t) <= DISPLAY_POINTS + 1 and view._dt == 1.
    assert view._display_dt > view._dt
    view._clock = 10 * view._view  # a late worker must not calculate invisible backlog
    view._request_response()
    ready(qtbot, view)
    assert len(view._t) <= DISPLAY_POINTS + 2
    assert all(n <= DISPLAY_POINTS + 2 and dt == 1. for n, dt in grids)


def test_std_live_summary_matches_the_transient_and_final_prediction(qtbot):
    view, _run, _model = make_view(qtbot, [FeatureSpec("x", window_s=600,
                                                     aggregation=Aggregation.STD)])
    view.show()
    ready(qtbot, view)
    assert view._baseline == pytest.approx(1.)
    view._controls[0].slider.setValue(750)
    qtbot.waitUntil(lambda: view._pending is None)
    assert view._target == pytest.approx(1.)
    assert view._controls[0].effect.text() == ""  # no false coefficient × raw step claim
    view._clock = 300.
    view._request_response()
    ready(qtbot, view)
    assert view._y[-1] > 1.
    assert "settling at <b>1</b>" in view.summary.text()
    assert "Standard-deviation features are 0 at steady state" in view.summary.text()
