from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtCore import QTimer

from inferential_studio.core.tasks import check, report
from inferential_studio.infrastructure.workers import TaskManager, friendly_message

pytestmark = pytest.mark.gui


def slow_sum(n, delay=0.01, *, progress=None, cancel=None):
    total = 0
    for i in range(n):
        check(cancel)
        time.sleep(delay)
        total += i
        report(progress, (i + 1) / n, f"step {i + 1}")
    return {"total": total, "thread": threading.get_ident()}


def failing(*, progress=None, cancel=None):
    raise ValueError("Ambiguous timestamp format: choose day-first or month-first.")


def crashing(*, progress=None, cancel=None):
    raise KeyError("internal")


@pytest.fixture
def manager(qtbot):
    m = TaskManager()
    yield m
    m.cancel_all()
    m.wait_for_done(5000)


def test_result_is_delivered_on_gui_thread(qtbot, manager):
    results, callback_threads, progress = [], [], []
    manager.task_progress.connect(lambda _id, _n, f, _m: progress.append(f))
    with qtbot.waitSignal(manager.busy_changed, timeout=5000) as busy:
        handle = manager.submit(
            "sum",
            slow_sum,
            20,
            on_result=lambda r: (results.append(r), callback_threads.append(threading.get_ident())),
        )
    assert busy.args == [True]
    qtbot.waitUntil(lambda: bool(results), timeout=10000)
    assert results[0]["total"] == sum(range(20))
    assert results[0]["thread"] != threading.get_ident()  # computed in the background
    assert callback_threads == [threading.get_ident()]  # delivered on the GUI thread
    assert progress and progress[-1] == 1.0
    assert not manager.busy and handle.task_id == 1


def test_gui_stays_responsive_while_task_runs(qtbot, manager):
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start(20)
    done = []
    manager.submit("slow", slow_sum, 30, 0.02, on_result=done.append)
    qtbot.waitUntil(lambda: bool(done), timeout=10000)
    timer.stop()
    assert len(ticks) >= 10  # the event loop kept running during ~0.6 s of work


def test_cancellation(qtbot, manager):
    cancelled, results = [], []
    handle = manager.submit(
        "long",
        slow_sum,
        1000,
        0.01,
        on_result=results.append,
        on_cancelled=lambda: cancelled.append(True),
    )
    qtbot.wait(50)
    with qtbot.waitSignal(manager.task_ended, timeout=5000) as ended:
        handle.cancel()
    assert ended.args[2] == "cancelled"
    assert cancelled == [True] and results == []
    assert not manager.busy


def test_user_facing_error(qtbot, manager):
    errors = []
    manager.submit("bad", failing, on_error=lambda msg, details: errors.append((msg, details)))
    qtbot.waitUntil(lambda: bool(errors), timeout=5000)
    message, details = errors[0]
    assert message.startswith("Ambiguous timestamp")
    assert "ValueError" in details


def test_unexpected_error_points_to_log(qtbot, manager):
    errors = []
    manager.submit("crash", crashing, on_error=lambda msg, details: errors.append(msg))
    qtbot.waitUntil(lambda: bool(errors), timeout=5000)
    assert "unexpected error" in errors[0] and "log" in errors[0]


def test_friendly_message():
    assert friendly_message(ValueError("Too few lab samples.")) == "Too few lab samples."
    assert "unexpected" in friendly_message(ZeroDivisionError("x"))
