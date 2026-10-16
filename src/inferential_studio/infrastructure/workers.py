"""Background execution of core functions without freezing the GUI.

Pattern:

    manager = TaskManager()
    manager.submit("Generate example", generate_fn, arg, on_result=show, on_error=report)

* The task function is a pure core function taking keyword arguments ``progress`` and
  ``cancel`` (see :mod:`inferential_studio.core.tasks`). It never touches widgets.
* Results, errors, progress and cancellation travel back as Qt signals. They are received by
  slots of :class:`TaskManager`, which lives in the GUI thread, so Qt delivers them through a
  queued connection and every callback runs on the GUI thread.
* Cancellation is cooperative: ``cancel()`` sets a flag that the core function checks.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from inferential_studio.core.persistence import ProjectFileError
from inferential_studio.core.schemas import SchemaError
from inferential_studio.core.tasks import CancelToken, TaskCancelled

log = logging.getLogger(__name__)

PROGRESS_MIN_INTERVAL_S = 0.05  # throttle progress signals so a chatty task cannot flood the GUI

USER_FACING_ERRORS: tuple[type[BaseException], ...] = (
    ProjectFileError,
    SchemaError,
    ValueError,
    OSError,
)


def friendly_message(exc: BaseException) -> str:
    """Message suitable for an error dialog. Unexpected errors point the user to the log."""
    if isinstance(exc, USER_FACING_ERRORS) and str(exc):
        return str(exc)
    return (
        f"An unexpected error occurred ({type(exc).__name__}: {exc}). "
        "Details have been written to the application log."
    )


class _TaskSignals(QObject):
    progress = Signal(int, float, str)
    succeeded = Signal(int, object)
    failed = Signal(int, str, str)
    cancelled = Signal(int)


class _Runnable(QRunnable):
    def __init__(
        self, task_id: int, fn: Callable[..., Any], args: tuple, kwargs: dict, token: CancelToken,
        discard_cancelled_result: bool = True,
    ):
        super().__init__()
        self.setAutoDelete(False)  # the manager owns the lifetime
        self.task_id = task_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.token = token
        self.discard_cancelled_result = discard_cancelled_result
        self.signals = _TaskSignals()
        self._last_progress = 0.0

    def _progress(self, fraction: float, message: str) -> None:
        now = time.monotonic()
        if fraction >= 1.0 or now - self._last_progress >= PROGRESS_MIN_INTERVAL_S:
            self._last_progress = now
            self.signals.progress.emit(self.task_id, float(fraction), message)

    def run(self) -> None:  # executes on a pool thread
        threading.current_thread().name = f"task-{self.task_id}"
        try:
            result = self.fn(*self.args, progress=self._progress, cancel=self.token, **self.kwargs)
        except TaskCancelled:
            self.signals.cancelled.emit(self.task_id)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the GUI
            log.exception("Background task %s failed", self.task_id)
            self.signals.failed.emit(self.task_id, friendly_message(exc), traceback.format_exc())
        else:
            if self.token.cancelled and self.discard_cancelled_result:
                self.signals.cancelled.emit(self.task_id)
            else:
                self.signals.succeeded.emit(self.task_id, result)


@dataclass
class TaskHandle:
    task_id: int
    name: str
    token: CancelToken
    on_result: Callable[[Any], None] | None = None
    on_error: Callable[[str, str], None] | None = None
    on_cancelled: Callable[[], None] | None = None
    started_at: float = field(default_factory=time.monotonic)

    def cancel(self) -> None:
        self.token.cancel()

    @property
    def cancel_requested(self) -> bool:
        return self.token.cancelled


class TaskManager(QObject):
    """Runs core functions on a thread pool and reports back on the GUI thread."""

    task_started = Signal(int, str)
    task_progress = Signal(int, str, float, str)  # id, name, fraction, message
    task_ended = Signal(int, str, str)  # id, name, outcome: "succeeded" | "failed" | "cancelled"
    busy_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None, max_threads: int = 2) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max_threads)
        self._ids = itertools.count(1)
        self._handles: dict[int, TaskHandle] = {}
        self._runnables: dict[int, _Runnable] = {}

    # -- public API ------------------------------------------------------------

    def submit(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None] | None = None,
        on_error: Callable[[str, str], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
        discard_cancelled_result: bool = True,
        **kwargs: Any,
    ) -> TaskHandle:
        """Atomic writers can keep a committed result despite a late cancellation request.

        With discard_cancelled_result=False the function must check cancellation itself,
        and raise TaskCancelled before committing. A successful return means it committed.
        """
        task_id = next(self._ids)
        token = CancelToken()
        handle = TaskHandle(task_id, name, token, on_result, on_error, on_cancelled)
        runnable = _Runnable(task_id, fn, args, kwargs, token, discard_cancelled_result)
        runnable.signals.progress.connect(self._on_progress)
        runnable.signals.succeeded.connect(self._on_succeeded)
        runnable.signals.failed.connect(self._on_failed)
        runnable.signals.cancelled.connect(self._on_cancelled)

        was_busy = self.busy
        self._handles[task_id] = handle
        self._runnables[task_id] = runnable
        log.info("Task %d started: %s", task_id, name)
        self.task_started.emit(task_id, name)
        if not was_busy:
            self.busy_changed.emit(True)
        self._pool.start(runnable)
        return handle

    @property
    def busy(self) -> bool:
        return bool(self._handles)

    @property
    def active(self) -> list[TaskHandle]:
        return list(self._handles.values())

    def cancel(self, task_id: int) -> None:
        handle = self._handles.get(task_id)
        if handle is not None:
            log.info("Task %d cancel requested", task_id)
            handle.cancel()

    def cancel_all(self) -> None:
        for task_id in list(self._handles):
            self.cancel(task_id)

    def wait_for_done(self, msecs: int = 5000) -> bool:
        """Block until pool threads finish (used at shutdown). Returns False on timeout."""
        return self._pool.waitForDone(msecs)

    # -- slots (always run on this object's thread, i.e. the GUI thread) -----

    @Slot(int, float, str)
    def _on_progress(self, task_id: int, fraction: float, message: str) -> None:
        handle = self._handles.get(task_id)
        if handle is not None:
            self.task_progress.emit(task_id, handle.name, fraction, message)

    @Slot(int, object)
    def _on_succeeded(self, task_id: int, result: Any) -> None:
        handle = self._finish(task_id, "succeeded")
        if handle and handle.on_result:
            handle.on_result(result)

    @Slot(int, str, str)
    def _on_failed(self, task_id: int, message: str, details: str) -> None:
        handle = self._finish(task_id, "failed")
        if handle and handle.on_error:
            handle.on_error(message, details)

    @Slot(int)
    def _on_cancelled(self, task_id: int) -> None:
        handle = self._finish(task_id, "cancelled")
        if handle and handle.on_cancelled:
            handle.on_cancelled()

    def _finish(self, task_id: int, outcome: str) -> TaskHandle | None:
        handle = self._handles.pop(task_id, None)
        self._runnables.pop(task_id, None)
        if handle is None:
            return None
        log.info(
            "Task %d %s after %.2f s: %s",
            task_id,
            outcome,
            time.monotonic() - handle.started_at,
            handle.name,
        )
        self.task_ended.emit(task_id, handle.name, outcome)
        if not self._handles:
            self.busy_changed.emit(False)
        return handle
