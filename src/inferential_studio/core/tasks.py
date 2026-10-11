"""Cancellation and progress primitives for long-running core functions.

Core functions that may take a while accept two optional keyword arguments:

    progress: ProgressCallback | None   called as progress(fraction_0_to_1, message)
    cancel:   CancelToken | None        checked periodically via cancel.raise_if_cancelled()

They stay free of Qt. The desktop app's background worker supplies implementations that
forward progress to the GUI through Qt signals; scripts and notebooks can pass nothing.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

ProgressCallback = Callable[[float, str], None]


class TaskCancelled(Exception):
    """Raised inside a task when the user has requested cancellation."""


class CancelToken:
    """Thread-safe, one-way cancellation flag."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise TaskCancelled()


def report(progress: ProgressCallback | None, fraction: float, message: str = "") -> None:
    """Call ``progress`` if supplied, clamping the fraction to [0, 1]."""
    if progress is not None:
        progress(min(max(float(fraction), 0.0), 1.0), message)


def check(cancel: CancelToken | None) -> None:
    """Raise :class:`TaskCancelled` if ``cancel`` has been triggered."""
    if cancel is not None:
        cancel.raise_if_cancelled()
