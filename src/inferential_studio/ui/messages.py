"""One place for every info / warning / error message the application shows.

Pages never put status text into their own layout (it would push the charts and tables around).
They report to the :class:`MessageCenter`, which the fixed message panel at the bottom of the
main window displays.

Two kinds of message:

* **conditions** — a state that holds until it changes, identified by a key, e.g.
  ``"processing.noncausal"``: "2 features read future data". ``set()`` replaces it, ``clear()``
  removes it. Shown pinned at the top while active.
* **events** — something that happened ("Saved project.imsproj", "Training failed: …"), kept in
  a history (newest first) until the user clears it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum

from PySide6.QtCore import QObject, Signal

MAX_EVENTS = 300
APP = "app"  # source for messages that do not belong to one stage


class Level(IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2


@dataclass
class Message:
    level: Level
    text: str
    source: str = APP  # stage key or APP
    details: str = ""
    key: str | None = None  # conditions only
    time: datetime = field(default_factory=datetime.now)
    also: tuple[str, ...] = ()  # further stages where the message is relevant

    def visible_in(self, stage: str | None) -> bool:
        return stage is None or self.source in (stage, APP) or stage in self.also

    @property
    def is_condition(self) -> bool:
        return self.key is not None


class MessageCenter(QObject):
    changed = Signal()
    posted = Signal(object)  # Message: a new event (the panel flashes for warnings/errors)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._conditions: dict[str, Message] = {}
        self._events: list[Message] = []

    # -- events ------------------------------------------------------------------

    def post(self, level: Level, text: str, source: str = APP, details: str = "") -> Message:
        msg = Message(Level(level), text, source, details)
        self._events.insert(0, msg)
        del self._events[MAX_EVENTS:]
        self.posted.emit(msg)
        self.changed.emit()
        return msg

    def info(self, text: str, source: str = APP) -> Message:
        return self.post(Level.INFO, text, source)

    def warning(self, text: str, source: str = APP, details: str = "") -> Message:
        return self.post(Level.WARNING, text, source, details)

    def error(self, text: str, source: str = APP, details: str = "") -> Message:
        return self.post(Level.ERROR, text, source, details)

    def clear_events(self) -> None:
        if self._events:
            self._events.clear()
            self.changed.emit()

    @property
    def events(self) -> list[Message]:
        return list(self._events)

    # -- conditions ----------------------------------------------------------------

    def set(
        self,
        key: str,
        level: Level,
        text: str,
        source: str | None = None,
        details: str = "",
        also: tuple[str, ...] = (),
    ) -> None:
        """Show (or update) a condition. An empty ``text`` clears it.

        ``source`` defaults to the key's prefix (``"processing.noncausal"`` → processing);
        ``also`` lists further stages where it should be shown.
        """
        if not text:
            self.clear(key)
            return
        source = source or key.split(".", 1)[0]
        new = Message(Level(level), text, source, details, key, also=tuple(also))
        old = self._conditions.get(key)
        if old is not None and (old.level, old.text, old.source, old.details, old.also) == (
            new.level,
            new.text,
            new.source,
            new.details,
            new.also,
        ):
            return
        self._conditions[key] = new
        self.changed.emit()

    def clear(self, key: str) -> None:
        if self._conditions.pop(key, None) is not None:
            self.changed.emit()

    def clear_prefix(self, prefix: str) -> None:
        keys = [k for k in self._conditions if k.startswith(prefix)]
        for k in keys:
            del self._conditions[k]
        if keys:
            self.changed.emit()

    def condition(self, key: str) -> Message | None:
        return self._conditions.get(key)

    def conditions(self, source: str | None = None) -> list[Message]:
        """Active conditions, most severe first (optionally for one stage plus app-wide ones)."""
        items = [m for m in self._conditions.values() if m.visible_in(source)]
        return sorted(items, key=lambda m: (-m.level, m.key or ""))

    def counts(self, source: str | None = None) -> dict[Level, int]:
        out = dict.fromkeys(Level, 0)
        for m in self.conditions(source):
            out[m.level] += 1
        for m in self._events:
            if m.visible_in(source):
                out[m.level] += 1
        return out

    def texts(self, source: str | None = None) -> str:
        """All visible text (handy for tests and copy-to-clipboard)."""
        rows = self.conditions(source) + [m for m in self._events if m.visible_in(source)]
        return "\n".join(m.text for m in rows)
