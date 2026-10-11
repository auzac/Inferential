"""Date axis that labels ticks in an explicit IANA timezone (not the PC's local time).

Plot x values are always UTC epoch seconds. Tick positions follow the timezone's offset at the
middle of the visible range; each label is converted with the offset valid at that instant, so
labels stay correct across daylight-saving changes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyqtgraph as pg


def to_epoch_seconds(index: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """UTC timestamps as float epoch seconds, the x unit of every trend plot."""
    values = pd.DatetimeIndex(index)
    if values.tz is None:
        raise ValueError("Timestamps must be timezone-aware (UTC).")
    return values.tz_convert("UTC").as_unit("ns").asi8 / 1e9


class TzDateAxisItem(pg.DateAxisItem):
    def __init__(self, timezone: str = "UTC", orientation: str = "bottom", **kwargs) -> None:
        super().__init__(orientation=orientation, **kwargs)
        self._tz = ZoneInfo(timezone)
        self.utcOffset = 0

    def set_timezone(self, timezone: str) -> None:
        self._tz = ZoneInfo(timezone)
        self.picture = None
        self.update()

    def _offset_at(self, epoch_s: float) -> float:
        try:
            moment = datetime.fromtimestamp(float(epoch_s), UTC)
        except (OverflowError, OSError, ValueError):
            return 0.0
        offset = moment.astimezone(self._tz).utcoffset()
        return -offset.total_seconds() if offset is not None else 0.0

    def tickValues(self, minVal, maxVal, size):  # noqa: N802 - Qt/pyqtgraph override
        self.utcOffset = self._offset_at((minVal + maxVal) / 2.0)
        return super().tickValues(minVal, maxVal, size)

    def tickStrings(self, values, scale, spacing):  # noqa: N802
        labels = []
        keep = self.utcOffset
        for v in values:
            self.utcOffset = self._offset_at(v)
            labels.extend(super().tickStrings([v], scale, spacing))
        self.utcOffset = keep
        return labels


def format_time(epoch_s: float, timezone: str, with_seconds: bool = True) -> str:
    fmt = "%Y-%m-%d %H:%M:%S" if with_seconds else "%Y-%m-%d %H:%M"
    return datetime.fromtimestamp(epoch_s, ZoneInfo(timezone)).strftime(fmt)
