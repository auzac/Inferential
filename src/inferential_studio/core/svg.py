"""Minimal, dependency-free SVG charts for self-contained HTML reports (no Qt, no JavaScript)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd

FONT = "font-family:Segoe UI,Arial,sans-serif;font-size:11px;fill:#333"
GRID = "#e6e6e6"


@dataclass
class Series:
    x: np.ndarray
    y: np.ndarray
    colour: str
    label: str = ""
    kind: str = "points"  # "points" or "line"


def nice_ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return []
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-9:
        ticks.append(round(v, 12))
        v += step
    return ticks


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def xy_chart(
    series: list[Series],
    *,
    width: int = 900,
    height: int = 330,
    x_label: str = "",
    y_label: str = "",
    time_zone: str | None = None,  # x values are epoch seconds when set
    identity: bool = False,
    hline: float | None = None,
) -> str:
    left, right, top, bottom = 64, 24, 26, 46
    xs = [s.x[np.isfinite(s.x) & np.isfinite(s.y)] for s in series]
    ys = [s.y[np.isfinite(s.x) & np.isfinite(s.y)] for s in series]
    allx = np.concatenate(xs) if xs else np.zeros(0)
    ally = np.concatenate(ys) if ys else np.zeros(0)
    if allx.size == 0:
        return f'<svg width="{width}" height="{height}" role="img"><text x="10" y="20" style="{FONT}">No data</text></svg>'
    x0, x1 = float(allx.min()), float(allx.max())
    y0, y1 = float(ally.min()), float(ally.max())
    if identity:
        x0 = y0 = min(x0, y0)
        x1 = y1 = max(x1, y1)
    if hline is not None:
        y0, y1 = min(y0, hline), max(y1, hline)
    pad_x = (x1 - x0) * 0.03 or 1.0
    pad_y = (y1 - y0) * 0.05 or 1.0
    x0, x1, y0, y1 = x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y
    pw, ph = width - left - right, height - top - bottom

    def px(v: float) -> float:
        return left + (v - x0) / (x1 - x0) * pw

    def py(v: float) -> float:
        return top + (1 - (v - y0) / (y1 - y0)) * ph

    out = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
           f'xmlns="http://www.w3.org/2000/svg">']  # fmt: skip
    out.append(
        f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="white" stroke="#bbb"/>'
    )
    for t in nice_ticks(y0, y1):
        y = py(t)
        out.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        out.append(
            f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" style="{FONT}">{_fmt(t)}</text>'
        )
    if time_zone:
        data_lo, data_hi = float(allx.min()), float(allx.max())
        ticks = pd.date_range(pd.Timestamp(data_lo, unit="s", tz="UTC"),
                              pd.Timestamp(data_hi, unit="s", tz="UTC"), periods=6)  # fmt: skip
        for i, t in enumerate(ticks):
            x = px(t.value / 1e9)
            label = t.tz_convert(time_zone).strftime("%Y-%m-%d")
            anchor = "start" if i == 0 else "end" if i == len(ticks) - 1 else "middle"
            out.append(
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + ph}" stroke="{GRID}"/>'
            )
            out.append(
                f'<text x="{x:.1f}" y="{top + ph + 16}" text-anchor="{anchor}" style="{FONT}">{label}</text>'
            )
    else:
        for t in nice_ticks(x0, x1):
            x = px(t)
            out.append(
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + ph}" stroke="{GRID}"/>'
            )
            out.append(
                f'<text x="{x:.1f}" y="{top + ph + 16}" text-anchor="middle" style="{FONT}">{_fmt(t)}</text>'
            )
    if identity:
        out.append(f'<line x1="{px(x0):.1f}" y1="{py(x0):.1f}" x2="{px(x1):.1f}" y2="{py(x1):.1f}" '
                   'stroke="#666" stroke-dasharray="4 3"/>')  # fmt: skip
    if hline is not None:
        out.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{py(hline):.1f}" y2="{py(hline):.1f}" stroke="#666"/>'
        )
    for s, x, y in zip(series, xs, ys, strict=True):
        if s.kind == "line" and x.size > 1:
            order = np.argsort(x)
            pts = " ".join(
                f"{px(a):.1f},{py(b):.1f}" for a, b in zip(x[order], y[order], strict=True)
            )
            out.append(
                f'<polyline points="{pts}" fill="none" stroke="{s.colour}" stroke-width="1.2"/>'
            )
        else:
            for a, b in zip(x, y, strict=True):
                out.append(
                    f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="2.6" fill="{s.colour}" fill-opacity="0.75"/>'
                )
    # legend
    lx = left
    for s in series:
        if not s.label:
            continue
        out.append(f'<rect x="{lx}" y="{top - 19}" width="10" height="10" fill="{s.colour}"/>')
        out.append(f'<text x="{lx + 14}" y="{top - 10}" style="{FONT}">{escape(s.label)}</text>')
        lx += 22 + 7 * len(s.label)
    out.append(
        f'<text x="{left + pw / 2}" y="{height - 6}" text-anchor="middle" style="{FONT}">{escape(x_label)}</text>'
    )
    out.append(
        f'<text transform="translate(14,{top + ph / 2}) rotate(-90)" text-anchor="middle" style="{FONT}">{escape(y_label)}</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def bar_chart(labels: list[str], values: list[float], *, width: int = 900, title: str = "") -> str:
    """Horizontal bars (blue positive, vermillion negative) with labels on the left."""
    row = 22
    left = min(240, 12 + 7 * max((len(n) for n in labels), default=4))
    right, top = 60, 20 if title else 6
    height = top + row * len(labels) + 24
    vals = np.nan_to_num(np.asarray(values, dtype=float))
    lim = float(np.max(np.abs(vals))) if vals.size else 1.0
    lim = lim or 1.0
    lo = -lim if (vals < 0).any() else 0.0
    pw = width - left - right

    def px(v: float) -> float:
        return left + (v - lo) / (lim - lo) * pw

    out = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
           f'xmlns="http://www.w3.org/2000/svg">']  # fmt: skip
    if title:
        out.append(f'<text x="{left}" y="13" style="{FONT};font-weight:600">{escape(title)}</text>')
    zero = px(0.0)
    for i, (name, v) in enumerate(zip(labels, vals, strict=True)):
        y = top + i * row
        x_a, x_b = sorted((zero, px(v)))
        colour = "#0072B2" if v >= 0 else "#D55E00"
        out.append(
            f'<text x="{left - 6}" y="{y + 15}" text-anchor="end" style="{FONT}">{escape(name)}</text>'
        )
        out.append(
            f'<rect x="{x_a:.1f}" y="{y + 4}" width="{max(x_b - x_a, 0.5):.1f}" height="{row - 8}" fill="{colour}"/>'
        )
        out.append(f'<text x="{x_b + 4 if v >= 0 else x_a - 4:.1f}" y="{y + 15}" '
                   f'text-anchor="{"start" if v >= 0 else "end"}" style="{FONT}">{v:.3g}</text>')  # fmt: skip
    out.append(
        f'<line x1="{zero:.1f}" x2="{zero:.1f}" y1="{top}" y2="{top + row * len(labels)}" stroke="#666"/>'
    )
    out.append("</svg>")
    return "\n".join(out)
