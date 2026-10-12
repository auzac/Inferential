"""Interface icons in the theme's colours.

The SVGs are Microsoft's Fluent UI System Icons (MIT License; ``assets/icons/fluent``). Each
has a single ``#212121`` fill, which is replaced by a theme colour before rendering. Icons are
rendered once per (name, colours, size) at 1x and 2x and cached, so asking for the same icon
again is free; after a theme switch the new colours simply give new cache entries.
"""

from __future__ import annotations

import logging
from functools import cache

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from inferential_studio.infrastructure.paths import resource_path
from inferential_studio.ui.theme import TOKENS

log = logging.getLogger(__name__)

ICON_DIR = "assets/icons/fluent"
SOURCE_FILL = "#212121"
_cache: dict[tuple, QIcon] = {}


@cache
def _svg(name: str) -> str:
    path = resource_path(f"{ICON_DIR}/{name}.svg")
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        log.warning("Icon %s not found at %s", name, path)
        return ""


def icon_exists(name: str) -> bool:
    return bool(_svg(name))


def _pixmap(name: str, colour: str, size: int, scale: int) -> QPixmap:
    image = QImage(size * scale, size * scale, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    svg = _svg(name)
    if svg:
        renderer = QSvgRenderer(QByteArray(svg.replace(SOURCE_FILL, colour).encode("utf-8")))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()
    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(scale)
    return pixmap


def icon(
    name: str,
    token: str = "text",
    *,
    size: int = 20,
    active: str | None = None,
) -> QIcon:
    """The named icon in the theme colour ``token``; disabled state in ``text_faint``.

    ``active`` is a literal colour for the hovered state (e.g. white on the red close button).
    """
    normal, faint = TOKENS[token], TOKENS["text_faint"]
    key = (name, normal, faint, active, size)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    result = QIcon()
    for scale in (1, 2):
        result.addPixmap(_pixmap(name, normal, size, scale), QIcon.Mode.Normal)
        result.addPixmap(_pixmap(name, active or normal, size, scale), QIcon.Mode.Active)
        result.addPixmap(_pixmap(name, normal, size, scale), QIcon.Mode.Selected)
        result.addPixmap(_pixmap(name, faint, size, scale), QIcon.Mode.Disabled)
    _cache[key] = result
    return result
