"""Draw the application icon and write assets/icons/app.png and app.ico (several sizes).

Run from the repository root: python packaging/make_icon.py (needs PySide6; works headless).
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIZES = (16, 24, 32, 48, 64, 128, 256)


def draw(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 256.0
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor("#0B3C5D")))
    p.drawRoundedRect(QRectF(8 * s, 8 * s, 240 * s, 240 * s), 44 * s, 44 * s)
    # Process trend (continuous) …
    path = QPainterPath(QPointF(36 * s, 168 * s))
    for x, y in ((70, 150), (100, 160), (130, 112), (160, 124), (190, 86), (222, 96)):
        path.lineTo(QPointF(x * s, y * s))
    p.setPen(QPen(QColor("#FFFFFF"), max(14 * s, 1.5), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                  Qt.PenJoinStyle.RoundJoin))  # fmt: skip
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    # … and sparse lab samples it estimates.
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor("#F5A623")))
    r = max(15 * s, 1.6)
    for x, y in ((62, 196), (132, 150), (204, 122)):
        p.drawEllipse(QPointF(x * s, y * s), r, r)
    p.end()
    return img


def png_bytes(img: QImage) -> bytes:
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(data)


def write_ico(images: list[tuple[int, bytes]], path: Path) -> None:
    """ICO with PNG-compressed entries (supported since Windows Vista)."""
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for size, blob in images:
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(blob), offset + len(blobs))
        blobs += blob
    path.write_bytes(header + entries + blobs)


def main() -> int:
    _app = QGuiApplication(sys.argv)
    out = ROOT / "assets" / "icons"
    out.mkdir(parents=True, exist_ok=True)
    images = [(size, png_bytes(draw(size))) for size in SIZES]
    (out / "app.png").write_bytes(dict(images)[256])
    write_ico(images, out / "app.ico")
    print(f"wrote {out / 'app.png'} and {out / 'app.ico'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
