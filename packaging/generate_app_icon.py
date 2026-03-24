#!/usr/bin/env python3
from __future__ import annotations

import math
import struct
from pathlib import Path

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient


ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
MASTER_SIZE = 1024
ICO_SIZES = (16, 32, 48, 64, 128, 256)
ICNS_SLOTS = (
    ("icp4", 16),
    ("ic11", 32),
    ("icp5", 32),
    ("ic12", 64),
    ("icp6", 64),
    ("ic07", 128),
    ("ic13", 256),
    ("ic08", 256),
    ("ic14", 512),
    ("ic09", 512),
    ("ic10", 1024),
)


def _superellipse_path(rect: QRectF, exponent: float = 5.2, samples: int = 256) -> QPainterPath:
    cx = rect.center().x()
    cy = rect.center().y()
    rx = rect.width() / 2.0
    ry = rect.height() / 2.0
    path = QPainterPath()
    for index in range(samples + 1):
        angle = (2.0 * math.pi * index) / samples
        cos_value = math.cos(angle)
        sin_value = math.sin(angle)
        x = cx + rx * math.copysign(abs(cos_value) ** (2.0 / exponent), cos_value)
        y = cy + ry * math.copysign(abs(sin_value) ** (2.0 / exponent), sin_value)
        point = QPointF(x, y)
        if index == 0:
            path.moveTo(point)
        else:
            path.lineTo(point)
    path.closeSubpath()
    return path


def _draw_waveform(painter: QPainter, size: int) -> None:
    waveform = QPainterPath()
    points = [
        QPointF(size * 0.24, size * 0.47),
        QPointF(size * 0.34, size * 0.47),
        QPointF(size * 0.405, size * 0.372),
        QPointF(size * 0.485, size * 0.557),
        QPointF(size * 0.565, size * 0.47),
        QPointF(size * 0.645, size * 0.47),
    ]
    waveform.moveTo(points[0])
    for point in points[1:]:
        waveform.lineTo(point)

    shadow_pen = QPen(QColor(7, 19, 45, 78), size * 0.064, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.save()
    painter.translate(0, size * 0.014)
    painter.setPen(shadow_pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(waveform)
    painter.restore()

    wave_gradient = QLinearGradient(QPointF(size * 0.24, size * 0.36), QPointF(size * 0.62, size * 0.58))
    wave_gradient.setColorAt(0.0, QColor("#ffffff"))
    wave_gradient.setColorAt(0.65, QColor("#f7fbff"))
    wave_gradient.setColorAt(1.0, QColor("#d7ecff"))
    painter.setPen(QPen(QBrush(wave_gradient), size * 0.056, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.drawPath(waveform)


def _draw_badge(painter: QPainter, size: int) -> None:
    badge_rect = QRectF(size * 0.585, size * 0.565, size * 0.22, size * 0.22)
    badge_center = badge_rect.center()

    painter.save()
    painter.translate(0, size * 0.01)
    painter.setBrush(QColor(8, 18, 40, 54))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(badge_rect)
    painter.restore()

    badge_fill = QRadialGradient(badge_center, badge_rect.width() * 0.68)
    badge_fill.setColorAt(0.0, QColor(255, 255, 255, 118))
    badge_fill.setColorAt(0.6, QColor(255, 255, 255, 74))
    badge_fill.setColorAt(1.0, QColor(255, 255, 255, 42))
    painter.setBrush(badge_fill)
    painter.setPen(QPen(QColor(255, 255, 255, 224), size * 0.009))
    painter.drawEllipse(badge_rect)

    check = QPainterPath()
    check.moveTo(size * 0.646, size * 0.681)
    check.lineTo(size * 0.684, size * 0.718)
    check.lineTo(size * 0.748, size * 0.648)
    painter.setPen(QPen(QColor("#ffffff"), size * 0.028, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(check)

    sparkle_pen = QPen(QColor(255, 255, 255, 235), size * 0.012, Qt.SolidLine, Qt.RoundCap)
    sparkle_center = QPointF(size * 0.746, size * 0.57)
    sparkle_long = size * 0.026
    sparkle_short = size * 0.017
    painter.setPen(sparkle_pen)
    painter.drawLine(
        QPointF(sparkle_center.x(), sparkle_center.y() - sparkle_long),
        QPointF(sparkle_center.x(), sparkle_center.y() + sparkle_long),
    )
    painter.drawLine(
        QPointF(sparkle_center.x() - sparkle_long, sparkle_center.y()),
        QPointF(sparkle_center.x() + sparkle_long, sparkle_center.y()),
    )
    painter.drawLine(
        QPointF(sparkle_center.x() - sparkle_short, sparkle_center.y() - sparkle_short),
        QPointF(sparkle_center.x() + sparkle_short, sparkle_center.y() + sparkle_short),
    )
    painter.drawLine(
        QPointF(sparkle_center.x() - sparkle_short, sparkle_center.y() + sparkle_short),
        QPointF(sparkle_center.x() + sparkle_short, sparkle_center.y() - sparkle_short),
    )


def render_master_icon(size: int = MASTER_SIZE) -> QImage:
    image = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)

    icon_rect = QRectF(size * 0.094, size * 0.086, size * 0.812, size * 0.812)
    icon_path = _superellipse_path(icon_rect)

    for spread, offset_y, alpha in ((68, 44, 26), (32, 24, 54), (12, 12, 88)):
        shadow_rect = icon_rect.adjusted(-spread, -spread * 0.42, spread, spread)
        shadow_path = _superellipse_path(shadow_rect)
        painter.save()
        painter.translate(0, offset_y)
        painter.fillPath(shadow_path, QColor(5, 12, 28, alpha))
        painter.restore()

    base_gradient = QLinearGradient(icon_rect.left(), icon_rect.top(), icon_rect.right(), icon_rect.bottom())
    base_gradient.setColorAt(0.0, QColor("#b8eeff"))
    base_gradient.setColorAt(0.18, QColor("#73ccff"))
    base_gradient.setColorAt(0.56, QColor("#347dff"))
    base_gradient.setColorAt(1.0, QColor("#1838b9"))
    painter.fillPath(icon_path, base_gradient)

    painter.save()
    painter.setClipPath(icon_path)

    top_glow = QRadialGradient(
        QPointF(icon_rect.left() + icon_rect.width() * 0.28, icon_rect.top() + icon_rect.height() * 0.16),
        icon_rect.width() * 0.7,
    )
    top_glow.setColorAt(0.0, QColor(255, 255, 255, 132))
    top_glow.setColorAt(0.4, QColor(255, 255, 255, 48))
    top_glow.setColorAt(1.0, QColor(255, 255, 255, 0))
    painter.fillRect(image.rect(), top_glow)

    lower_tint = QLinearGradient(icon_rect.left(), icon_rect.top(), icon_rect.left(), icon_rect.bottom())
    lower_tint.setColorAt(0.0, QColor(255, 255, 255, 0))
    lower_tint.setColorAt(0.54, QColor(12, 34, 124, 0))
    lower_tint.setColorAt(1.0, QColor(6, 18, 78, 92))
    painter.fillRect(image.rect(), lower_tint)

    sheen_rect = QRectF(
        icon_rect.left() + icon_rect.width() * 0.09,
        icon_rect.top() + icon_rect.height() * 0.07,
        icon_rect.width() * 0.82,
        icon_rect.height() * 0.3,
    )
    painter.setBrush(QColor(255, 255, 255, 30))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(sheen_rect, size * 0.06, size * 0.06)
    painter.restore()

    painter.setPen(QPen(QColor(255, 255, 255, 108), size * 0.0048))
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(icon_path)

    _draw_waveform(painter, size)
    _draw_badge(painter, size)
    painter.end()
    return image


def _png_bytes(image: QImage) -> bytes:
    buffer = QByteArray()
    device = QBuffer(buffer)
    device.open(QIODevice.WriteOnly)
    image.save(device, "PNG")
    device.close()
    return bytes(buffer)


def _scaled_image(image: QImage, size: int) -> QImage:
    return image.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def write_png(image: QImage, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), "PNG")


def write_ico(master: QImage, path: Path) -> None:
    frames: list[tuple[int, int, bytes]] = []
    for size in ICO_SIZES:
        scaled = _scaled_image(master, size)
        png_data = _png_bytes(scaled)
        frames.append((size, size, png_data))

    header = struct.pack("<HHH", 0, 1, len(frames))
    entries = bytearray()
    offset = 6 + 16 * len(frames)
    payload = bytearray()
    for width, height, png_data in frames:
        entries.extend(
            struct.pack(
                "<BBBBHHII",
                0 if width >= 256 else width,
                0 if height >= 256 else height,
                0,
                0,
                1,
                32,
                len(png_data),
                offset,
            )
        )
        payload.extend(png_data)
        offset += len(png_data)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + entries + payload)


def write_icns(master: QImage, path: Path) -> None:
    chunks: list[bytes] = []
    for icon_type, size in ICNS_SLOTS:
        png_data = _png_bytes(_scaled_image(master, size))
        chunks.append(icon_type.encode("ascii") + struct.pack(">I", len(png_data) + 8) + png_data)
    payload = b"".join(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"icns" + struct.pack(">I", len(payload) + 8) + payload)


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication([])
    master = render_master_icon()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    write_png(master, ASSETS_DIR / "app.png")
    write_ico(master, ASSETS_DIR / "app.ico")
    write_ico(master, ROOT / "app.ico")
    write_ico(master, ROOT / "packaging" / "app.ico")

    try:
        write_icns(master, ASSETS_DIR / "app.icns")
    except Exception as exc:
        print(f"[warn] Skipped app.icns generation: {exc}")
    else:
        print(f"[ok] Wrote {ASSETS_DIR / 'app.icns'}")

    print(f"[ok] Wrote {ASSETS_DIR / 'app.png'}")
    print(f"[ok] Wrote {ASSETS_DIR / 'app.ico'}")
    print(f"[ok] Wrote {ROOT / 'app.ico'}")
    print(f"[ok] Wrote {ROOT / 'packaging' / 'app.ico'}")
    _ = app


if __name__ == "__main__":
    main()
