from __future__ import annotations

from pathlib import Path
import sys
from typing import List

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QLinearGradient, QMovie, QPainter, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import QFrame, QPushButton, QSizePolicy, QWidget

from config import ALL_MEDIA_EXTENSIONS


def _ui_font_family() -> str:
    if sys.platform == "darwin":
        return "PingFang SC"
    if sys.platform == "win32":
        return "Microsoft YaHei UI"
    return "Noto Sans CJK SC"


def _display_font_family() -> str:
    if sys.platform == "darwin":
        return "PingFang SC"
    return _ui_font_family()


def _mix_color(base: QColor, other: QColor, ratio: float) -> QColor:
    t = max(0.0, min(1.0, float(ratio)))
    inv = 1.0 - t
    return QColor(
        int(base.red() * inv + other.red() * t),
        int(base.green() * inv + other.green() * t),
        int(base.blue() * inv + other.blue() * t),
        int(base.alpha() * inv + other.alpha() * t),
    )


class PictureBackgroundFrame(QFrame):
    """Background frame that paints a cover image with a soft tinted overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._background = QPixmap()
        self._dark_mode = False
        self._accent = QColor("#7db8ff")
        self.setAttribute(Qt.WA_StyledBackground, True)

    def set_background_theme(
        self,
        image_path: str | Path | None,
        *,
        dark_mode: bool,
        accent: QColor | str | None = None,
    ) -> None:
        self._dark_mode = bool(dark_mode)
        if accent is not None:
            try:
                parsed = QColor(accent)
            except Exception:
                parsed = QColor("#7db8ff")
            if parsed.isValid():
                self._accent = parsed
        path = Path(str(image_path or "")).expanduser()
        if path.exists():
            pixmap = QPixmap(str(path))
            self._background = pixmap if not pixmap.isNull() else QPixmap()
        else:
            self._background = QPixmap()
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        if not painter.isActive():
            return
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)

            rect = self.rect()
            if rect.width() <= 0 or rect.height() <= 0:
                return

            if not self._background.isNull():
                scaled = self._background.scaled(
                    rect.size(),
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation,
                )
                x = int((rect.width() - scaled.width()) / 2)
                y = int((rect.height() - scaled.height()) / 2)
                painter.drawPixmap(x, y, scaled)
            else:
                fallback = QColor("#131a26") if self._dark_mode else QColor("#edf4fb")
                painter.fillRect(rect, fallback)

            accent = QColor(self._accent)
            if not accent.isValid():
                accent = QColor("#7db8ff")

            sky = _mix_color(accent, QColor("#9fd8ff"), 0.36 if self._dark_mode else 0.52)
            candy = QColor("#ffb7cc" if self._dark_mode else "#ffd5e4")
            mint = QColor("#98f3db" if self._dark_mode else "#d8fff5")
            halo = QColor("#fff9ef")

            top_glow = QRadialGradient(rect.width() * 0.18, rect.height() * 0.10, rect.width() * 0.72)
            top_color = QColor(accent)
            top_color.setAlpha(100 if self._dark_mode else 84)
            top_glow.setColorAt(0.0, top_color)
            top_glow.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.fillRect(rect, top_glow)

            candy_glow = QRadialGradient(rect.width() * 0.86, rect.height() * 0.18, rect.width() * 0.48)
            candy_color = QColor(candy)
            candy_color.setAlpha(92 if self._dark_mode else 74)
            candy_glow.setColorAt(0.0, candy_color)
            candy_glow.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.fillRect(rect, candy_glow)

            mint_glow = QRadialGradient(rect.width() * 0.74, rect.height() * 0.82, rect.width() * 0.52)
            mint_color = QColor(mint)
            mint_color.setAlpha(86 if self._dark_mode else 70)
            mint_glow.setColorAt(0.0, mint_color)
            mint_glow.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.fillRect(rect, mint_glow)

            wash = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            if self._dark_mode:
                wash.setColorAt(0.0, QColor(8, 12, 21, 98))
                wash.setColorAt(0.32, QColor(18, 24, 38, 68))
                wash.setColorAt(0.74, QColor(11, 15, 24, 128))
                wash.setColorAt(1.0, QColor(8, 11, 18, 168))
            else:
                wash.setColorAt(0.0, QColor(255, 255, 255, 84))
                wash.setColorAt(0.34, QColor(255, 247, 252, 58))
                wash.setColorAt(0.72, QColor(243, 248, 255, 112))
                wash.setColorAt(1.0, QColor(239, 244, 252, 148))
            painter.fillRect(rect, wash)

            ribbon = QLinearGradient(rect.topLeft(), rect.bottomRight())
            ribbon.setColorAt(0.0, QColor(sky.red(), sky.green(), sky.blue(), 26 if self._dark_mode else 22))
            ribbon.setColorAt(0.45, QColor(halo.red(), halo.green(), halo.blue(), 12 if self._dark_mode else 18))
            ribbon.setColorAt(1.0, QColor(candy.red(), candy.green(), candy.blue(), 24 if self._dark_mode else 20))
            painter.fillRect(rect, ribbon)

            glass_haze = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            glass_haze.setColorAt(0.0, QColor(255, 255, 255, 34 if self._dark_mode else 40))
            glass_haze.setColorAt(0.18, QColor(255, 255, 255, 10 if self._dark_mode else 20))
            glass_haze.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.fillRect(rect, glass_haze)
        finally:
            painter.end()


class AnimatedSticker(QWidget):
    """Transparent widget that paints an animated image with aspect-fit scaling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._movie: QMovie | None = None
        self._fallback = QPixmap()
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_movie(self, movie: QMovie | None) -> None:
        old = self._movie
        if old is not None:
            try:
                old.frameChanged.disconnect(self._on_frame_changed)
            except Exception:
                pass
        self._movie = movie
        if movie is not None:
            try:
                movie.frameChanged.connect(self._on_frame_changed)
            except Exception:
                pass
        self.update()

    def set_fallback_pixmap(self, pixmap: QPixmap | None) -> None:
        self._fallback = QPixmap(pixmap) if pixmap is not None else QPixmap()
        self.update()

    def has_media(self) -> bool:
        if self._movie is not None:
            pix = self._movie.currentPixmap()
            if not pix.isNull():
                return True
        return not self._fallback.isNull()

    def current_source_size(self) -> QSize:
        if self._movie is not None:
            pix = self._movie.currentPixmap()
            if not pix.isNull():
                return pix.size()
        if not self._fallback.isNull():
            return self._fallback.size()
        return QSize()

    def _current_pixmap(self) -> QPixmap:
        if self._movie is not None:
            pix = self._movie.currentPixmap()
            if not pix.isNull():
                return pix
        return QPixmap(self._fallback)

    def _on_frame_changed(self, _frame: int) -> None:
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        pix = self._current_pixmap()
        if pix.isNull():
            return

        painter = QPainter(self)
        if not painter.isActive():
            return
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)

            target = pix.size().scaled(self.size(), Qt.KeepAspectRatio)
            if target.width() <= 0 or target.height() <= 0:
                return
            x = int((self.width() - target.width()) / 2)
            y = int((self.height() - target.height()) / 2)
            painter.drawPixmap(x, y, target.width(), target.height(), pix)
        finally:
            painter.end()


class ColorToggleButton(QPushButton):
    """Compact color toggle with per-instance accent colors."""

    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.setFlat(True)
        self.setText("")

        self._on_color = QColor("#2fa3ff")
        self._off_color = QColor("#768390")
        self._thumb_position = 1.0 if checked else 0.0
        self._thumb_anim = QPropertyAnimation(self, b"thumbPosition", self)
        self._thumb_anim.setDuration(150)
        self._thumb_anim.setEasingCurve(QEasingCurve.OutCubic)

        self.toggled.connect(self._on_toggled)
        self._sync_size()
        self.setChecked(bool(checked))
        self._on_toggled(bool(checked))

    def set_palette(self, on_color: str, off_color: str = "#768390") -> None:
        try:
            self._on_color = QColor(on_color)
        except Exception:
            self._on_color = QColor("#2fa3ff")
        try:
            self._off_color = QColor(off_color)
        except Exception:
            self._off_color = QColor("#768390")
        self.update()

    def _computed_size(self) -> QSize:
        # Keep dimensions stable across 100%/125%/150% UI scaling.
        h = max(22, int(round(self.fontMetrics().height() * 1.5)))
        w = max(40, int(round(h * 1.9)))
        return QSize(w, h)

    def sizeHint(self) -> QSize:
        return self._computed_size()

    def minimumSizeHint(self) -> QSize:
        return self._computed_size()

    def _sync_size(self) -> None:
        hint = self._computed_size()
        self.setMinimumSize(hint)
        self.setMaximumHeight(hint.height())
        self.updateGeometry()

    @Property(float)
    def thumbPosition(self):
        return self._thumb_position

    @thumbPosition.setter
    def thumbPosition(self, value):
        self._thumb_position = max(0.0, min(1.0, float(value)))
        self.update()

    @Slot(bool)
    def _on_toggled(self, checked: bool) -> None:
        self.setToolTip("已启用" if checked else "已禁用")
        self._thumb_anim.stop()
        self._thumb_anim.setStartValue(self._thumb_position)
        self._thumb_anim.setEndValue(1.0 if checked else 0.0)
        self._thumb_anim.start()
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        if not painter.isActive():
            return
        try:
            painter.setRenderHint(QPainter.Antialiasing)

            rect = self.rect().adjusted(1, 1, -1, -1)
            if rect.width() <= 2 or rect.height() <= 2:
                return

            dark_mode = bool(self.property("darkMode"))

            accent = QColor(self._on_color)
            off_tint = QColor(self._off_color)

            track_fill = QColor(accent if self.isChecked() else off_tint)
            track_fill.setAlpha(162 if dark_mode else 118)
            track_edge = QColor(accent if self.isChecked() else off_tint)
            track_edge.setAlpha(214 if dark_mode else 166)
            inner_gloss = QColor(255, 255, 255, 38 if dark_mode else 92)
            if not self.isChecked():
                track_fill = QColor(77, 88, 102, 146) if dark_mode else QColor(205, 212, 223, 184)
                track_edge = QColor(123, 135, 151, 176) if dark_mode else QColor(168, 178, 192, 196)

            radius = rect.height() / 2.0
            track_rect = rect.adjusted(0, 0, 0, 0)

            painter.setPen(QPen(track_edge, 1.2))
            painter.setBrush(track_fill)
            painter.drawRoundedRect(track_rect, radius, radius)

            gloss_rect = QRectF(track_rect.left() + 1, track_rect.top() + 1, track_rect.width() - 2, max(4.0, track_rect.height() * 0.48))
            painter.setPen(Qt.NoPen)
            painter.setBrush(inner_gloss)
            painter.drawRoundedRect(gloss_rect, max(3.0, radius - 1.5), max(3.0, radius - 1.5))

            thumb_margin = 2.0
            thumb_d = max(12.0, track_rect.height() - (thumb_margin * 2.0))
            travel = max(0.0, track_rect.width() - thumb_d - (thumb_margin * 2.0))
            thumb_x = track_rect.left() + thumb_margin + (travel * self._thumb_position)
            thumb_y = track_rect.top() + thumb_margin
            thumb_rect = QRectF(thumb_x, thumb_y, thumb_d, thumb_d)

            shadow_color = QColor(8, 12, 18, 54 if dark_mode else 28)
            painter.setBrush(shadow_color)
            painter.drawEllipse(thumb_rect.adjusted(0.0, 1.2, 0.0, 1.2))

            thumb_fill = QColor("#ffffff" if not dark_mode else "#fdfcff")
            painter.setBrush(thumb_fill)
            painter.setPen(QPen(QColor(255, 255, 255, 130 if dark_mode else 162), 1.0))
            painter.drawEllipse(thumb_rect)

            top_sheen = QLinearGradient(thumb_rect.topLeft(), thumb_rect.bottomLeft())
            top_sheen.setColorAt(0.0, QColor(255, 255, 255, 146))
            top_sheen.setColorAt(1.0, QColor(255, 255, 255, 16))
            painter.setBrush(top_sheen)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(thumb_rect.adjusted(1.0, 1.0, -1.0, -thumb_rect.height() * 0.42))
        finally:
            painter.end()


class HoldToDeleteButton(QPushButton):
    """Press-and-hold button with progress fill for destructive actions."""

    confirmed = Signal()

    def __init__(self, text: str = "长按删除", hold_ms: int = 1200, parent=None):
        super().__init__(text, parent)
        self._hold_ms = max(300, int(hold_ms))
        self._hold_progress = 0.0
        self._hold_active = False
        self._hold_fired = False
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick_hold)
        self._elapsed_ms = 0

    @Property(float)
    def hold_progress(self):
        return self._hold_progress

    @hold_progress.setter
    def hold_progress(self, value):
        self._hold_progress = max(0.0, min(1.0, float(value)))
        self.update()

    def _reset_hold(self) -> None:
        self._timer.stop()
        self._hold_active = False
        self._hold_fired = False
        self._elapsed_ms = 0
        self.hold_progress = 0.0

    def _tick_hold(self) -> None:
        if not self._hold_active:
            self._reset_hold()
            return

        self._elapsed_ms += self._timer.interval()
        progress = min(1.0, self._elapsed_ms / self._hold_ms)
        self.hold_progress = progress

        if progress >= 1.0 and not self._hold_fired:
            self._hold_fired = True
            self.confirmed.emit()
            self._reset_hold()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.isEnabled():
            self._hold_active = True
            self._hold_fired = False
            self.hold_progress = 0.0
            self._elapsed_ms = 0
            self._timer.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        self._reset_hold()

    def leaveEvent(self, event) -> None:
        if self._hold_active and not self._hold_fired:
            self._reset_hold()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        if self._hold_progress <= 0.0:
            return

        painter = QPainter(self)
        if not painter.isActive():
            return
        try:
            painter.setRenderHint(QPainter.Antialiasing)

            fill_rect = self.rect().adjusted(1, 1, -1, -1)
            fill_width = max(1, int(fill_rect.width() * self._hold_progress))
            fill_rect.setWidth(fill_width)

            fill_grad = QLinearGradient(fill_rect.topLeft(), fill_rect.topRight())
            left = QColor("#ff8bab")
            right = QColor("#ff5f83")
            left.setAlpha(86 if self._hold_progress < 1.0 else 126)
            right.setAlpha(112 if self._hold_progress < 1.0 else 156)
            fill_grad.setColorAt(0.0, left)
            fill_grad.setColorAt(1.0, right)
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill_grad)
            painter.drawRoundedRect(fill_rect, 7, 7)

            gloss_rect = fill_rect.adjusted(1, 1, -1, -max(2, int(fill_rect.height() * 0.45)))
            painter.setBrush(QColor(255, 255, 255, 36))
            painter.drawRoundedRect(gloss_rect, 6, 6)
        finally:
            painter.end()


class DropArea(QFrame):
    """Drag-and-drop file area with a large plus icon."""

    files_dropped = Signal(list)
    browse_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(160)
        self.setCursor(Qt.PointingHandCursor)
        self._glow = 0.0
        self._accent_color = QColor("#5d9fff")
        self._anim = QPropertyAnimation(self, b"glow", self)
        self._anim.setDuration(260)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    @Property(float)
    def glow(self):
        return self._glow

    @glow.setter
    def glow(self, value):
        self._glow = max(0.0, min(1.0, float(value)))
        self.update()

    def _animate_glow(self, value: float) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._glow)
        self._anim.setEndValue(value)
        self._anim.start()

    def set_accent_color(self, color: QColor | str) -> None:
        try:
            parsed = QColor(color)
        except Exception:
            parsed = QColor("#5d9fff")
        if not parsed.isValid():
            parsed = QColor("#5d9fff")
        self._accent_color = parsed
        self.update()

    @staticmethod
    def _extract_media_paths(mime_data) -> List[str]:
        if not mime_data.hasUrls():
            return []
        files = []
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() in ALL_MEDIA_EXTENSIONS:
                files.append(str(path.resolve()))
        return files

    def dragEnterEvent(self, event) -> None:
        files = self._extract_media_paths(event.mimeData())
        if files:
            event.acceptProposedAction()
            self._animate_glow(1.0)
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        _ = event
        self._animate_glow(0.0)

    def dropEvent(self, event) -> None:
        files = self._extract_media_paths(event.mimeData())
        if files:
            self.files_dropped.emit(files)
            event.acceptProposedAction()
        else:
            event.ignore()
        self._animate_glow(0.0)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.browse_requested.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        if not painter.isActive():
            return
        try:
            painter.setRenderHint(QPainter.Antialiasing)

            rect = self.rect().adjusted(3, 3, -3, -3)
            dark_mode = bool(self.property("darkMode"))
            accent = QColor(self._accent_color)
            if not accent.isValid():
                accent = QColor("#5d9fff")
            candy = _mix_color(accent, QColor("#ffb9d3"), 0.52 if dark_mode else 0.64)
            if dark_mode:
                base = QColor(26, 31, 44, 198)
                glow = QColor(candy)
                glow.setAlpha(66)
                border_idle = QColor(138, 150, 172, 116)
                border_hot = QColor(candy)
                border_hot.setAlpha(232)
                title_color = QColor("#f5efff")
                subtitle_color = QColor("#d6c9e4")
                plus_pen = QColor("#fdfcff")
                orb_fill = QColor(255, 255, 255, 34)
                orb_edge = QColor(255, 255, 255, 62)
            else:
                base = QColor(255, 251, 255, 202)
                glow = QColor(candy)
                glow.setAlpha(54)
                border_idle = QColor("#e8dbe9")
                border_hot = QColor(candy)
                border_hot.setAlpha(210)
                title_color = QColor("#4d4561")
                subtitle_color = QColor("#7d748f")
                plus_pen = QColor("#fffdfd")
                orb_fill = QColor(255, 255, 255, 118)
                orb_edge = QColor(255, 255, 255, 168)

            mix = QColor(
                int(base.red() + (glow.red() - base.red()) * self._glow),
                int(base.green() + (glow.green() - base.green()) * self._glow),
                int(base.blue() + (glow.blue() - base.blue()) * self._glow),
                int(base.alpha() + (glow.alpha() - base.alpha()) * self._glow),
            )
            border = QColor(
                int(border_idle.red() + (border_hot.red() - border_idle.red()) * self._glow),
                int(border_idle.green() + (border_hot.green() - border_idle.green()) * self._glow),
                int(border_idle.blue() + (border_hot.blue() - border_idle.blue()) * self._glow),
                int(border_idle.alpha() + (border_hot.alpha() - border_idle.alpha()) * self._glow),
            )

            fill = QLinearGradient(rect.topLeft(), rect.bottomRight())
            fill.setColorAt(0.0, QColor(mix.red(), mix.green(), mix.blue(), mix.alpha()))
            fill.setColorAt(0.45, QColor(_mix_color(mix, QColor("#fffafd"), 0.28)))
            fill.setColorAt(1.0, mix.darker(104))

            painter.setPen(QPen(border, 1.8, Qt.SolidLine))
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 24, 24)

            haze = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            haze.setColorAt(0.0, QColor(255, 255, 255, 46 if dark_mode else 72))
            haze.setColorAt(0.34, QColor(255, 255, 255, 20 if dark_mode else 34))
            haze.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.setBrush(haze)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 23, 23)

            orb_size = max(42, int(min(rect.width(), rect.height()) * 0.24))
            orb_rect = QRectF(
                rect.center().x() - (orb_size / 2.0),
                rect.top() + max(16, rect.height() * 0.14),
                orb_size,
                orb_size,
            )
            orb_grad = QLinearGradient(orb_rect.topLeft(), orb_rect.bottomLeft())
            orb_grad.setColorAt(0.0, orb_fill)
            orb_grad.setColorAt(1.0, QColor(accent.red(), accent.green(), accent.blue(), 42 if dark_mode else 36))
            painter.setBrush(orb_grad)
            painter.setPen(QPen(orb_edge, 1.2))
            painter.drawEllipse(orb_rect)

            painter.setPen(QPen(plus_pen, 3))
            cx = orb_rect.center().x()
            cy = orb_rect.center().y()
            arm = max(10, int(orb_rect.width() * 0.18))
            painter.drawLine(int(cx - arm), int(cy), int(cx + arm), int(cy))
            painter.drawLine(int(cx), int(cy - arm), int(cx), int(cy + arm))

            painter.setPen(title_color)
            title_font = QFont(_display_font_family(), 12, QFont.DemiBold)
            painter.setFont(title_font)
            painter.drawText(
                rect.adjusted(0, int(rect.height() * 0.04), 0, 0),
                Qt.AlignHCenter,
                "把文件拖放进来吧",
            )

            sub_font = QFont(_ui_font_family(), 9)
            painter.setFont(sub_font)
            painter.setPen(subtitle_color)
            subtitle_text = "也可以点一下并从 Finder 里挑" if sys.platform == "darwin" else "也可以点一下并从文件夹里挑"
            painter.drawText(
                rect.adjusted(0, int(rect.height() * 0.50), 0, 0),
                Qt.AlignHCenter,
                subtitle_text,
            )
        finally:
            painter.end()
