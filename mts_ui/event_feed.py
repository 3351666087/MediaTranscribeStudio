from __future__ import annotations

import time
from typing import List, Optional

from PySide6.QtCore import QEasingCurve, QParallelAnimationGroup, QPropertyAnimation, QTimer, Qt, QRectF
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class _BurstItem(QFrame):
    def __init__(
        self,
        text: str,
        level: str,
        *,
        role: str = "event",
        overlay_stack: Optional[bool] = None,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.role = str(role or "event")
        self._overlay_mode = (self.role == "log") if overlay_stack is None else bool(overlay_stack)
        self.level = level
        self.setObjectName(f"BurstItem_{level}")
        self.setProperty("role", self.role)
        self.setProperty("overlayStack", self._overlay_mode)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215 if self._overlay_mode else 0)
        self.setToolTip(text)
        self._stack_index = 0
        self._stack_visible_count = 1
        self._body_lines_override: Optional[int] = None
        self._is_front = True
        if self._overlay_mode:
            self.setFrameShape(QFrame.NoFrame)
            self.setAttribute(Qt.WA_TranslucentBackground, True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)

        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.setSpacing(8)

        chip = QLabel(self._chip_text(level), self)
        chip.setObjectName("BurstChip")
        chip.setProperty("level", level)
        chip.setProperty("role", self.role)
        meta_row.addWidget(chip, 0, Qt.AlignLeft)
        meta_row.addStretch(1)

        stamp = QLabel(time.strftime("%H:%M:%S"), self)
        stamp.setObjectName("BurstStamp")
        stamp.setProperty("role", self.role)
        meta_row.addWidget(stamp, 0, Qt.AlignRight)
        lay.addLayout(meta_row)

        label = QLabel(text, self)
        label.setObjectName("BurstText")
        label.setProperty("role", self.role)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(label)
        self._chip = chip
        self._stamp = stamp
        self._label = label
        self._layout = lay

        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(0.0)
        self.setGraphicsEffect(effect)
        self._effect = effect
        self._show_anim: QParallelAnimationGroup | None = None
        self._hide_anim: QParallelAnimationGroup | None = None

    @staticmethod
    def _chip_text(level: str) -> str:
        if level == "ok":
            return "完成"
        if level == "warn":
            return "提醒"
        if level == "err":
            return "异常"
        return "进行中"

    @staticmethod
    def _dot_color(level: str) -> str:
        if level == "ok":
            return "#2fbf71"
        if level == "warn":
            return "#d18a00"
        if level == "err":
            return "#cc3f3f"
        return "#3f8cff"

    def stop_animations(self) -> None:
        if self._show_anim is not None:
            try:
                self._show_anim.stop()
            except Exception:
                pass
            self._show_anim = None
        if self._hide_anim is not None:
            try:
                self._hide_anim.stop()
            except Exception:
                pass
            self._hide_anim = None

    def animate_in(self) -> None:
        self.stop_animations()
        if self._overlay_mode:
            op_anim = QPropertyAnimation(self._effect, b"opacity", self)
            op_anim.setDuration(220)
            op_anim.setStartValue(0.0)
            op_anim.setEndValue(1.0)
            op_anim.setEasingCurve(QEasingCurve.OutCubic)

            group = QParallelAnimationGroup(self)
            group.addAnimation(op_anim)
            group.finished.connect(lambda: setattr(self, "_show_anim", None))
            group.start()
            self._show_anim = group
            return

        target_h = max(52 if self.role == "log" else 46, self.sizeHint().height())

        h_anim = QPropertyAnimation(self, b"maximumHeight", self)
        h_anim.setDuration(220)
        h_anim.setStartValue(0)
        h_anim.setEndValue(target_h)
        h_anim.setEasingCurve(QEasingCurve.OutCubic)

        op_anim = QPropertyAnimation(self._effect, b"opacity", self)
        op_anim.setDuration(220)
        op_anim.setStartValue(0.0)
        op_anim.setEndValue(1.0)
        op_anim.setEasingCurve(QEasingCurve.OutCubic)

        group = QParallelAnimationGroup(self)
        group.addAnimation(h_anim)
        group.addAnimation(op_anim)
        group.finished.connect(lambda: setattr(self, "_show_anim", None))
        group.start()
        self._show_anim = group

    def animate_out(self, on_finished=None, *, immediate: bool = False) -> None:
        if immediate:
            self.stop_animations()
            if on_finished is not None:
                on_finished()
            return

        if self._hide_anim is not None:
            if on_finished is not None:
                self._hide_anim.finished.connect(on_finished)
            return

        if self._show_anim is not None:
            try:
                self._show_anim.stop()
            except Exception:
                pass
            self._show_anim = None

        if self._overlay_mode:
            op_anim = QPropertyAnimation(self._effect, b"opacity", self)
            op_anim.setDuration(180)
            op_anim.setStartValue(self._effect.opacity())
            op_anim.setEndValue(0.0)
            op_anim.setEasingCurve(QEasingCurve.InCubic)

            group = QParallelAnimationGroup(self)
            group.addAnimation(op_anim)
            if on_finished is not None:
                group.finished.connect(on_finished)
            group.finished.connect(lambda: setattr(self, "_hide_anim", None))
            group.start()
            self._hide_anim = group
            return

        h_anim = QPropertyAnimation(self, b"maximumHeight", self)
        h_anim.setDuration(300)
        h_anim.setStartValue(self.maximumHeight())
        h_anim.setEndValue(0)
        h_anim.setEasingCurve(QEasingCurve.InBack)

        op_anim = QPropertyAnimation(self._effect, b"opacity", self)
        op_anim.setDuration(280)
        op_anim.setStartValue(self._effect.opacity())
        op_anim.setEndValue(0.0)
        op_anim.setEasingCurve(QEasingCurve.InCubic)

        group = QParallelAnimationGroup(self)
        group.addAnimation(h_anim)
        group.addAnimation(op_anim)
        if on_finished is not None:
            group.finished.connect(on_finished)
        group.finished.connect(lambda: setattr(self, "_hide_anim", None))
        group.start()
        self._hide_anim = group

    def _body_line_height(self) -> int:
        return max(14, int(self._label.fontMetrics().lineSpacing()))

    def _header_height(self) -> int:
        return max(self._chip.sizeHint().height(), self._stamp.sizeHint().height())

    def _apply_log_stack_mode(self, idx: int, visible_count: int) -> None:
        if not self._overlay_mode:
            return

        self._stack_index = max(0, int(idx))
        self._stack_visible_count = max(1, int(visible_count))
        self._is_front = self._stack_index == 0

        if self._is_front:
            self._body_lines_override = None
            self._label.setVisible(True)
            self._label.setMaximumHeight(16777215)
            self._layout.setContentsMargins(16, 14, 16, 14)
            self._layout.setSpacing(8)
        elif self._stack_index == 1:
            self._body_lines_override = 1
            self._label.setVisible(True)
            self._label.setMaximumHeight(self._body_line_height() + 6)
            self._layout.setContentsMargins(14, 11, 14, 11)
            self._layout.setSpacing(5)
        else:
            self._body_lines_override = 0
            self._label.setVisible(False)
            self._label.setMaximumHeight(0)
            self._layout.setContentsMargins(13, 10, 13, 10)
            self._layout.setSpacing(0)

        self.updateGeometry()
        self.update()

    def content_height_for_width(self, width: int, *, body_lines: Optional[int] = None) -> int:
        usable_width = max(120, int(width))
        margins = self._layout.contentsMargins()
        inner_width = max(72, usable_width - margins.left() - margins.right())
        resolved_lines = self._body_lines_override if body_lines is None else body_lines
        label_height = 0
        if resolved_lines != 0:
            label_height = self._label.heightForWidth(inner_width)
            if label_height <= 0:
                label_height = self._label.sizeHint().height()
            if resolved_lines is not None and resolved_lines > 0:
                label_height = min(label_height, (self._body_line_height() * resolved_lines) + 6)
        meta_height = max(self._chip.sizeHint().height(), self._stamp.sizeHint().height())
        spacing = self._layout.spacing() if label_height > 0 else 0
        return int(margins.top() + meta_height + spacing + label_height + margins.bottom())

    def minimum_front_height_for_width(self, width: int) -> int:
        return self.content_height_for_width(width, body_lines=2)

    def compact_height_for_width(self, width: int, *, stack_index: int) -> int:
        body_lines = 1 if stack_index == 1 else 0
        return self.content_height_for_width(width, body_lines=body_lines)

    def _glass_tint(self) -> QColor:
        if self.level == "ok":
            return QColor("#7dd9ab")
        if self.level == "warn":
            return QColor("#ffc06b")
        if self.level == "err":
            return QColor("#ff8fa9")
        return QColor("#7ab8ff")

    def paintEvent(self, event) -> None:
        if not self._overlay_mode:
            super().paintEvent(event)
            return

        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        if rect.width() <= 2.0 or rect.height() <= 2.0:
            return

        radius = 24.0 if self._is_front else 21.0
        depth = min(self._stack_index, 4)
        tint = self._glass_tint()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        shell = QPainterPath()
        shell.addRoundedRect(rect, radius, radius)

        wash = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        wash.setColorAt(0.0, QColor(255, 255, 255, max(90, 228 - (depth * 24))))
        wash.setColorAt(0.52, QColor(255, 255, 255, max(54, 164 - (depth * 18))))
        wash.setColorAt(1.0, QColor(tint.red(), tint.green(), tint.blue(), max(34, 88 - (depth * 10))))
        painter.fillPath(shell, wash)

        tint_layer = QLinearGradient(rect.topLeft(), rect.bottomRight())
        tint_layer.setColorAt(0.0, QColor(tint.red(), tint.green(), tint.blue(), max(24, 74 - (depth * 8))))
        tint_layer.setColorAt(1.0, QColor(255, 255, 255, max(18, 52 - (depth * 6))))
        painter.fillPath(shell, tint_layer)

        highlight_rect = QRectF(rect.left() + 2.0, rect.top() + 2.0, rect.width() - 4.0, rect.height() * (0.54 if self._is_front else 0.42))
        highlight = QPainterPath()
        highlight.addRoundedRect(highlight_rect, max(12.0, radius - 5.0), max(12.0, radius - 5.0))
        painter.fillPath(highlight, QColor(255, 255, 255, max(16, 48 - (depth * 8))))

        painter.setPen(QPen(QColor(255, 255, 255, max(48, 118 - (depth * 16))), 1.15))
        painter.drawPath(shell)

        inner_rect = rect.adjusted(1.6, 1.6, -1.6, -1.6)
        inner = QPainterPath()
        inner.addRoundedRect(inner_rect, max(10.0, radius - 2.0), max(10.0, radius - 2.0))
        painter.setPen(QPen(QColor(tint.red(), tint.green(), tint.blue(), max(18, 56 - (depth * 10))), 0.85))
        painter.drawPath(inner)


class BurstEventFeed(QFrame):
    """Stacked event feed with bounded memory and overflow protection."""

    def __init__(
        self,
        parent=None,
        max_items: int = 8,
        *,
        role: str = "event",
        default_ttl_ms: int = 7800,
        overlay_stack: Optional[bool] = None,
    ):
        super().__init__(parent)
        self.setObjectName("BurstFeed")
        self.role = str(role or "event")
        self.setProperty("role", self.role)
        self._overlay_stack = (self.role == "log") if overlay_stack is None else bool(overlay_stack)
        self.setProperty("overlayStack", self._overlay_stack)
        self._max_items = max(3, int(max_items))
        self._hard_limit = max(self._max_items * 5, 32)
        self._default_ttl_ms = int(default_ttl_ms)
        self._items: List[_BurstItem] = []
        self._last_overflow_notice_ts = 0.0
        self._layout = None

        if not self._overlay_stack:
            lay = QVBoxLayout(self)
            lay.setContentsMargins(6, 6, 6, 6)
            lay.setSpacing(8)
            lay.addStretch(1)
            self._layout = lay

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reflow_items()

    def clear(self) -> None:
        for item in list(self._items):
            self._items.remove(item)
            self._finalize_item_removal(item)
        self._items.clear()

    def _finalize_item_removal(self, item: _BurstItem) -> None:
        try:
            item.stop_animations()
        except Exception:
            pass
        try:
            item.hide()
        except Exception:
            pass
        try:
            if self._layout is not None:
                self._layout.removeWidget(item)
        except Exception:
            pass
        item.deleteLater()
        self._reflow_items()

    def _schedule_item_removal(self, item: _BurstItem, *, animate: bool) -> None:
        if item not in self._items:
            return
        try:
            self._items.remove(item)
        except ValueError:
            return

        if animate:
            item.animate_out(on_finished=lambda w=item: self._finalize_item_removal(w))
            return
        self._finalize_item_removal(item)

    def _is_visually_overflowing(self) -> bool:
        if self._overlay_stack:
            return False
        if not self._items:
            return False
        height = int(self.height())
        if height <= 0:
            return False
        spacing = int(self._layout.spacing())
        visible_items = min(len(self._items), self._max_items + 6)
        expected_height = sum(
            max(36, int(self._items[i].sizeHint().height())) + spacing
            for i in range(visible_items)
        )
        return expected_height > int(height * 1.4)

    def _trim_to_capacity(self, *, animate: bool, limit: Optional[int] = None) -> None:
        target_limit = self._max_items if limit is None else max(1, int(limit))
        while len(self._items) > target_limit:
            old = self._items[-1]
            self._schedule_item_removal(old, animate=animate)

    def _maybe_push_overflow_notice(self) -> None:
        now = time.monotonic()
        if now - self._last_overflow_notice_ts < 6.0:
            return
        self._last_overflow_notice_ts = now
        self.push_event(
            "消息过多，较早的气泡已自动挤出。",
            level="warn",
            ttl_ms=6500,
            _internal=True,
        )

    def push_event(
        self,
        text: str,
        level: str = "info",
        ttl_ms: Optional[int] = 7800,
        _internal: bool = False,
    ) -> None:
        clean = str(text or "").strip()
        if not clean:
            return

        if not _internal and len(self._items) >= self._hard_limit:
            self._trim_to_capacity(animate=False, limit=self._max_items)
            self._maybe_push_overflow_notice()

        if not _internal and self._is_visually_overflowing() and len(self._items) > self._max_items:
            self._trim_to_capacity(animate=False)

        item = _BurstItem(clean, level, role=self.role, overlay_stack=self._overlay_stack, parent=self)
        if self._layout is not None:
            self._layout.insertWidget(0, item)
        self._items.insert(0, item)
        item.animate_in()
        self._reflow_items()

        resolved_ttl = self._default_ttl_ms if ttl_ms is None else int(ttl_ms)
        if resolved_ttl > 0:
            def _drop() -> None:
                self._schedule_item_removal(item, animate=True)

            QTimer.singleShot(max(1200, resolved_ttl), _drop)

        self._trim_to_capacity(animate=not self._is_visually_overflowing())

    def _remove_item(self, item: _BurstItem) -> None:
        # Backward-compat for old callbacks.
        self._schedule_item_removal(item, animate=False)

    def _overlay_visible_count(self) -> int:
        height = max(0, int(self.height()))
        if height < 120:
            return 1
        if height < 190:
            return 2
        if height < 280:
            return 3
        if height < 380:
            return 4
        return 5

    def _reflow_items(self) -> None:
        if not self._overlay_stack:
            return
        if not self._items:
            return

        frame_left = 8
        frame_top = 8
        frame_width = max(0, self.width() - 16)
        frame_height = max(0, self.height() - 16)
        if frame_width <= 0 or frame_height <= 0:
            return

        visible_count = min(len(self._items), self._overlay_visible_count())
        peek_y = 16
        side_step = 10
        front = self._items[0]
        front_width = frame_width
        desired_front_height = front.content_height_for_width(front_width - 26, body_lines=None)
        minimum_front_height = front.minimum_front_height_for_width(front_width - 26)
        while visible_count > 1:
            reserved_peek = max(0, visible_count - 1) * peek_y
            available_front = max(84, frame_height - reserved_peek)
            if available_front >= minimum_front_height:
                break
            visible_count -= 1
        reserved_peek = max(0, visible_count - 1) * peek_y
        available_front = max(72, frame_height - reserved_peek)
        front_height = min(frame_height, max(minimum_front_height, min(desired_front_height, available_front)))

        for idx, item in enumerate(self._items):
            if idx >= visible_count:
                item.hide()
                continue

            item._apply_log_stack_mode(idx, visible_count)
            inset = min(idx * side_step, 32)
            x = frame_left + inset
            y = frame_top + max(0, visible_count - idx - 1) * peek_y
            width = max(180, frame_width - inset * 2)

            if idx == 0:
                height = front_height
            else:
                height = item.compact_height_for_width(width - 24, stack_index=idx)
                height = max(44, min(height, 74 - (idx * 6)))

            max_height_here = max(36, frame_height - (max(0, visible_count - idx - 1) * peek_y))
            height = max(36, min(height, max_height_here))

            item.setGeometry(x, y, width, height)
            item.show()
            effect = item.graphicsEffect()
            if effect is not None:
                try:
                    if getattr(item, "_show_anim", None) is None:
                        effect.setOpacity(max(0.24, 1.0 - (idx * 0.19)))
                except Exception:
                    pass

        for item in reversed(self._items[:visible_count]):
            item.raise_()
