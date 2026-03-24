from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


StepTuple = Tuple[QWidget, str, str]


class StepOnboardingOverlay(QWidget):
    """First-run overlay that highlights one target widget at a time."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("OnboardingOverlay")
        self.setAttribute(Qt.WA_StyledBackground, False)
        self.setAttribute(Qt.WA_NoSystemBackground, False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.hide()

        self._steps: List[StepTuple] = []
        self._index = 0
        self._finish_cb: Optional[Callable[[], None]] = None
        self._highlight_color = QColor("#70a9ff")

        # Keep overlay aligned while parent layout is recalculated during resize/screen scale changes.
        self._track_timer = QTimer(self)
        self._track_timer.setInterval(90)
        self._track_timer.timeout.connect(self._track_target_geometry)

        panel = QFrame(self)
        panel.setObjectName("OnboardingPanel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        self.title_label = QLabel(panel)
        self.title_label.setObjectName("OnboardingTitle")
        self.title_label.setWordWrap(True)
        lay.addWidget(self.title_label)

        self.body_label = QLabel(panel)
        self.body_label.setObjectName("OnboardingBody")
        self.body_label.setWordWrap(True)
        lay.addWidget(self.body_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.skip_btn = QPushButton("跳过", panel)
        self.skip_btn.setObjectName("GhostBtnMin")
        self.skip_btn.clicked.connect(self._finish)
        self.next_btn = QPushButton("下一步", panel)
        self.next_btn.setObjectName("PrimaryBtn")
        self.next_btn.clicked.connect(self._next_step)
        btn_row.addWidget(self.skip_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.next_btn)
        lay.addLayout(btn_row)

        self.panel = panel
        self.panel.resize(380, 180)

    def set_accent_color(self, color: QColor | str) -> None:
        try:
            parsed = QColor(color)
        except Exception:
            parsed = QColor("#70a9ff")
        if not parsed.isValid():
            parsed = QColor("#70a9ff")
        self._highlight_color = parsed
        self.update()

    def start(self, steps: List[StepTuple], on_finished: Optional[Callable[[], None]] = None) -> None:
        valid: List[StepTuple] = []
        for target, title, body in steps:
            if target is None:
                continue
            valid.append((target, title, body))
        if not valid:
            return
        self._steps = valid
        self._index = 0
        self._finish_cb = on_finished
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())
        self._refresh_step()
        self.show()
        self.raise_()
        self._track_timer.start()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_panel()

    def _track_target_geometry(self) -> None:
        if not self.isVisible():
            return
        parent = self.parentWidget()
        if parent is not None and self.geometry() != parent.rect():
            self.setGeometry(parent.rect())
        self._position_panel()
        self.update()

    def _current_target_rect(self) -> QRect:
        if not self._steps:
            return QRect()
        target = self._steps[self._index][0]
        if target is None or not target.isVisible():
            return QRect()
        global_top_left = target.mapToGlobal(target.rect().topLeft())
        local_top_left = self.mapFromGlobal(global_top_left)
        rect = QRect(local_top_left, target.rect().size())
        return rect

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        if not painter.isActive():
            return
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor(8, 12, 20, 138))

            target_rect = self._current_target_rect()
            if target_rect.isValid():
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(Qt.transparent, 1))
                painter.drawRoundedRect(target_rect, 12, 12)
                painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                ring = QColor(self._highlight_color)
                ring.setAlpha(255)
                painter.setPen(QPen(ring, 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(target_rect, 12, 12)
        finally:
            painter.end()

    def _position_panel(self) -> None:
        target_rect = self._current_target_rect()
        panel = self.panel
        margin = 18
        x = margin
        y = margin
        if target_rect.isValid():
            max_x = max(margin, self.width() - panel.width() - margin)
            max_y = max(margin, self.height() - panel.height() - margin)

            def clamp(value: int, low: int, high: int) -> int:
                if high <= low:
                    return low
                return max(low, min(high, int(value)))

            def panel_rect_at(raw_x: int, raw_y: int) -> QRect:
                px = clamp(raw_x, margin, max_x)
                py = clamp(raw_y, margin, max_y)
                return QRect(px, py, panel.width(), panel.height())

            target_x = target_rect.left()
            target_y = target_rect.top()
            target_mid_y = target_rect.center().y() - (panel.height() // 2)
            candidates = [
                panel_rect_at(target_x, target_rect.bottom() + margin),
                panel_rect_at(target_x, target_rect.top() - panel.height() - margin),
                panel_rect_at(target_rect.right() + margin, target_mid_y),
                panel_rect_at(target_rect.left() - panel.width() - margin, target_mid_y),
                panel_rect_at(target_x, target_y),
            ]

            for candidate in candidates:
                if not candidate.intersects(target_rect):
                    panel.move(candidate.topLeft())
                    return

            best_rect = candidates[0]
            best_overlap = None
            for candidate in candidates:
                overlap = candidate.intersected(target_rect)
                overlap_area = 0
                if overlap.isValid():
                    overlap_area = max(0, overlap.width()) * max(0, overlap.height())
                if best_overlap is None or overlap_area < best_overlap:
                    best_overlap = overlap_area
                    best_rect = candidate
            panel.move(best_rect.topLeft())
            return
        panel.move(int(x), int(y))

    def _refresh_step(self) -> None:
        if not self._steps:
            self._finish()
            return
        _, title, body = self._steps[self._index]
        self.title_label.setText(title)
        self.body_label.setText(body)
        if self._index >= len(self._steps) - 1:
            self.next_btn.setText("完成")
        else:
            self.next_btn.setText("下一步")
        self._position_panel()
        self.update()

    def _next_step(self) -> None:
        if self._index >= len(self._steps) - 1:
            self._finish()
            return
        self._index += 1
        self._refresh_step()

    def _finish(self) -> None:
        self._track_timer.stop()
        self.hide()
        cb = self._finish_cb
        self._finish_cb = None
        if cb is not None:
            try:
                cb()
            except Exception:
                pass
