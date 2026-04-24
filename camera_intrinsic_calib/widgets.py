from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget


class ImageCanvas(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._image = QImage()
        self._markers: list[tuple[float, float, str, QColor]] = []
        self._status_lines: list[str] = []
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._last_drag_pos: QPoint | None = None
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def set_markers(self, markers: list[tuple[float, float, str, QColor]]) -> None:
        self._markers = markers
        self.update()

    def set_status_lines(self, lines: list[str]) -> None:
        self._status_lines = lines
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def _image_rect_qrectf(self):
        from PySide6.QtCore import QRectF
        if self._image.isNull():
            return QRectF()
        width = max(1, self.width())
        height = max(1, self.height())
        scale = min(width / self._image.width(), height / self._image.height()) * self._zoom
        draw_w = self._image.width() * scale
        draw_h = self._image.height() * scale
        x = (width - draw_w) * 0.5 + self._pan.x()
        y = (height - draw_h) * 0.5 + self._pan.y()
        return QRectF(x, y, draw_w, draw_h)

    def image_to_widget(self, u: float, v: float) -> QPointF | None:
        rect = self._image_rect_qrectf()
        if rect.isNull():
            return None
        x = rect.left() + (u / self._image.width()) * rect.width()
        y = rect.top() + (v / self._image.height()) * rect.height()
        return QPointF(x, y)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 18))

        painter.setPen(QColor(240, 240, 240))
        painter.drawText(12, 22, self._title)

        if self._image.isNull():
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignCenter, "未加载图像")
            return

        rect = self._image_rect_qrectf()
        painter.drawImage(rect, self._image)

        # Draw markers (corner points)
        marker_pen = QPen(QColor(255, 255, 255))
        marker_pen.setWidth(2)
        painter.setPen(marker_pen)
        for u, v, label, color in self._markers:
            pos = self.image_to_widget(u, v)
            if pos is None:
                continue
            painter.setBrush(color)
            painter.drawEllipse(pos, 4.0, 4.0)
            painter.setPen(color)
            painter.drawText(pos + QPointF(6.0, -6.0), label)
            painter.setPen(marker_pen)

        if self._status_lines:
            painter.setPen(QColor(220, 220, 220))
            y = self.height() - 12 * len(self._status_lines) - 10
            for line in self._status_lines:
                painter.drawText(12, y, line)
                y += 14

    def wheelEvent(self, event) -> None:
        if self._image.isNull():
            return
        old_rect = self._image_rect_qrectf()
        zoom_factor = 1.12 if event.angleDelta().y() > 0 else 1.0 / 1.12
        self._zoom = max(0.2, min(20.0, self._zoom * zoom_factor))
        new_rect = self._image_rect_qrectf()
        if old_rect.isNull() or new_rect.isNull():
            self.update()
            return

        mouse_pos = QPointF(event.position())
        if old_rect.contains(mouse_pos):
            frac_x = (mouse_pos.x() - old_rect.left()) / old_rect.width()
            frac_y = (mouse_pos.y() - old_rect.top()) / old_rect.height()
            new_x = new_rect.left() + frac_x * new_rect.width()
            new_y = new_rect.top() + frac_y * new_rect.height()
            self._pan += QPointF(mouse_pos.x() - new_x, mouse_pos.y() - new_y)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            self._last_drag_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_drag_pos is None:
            return
        delta = event.pos() - self._last_drag_pos
        self._pan += QPointF(delta.x(), delta.y())
        self._last_drag_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            self._last_drag_pos = None
            self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, _event: QMouseEvent) -> None:
        self.reset_view()
