from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget


class ImageCanvas(QWidget):
    pointPicked = Signal(float, float)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._image = QImage()
        self._overlay_points = np.zeros((0, 2), dtype=np.float64)
        self._overlay_colors = np.zeros((0, 3), dtype=np.uint8)
        self._markers: list[tuple[float, float, str, QColor]] = []
        self._status_lines: list[str] = []
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._last_drag_pos: QPoint | None = None
        self._pick_enabled = False
        self._flip_mode = 0
        self.setMinimumSize(400, 280)
        self.setMouseTracking(True)

    def set_pick_enabled(self, enabled: bool) -> None:
        self._pick_enabled = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def set_status_lines(self, lines: list[str]) -> None:
        self._status_lines = lines
        self.update()

    def set_flip_mode(self, flip_mode: int) -> None:
        self._flip_mode = flip_mode
        self.update()

    def _apply_flip_to_image(self, image: QImage) -> QImage:
        if self._flip_mode == 1:
            return image.mirrored(True, False)
        if self._flip_mode == 2:
            return image.mirrored(False, True)
        if self._flip_mode == 3:
            return image.mirrored(True, True)
        return image

    def set_image(self, image: QImage) -> None:
        self._image = self._apply_flip_to_image(image)
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def set_overlay(self, points_uv: np.ndarray, colors_rgb: np.ndarray) -> None:
        self._overlay_points = np.asarray(points_uv, dtype=np.float64)
        self._overlay_colors = np.asarray(colors_rgb, dtype=np.uint8)
        self.update()

    def set_markers(self, markers: list[tuple[float, float, str, QColor]]) -> None:
        self._markers = markers
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def _image_rect(self) -> QRectF:
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

    def widget_to_image(self, widget_pos: QPointF) -> QPointF | None:
        rect = self._image_rect()
        if rect.isNull() or not rect.contains(widget_pos):
            return None
        u = (widget_pos.x() - rect.left()) / rect.width() * self._image.width()
        v = (widget_pos.y() - rect.top()) / rect.height() * self._image.height()
        return QPointF(u, v)

    def image_to_widget(self, u: float, v: float) -> QPointF | None:
        rect = self._image_rect()
        if rect.isNull():
            return None
        x = rect.left() + (u / self._image.width()) * rect.width()
        y = rect.top() + (v / self._image.height()) * rect.height()
        return QPointF(x, y)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 18))

        title_pen = QPen(QColor(240, 240, 240))
        painter.setPen(title_pen)
        painter.drawText(12, 22, self._title)

        if self._image.isNull():
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignCenter, "未加载图像")
            return

        rect = self._image_rect()
        painter.drawImage(rect, self._image)

        if self._overlay_points.size > 0:
            for index, (u, v) in enumerate(self._overlay_points):
                pos = self.image_to_widget(float(u), float(v))
                if pos is None:
                    continue
                color = QColor(*[int(c) for c in self._overlay_colors[index]])
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(pos, 2.0, 2.0)

        marker_pen = QPen(QColor(255, 255, 255))
        marker_pen.setWidth(2)
        painter.setPen(marker_pen)
        for u, v, label, color in self._markers:
            pos = self.image_to_widget(u, v)
            if pos is None:
                continue
            painter.setBrush(color)
            painter.drawEllipse(pos, 5.0, 5.0)
            painter.setPen(color)
            painter.drawText(pos + QPointF(8.0, -8.0), label)
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
        old_rect = self._image_rect()
        image_pos = self.widget_to_image(QPointF(event.position()))
        if image_pos is None:
            self._zoom *= 1.1 if event.angleDelta().y() > 0 else 1.0 / 1.1
            self._zoom = max(0.2, min(20.0, self._zoom))
            self.update()
            return

        zoom_factor = 1.12 if event.angleDelta().y() > 0 else 1.0 / 1.12
        self._zoom = max(0.2, min(20.0, self._zoom * zoom_factor))
        new_rect = self._image_rect()
        if old_rect.isNull() or new_rect.isNull():
            self.update()
            return

        old_x = old_rect.left() + image_pos.x() / self._image.width() * old_rect.width()
        old_y = old_rect.top() + image_pos.y() / self._image.height() * old_rect.height()
        new_x = new_rect.left() + image_pos.x() / self._image.width() * new_rect.width()
        new_y = new_rect.top() + image_pos.y() / self._image.height() * new_rect.height()
        self._pan += QPointF(old_x - new_x, old_y - new_y)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            self._last_drag_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton and self._pick_enabled:
            image_pos = self.widget_to_image(QPointF(event.position()))
            if image_pos is not None:
                self.pointPicked.emit(float(image_pos.x()), float(image_pos.y()))

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
            self.setCursor(Qt.CrossCursor if self._pick_enabled else Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, _event: QMouseEvent) -> None:
        self.reset_view()


class PointCloudBevCanvas(QWidget):
    pointPicked = Signal(float, float, float)
    regionSelected = Signal(float, float, float, float)  # x_min, x_max, y_min, y_max

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._points = np.zeros((0, 3), dtype=np.float32)
        self._colors = np.zeros((0, 3), dtype=np.uint8)
        self._markers: list[tuple[float, float, float, str, QColor]] = []
        self._status_lines: list[str] = []
        self._ranges = {"x_min": -10.0, "x_max": 60.0, "y_min": -30.0, "y_max": 30.0}
        self._frustum: list[np.ndarray] = []
        self._pick_enabled = False
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._last_drag_pos: QPoint | None = None
        self._is_selecting = False
        self._selection_start = QPointF()
        self._selection_end = QPointF()
        self.setMinimumSize(400, 280)
        self.setMouseTracking(True)

    def set_pick_enabled(self, enabled: bool) -> None:
        self._pick_enabled = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def set_status_lines(self, lines: list[str]) -> None:
        self._status_lines = lines
        self.update()

    def set_ranges(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        self._ranges = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
        self.update()

    def set_points(self, points_xyz: np.ndarray, colors_rgb: np.ndarray) -> None:
        self._points = np.asarray(points_xyz, dtype=np.float32)
        self._colors = np.asarray(colors_rgb, dtype=np.uint8)
        self.update()

    def set_markers(self, markers: list[tuple[float, float, float, str, QColor]]) -> None:
        self._markers = markers
        self.update()

    def set_frustum(self, corners: list[np.ndarray]) -> None:
        self._frustum = corners
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def _base_world_to_widget(self, x: float, y: float) -> QPointF:
        x_min = self._ranges["x_min"]
        x_max = self._ranges["x_max"]
        y_min = self._ranges["y_min"]
        y_max = self._ranges["y_max"]
        width = max(1.0, float(self.width()))
        height = max(1.0, float(self.height()))
        sx = (y_max - y) / max(1e-6, (y_max - y_min)) * width
        sy = (x_max - x) / max(1e-6, (x_max - x_min)) * height
        return QPointF(sx, sy)

    def _world_to_widget(self, x: float, y: float) -> QPointF:
        base = self._base_world_to_widget(x, y)
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        delta = base - center
        scaled = QPointF(delta.x() * self._zoom, delta.y() * self._zoom)
        return center + scaled + self._pan

    def _widget_to_world(self, point: QPointF) -> tuple[float, float]:
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        delta = point - center - self._pan
        base = QPointF(delta.x() / self._zoom, delta.y() / self._zoom) + center
        x_min = self._ranges["x_min"]
        x_max = self._ranges["x_max"]
        y_min = self._ranges["y_min"]
        y_max = self._ranges["y_max"]
        x = x_max - base.y() / max(1.0, float(self.height())) * (x_max - x_min)
        y = y_max - base.x() / max(1.0, float(self.width())) * (y_max - y_min)
        return float(x), float(y)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 18))
        painter.setPen(QColor(240, 240, 240))
        painter.drawText(12, 22, self._title)

        if self._points.shape[0] == 0:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignCenter, "未加载点云")
            return

        axis_pen = QPen(QColor(80, 80, 80))
        axis_pen.setWidth(1)
        painter.setPen(axis_pen)
        y0 = self._world_to_widget(0.0, 0.0).y()
        x0 = self._world_to_widget(0.0, 0.0).x()
        painter.drawLine(0, int(y0), self.width(), int(y0))
        painter.drawLine(int(x0), 0, int(x0), self.height())

        for point, color in zip(self._points, self._colors):
            pos = self._world_to_widget(float(point[0]), float(point[1]))
            painter.setPen(QColor(int(color[0]), int(color[1]), int(color[2])))
            painter.drawPoint(int(pos.x()), int(pos.y()))

        if self._frustum:
            camera_origin = self._frustum[0]
            origin_pos = self._world_to_widget(float(camera_origin[0]), float(camera_origin[1]))
            painter.setPen(QPen(QColor(255, 200, 0), 2))
            painter.setBrush(QColor(255, 200, 0))
            painter.drawEllipse(origin_pos, 4.0, 4.0)
            for corner in self._frustum[1:]:
                corner_pos = self._world_to_widget(float(corner[0]), float(corner[1]))
                painter.drawLine(origin_pos, corner_pos)
            if len(self._frustum) == 5:
                painter.drawLine(self._world_to_widget(float(self._frustum[1][0]), float(self._frustum[1][1])), self._world_to_widget(float(self._frustum[2][0]), float(self._frustum[2][1])))
                painter.drawLine(self._world_to_widget(float(self._frustum[2][0]), float(self._frustum[2][1])), self._world_to_widget(float(self._frustum[3][0]), float(self._frustum[3][1])))
                painter.drawLine(self._world_to_widget(float(self._frustum[3][0]), float(self._frustum[3][1])), self._world_to_widget(float(self._frustum[4][0]), float(self._frustum[4][1])))
                painter.drawLine(self._world_to_widget(float(self._frustum[4][0]), float(self._frustum[4][1])), self._world_to_widget(float(self._frustum[1][0]), float(self._frustum[1][1])))

        for x, y, _z, label, color in self._markers:
            pos = self._world_to_widget(x, y)
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            painter.drawEllipse(pos, 5.0, 5.0)
            painter.drawText(pos + QPointF(8.0, -8.0), label)

        if self._is_selecting:
            rect = QRectF(self._selection_start, self._selection_end).normalized()
            painter.setPen(QPen(QColor(0, 255, 128), 2, Qt.DashLine))
            painter.setBrush(QColor(0, 255, 128, 40))
            painter.drawRect(rect)

        if self._status_lines:
            painter.setPen(QColor(220, 220, 220))
            y = self.height() - 12 * len(self._status_lines) - 10
            for line in self._status_lines:
                painter.drawText(12, y, line)
                y += 14

    def wheelEvent(self, event) -> None:
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        old_zoom = self._zoom
        self._zoom = max(0.1, min(20.0, self._zoom * zoom_factor))
        actual_factor = self._zoom / old_zoom
        mouse_pos = QPointF(event.position())
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        self._pan = center + (self._pan - center + mouse_pos - center) * actual_factor - (mouse_pos - center)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            self._last_drag_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton and self._pick_enabled and self._points.shape[0] > 0:
            x_world, y_world = self._widget_to_world(QPointF(event.position()))
            screen_points = np.array(
                [[self._world_to_widget(float(point[0]), float(point[1])).x(), self._world_to_widget(float(point[0]), float(point[1])).y()] for point in self._points],
                dtype=np.float64,
            )
            click = np.array([event.position().x(), event.position().y()], dtype=np.float64)
            distances = np.linalg.norm(screen_points - click[None, :], axis=1)
            nearest_index = int(np.argmin(distances))
            if distances[nearest_index] <= 24.0:
                selected = self._points[nearest_index]
                self.pointPicked.emit(float(selected[0]), float(selected[1]), float(selected[2]))
            return

        if event.button() == Qt.LeftButton and not self._pick_enabled:
            self._is_selecting = True
            self._selection_start = QPointF(event.position())
            self._selection_end = QPointF(event.position())
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_drag_pos is not None:
            delta = event.pos() - self._last_drag_pos
            self._pan += QPointF(delta.x(), delta.y())
            self._last_drag_pos = event.pos()
            self.update()
            return

        if self._is_selecting:
            self._selection_end = QPointF(event.position())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            self._last_drag_pos = None
            self.setCursor(Qt.CrossCursor if self._pick_enabled else Qt.ArrowCursor)
            return

        if event.button() == Qt.LeftButton and self._is_selecting:
            self._is_selecting = False
            start_world = self._widget_to_world(self._selection_start)
            end_world = self._widget_to_world(self._selection_end)
            x_min = min(start_world[0], end_world[0])
            x_max = max(start_world[0], end_world[0])
            y_min = min(start_world[1], end_world[1])
            y_max = max(start_world[1], end_world[1])
            if abs(x_max - x_min) > 1e-3 and abs(y_max - y_min) > 1e-3:
                self.regionSelected.emit(x_min, x_max, y_min, y_max)
            self.update()

    def mouseDoubleClickEvent(self, _event: QMouseEvent) -> None:
        self.reset_view()


class PointCloud3DCanvas(QWidget):
    pointPicked = Signal(float, float, float)
    viewChanged = Signal()

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._points = np.zeros((0, 3), dtype=np.float32)
        self._colors = np.zeros((0, 3), dtype=np.uint8)
        self._markers: list[tuple[float, float, float, str, QColor]] = []
        self._status_lines: list[str] = []
        self._pick_enabled = False

        # Camera orbit parameters
        self.azimuth_deg = 45.0
        self.elevation_deg = 30.0
        self.distance = 60.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.fov_deg = 60.0

        self._last_drag_pos: QPoint | None = None
        self._drag_mode: str | None = None  # "rotate" or "pan"
        self.setMinimumSize(400, 280)
        self.setMouseTracking(True)

    def set_pick_enabled(self, enabled: bool) -> None:
        self._pick_enabled = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def set_status_lines(self, lines: list[str]) -> None:
        self._status_lines = lines
        self.update()

    def set_points(self, points_xyz: np.ndarray, colors_rgb: np.ndarray) -> None:
        self._points = np.asarray(points_xyz, dtype=np.float32)
        self._colors = np.asarray(colors_rgb, dtype=np.uint8)
        self.update()

    def set_markers(self, markers: list[tuple[float, float, float, str, QColor]]) -> None:
        self._markers = markers
        self.update()

    def get_view_state(self) -> dict:
        return {
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "distance": self.distance,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y,
        }

    def set_view_state(self, state: dict) -> None:
        self.azimuth_deg = float(state.get("azimuth_deg", self.azimuth_deg))
        self.elevation_deg = float(state.get("elevation_deg", self.elevation_deg))
        self.distance = float(state.get("distance", self.distance))
        self.pan_x = float(state.get("pan_x", self.pan_x))
        self.pan_y = float(state.get("pan_y", self.pan_y))
        self.update()

    def reset_view(self) -> None:
        self.azimuth_deg = 45.0
        self.elevation_deg = 30.0
        self.distance = 60.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.viewChanged.emit()
        self.update()

    def _camera_pos(self) -> np.ndarray:
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elevation_deg)
        dx = math.cos(el) * math.cos(az)
        dy = math.cos(el) * math.sin(az)
        dz = math.sin(el)
        return np.array([self.pan_x + self.distance * dx, self.pan_y + self.distance * dy, self.distance * dz], dtype=np.float64)

    def _look_at_matrix(self) -> np.ndarray:
        cam = self._camera_pos()
        target = np.array([self.pan_x, self.pan_y, 0.0], dtype=np.float64)
        forward = target - cam
        norm = np.linalg.norm(forward)
        if norm < 1e-12:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            forward /= norm
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(forward, world_up)) > 0.9999:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        rot = np.array(
            [
                [right[0], right[1], right[2], 0.0],
                [up[0], up[1], up[2], 0.0],
                [-forward[0], -forward[1], -forward[2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        trans = np.array(
            [
                [1.0, 0.0, 0.0, -cam[0]],
                [0.0, 1.0, 0.0, -cam[1]],
                [0.0, 0.0, 1.0, -cam[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return rot @ trans

    def _perspective_matrix(self) -> np.ndarray:
        aspect = max(1.0, float(self.width())) / max(1.0, float(self.height()))
        f = 1.0 / math.tan(math.radians(self.fov_deg) * 0.5)
        near = 0.1
        far = 200.0
        return np.array(
            [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )

    def _project_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if points.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        mvp = self._perspective_matrix() @ self._look_at_matrix()
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        homo = np.concatenate([points, ones], axis=1)
        clip = (mvp @ homo.T).T
        w = clip[:, 3]
        valid = np.abs(w) > 1e-9
        ndc = np.zeros_like(clip[:, :2])
        ndc[valid] = clip[valid, :2] / w[valid, None]
        width = max(1.0, float(self.width()))
        height = max(1.0, float(self.height()))
        screen = np.zeros_like(ndc)
        screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * width
        screen[:, 1] = (0.5 - ndc[:, 1] * 0.5) * height
        # camera-space z for sorting (view matrix only)
        view = self._look_at_matrix()
        cam_z = (view @ homo.T).T[:, 2]
        return screen, cam_z

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 18))
        painter.setPen(QColor(240, 240, 240))
        painter.drawText(12, 22, self._title)

        if self._points.shape[0] == 0:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignCenter, "未加载点云")
            return

        screen_pts, cam_z = self._project_points(self._points)
        sort_idx = np.argsort(cam_z)

        for idx in sort_idx:
            sx, sy = screen_pts[idx]
            if not (0 <= sx <= self.width() and 0 <= sy <= self.height()):
                continue
            color = self._colors[idx]
            painter.setPen(QColor(int(color[0]), int(color[1]), int(color[2])))
            painter.drawPoint(int(sx), int(sy))

        # Draw axes
        axis_points = np.array(
            [
                [0.0, 0.0, 0.0], [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0], [0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0], [0.0, 0.0, 5.0],
            ],
            dtype=np.float32,
        )
        axis_screen, _ = self._project_points(axis_points)
        axis_colors = [QColor(255, 60, 60), QColor(60, 255, 60), QColor(60, 120, 255)]
        for i in range(3):
            if i * 2 + 1 < axis_screen.shape[0]:
                painter.setPen(QPen(axis_colors[i], 2))
                p0 = QPointF(axis_screen[i * 2, 0], axis_screen[i * 2, 1])
                p1 = QPointF(axis_screen[i * 2 + 1, 0], axis_screen[i * 2 + 1, 1])
                painter.drawLine(p0, p1)

        # Draw markers
        if self._markers:
            marker_pts = np.array([[m[0], m[1], m[2]] for m in self._markers], dtype=np.float32)
            marker_screen, _ = self._project_points(marker_pts)
            for i, (x, y, z, label, color) in enumerate(self._markers):
                if i >= marker_screen.shape[0]:
                    continue
                sx, sy = marker_screen[i]
                if not (0 <= sx <= self.width() and 0 <= sy <= self.height()):
                    continue
                pos = QPointF(sx, sy)
                painter.setPen(QPen(color, 2))
                painter.setBrush(color)
                painter.drawEllipse(pos, 5.0, 5.0)
                painter.drawText(pos + QPointF(8.0, -8.0), label)

        if self._status_lines:
            painter.setPen(QColor(220, 220, 220))
            y = self.height() - 12 * len(self._status_lines) - 10
            for line in self._status_lines:
                painter.drawText(12, y, line)
                y += 14

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.distance = max(1.0, min(500.0, self.distance * factor))
        self.viewChanged.emit()
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            self._last_drag_pos = event.pos()
            self._drag_mode = "pan"
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton and self._pick_enabled and self._points.shape[0] > 0:
            screen_pts, _ = self._project_points(self._points)
            click = np.array([event.position().x(), event.position().y()], dtype=np.float64)
            valid = (screen_pts[:, 0] >= 0) & (screen_pts[:, 0] <= self.width()) & (screen_pts[:, 1] >= 0) & (screen_pts[:, 1] <= self.height())
            if np.any(valid):
                distances = np.linalg.norm(screen_pts[valid] - click[None, :], axis=1)
                nearest_local = int(np.argmin(distances))
                if distances[nearest_local] <= 24.0:
                    indices = np.where(valid)[0]
                    selected = self._points[indices[nearest_local]]
                    self.pointPicked.emit(float(selected[0]), float(selected[1]), float(selected[2]))
            return

        if event.button() == Qt.LeftButton and not self._pick_enabled:
            self._last_drag_pos = event.pos()
            self._drag_mode = "rotate"

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_drag_pos is None or self._drag_mode is None:
            return
        delta = event.pos() - self._last_drag_pos
        if self._drag_mode == "rotate":
            self.azimuth_deg -= delta.x() * 0.3
            self.elevation_deg += delta.y() * 0.3
            self.elevation_deg = max(-89.0, min(89.0, self.elevation_deg))
        elif self._drag_mode == "pan":
            scale = self.distance * math.tan(math.radians(self.fov_deg) * 0.5) * 2.0 / max(1.0, float(self.height()))
            dx = -delta.x() * scale
            dy = -delta.y() * scale
            cam = self._camera_pos()
            target = np.array([self.pan_x, self.pan_y, 0.0], dtype=np.float64)
            forward = target - cam
            norm = np.linalg.norm(forward)
            if norm > 1e-12:
                forward /= norm
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(np.dot(forward, world_up)) > 0.9999:
                world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            right = np.cross(forward, world_up)
            rnorm = np.linalg.norm(right)
            if rnorm > 1e-12:
                right /= rnorm
            up = np.cross(right, forward)
            unorm = np.linalg.norm(up)
            if unorm > 1e-12:
                up /= unorm
            self.pan_x += dx * right[0] + dy * up[0]
            self.pan_y += dx * right[1] + dy * up[1]
        self._last_drag_pos = event.pos()
        self.viewChanged.emit()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton or event.button() == Qt.LeftButton:
            self._last_drag_pos = None
            self._drag_mode = None
            self.setCursor(Qt.CrossCursor if self._pick_enabled else Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, _event: QMouseEvent) -> None:
        self.reset_view()

