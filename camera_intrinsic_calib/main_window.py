from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .calibrator import (
    CalibrationResult,
    calibrate_camera,
    draw_detected_corners,
    save_calibration_json,
    undistort_image,
)
from .widgets import ImageCanvas


def cv_image_to_qimage(cv_img: np.ndarray) -> QImage:
    if len(cv_img.shape) == 2:
        h, w = cv_img.shape
        return QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
    h, w, c = cv_img.shape
    if c == 3:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    if c == 4:
        rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
        return QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
    h, w = cv_img.shape[:2]
    return QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)


def load_image_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    return img


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Camera Intrinsic Calibrator")
        self.resize(1600, 1020)

        self.image_paths: list[Path] = []
        self.detection_results: dict[str, object] = {}
        self.calibration_result: CalibrationResult | None = None
        self.selected_image_index: int = -1
        self._current_cv_image: np.ndarray | None = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setMinimumWidth(390)
        sidebar_scroll.setMaximumWidth(480)

        sidebar = QWidget()
        self.sidebar_layout = QVBoxLayout(sidebar)
        self.sidebar_layout.setAlignment(Qt.AlignTop)
        sidebar_scroll.setWidget(sidebar)
        root_layout.addWidget(sidebar_scroll)

        right_splitter = QSplitter(Qt.Vertical)
        root_layout.addWidget(right_splitter, 1)

        self.raw_canvas = ImageCanvas("原始图像（角点检测结果）")
        self.undistort_canvas = ImageCanvas("去畸变预览")

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(self._wrap_widget("原始图像", self.raw_canvas))
        top_splitter.addWidget(self._wrap_widget("去畸变预览", self.undistort_canvas))
        right_splitter.addWidget(top_splitter)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(160)
        right_splitter.addWidget(self.log_output)
        right_splitter.setSizes([600, 220])

        self._build_sidebar()

    def _build_sidebar(self) -> None:
        file_group = QGroupBox("数据源")
        file_layout = QFormLayout(file_group)
        self.image_dir_edit = QLineEdit()
        file_layout.addRow("图片文件夹", self._line_with_button(self.image_dir_edit, self._choose_image_dir))

        load_row = QHBoxLayout()
        self.load_images_button = QPushButton("加载图片")
        load_row.addWidget(self.load_images_button)
        file_layout.addRow(load_row)
        self.sidebar_layout.addWidget(file_group)

        pattern_group = QGroupBox("标定板参数")
        pattern_layout = QFormLayout(pattern_group)
        self.pattern_rows_spin = QSpinBox()
        self.pattern_rows_spin.setRange(2, 50)
        self.pattern_rows_spin.setValue(6)
        self.pattern_cols_spin = QSpinBox()
        self.pattern_cols_spin.setRange(2, 50)
        self.pattern_cols_spin.setValue(9)
        self.square_size_spin = self._make_double_spin(1.0, 1000.0, 25.0, 1.0, 2)
        self.pattern_type_combo = QComboBox()
        self.pattern_type_combo.addItem("棋盘格", "chessboard")
        self.pattern_type_combo.addItem("对称圆点格", "symmetric_circles")
        self.pattern_type_combo.addItem("非对称圆点格", "asymmetric_circles")

        pattern_layout.addRow("内角点行数", self.pattern_rows_spin)
        pattern_layout.addRow("内角点列数", self.pattern_cols_spin)
        pattern_layout.addRow("方格/圆点间距 (mm)", self.square_size_spin)
        pattern_layout.addRow("标定板类型", self.pattern_type_combo)
        self.sidebar_layout.addWidget(pattern_group)

        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        self.detect_button = QPushButton("检测角点")
        self.calibrate_button = QPushButton("运行标定")
        self.calibrate_button.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        self.undistort_button = QPushButton("预览去畸变")
        action_layout.addWidget(self.detect_button)
        action_layout.addWidget(self.calibrate_button)
        action_layout.addWidget(self.undistort_button)
        self.sidebar_layout.addWidget(action_group)

        list_group = QGroupBox("图片列表")
        list_layout = QVBoxLayout(list_group)
        self.image_table = QTableWidget(0, 3)
        self.image_table.setHorizontalHeaderLabels(["文件名", "状态", "重投影误差(px)"])
        self.image_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.image_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.image_table.setAlternatingRowColors(True)
        self.image_table.verticalHeader().setVisible(False)
        list_layout.addWidget(self.image_table)
        self.sidebar_layout.addWidget(list_group)

        result_group = QGroupBox("标定结果")
        result_layout = QVBoxLayout(result_group)
        self.result_output = QPlainTextEdit()
        self.result_output.setReadOnly(True)
        self.result_output.setMinimumHeight(220)
        result_layout.addWidget(self.result_output)

        save_row = QHBoxLayout()
        self.save_json_button = QPushButton("保存内参 JSON")
        save_row.addWidget(self.save_json_button)
        result_layout.addLayout(save_row)
        self.sidebar_layout.addWidget(result_group)

    def _connect_signals(self) -> None:
        self.load_images_button.clicked.connect(self._load_images)
        self.detect_button.clicked.connect(self._detect_corners)
        self.calibrate_button.clicked.connect(self._run_calibration)
        self.undistort_button.clicked.connect(self._preview_undistort)
        self.save_json_button.clicked.connect(self._save_calibration_json)
        self.image_table.itemSelectionChanged.connect(self._on_image_selection_changed)

    def _wrap_widget(self, title: str, widget: QWidget) -> QWidget:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _line_with_button(self, line_edit: QLineEdit, callback) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton("浏览")
        button.clicked.connect(callback)
        layout.addWidget(button)
        return wrapper

    def _make_double_spin(self, minimum: float, maximum: float, value: float, step: float, decimals: int):
        from PySide6.QtWidgets import QDoubleSpinBox
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        return spin

    def _choose_image_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", self.image_dir_edit.text() or str(Path.home()))
        if path:
            self.image_dir_edit.setText(path)

    def _load_images(self) -> None:
        image_dir = self.image_dir_edit.text().strip()
        if not image_dir:
            QMessageBox.warning(self, "缺少路径", "请先选择图片文件夹。")
            return

        dir_path = Path(image_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            QMessageBox.warning(self, "路径错误", "所选路径不存在或不是文件夹。")
            return

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        self.image_paths = sorted([p for p in dir_path.iterdir() if p.suffix.lower() in exts])
        if not self.image_paths:
            QMessageBox.warning(self, "无图片", "所选文件夹中没有支持的图片文件。")
            return

        self.detection_results.clear()
        self.calibration_result = None
        self._populate_image_table()
        self._append_log(f"加载完成: 共 {len(self.image_paths)} 张图片。")
        self.selected_image_index = 0
        self.image_table.selectRow(0)
        self._update_canvases()

    def _populate_image_table(self) -> None:
        self.image_table.setRowCount(len(self.image_paths))
        for row, path in enumerate(self.image_paths):
            self.image_table.setItem(row, 0, QTableWidgetItem(path.name))
            self.image_table.setItem(row, 1, QTableWidgetItem("未检测"))
            self.image_table.setItem(row, 2, QTableWidgetItem("--"))
        self.image_table.resizeColumnsToContents()

    def _on_image_selection_changed(self) -> None:
        items = self.image_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        if row != self.selected_image_index:
            self.selected_image_index = row
            self._update_canvases()

    def _get_pattern_params(self) -> tuple[tuple[int, int], float, Literal["chessboard", "symmetric_circles", "asymmetric_circles"]]:
        rows = self.pattern_rows_spin.value()
        cols = self.pattern_cols_spin.value()
        square_size = self.square_size_spin.value() / 1000.0  # mm to m
        pattern_type = self.pattern_type_combo.currentData()
        return (rows, cols), square_size, pattern_type

    def _detect_corners(self) -> None:
        if not self.image_paths:
            QMessageBox.warning(self, "无图片", "请先加载图片。")
            return

        pattern_size, _, pattern_type = self._get_pattern_params()

        self._append_log("开始检测角点...")
        success_count = 0
        from .calibrator import detect_chessboard_corners, detect_circle_grid_corners

        for row, path in enumerate(self.image_paths):
            if pattern_type == "chessboard":
                result = detect_chessboard_corners(path, pattern_size)
            elif pattern_type == "symmetric_circles":
                result = detect_circle_grid_corners(path, pattern_size, symmetric=True)
            else:
                result = detect_circle_grid_corners(path, pattern_size, symmetric=False)

            self.detection_results[path.name] = result

            status_item = QTableWidgetItem("成功" if result.success else "失败")
            status_item.setForeground(QColor(60, 200, 60) if result.success else QColor(255, 80, 80))
            self.image_table.setItem(row, 1, status_item)

            msg_item = QTableWidgetItem(result.message)
            self.image_table.setItem(row, 2, msg_item)

            if result.success:
                success_count += 1

        self._append_log(f"角点检测完成: {success_count}/{len(self.image_paths)} 张成功。")
        self._update_canvases()

    def _run_calibration(self) -> None:
        if not self.image_paths:
            QMessageBox.warning(self, "无图片", "请先加载图片。")
            return

        pattern_size, square_size, pattern_type = self._get_pattern_params()

        valid_paths = [p for p in self.image_paths if p.name in self.detection_results and self.detection_results[p.name].success]
        if not valid_paths:
            QMessageBox.warning(self, "未检测到角点", "请先运行角点检测，或检查标定板参数是否正确。")
            return

        self._append_log("开始标定...")
        try:
            result, all_results = calibrate_camera(valid_paths, pattern_size, square_size, pattern_type)
            self.calibration_result = result
            self._append_log(f"标定完成，RMS={result.rms_error:.4f}px")
            self._update_result_output()

            # Update per-image errors in table
            error_map = {name: err for name, err in result.per_image_errors}
            for row, path in enumerate(self.image_paths):
                if path.name in error_map:
                    err_item = QTableWidgetItem(f"{error_map[path.name]:.4f}")
                    self.image_table.setItem(row, 2, err_item)

        except Exception as exc:
            QMessageBox.critical(self, "标定失败", str(exc))
            self._append_log(f"标定失败: {exc}")

    def _preview_undistort(self) -> None:
        if self.calibration_result is None or self.calibration_result.camera_matrix is None:
            QMessageBox.warning(self, "无标定结果", "请先运行标定。")
            return

        if self.selected_image_index < 0 or self.selected_image_index >= len(self.image_paths):
            return

        path = self.image_paths[self.selected_image_index]
        img = load_image_bgr(path)
        if img is None:
            return

        camera_matrix = self.calibration_result.camera_matrix
        dist = np.array(self.calibration_result.distortion, dtype=np.float64)
        undistorted = undistort_image(img, camera_matrix, dist)
        self.undistort_canvas.set_image(cv_image_to_qimage(undistorted))
        self.undistort_canvas.set_status_lines([f"去畸变: {path.name}", f"尺寸: {undistorted.shape[1]}x{undistorted.shape[0]}"])

    def _update_canvases(self) -> None:
        if self.selected_image_index < 0 or self.selected_image_index >= len(self.image_paths):
            self.raw_canvas.set_image(QImage())
            self.undistort_canvas.set_image(QImage())
            self.raw_canvas.set_status_lines(["未选择图片"])
            self.undistort_canvas.set_status_lines(["未选择图片"])
            return

        path = self.image_paths[self.selected_image_index]
        img = load_image_bgr(path)
        if img is None:
            self.raw_canvas.set_image(QImage())
            self.raw_canvas.set_status_lines(["无法读取图像"])
            return

        self._current_cv_image = img

        # Draw detected corners if available
        markers: list[tuple[float, float, str, QColor]] = []
        result = self.detection_results.get(path.name)
        if result is not None and result.corners is not None:
            display_img = draw_detected_corners(img, result.corners)
            for i, corner in enumerate(result.corners):
                u, v = float(corner[0][0]), float(corner[0][1])
                if i < 5 or i == len(result.corners) - 1:
                    markers.append((u, v, str(i), QColor(0, 255, 128)))
        else:
            display_img = img

        self.raw_canvas.set_image(cv_image_to_qimage(display_img))

        status_lines = [f"文件: {path.name}", f"尺寸: {img.shape[1]}x{img.shape[0]}"]
        if result is not None:
            status_lines.append(f"状态: {result.message}")
        self.raw_canvas.set_markers(markers)
        self.raw_canvas.set_status_lines(status_lines)

        # Also update undistort if we have calibration result
        if self.calibration_result is not None and self.calibration_result.camera_matrix is not None:
            dist = np.array(self.calibration_result.distortion, dtype=np.float64)
            undistorted = undistort_image(img, self.calibration_result.camera_matrix, dist)
            self.undistort_canvas.set_image(cv_image_to_qimage(undistorted))
            self.undistort_canvas.set_status_lines([f"去畸变预览: {path.name}", f"尺寸: {undistorted.shape[1]}x{undistorted.shape[0]}"])
        else:
            self.undistort_canvas.set_image(QImage())
            self.undistort_canvas.set_status_lines(["尚未标定"])

    def _update_result_output(self) -> None:
        if self.calibration_result is None:
            self.result_output.setPlainText("尚未运行标定。")
            return

        r = self.calibration_result
        lines = []
        lines.append(f"fx={r.fx:.4f}")
        lines.append(f"fy={r.fy:.4f}")
        lines.append(f"cx={r.cx:.4f}")
        lines.append(f"cy={r.cy:.4f}")
        lines.append(f"width={r.width}")
        lines.append(f"height={r.height}")
        lines.append("")
        lines.append("distortion (k1,k2,p1,p2,k3,k4,k5,k6):")
        lines.append(", ".join(f"{v:.8f}" for v in r.distortion))
        lines.append("")
        lines.append(f"RMS reprojection error={r.rms_error:.6f} px")
        lines.append("")
        lines.append("per-image reprojection errors:")
        for name, err in r.per_image_errors:
            lines.append(f"  {name}: {err:.4f} px")
        lines.append("")
        lines.append("camera_matrix:")
        if r.camera_matrix is not None:
            lines.append(json.dumps(r.camera_matrix.tolist(), indent=2))

        self.result_output.setPlainText("\n".join(lines))

    def _save_calibration_json(self) -> None:
        if self.calibration_result is None:
            QMessageBox.warning(self, "无结果", "请先运行标定。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存内参 JSON", str(Path.home() / "camera_intrinsics.json"), "JSON (*.json)"
        )
        if not path:
            return
        try:
            save_calibration_json(Path(path), self.calibration_result)
            self._append_log(f"已保存内参到 {path}")
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    def _append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)
