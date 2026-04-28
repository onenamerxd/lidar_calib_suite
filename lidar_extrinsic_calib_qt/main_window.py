from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
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
    QVBoxLayout,
    QWidget,
)

from .calibrator import (
    build_registration_stages,
    load_point_cloud,
    load_transform_matrix,
    matrix_to_xyz_quat,
    merge_point_clouds,
    register_multiscale,
    RegistrationResult,
    save_transform_matrix,
)
from .math_utils import compute_bounding_box, depth_to_rgb
from .widgets import PointCloud3DCanvas


class FullScreenPointCloudWindow(QDialog):
    def __init__(self, source_canvas: PointCloud3DCanvas, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("点云全屏显示")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = PointCloud3DCanvas("")
        self.canvas.copy_from(source_canvas)
        self.canvas.set_fullscreen_button_tooltip("退出全屏")
        self.canvas.fullScreenRequested.connect(self.close)
        layout.addWidget(self.canvas)

        self._escape_shortcut = QShortcut(QKeySequence("Esc"), self)
        self._escape_shortcut.activated.connect(self.close)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LiDAR Extrinsic Calibrator")
        self.resize(1600, 1020)

        self.target_cloud: o3d.geometry.PointCloud | None = None
        self.source_cloud: o3d.geometry.PointCloud | None = None
        self.init_transform: np.ndarray | None = None
        self.result_transform: np.ndarray | None = None
        self.result: RegistrationResult | None = None
        self.fullscreen_window: FullScreenPointCloudWindow | None = None

        self._build_ui()
        self._connect_signals()
        self._set_sample_paths_if_available()
        self._update_visuals()

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

        top_splitter = QSplitter(Qt.Horizontal)
        right_splitter.addWidget(top_splitter)

        self.before_3d = PointCloud3DCanvas("配准前 (初始变换)")
        self.after_3d = PointCloud3DCanvas("配准后 (优化结果)")

        top_splitter.addWidget(self._wrap_widget("配准前 3D", self.before_3d))
        top_splitter.addWidget(self._wrap_widget("配准后 3D", self.after_3d))

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(180)
        right_splitter.addWidget(self.log_output)
        right_splitter.setSizes([700, 260])

        self._build_sidebar()

    def _build_sidebar(self) -> None:
        file_group = QGroupBox("数据文件")
        file_layout = QFormLayout(file_group)
        self.target_edit = QLineEdit()
        self.source_edit = QLineEdit()
        self.init_edit = QLineEdit()
        file_layout.addRow("Target PCD", self._line_with_button(self.target_edit, self._choose_target_pcd))
        file_layout.addRow("Source PCD", self._line_with_button(self.source_edit, self._choose_source_pcd))
        file_layout.addRow("初始变换 JSON", self._line_with_button(self.init_edit, self._choose_init_json))
        file_button_row = QHBoxLayout()
        self.sample_button = QPushButton("填充样例路径")
        self.load_button = QPushButton("加载数据")
        file_button_row.addWidget(self.sample_button)
        file_button_row.addWidget(self.load_button)
        file_layout.addRow(file_button_row)
        self.sidebar_layout.addWidget(file_group)

        param_group = QGroupBox("配准参数")
        param_layout = QFormLayout(param_group)
        self.voxel_sizes_edit = QLineEdit("1.0,0.5,0.2,0.1")
        self.max_corr_edit = QLineEdit("3.0,1.5,0.8,0.4")
        self.max_iters_edit = QLineEdit("200,300,400,500")
        param_layout.addRow("Voxel sizes", self.voxel_sizes_edit)
        param_layout.addRow("Max corr", self.max_corr_edit)
        param_layout.addRow("Max iters", self.max_iters_edit)
        self.sidebar_layout.addWidget(param_group)

        prep_group = QGroupBox("预处理")
        prep_layout = QFormLayout(prep_group)
        self.crop_range_spin = self._make_double_spin(0.0, 500.0, 0.0, 0.5, 2)
        self.crop_range_spin.setSpecialValueText("无")
        self.z_min_spin = self._make_double_spin(-100.0, 100.0, -5.0, 0.1, 2)
        self.z_max_spin = self._make_double_spin(-100.0, 100.0, 5.0, 0.1, 2)
        self.method_combo = QComboBox()
        self.method_combo.addItem("point_to_plane", "point_to_plane")
        self.method_combo.addItem("point_to_point", "point_to_point")
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(1000, 500000)
        self.max_points_spin.setSingleStep(5000)
        self.max_points_spin.setValue(50000)
        self.color_by_height_checkbox = QCheckBox("按高度着色（关闭时 Target 红色 / Source 绿色）")
        self.color_by_height_checkbox.setChecked(False)
        prep_layout.addRow("XY 裁剪半径", self.crop_range_spin)
        prep_layout.addRow("Z 最小值", self.z_min_spin)
        prep_layout.addRow("Z 最大值", self.z_max_spin)
        prep_layout.addRow("估计方法", self.method_combo)
        prep_layout.addRow("最大显示点数", self.max_points_spin)
        prep_layout.addRow("", self.color_by_height_checkbox)
        self.sidebar_layout.addWidget(prep_group)

        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        self.run_button = QPushButton("运行配准")
        self.run_button.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        action_layout.addWidget(self.run_button)
        reset_row = QHBoxLayout()
        self.reset_before_view_button = QPushButton("重置左侧视角")
        self.reset_after_view_button = QPushButton("重置右侧视角")
        reset_row.addWidget(self.reset_before_view_button)
        reset_row.addWidget(self.reset_after_view_button)
        action_layout.addLayout(reset_row)
        self.sidebar_layout.addWidget(action_group)

        save_group = QGroupBox("结果保存")
        save_layout = QVBoxLayout(save_group)
        self.save_matrix_button = QPushButton("保存变换矩阵 JSON")
        self.save_aligned_button = QPushButton("保存对齐点云 PCD")
        self.save_merged_button = QPushButton("保存合并点云 PCD")
        save_layout.addWidget(self.save_matrix_button)
        save_layout.addWidget(self.save_aligned_button)
        save_layout.addWidget(self.save_merged_button)
        self.sidebar_layout.addWidget(save_group)

        output_group = QGroupBox("结果输出")
        output_layout = QVBoxLayout(output_group)
        self.result_output = QPlainTextEdit()
        self.result_output.setReadOnly(True)
        self.result_output.setMinimumHeight(220)
        output_layout.addWidget(self.result_output)
        self.sidebar_layout.addWidget(output_group)

    def _connect_signals(self) -> None:
        self.sample_button.clicked.connect(self._set_sample_paths_if_available)
        self.load_button.clicked.connect(self._load_data)
        self.run_button.clicked.connect(self._run_registration)

        self.reset_before_view_button.clicked.connect(self._reset_before_views)
        self.reset_after_view_button.clicked.connect(self._reset_after_views)
        self.before_3d.fullScreenRequested.connect(lambda: self._open_fullscreen_view(self.before_3d))
        self.after_3d.fullScreenRequested.connect(lambda: self._open_fullscreen_view(self.after_3d))

        self.save_matrix_button.clicked.connect(self._save_matrix)
        self.save_aligned_button.clicked.connect(self._save_aligned)
        self.save_merged_button.clicked.connect(self._save_merged)

        for widget in [
            self.crop_range_spin,
            self.z_min_spin,
            self.z_max_spin,
            self.max_points_spin,
        ]:
            widget.valueChanged.connect(self._update_visuals)

        self.color_by_height_checkbox.toggled.connect(self._update_visuals)
        self.method_combo.currentIndexChanged.connect(self._update_visuals)

    def _open_fullscreen_view(self, canvas: PointCloud3DCanvas) -> None:
        if self.fullscreen_window is not None:
            self.fullscreen_window.close()

        window = FullScreenPointCloudWindow(canvas, self)
        self.fullscreen_window = window
        window.finished.connect(lambda _result, closed_window=window: self._clear_fullscreen_window(closed_window))
        window.showFullScreen()

    def _clear_fullscreen_window(self, window: FullScreenPointCloudWindow) -> None:
        if self.fullscreen_window is window:
            self.fullscreen_window = None

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

    def _make_double_spin(self, minimum: float, maximum: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        return spin

    def _choose_target_pcd(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 Target PCD", self.target_edit.text() or str(Path.home()), "PCD (*.pcd)")
        if path:
            self.target_edit.setText(path)

    def _choose_source_pcd(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 Source PCD", self.source_edit.text() or str(Path.home()), "PCD (*.pcd)")
        if path:
            self.source_edit.setText(path)

    def _choose_init_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择初始变换 JSON", self.init_edit.text() or str(Path.home()), "JSON (*.json)")
        if path:
            self.init_edit.setText(path)

    def _set_sample_paths_if_available(self) -> None:
        base = Path("/home/rxd/lidar_extrinsic_calibrator/runs")
        target = base / "main.pcd"
        source = base / "left.pcd"
        init = base / "init_main_left.json"
        if target.exists():
            self.target_edit.setText(str(target))
        if source.exists():
            self.source_edit.setText(str(source))
        if init.exists():
            self.init_edit.setText(str(init))

    def _load_data(self) -> None:
        target_path = self.target_edit.text().strip()
        source_path = self.source_edit.text().strip()
        init_path = self.init_edit.text().strip()
        if not target_path or not source_path or not init_path:
            QMessageBox.warning(self, "缺少文件", "请先选择 Target PCD、Source PCD 和初始变换 JSON。")
            return

        try:
            self.target_cloud = load_point_cloud(Path(target_path))
            self.source_cloud = load_point_cloud(Path(source_path))
            self.init_transform = load_transform_matrix(Path(init_path))
            self.result_transform = None
            self.result = None
            self._append_log(f"加载完成: target={target_path}, source={source_path}, init={init_path}")
            self._update_visuals()
            self._fit_views()
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))

    def _fit_views(self) -> None:
        if self.target_cloud is None or self.source_cloud is None or self.init_transform is None:
            return
        target_pts = np.asarray(self.target_cloud.points)
        source_init = np.asarray(self.source_cloud.points).copy()
        ones = np.ones((source_init.shape[0], 1))
        homo = np.concatenate([source_init, ones], axis=1)
        source_init = (self.init_transform @ homo.T).T[:, :3]
        all_pts = np.concatenate([target_pts, source_init], axis=0)

        self.before_3d.fit_to_points(all_pts)
        self.after_3d.fit_to_points(all_pts)

    def _reset_before_views(self) -> None:
        if self.target_cloud is not None and self.source_cloud is not None and self.init_transform is not None:
            target_pts = np.asarray(self.target_cloud.points)
            source_init = np.asarray(self.source_cloud.points).copy()
            ones = np.ones((source_init.shape[0], 1))
            homo = np.concatenate([source_init, ones], axis=1)
            source_init = (self.init_transform @ homo.T).T[:, :3]
            all_pts = np.concatenate([target_pts, source_init], axis=0)
            self.before_3d.fit_to_points(all_pts)
        else:
            self.before_3d.reset_view()

    def _reset_after_views(self) -> None:
        if self.target_cloud is not None and self.source_cloud is not None:
            all_pts = np.concatenate([np.asarray(self.target_cloud.points), np.asarray(self.source_cloud.points)], axis=0)
            self.after_3d.fit_to_points(all_pts)
        else:
            self.after_3d.reset_view()

    def _run_registration(self) -> None:
        if self.target_cloud is None or self.source_cloud is None or self.init_transform is None:
            QMessageBox.warning(self, "数据未加载", "请先加载数据文件。")
            return

        try:
            stages = build_registration_stages(
                self.voxel_sizes_edit.text(),
                self.max_corr_edit.text(),
                self.max_iters_edit.text(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "参数错误", str(exc))
            return

        crop_range = self.crop_range_spin.value() if self.crop_range_spin.value() > 0 else None
        z_min = self.z_min_spin.value()
        z_max = self.z_max_spin.value()
        z_range = (z_min, z_max)
        method = self.method_combo.currentData()

        self._append_log("开始配准...")
        try:
            self.result = register_multiscale(
                target_cloud=self.target_cloud,
                source_cloud=self.source_cloud,
                init_transform=self.init_transform,
                stages=stages,
                crop_range=crop_range,
                z_range=z_range,
                estimation_method=method,
            )
            self.result_transform = self.result.transform
            self._append_log("配准完成。")
            self._update_result_output()
            self._update_visuals()
            self._reset_after_views()
        except Exception as exc:
            QMessageBox.critical(self, "配准失败", str(exc))

    def _update_result_output(self) -> None:
        if self.result is None or self.result_transform is None:
            self.result_output.setPlainText("尚未运行配准。")
            return

        xyz_quat = matrix_to_xyz_quat(self.result_transform)
        lines = []
        lines.append(f"fitness={self.result.fitness:.6f}")
        lines.append(f"inlier_rmse={self.result.inlier_rmse:.6f}")
        lines.append(f"converged={self.result.converged}")
        lines.append("")
        lines.append("stage_metrics:")
        for metric in self.result.stage_metrics:
            lines.append(
                "stage={stage} voxel={voxel_size:.3f} max_corr={max_correspondence_distance:.3f} "
                "iters={max_iterations} fitness={fitness:.6f} rmse={inlier_rmse:.6f}".format(**metric)
            )
        lines.append("")
        lines.append("transform (4x4):")
        lines.append(json.dumps(self.result_transform.tolist(), indent=2))
        lines.append("")
        lines.append(f"xyz_quat={json.dumps(xyz_quat)}")
        self.result_output.setPlainText("\n".join(lines))

    def _make_combined_arrays(
        self, target_cloud: o3d.geometry.PointCloud, source_cloud: o3d.geometry.PointCloud, transform: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        target_pts = np.asarray(target_cloud.points, dtype=np.float32)
        source_pts = np.asarray(source_cloud.points, dtype=np.float32)

        if transform is not None:
            ones = np.ones((source_pts.shape[0], 1), dtype=np.float32)
            homo = np.concatenate([source_pts, ones], axis=1)
            source_pts = (transform @ homo.T).T[:, :3].astype(np.float32)

        all_pts = np.concatenate([target_pts, source_pts], axis=0)

        if self.color_by_height_checkbox.isChecked():
            all_colors = depth_to_rgb(all_pts[:, 2])
        else:
            target_color = np.full((target_pts.shape[0], 3), [255, 90, 80], dtype=np.uint8)
            source_color = np.full((source_pts.shape[0], 3), [80, 255, 120], dtype=np.uint8)
            all_colors = np.concatenate([target_color, source_color], axis=0)

        max_points = self.max_points_spin.value()
        if all_pts.shape[0] > max_points:
            stride = max(1, int(np.ceil(all_pts.shape[0] / max_points)))
            all_pts = all_pts[::stride]
            all_colors = all_colors[::stride]

        return all_pts, all_colors

    def _update_visuals(self) -> None:
        has_data = self.target_cloud is not None and self.source_cloud is not None and self.init_transform is not None

        if not has_data:
            self.before_3d.set_points(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))
            self.after_3d.set_points(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))
            self.before_3d.set_status_lines(["未加载数据"])
            self.after_3d.set_status_lines(["未加载数据"])
            return

        before_pts, before_colors = self._make_combined_arrays(
            self.target_cloud, self.source_cloud, self.init_transform
        )
        after_transform = self.result_transform if self.result_transform is not None else self.init_transform
        after_pts, after_colors = self._make_combined_arrays(
            self.target_cloud, self.source_cloud, after_transform
        )

        self.before_3d.set_points(before_pts, before_colors)
        self.after_3d.set_points(after_pts, after_colors)

        target_count = len(self.target_cloud.points)
        source_count = len(self.source_cloud.points)

        self.before_3d.set_status_lines([f"Target(红): {target_count} 点", f"Source(绿): {source_count} 点"])

        if self.result_transform is not None:
            self.after_3d.set_status_lines([f"Target(红): {target_count} 点", f"Aligned(绿): {source_count} 点", "配准已完成"])
        else:
            self.after_3d.set_status_lines([f"Target(红): {target_count} 点", f"Aligned(绿): {source_count} 点", "尚未配准"])

    def _save_matrix(self) -> None:
        if self.result_transform is None:
            QMessageBox.warning(self, "无结果", "请先运行配准。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存变换矩阵", str(Path.home() / "T_main_source.json"), "JSON (*.json)")
        if not path:
            return
        try:
            save_transform_matrix(Path(path), self.result_transform)
            self._append_log(f"已保存变换矩阵: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    def _save_aligned(self) -> None:
        if self.result_transform is None or self.source_cloud is None:
            QMessageBox.warning(self, "无结果", "请先运行配准。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存对齐点云", str(Path.home() / "aligned.pcd"), "PCD (*.pcd)")
        if not path:
            return
        try:
            aligned = o3d.geometry.PointCloud(self.source_cloud)
            aligned.transform(self.result_transform)
            o3d.io.write_point_cloud(str(path), aligned)
            self._append_log(f"已保存对齐点云: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    def _save_merged(self) -> None:
        if self.result_transform is None or self.target_cloud is None or self.source_cloud is None:
            QMessageBox.warning(self, "无结果", "请先运行配准。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存合并点云", str(Path.home() / "merged.pcd"), "PCD (*.pcd)")
        if not path:
            return
        try:
            merged = merge_point_clouds(self.target_cloud, self.source_cloud, self.result_transform)
            o3d.io.write_point_cloud(str(path), merged)
            self._append_log(f"已保存合并点云: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    def _append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)
