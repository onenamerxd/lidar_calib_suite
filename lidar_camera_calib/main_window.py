from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .io_utils import build_frame_pairs, load_camera_json, load_pcd, load_qimage
from .math_utils import (
    export_extrinsics,
    camera_frustum_in_lidar,
    depth_to_rgb,
    project_lidar_to_image,
    solve_extrinsics_from_correspondences,
)
from .models import CalibrationCorrespondence, CameraIntrinsics, Extrinsics, FramePair, PointCloudData
from .settings_store import load_settings, save_settings
from .widgets import ImageCanvas, PointCloud3DCanvas, PointCloudBevCanvas


EXPORT_DIRECTION_OPTIONS = [
    ("LiDAR -> Camera", "lidar_to_camera"),
    ("Camera -> LiDAR", "camera_to_lidar"),
]

LIDAR_AXIS_OPTIONS = [
    ("Apollo LS LiDAR / RFU (x右 y前 z上)", "apollo_lslidar_rfu"),
]


class FullScreenProjectionWindow(QDialog):
    extrinsicsChanged = Signal(object)

    def __init__(
        self,
        source_canvas: ImageCanvas,
        extrinsics: Extrinsics,
        step_index: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("点云投影全屏微调")
        self._syncing = False

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.canvas = ImageCanvas("")
        self.canvas.copy_from(source_canvas)
        self.canvas.set_fullscreen_enabled(True)
        self.canvas.set_fullscreen_button_tooltip("退出全屏")
        self.canvas.fullScreenRequested.connect(self.close)
        root_layout.addWidget(self.canvas, 1)

        self.panel = QWidget()
        self.panel.setFixedWidth(330)
        self.panel.setStyleSheet(
            "QWidget { background: #202020; color: #f0f0f0; }"
            "QGroupBox { border: 1px solid #444; margin-top: 10px; padding-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
            "QDoubleSpinBox, QComboBox { background: #2b2b2b; border: 1px solid #555; padding: 4px; }"
            "QPushButton { background: #303030; border: 1px solid #666; padding: 7px; }"
            "QPushButton:hover { background: #3b3b3b; }"
        )
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(14, 14, 14, 14)
        panel_layout.setSpacing(10)

        title = QLabel("外参实时微调")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        panel_layout.addWidget(title)

        self.step_mode_combo = QComboBox()
        self.step_mode_combo.addItem("精细 (T:0.001, R:0.01 deg)", (0.001, 0.01))
        self.step_mode_combo.addItem("普通 (T:0.01, R:0.1 deg)", (0.01, 0.1))
        self.step_mode_combo.addItem("粗略 (T:0.1, R:1 deg)", (0.1, 1.0))
        self.step_mode_combo.setCurrentIndex(max(0, min(step_index, self.step_mode_combo.count() - 1)))

        self.transform_group = QGroupBox("LiDAR -> Camera")
        transform_layout = QFormLayout(self.transform_group)
        self.tx_spin = self._make_double_spin(-100.0, 100.0, 0.0, 0.01, 5)
        self.ty_spin = self._make_double_spin(-100.0, 100.0, 0.0, 0.01, 5)
        self.tz_spin = self._make_double_spin(-100.0, 100.0, 0.0, 0.01, 5)
        self.roll_spin = self._make_double_spin(-180.0, 180.0, 0.0, 0.1, 5)
        self.pitch_spin = self._make_double_spin(-180.0, 180.0, 0.0, 0.1, 5)
        self.yaw_spin = self._make_double_spin(-180.0, 180.0, 0.0, 0.1, 5)
        transform_layout.addRow("调节步长", self.step_mode_combo)
        transform_layout.addRow("x / tx (m)", self.tx_spin)
        transform_layout.addRow("y / ty (m)", self.ty_spin)
        transform_layout.addRow("z / tz (m)", self.tz_spin)
        transform_layout.addRow("roll (deg)", self.roll_spin)
        transform_layout.addRow("pitch (deg)", self.pitch_spin)
        transform_layout.addRow("yaw (deg)", self.yaw_spin)
        panel_layout.addWidget(self.transform_group)

        self.projection_label = QLabel("投影状态: --")
        self.projection_label.setWordWrap(True)
        panel_layout.addWidget(self.projection_label)
        panel_layout.addStretch(1)

        self.reset_view_button = QPushButton("重置视图")
        self.close_button = QPushButton("退出全屏")
        panel_layout.addWidget(self.reset_view_button)
        panel_layout.addWidget(self.close_button)
        root_layout.addWidget(self.panel)

        self._set_extrinsics_controls(extrinsics)
        self._on_step_mode_changed(self.step_mode_combo.currentIndex())
        self._connect_signals()

        self._escape_shortcut = QShortcut(QKeySequence("Esc"), self)
        self._escape_shortcut.activated.connect(self.close)

    def _connect_signals(self) -> None:
        for widget in self._extrinsic_spins():
            widget.valueChanged.connect(self._on_extrinsics_changed)
        self.step_mode_combo.currentIndexChanged.connect(self._on_step_mode_changed)
        self.reset_view_button.clicked.connect(self.canvas.reset_view)
        self.close_button.clicked.connect(self.close)

    def _make_double_spin(self, minimum: float, maximum: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        return spin

    def _extrinsic_spins(self) -> list[QDoubleSpinBox]:
        return [self.tx_spin, self.ty_spin, self.tz_spin, self.roll_spin, self.pitch_spin, self.yaw_spin]

    def _on_step_mode_changed(self, _index: int) -> None:
        t_step, r_step = self.step_mode_combo.currentData()
        for spin in [self.tx_spin, self.ty_spin, self.tz_spin]:
            spin.setSingleStep(t_step)
        for spin in [self.roll_spin, self.pitch_spin, self.yaw_spin]:
            spin.setSingleStep(r_step)

    def _current_extrinsics(self) -> Extrinsics:
        return Extrinsics(
            tx=self.tx_spin.value(),
            ty=self.ty_spin.value(),
            tz=self.tz_spin.value(),
            roll_deg=self.roll_spin.value(),
            pitch_deg=self.pitch_spin.value(),
            yaw_deg=self.yaw_spin.value(),
        )

    def _set_extrinsics_controls(self, extrinsics: Extrinsics) -> None:
        self._syncing = True
        for spin in self._extrinsic_spins():
            spin.blockSignals(True)
        try:
            self.tx_spin.setValue(extrinsics.tx)
            self.ty_spin.setValue(extrinsics.ty)
            self.tz_spin.setValue(extrinsics.tz)
            self.roll_spin.setValue(extrinsics.roll_deg)
            self.pitch_spin.setValue(extrinsics.pitch_deg)
            self.yaw_spin.setValue(extrinsics.yaw_deg)
        finally:
            for spin in self._extrinsic_spins():
                spin.blockSignals(False)
            self._syncing = False

    def _on_extrinsics_changed(self, _value: float) -> None:
        if self._syncing:
            return
        self.extrinsicsChanged.emit(self._current_extrinsics())

    def sync_from_source(self, source_canvas: ImageCanvas, extrinsics: Extrinsics) -> None:
        self.canvas.copy_from(source_canvas, preserve_view=True)
        self.canvas.set_fullscreen_enabled(True)
        self.canvas.set_fullscreen_button_tooltip("退出全屏")
        self._set_extrinsics_controls(extrinsics)
        self.projection_label.setText(" / ".join(source_canvas._status_lines) if source_canvas._status_lines else "投影状态: --")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LiDAR To Camera Calibration Tool")
        self.resize(1740, 1020)

        self.frame_pairs: list[FramePair] = []
        self.current_pair_index = -1
        self.current_intrinsics = CameraIntrinsics()
        self.current_extrinsics = Extrinsics()
        self.loaded_camera_json_intrinsics: CameraIntrinsics | None = None
        self.loaded_camera_json_extrinsics: Extrinsics | None = None
        self.loaded_camera_json_payload: dict | None = None
        self.correspondences: list[CalibrationCorrespondence] = []
        self.pending_image_point: tuple[float, float] | None = None
        self.pending_lidar_point: tuple[float, float, float] | None = None
        self.image_cache: dict[str, object] = {}
        self.pcd_cache: dict[str, PointCloudData] = {}
        self.current_filtered_points = np.zeros((0, 3), dtype=np.float32)
        self.current_filtered_intensity = np.zeros((0,), dtype=np.float32)
        self.projection_fullscreen_window: FullScreenProjectionWindow | None = None

        self._build_ui()
        self._connect_signals()
        self._refresh_extrinsics_output()
        self._load_3d_view_state()

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

        left_column = QWidget()
        left_column_layout = QVBoxLayout(left_column)
        right_column = QWidget()
        right_column_layout = QVBoxLayout(right_column)
        top_splitter.addWidget(left_column)
        top_splitter.addWidget(right_column)

        self.raw_image_canvas = ImageCanvas("原始相机图像")
        self.overlay_canvas = ImageCanvas("点云投影叠加")
        self.overlay_canvas.set_fullscreen_enabled(True)
        self.raw_bev_canvas = PointCloud3DCanvas("原始雷达 3D")
        self.frustum_bev_canvas = PointCloudBevCanvas("BEV 对齐视图（含相机视场）")

        left_column_layout.addWidget(self._wrap_widget("原始相机图像", self.raw_image_canvas))
        left_column_layout.addWidget(self._wrap_widget("点云投影叠加", self.overlay_canvas))
        right_column_layout.addWidget(self._wrap_widget("原始雷达 3D", self.raw_bev_canvas))
        right_column_layout.addWidget(self._wrap_widget("BEV 对齐视图（含相机视场）", self.frustum_bev_canvas))

        bottom_tabs = QTabWidget()
        right_splitter.addWidget(bottom_tabs)
        right_splitter.setSizes([760, 260])

        self.pair_table = QTableWidget(0, 5)
        self.pair_table.setHorizontalHeaderLabels(["序号", "图像", "点云", "时间差(ms)", "时间戳"])
        self.pair_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.pair_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pair_table.setAlternatingRowColors(True)
        self.pair_table.verticalHeader().setVisible(False)
        bottom_tabs.addTab(self.pair_table, "帧配对")

        corr_page = QWidget()
        corr_layout = QVBoxLayout(corr_page)
        corr_button_row = QHBoxLayout()
        self.pick_image_button = QPushButton("选图像点")
        self.pick_image_button.setCheckable(True)
        self.pick_lidar_button = QPushButton("选雷达点")
        self.pick_lidar_button.setCheckable(True)
        self.add_corr_button = QPushButton("加入对应点")
        self.remove_corr_button = QPushButton("删除选中")
        self.clear_corr_button = QPushButton("清空对应点")
        self.solve_corr_button = QPushButton("根据对应点求外参")
        corr_button_row.addWidget(self.pick_image_button)
        corr_button_row.addWidget(self.pick_lidar_button)
        corr_button_row.addWidget(self.add_corr_button)
        corr_button_row.addWidget(self.remove_corr_button)
        corr_button_row.addWidget(self.clear_corr_button)
        corr_button_row.addWidget(self.solve_corr_button)
        corr_layout.addLayout(corr_button_row)

        self.pending_pick_label = QLabel("待加入: 图像点=无, 雷达点=无")
        corr_layout.addWidget(self.pending_pick_label)

        self.corr_table = QTableWidget(0, 6)
        self.corr_table.setHorizontalHeaderLabels(["#", "u", "v", "x", "y", "z"])
        self.corr_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.corr_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.corr_table.setAlternatingRowColors(True)
        self.corr_table.verticalHeader().setVisible(False)
        corr_layout.addWidget(self.corr_table)
        bottom_tabs.addTab(corr_page, "对应点")

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        bottom_tabs.addTab(self.log_output, "日志")

        self._build_sidebar()

    def _build_sidebar(self) -> None:
        dataset_group = QGroupBox("数据源")
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_form = QFormLayout()
        self.image_dir_edit = QLineEdit()
        self.lidar_dir_edit = QLineEdit()
        self.camera_json_edit = QLineEdit()
        dataset_form.addRow("图片文件夹", self._line_with_button(self.image_dir_edit, self._choose_image_dir))
        dataset_form.addRow("PCD 文件夹", self._line_with_button(self.lidar_dir_edit, self._choose_lidar_dir))
        dataset_form.addRow("相机 JSON", self._line_with_button(self.camera_json_edit, self._choose_camera_json))
        dataset_layout.addLayout(dataset_form)

        dataset_button_row = QHBoxLayout()
        self.load_button = QPushButton("加载数据")
        dataset_button_row.addWidget(self.load_button)
        dataset_layout.addLayout(dataset_button_row)

        sync_row = QHBoxLayout()
        self.time_offset_spin = QDoubleSpinBox()
        self.time_offset_spin.setRange(-5.0, 5.0)
        self.time_offset_spin.setDecimals(6)
        self.time_offset_spin.setSingleStep(0.001)
        self.time_offset_spin.setValue(0.0)
        self.jump_best_delta_button = QPushButton("跳到最小时差帧")
        sync_row.addWidget(QLabel("雷达时间偏移(s)"))
        sync_row.addWidget(self.time_offset_spin, 1)
        sync_row.addWidget(self.jump_best_delta_button)
        dataset_layout.addLayout(sync_row)

        self.sidebar_layout.addWidget(dataset_group)

        frame_group = QGroupBox("帧浏览")
        frame_layout = QVBoxLayout(frame_group)
        top_row = QHBoxLayout()
        self.prev_frame_button = QPushButton("上一帧")
        self.next_frame_button = QPushButton("下一帧")
        self.current_frame_label = QLabel("当前: 未加载")
        top_row.addWidget(self.prev_frame_button)
        top_row.addWidget(self.next_frame_button)
        top_row.addWidget(self.current_frame_label, 1)
        frame_layout.addLayout(top_row)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        frame_layout.addWidget(self.frame_slider)
        self.frame_delta_label = QLabel("时间差: --")
        frame_layout.addWidget(self.frame_delta_label)
        self.sidebar_layout.addWidget(frame_group)

        intrinsics_group = QGroupBox("相机内参")
        intrinsics_layout = QFormLayout(intrinsics_group)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 10000)
        self.width_spin.setValue(self.current_intrinsics.width)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 10000)
        self.height_spin.setValue(self.current_intrinsics.height)
        self.fx_spin = self._make_double_spin(0.0, 100000.0, self.current_intrinsics.fx, 1.0, 4)
        self.fy_spin = self._make_double_spin(0.0, 100000.0, self.current_intrinsics.fy, 1.0, 4)
        self.cx_spin = self._make_double_spin(-10000.0, 100000.0, self.current_intrinsics.cx, 1.0, 4)
        self.cy_spin = self._make_double_spin(-10000.0, 100000.0, self.current_intrinsics.cy, 1.0, 4)
        self.distortion_edit = QLineEdit(",".join(["0"] * 8))
        self.use_distortion_checkbox = QCheckBox("投影时使用畸变参数")
        self.use_distortion_checkbox.setChecked(True)
        self.flip_mode_combo = QComboBox()
        self.flip_mode_combo.addItem("无", 0)
        self.flip_mode_combo.addItem("水平翻转", 1)
        self.flip_mode_combo.addItem("垂直翻转", 2)
        self.flip_mode_combo.addItem("旋转180°", 3)
        intrinsics_layout.addRow("图像宽", self.width_spin)
        intrinsics_layout.addRow("图像高", self.height_spin)
        intrinsics_layout.addRow("fx", self.fx_spin)
        intrinsics_layout.addRow("fy", self.fy_spin)
        intrinsics_layout.addRow("cx", self.cx_spin)
        intrinsics_layout.addRow("cy", self.cy_spin)
        intrinsics_layout.addRow("D(k1,k2,p1,p2,k3,k4,k5,k6)", self.distortion_edit)
        intrinsics_layout.addRow("", self.use_distortion_checkbox)
        intrinsics_layout.addRow("图像翻转/旋转", self.flip_mode_combo)
        self.sidebar_layout.addWidget(intrinsics_group)

        filter_group = QGroupBox("点云筛选")
        filter_layout = QFormLayout(filter_group)
        self.x_min_spin = self._make_double_spin(-200.0, 200.0, -10.0, 0.5, 2)
        self.x_max_spin = self._make_double_spin(-200.0, 200.0, 80.0, 0.5, 2)
        self.y_min_spin = self._make_double_spin(-200.0, 200.0, -40.0, 0.5, 2)
        self.y_max_spin = self._make_double_spin(-200.0, 200.0, 40.0, 0.5, 2)
        self.z_min_spin = self._make_double_spin(-20.0, 50.0, -3.0, 0.2, 2)
        self.z_max_spin = self._make_double_spin(-20.0, 50.0, 3.0, 0.2, 2)
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(100, 300000)
        self.max_points_spin.setSingleStep(500)
        self.max_points_spin.setValue(20000)
        filter_layout.addRow("x 最小", self.x_min_spin)
        filter_layout.addRow("x 最大", self.x_max_spin)
        filter_layout.addRow("y 最小", self.y_min_spin)
        filter_layout.addRow("y 最大", self.y_max_spin)
        filter_layout.addRow("z 最小", self.z_min_spin)
        filter_layout.addRow("z 最大", self.z_max_spin)
        filter_layout.addRow("最多显示点数", self.max_points_spin)
        self.sidebar_layout.addWidget(filter_group)

        view3d_group = QGroupBox("3D 视图控制")
        view3d_layout = QVBoxLayout(view3d_group)
        self.reset_view3d_button = QPushButton("重置3D视角")
        view3d_layout.addWidget(self.reset_view3d_button)
        self.sidebar_layout.addWidget(view3d_group)

        extr_group = QGroupBox("LiDAR -> Camera 外参")
        extr_layout = QFormLayout(extr_group)

        self.step_mode_combo = QComboBox()
        self.step_mode_combo.addItem("精细 (T:0.001, R:0.01°)", (0.001, 0.01))
        self.step_mode_combo.addItem("普通 (T:0.01, R:0.1°)", (0.01, 0.1))
        self.step_mode_combo.addItem("粗略 (T:0.1, R:1°)", (0.1, 1.0))
        self.step_mode_combo.setCurrentIndex(1)
        extr_layout.addRow("调节步长", self.step_mode_combo)

        self.tx_spin = self._make_double_spin(-100.0, 100.0, 0.0, 0.01, 5)
        self.ty_spin = self._make_double_spin(-100.0, 100.0, 0.0, 0.01, 5)
        self.tz_spin = self._make_double_spin(-100.0, 100.0, 0.0, 0.01, 5)
        self.roll_spin = self._make_double_spin(-180.0, 180.0, 0.0, 0.1, 5)
        self.pitch_spin = self._make_double_spin(-180.0, 180.0, 0.0, 0.1, 5)
        self.yaw_spin = self._make_double_spin(-180.0, 180.0, 0.0, 0.1, 5)
        extr_layout.addRow("tx (m)", self.tx_spin)
        extr_layout.addRow("ty (m)", self.ty_spin)
        extr_layout.addRow("tz (m)", self.tz_spin)
        extr_layout.addRow("roll (deg)", self.roll_spin)
        extr_layout.addRow("pitch (deg)", self.pitch_spin)
        extr_layout.addRow("yaw (deg)", self.yaw_spin)

        self.export_direction_combo = QComboBox()
        for label, value in EXPORT_DIRECTION_OPTIONS:
            self.export_direction_combo.addItem(label, value)
        extr_layout.addRow("导出方向", self.export_direction_combo)

        self.adjust_lidar_axis_checkbox = QCheckBox("导出时调整雷达实际朝向")
        self.adjust_lidar_axis_checkbox.setChecked(False)
        extr_layout.addRow("", self.adjust_lidar_axis_checkbox)

        self.lidar_axis_combo = QComboBox()
        for label, value in LIDAR_AXIS_OPTIONS:
            self.lidar_axis_combo.addItem(label, value)
        self.lidar_axis_combo.setEnabled(False)
        extr_layout.addRow("雷达朝向预设", self.lidar_axis_combo)

        self.export_hint_label = QLabel("导出设置只影响输出 JSON，不影响当前投影和求解。")
        self.export_hint_label.setWordWrap(True)
        extr_layout.addRow("", self.export_hint_label)

        extr_button_row = QHBoxLayout()
        self.reset_extr_button = QPushButton("重置外参")
        self.json_extr_button = QPushButton("用 JSON 外参初始化")
        self.export_extr_button = QPushButton("导出外参")
        extr_button_row.addWidget(self.reset_extr_button)
        extr_button_row.addWidget(self.json_extr_button)
        extr_button_row.addWidget(self.export_extr_button)
        extr_layout.addRow("", extr_button_row)
        self.sidebar_layout.addWidget(extr_group)

        output_group = QGroupBox("外参输出")
        output_layout = QVBoxLayout(output_group)
        self.extr_output = QPlainTextEdit()
        self.extr_output.setReadOnly(True)
        self.extr_output.setMinimumHeight(220)
        output_layout.addWidget(self.extr_output)
        self.sidebar_layout.addWidget(output_group)

    def _connect_signals(self) -> None:
        self.load_button.clicked.connect(self._load_dataset)
        self.time_offset_spin.valueChanged.connect(self._reload_frame_pairs_only)
        self.jump_best_delta_button.clicked.connect(self._jump_to_smallest_delta_pair)

        self.prev_frame_button.clicked.connect(lambda: self._step_frame(-1))
        self.next_frame_button.clicked.connect(lambda: self._step_frame(1))
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.pair_table.itemSelectionChanged.connect(self._on_pair_table_selection_changed)

        for widget in [
            self.width_spin,
            self.height_spin,
            self.fx_spin,
            self.fy_spin,
            self.cx_spin,
            self.cy_spin,
            self.x_min_spin,
            self.x_max_spin,
            self.y_min_spin,
            self.y_max_spin,
            self.z_min_spin,
            self.z_max_spin,
            self.max_points_spin,
            self.tx_spin,
            self.ty_spin,
            self.tz_spin,
            self.roll_spin,
            self.pitch_spin,
            self.yaw_spin,
        ]:
            widget.valueChanged.connect(self._update_visuals)

        self.distortion_edit.editingFinished.connect(self._update_visuals)
        self.use_distortion_checkbox.toggled.connect(self._update_visuals)
        self.flip_mode_combo.currentIndexChanged.connect(self._update_visuals)

        self.step_mode_combo.currentIndexChanged.connect(self._on_step_mode_changed)
        self.reset_extr_button.clicked.connect(self._reset_extrinsics)
        self.json_extr_button.clicked.connect(self._apply_loaded_json_extrinsics)
        self.export_extr_button.clicked.connect(self._export_extrinsics)
        self.export_direction_combo.currentIndexChanged.connect(self._on_export_settings_changed)
        self.adjust_lidar_axis_checkbox.toggled.connect(self._on_export_settings_changed)
        self.adjust_lidar_axis_checkbox.toggled.connect(self.lidar_axis_combo.setEnabled)
        self.lidar_axis_combo.currentIndexChanged.connect(self._on_export_settings_changed)

        self.reset_view3d_button.clicked.connect(self.raw_bev_canvas.reset_view)
        self.raw_bev_canvas.viewChanged.connect(self._save_3d_view_state)
        self.overlay_canvas.fullScreenRequested.connect(self._open_projection_fullscreen)

        self.pick_image_button.toggled.connect(self.raw_image_canvas.set_pick_enabled)
        self.pick_lidar_button.toggled.connect(self.raw_bev_canvas.set_pick_enabled)
        self.raw_image_canvas.pointPicked.connect(self._on_image_point_picked)
        self.raw_bev_canvas.pointPicked.connect(self._on_lidar_point_picked)
        self.frustum_bev_canvas.regionSelected.connect(self._on_bev_region_selected)
        self.add_corr_button.clicked.connect(self._add_pending_correspondence)
        self.remove_corr_button.clicked.connect(self._remove_selected_correspondence)
        self.clear_corr_button.clicked.connect(self._clear_correspondences)
        self.solve_corr_button.clicked.connect(self._solve_from_correspondences)

    def _open_projection_fullscreen(self) -> None:
        if self.projection_fullscreen_window is not None:
            self.projection_fullscreen_window.close()

        window = FullScreenProjectionWindow(
            self.overlay_canvas,
            self._current_extrinsics(),
            self.step_mode_combo.currentIndex(),
            self,
        )
        self.projection_fullscreen_window = window
        window.extrinsicsChanged.connect(self._apply_fullscreen_extrinsics)
        window.finished.connect(lambda _result, closed_window=window: self._clear_projection_fullscreen(closed_window))
        window.showFullScreen()

    def _clear_projection_fullscreen(self, window: FullScreenProjectionWindow) -> None:
        if self.projection_fullscreen_window is window:
            self.projection_fullscreen_window = None

    def _apply_fullscreen_extrinsics(self, extrinsics: Extrinsics) -> None:
        self._set_extrinsics_controls(extrinsics)
        self._update_visuals()

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

    def _on_step_mode_changed(self, _index: int) -> None:
        t_step, r_step = self.step_mode_combo.currentData()
        for spin in [self.tx_spin, self.ty_spin, self.tz_spin]:
            spin.setSingleStep(t_step)
        for spin in [self.roll_spin, self.pitch_spin, self.yaw_spin]:
            spin.setSingleStep(r_step)

    def _make_double_spin(self, minimum: float, maximum: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        return spin

    def _on_export_settings_changed(self, *_args) -> None:
        self._refresh_extrinsics_output()

    def _current_export_direction(self) -> str:
        return str(self.export_direction_combo.currentData())

    def _current_lidar_axis_mode(self) -> str:
        if not self.adjust_lidar_axis_checkbox.isChecked():
            return "default"
        return str(self.lidar_axis_combo.currentData())

    def _choose_image_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", self.image_dir_edit.text() or str(Path.home()))
        if path:
            self.image_dir_edit.setText(path)

    def _choose_lidar_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择点云文件夹", self.lidar_dir_edit.text() or str(Path.home()))
        if path:
            self.lidar_dir_edit.setText(path)

    def _choose_camera_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择相机 JSON", self.camera_json_edit.text() or str(Path.home()), "JSON (*.json)")
        if path:
            self.camera_json_edit.setText(path)

    def _load_3d_view_state(self) -> None:
        settings = load_settings()
        view_state = settings.get("view3d", {})
        if view_state:
            self.raw_bev_canvas.set_view_state(view_state)

    def _save_3d_view_state(self) -> None:
        settings = load_settings()
        settings["view3d"] = self.raw_bev_canvas.get_view_state()
        save_settings(settings)

    def _load_dataset(self) -> None:
        image_dir = self.image_dir_edit.text().strip()
        lidar_dir = self.lidar_dir_edit.text().strip()
        if not image_dir or not lidar_dir:
            QMessageBox.warning(self, "缺少路径", "请先选择图片文件夹和 PCD 文件夹。")
            return

        try:
            self.image_cache.clear()
            self.pcd_cache.clear()
            self.correspondences.clear()
            self.pending_image_point = None
            self.pending_lidar_point = None
            self._refresh_correspondence_views()

            camera_json = self.camera_json_edit.text().strip()
            if camera_json:
                intrinsics, extrinsics, payload = load_camera_json(camera_json)
                self.loaded_camera_json_intrinsics = intrinsics
                self.loaded_camera_json_extrinsics = extrinsics
                self.loaded_camera_json_payload = payload
                if intrinsics is not None:
                    self._set_intrinsics_controls(intrinsics)
                if extrinsics is not None:
                    self._set_extrinsics_controls(extrinsics)
            else:
                self.loaded_camera_json_intrinsics = None
                self.loaded_camera_json_extrinsics = None
                self.loaded_camera_json_payload = None

            self.frame_pairs = build_frame_pairs(image_dir, lidar_dir, self.time_offset_spin.value())
            if not self.frame_pairs:
                raise ValueError("没有找到可配对的 PNG 和 PCD 文件。")

            self._populate_pair_table()
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, max(0, len(self.frame_pairs) - 1))
            self.frame_slider.blockSignals(False)
            self.current_pair_index = 0
            self.frame_slider.setValue(0)
            self._select_pair_row(0)
            self._append_log(f"加载完成: 共 {len(self.frame_pairs)} 组图像/点云配对。")
            self._update_visuals()
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))

    def _reload_frame_pairs_only(self) -> None:
        if not self.image_dir_edit.text().strip() or not self.lidar_dir_edit.text().strip():
            return
        if not self.frame_pairs:
            return
        current_image_path = self.frame_pairs[self.current_pair_index].image_path if 0 <= self.current_pair_index < len(self.frame_pairs) else None
        self.frame_pairs = build_frame_pairs(self.image_dir_edit.text().strip(), self.lidar_dir_edit.text().strip(), self.time_offset_spin.value())
        self._populate_pair_table()
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, len(self.frame_pairs) - 1))
        self.frame_slider.blockSignals(False)
        new_index = 0
        if current_image_path is not None:
            for idx, pair in enumerate(self.frame_pairs):
                if pair.image_path == current_image_path:
                    new_index = idx
                    break
        self.current_pair_index = new_index
        self.frame_slider.setValue(new_index)
        self._select_pair_row(new_index)
        self._update_visuals()

    def _populate_pair_table(self) -> None:
        self.pair_table.setRowCount(len(self.frame_pairs))
        for row, pair in enumerate(self.frame_pairs):
            timestamp_desc = "-"
            if pair.image_timestamp is not None and pair.lidar_timestamp is not None:
                timestamp_desc = f"{pair.image_timestamp:.6f} / {pair.lidar_timestamp:.6f}"
            delta_ms = "-" if pair.delta_seconds is None else f"{pair.delta_seconds * 1000.0:.3f}"
            values = [
                str(row),
                pair.image_path.name,
                pair.lidar_path.name,
                delta_ms,
                timestamp_desc,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                self.pair_table.setItem(row, column, item)
        self.pair_table.resizeColumnsToContents()

    def _select_pair_row(self, row: int) -> None:
        self.pair_table.blockSignals(True)
        self.pair_table.clearSelection()
        if 0 <= row < self.pair_table.rowCount():
            self.pair_table.selectRow(row)
        self.pair_table.blockSignals(False)

    def _on_pair_table_selection_changed(self) -> None:
        items = self.pair_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        if row != self.current_pair_index:
            self.current_pair_index = row
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(row)
            self.frame_slider.blockSignals(False)
            self._update_visuals()

    def _on_frame_slider_changed(self, value: int) -> None:
        self.current_pair_index = value
        self._select_pair_row(value)
        self._update_visuals()

    def _step_frame(self, step: int) -> None:
        if not self.frame_pairs:
            return
        next_index = max(0, min(len(self.frame_pairs) - 1, self.current_pair_index + step))
        self.frame_slider.setValue(next_index)

    def _jump_to_smallest_delta_pair(self) -> None:
        if not self.frame_pairs:
            return
        valid = [(index, abs(pair.delta_seconds)) for index, pair in enumerate(self.frame_pairs) if pair.delta_seconds is not None]
        if not valid:
            return
        best_index = min(valid, key=lambda item: item[1])[0]
        self.frame_slider.setValue(best_index)

    def _set_intrinsics_controls(self, intrinsics: CameraIntrinsics) -> None:
        self.width_spin.setValue(int(intrinsics.width))
        self.height_spin.setValue(int(intrinsics.height))
        self.fx_spin.setValue(float(intrinsics.fx))
        self.fy_spin.setValue(float(intrinsics.fy))
        self.cx_spin.setValue(float(intrinsics.cx))
        self.cy_spin.setValue(float(intrinsics.cy))
        self.distortion_edit.setText(",".join(f"{value:.10f}" for value in intrinsics.clipped_distortion()))
        index = self.flip_mode_combo.findData(int(intrinsics.flip_mode))
        self.flip_mode_combo.setCurrentIndex(index if index >= 0 else 0)

    def _set_extrinsics_controls(self, extrinsics: Extrinsics) -> None:
        spins = [self.tx_spin, self.ty_spin, self.tz_spin, self.roll_spin, self.pitch_spin, self.yaw_spin]
        for spin in spins:
            spin.blockSignals(True)
        try:
            self.tx_spin.setValue(extrinsics.tx)
            self.ty_spin.setValue(extrinsics.ty)
            self.tz_spin.setValue(extrinsics.tz)
            self.roll_spin.setValue(extrinsics.roll_deg)
            self.pitch_spin.setValue(extrinsics.pitch_deg)
            self.yaw_spin.setValue(extrinsics.yaw_deg)
        finally:
            for spin in spins:
                spin.blockSignals(False)

    def _current_intrinsics(self) -> CameraIntrinsics:
        distortion_values = []
        for token in self.distortion_edit.text().replace(" ", "").split(","):
            if token:
                try:
                    distortion_values.append(float(token))
                except ValueError:
                    pass
        return CameraIntrinsics(
            fx=self.fx_spin.value(),
            fy=self.fy_spin.value(),
            cx=self.cx_spin.value(),
            cy=self.cy_spin.value(),
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            distortion=distortion_values,
            flip_mode=int(self.flip_mode_combo.currentData()),
        )

    def _current_extrinsics(self) -> Extrinsics:
        return Extrinsics(
            tx=self.tx_spin.value(),
            ty=self.ty_spin.value(),
            tz=self.tz_spin.value(),
            roll_deg=self.roll_spin.value(),
            pitch_deg=self.pitch_spin.value(),
            yaw_deg=self.yaw_spin.value(),
        )

    def _get_cached_image(self, image_path: Path):
        key = str(image_path)
        if key not in self.image_cache:
            self.image_cache[key] = load_qimage(image_path)
        return self.image_cache[key]

    def _get_cached_pcd(self, lidar_path: Path) -> PointCloudData:
        key = str(lidar_path)
        if key not in self.pcd_cache:
            self.pcd_cache[key] = load_pcd(lidar_path)
        return self.pcd_cache[key]

    def _filter_current_point_cloud(self, point_cloud: PointCloudData) -> tuple[np.ndarray, np.ndarray]:
        points = point_cloud.points_xyz
        intensity = point_cloud.intensity
        mask = (
            (points[:, 0] >= self.x_min_spin.value())
            & (points[:, 0] <= self.x_max_spin.value())
            & (points[:, 1] >= self.y_min_spin.value())
            & (points[:, 1] <= self.y_max_spin.value())
            & (points[:, 2] >= self.z_min_spin.value())
            & (points[:, 2] <= self.z_max_spin.value())
        )
        filtered_points = points[mask]
        filtered_intensity = intensity[mask]
        max_points = max(1, self.max_points_spin.value())
        if filtered_points.shape[0] > max_points:
            stride = max(1, int(np.ceil(filtered_points.shape[0] / max_points)))
            filtered_points = filtered_points[::stride]
            filtered_intensity = filtered_intensity[::stride]
        return filtered_points.astype(np.float32), filtered_intensity.astype(np.float32)

    def _colorize_height(self, points_xyz: np.ndarray) -> np.ndarray:
        if points_xyz.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.uint8)
        z = points_xyz[:, 2]
        return depth_to_rgb(z)

    def _update_visuals(self) -> None:
        self.current_intrinsics = self._current_intrinsics()
        self.current_extrinsics = self._current_extrinsics()
        self._refresh_extrinsics_output()
        self._refresh_correspondence_views()

        if not self.frame_pairs or not (0 <= self.current_pair_index < len(self.frame_pairs)):
            return

        pair = self.frame_pairs[self.current_pair_index]
        image = self._get_cached_image(pair.image_path)
        point_cloud = self._get_cached_pcd(pair.lidar_path)
        if image.width() > 0 and image.height() > 0 and self.loaded_camera_json_intrinsics is None:
            if self.width_spin.value() != image.width() or self.height_spin.value() != image.height():
                self.width_spin.blockSignals(True)
                self.height_spin.blockSignals(True)
                self.width_spin.setValue(image.width())
                self.height_spin.setValue(image.height())
                self.width_spin.blockSignals(False)
                self.height_spin.blockSignals(False)
                self.current_intrinsics = self._current_intrinsics()

        self.current_filtered_points, self.current_filtered_intensity = self._filter_current_point_cloud(point_cloud)
        raw_colors = self._colorize_height(self.current_filtered_points)

        projected_uv, projected_depth, mask = project_lidar_to_image(
            self.current_filtered_points,
            self.current_intrinsics,
            self.current_extrinsics,
            use_distortion=self.use_distortion_checkbox.isChecked(),
            flip_mode=self.current_intrinsics.flip_mode,
        )
        overlay_points = projected_uv[mask]
        overlay_colors = depth_to_rgb(projected_depth[mask]) if np.any(mask) else np.zeros((0, 3), dtype=np.uint8)
        frustum = camera_frustum_in_lidar(self.current_intrinsics, self.current_extrinsics, depth_m=max(10.0, min(self.x_max_spin.value(), 80.0)))

        image_markers, lidar_markers = self._build_marker_lists()
        flip_desc = {0: "无", 1: "水平翻转", 2: "垂直翻转", 3: "旋转180°"}.get(self.current_intrinsics.flip_mode, "未知")
        self.raw_image_canvas.set_flip_mode(self.current_intrinsics.flip_mode)
        self.raw_image_canvas.set_image(image)
        self.raw_image_canvas.set_overlay(np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8))
        self.raw_image_canvas.set_markers(image_markers)
        self.raw_image_canvas.set_status_lines(
            [
                f"文件: {pair.image_path.name}",
                f"尺寸: {image.width()}x{image.height()}",
            ]
        )

        self.overlay_canvas.set_flip_mode(self.current_intrinsics.flip_mode)
        self.overlay_canvas.set_image(image)
        self.overlay_canvas.set_overlay(overlay_points, overlay_colors)
        self.overlay_canvas.set_markers(image_markers)
        self.overlay_canvas.set_status_lines(
            [
                f"投影点数: {overlay_points.shape[0]} / {self.current_filtered_points.shape[0]}",
                f"畸变: {'开' if self.use_distortion_checkbox.isChecked() else '关'}",
                f"翻转: {flip_desc}",
            ]
        )

        self.raw_bev_canvas.set_points(self.current_filtered_points, raw_colors)
        self.raw_bev_canvas.set_markers(lidar_markers)
        self.raw_bev_canvas.set_status_lines([f"点云: {self.current_filtered_points.shape[0]} 点"])

        self.frustum_bev_canvas.set_ranges(self.x_min_spin.value(), self.x_max_spin.value(), self.y_min_spin.value(), self.y_max_spin.value())
        self.frustum_bev_canvas.set_points(self.current_filtered_points, raw_colors)
        self.frustum_bev_canvas.set_markers(lidar_markers)
        self.frustum_bev_canvas.set_frustum(frustum)
        self.frustum_bev_canvas.set_status_lines(["黄色锥体: 当前相机视场"])

        self.current_frame_label.setText(f"当前: {self.current_pair_index + 1}/{len(self.frame_pairs)}")
        if pair.delta_seconds is None:
            self.frame_delta_label.setText("时间差: 索引配对")
        else:
            self.frame_delta_label.setText(f"时间差: {pair.delta_seconds * 1000.0:.3f} ms")

        if self.projection_fullscreen_window is not None:
            self.projection_fullscreen_window.sync_from_source(self.overlay_canvas, self.current_extrinsics)

    def _build_marker_lists(self) -> tuple[list[tuple[float, float, str, QColor]], list[tuple[float, float, float, str, QColor]]]:
        image_markers: list[tuple[float, float, str, QColor]] = []
        lidar_markers: list[tuple[float, float, float, str, QColor]] = []
        for index, corr in enumerate(self.correspondences, start=1):
            color = QColor(60, 255, 170)
            image_markers.append((corr.image_u, corr.image_v, f"#{index}", color))
            lidar_markers.append((corr.lidar_x, corr.lidar_y, corr.lidar_z, f"#{index}", color))

        if self.pending_image_point is not None:
            image_markers.append((self.pending_image_point[0], self.pending_image_point[1], "待配对", QColor(255, 160, 60)))
        if self.pending_lidar_point is not None:
            lidar_markers.append((self.pending_lidar_point[0], self.pending_lidar_point[1], self.pending_lidar_point[2], "待配对", QColor(255, 160, 60)))
        return image_markers, lidar_markers

    def _refresh_extrinsics_output(self) -> None:
        extrinsics = self._current_extrinsics()
        try:
            payload = export_extrinsics(
                extrinsics,
                lidar_axis_mode=self._current_lidar_axis_mode(),
                direction=self._current_export_direction(),
            )
        except ValueError as exc:
            payload = {
                "export_error": str(exc),
                "metadata": {
                    "export_direction": self._current_export_direction(),
                    "lidar_axis_mode": self._current_lidar_axis_mode(),
                },
            }
        self.extr_output.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))

    def _refresh_correspondence_views(self) -> None:
        image_text = "无" if self.pending_image_point is None else f"({self.pending_image_point[0]:.2f}, {self.pending_image_point[1]:.2f})"
        lidar_text = "无" if self.pending_lidar_point is None else f"({self.pending_lidar_point[0]:.3f}, {self.pending_lidar_point[1]:.3f}, {self.pending_lidar_point[2]:.3f})"
        self.pending_pick_label.setText(f"待加入: 图像点={image_text}, 雷达点={lidar_text}")
        self.corr_table.setRowCount(len(self.correspondences))
        for row, corr in enumerate(self.correspondences):
            values = [
                str(row + 1),
                f"{corr.image_u:.3f}",
                f"{corr.image_v:.3f}",
                f"{corr.lidar_x:.3f}",
                f"{corr.lidar_y:.3f}",
                f"{corr.lidar_z:.3f}",
            ]
            for column, value in enumerate(values):
                self.corr_table.setItem(row, column, QTableWidgetItem(value))
        self.corr_table.resizeColumnsToContents()

    def _on_image_point_picked(self, u: float, v: float) -> None:
        self.pending_image_point = (u, v)
        self.pick_image_button.setChecked(False)
        self._refresh_correspondence_views()
        self._update_visuals()

    def _on_lidar_point_picked(self, x: float, y: float, z: float) -> None:
        self.pending_lidar_point = (x, y, z)
        self.pick_lidar_button.setChecked(False)
        self._refresh_correspondence_views()
        self._update_visuals()

    def _on_bev_region_selected(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        self.x_min_spin.setValue(x_min)
        self.x_max_spin.setValue(x_max)
        self.y_min_spin.setValue(y_min)
        self.y_max_spin.setValue(y_max)
        self._append_log(f"BEV 框选筛选范围: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
        self._update_visuals()

    def _add_pending_correspondence(self) -> None:
        if self.pending_image_point is None or self.pending_lidar_point is None:
            QMessageBox.information(self, "对应点不足", "请先各选一个图像点和一个雷达点。")
            return
        self.correspondences.append(
            CalibrationCorrespondence(
                image_u=self.pending_image_point[0],
                image_v=self.pending_image_point[1],
                lidar_x=self.pending_lidar_point[0],
                lidar_y=self.pending_lidar_point[1],
                lidar_z=self.pending_lidar_point[2],
            )
        )
        self.pending_image_point = None
        self.pending_lidar_point = None
        self._refresh_correspondence_views()
        self._append_log(f"已加入第 {len(self.correspondences)} 组对应点。")
        self._update_visuals()

    def _remove_selected_correspondence(self) -> None:
        selected = self.corr_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        if 0 <= row < len(self.correspondences):
            self.correspondences.pop(row)
            self._refresh_correspondence_views()
            self._update_visuals()

    def _clear_correspondences(self) -> None:
        self.correspondences.clear()
        self.pending_image_point = None
        self.pending_lidar_point = None
        self._refresh_correspondence_views()
        self._update_visuals()

    def _solve_from_correspondences(self) -> None:
        intrinsics = self._current_intrinsics()
        success, extrinsics, rmse, message = solve_extrinsics_from_correspondences(
            self.correspondences,
            intrinsics,
            self._current_extrinsics(),
            use_distortion=self.use_distortion_checkbox.isChecked(),
            flip_mode=intrinsics.flip_mode,
        )
        self._set_extrinsics_controls(extrinsics)
        self._append_log(f"{message} RMSE={rmse:.3f}px")
        self._update_visuals()
        if not success:
            QMessageBox.warning(self, "求解提示", f"{message}\nRMSE={rmse:.3f}px")

    def _reset_extrinsics(self) -> None:
        self._set_extrinsics_controls(Extrinsics())
        self._update_visuals()

    def _apply_loaded_json_extrinsics(self) -> None:
        if self.loaded_camera_json_extrinsics is None:
            QMessageBox.information(self, "没有 JSON 外参", "请先加载带 rotation/translation 的相机 JSON。")
            return
        self._set_extrinsics_controls(self.loaded_camera_json_extrinsics)
        self._append_log("已用 JSON 中的外参初始化界面。")
        self._update_visuals()

    def _export_extrinsics(self) -> None:
        default_name = (
            "lidar_to_camera_extrinsics.json"
            if self._current_export_direction() == "lidar_to_camera"
            else "camera_to_lidar_extrinsics.json"
        )
        path, _ = QFileDialog.getSaveFileName(self, "导出外参 JSON", str(Path.home() / default_name), "JSON (*.json)")
        if not path:
            return
        payload = json.loads(self.extr_output.toPlainText())
        if "export_error" in payload:
            QMessageBox.warning(self, "无法导出外参", payload["export_error"])
            return
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        axis_desc = "默认朝向" if self._current_lidar_axis_mode() == "default" else self.lidar_axis_combo.currentText()
        self._append_log(
            f"已导出外参到 {path} "
            f"(方向: {self.export_direction_combo.currentText()}, 雷达朝向: {axis_desc})"
        )

    def _append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Left, Qt.Key_A):
            self._step_frame(-1)
            return
        if event.key() in (Qt.Key_Right, Qt.Key_D):
            self._step_frame(1)
            return
        super().keyPressEvent(event)
