from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .calibrator import LidarImuCalibrationResult, calibrate_lidar_imu, calibrate_lidar_imu_from_pcd_folder, result_to_json


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LiDAR → IMU 外参标定")
        self.resize(980, 760)
        self.result: LidarImuCalibrationResult | None = None

        self._build_ui()
        self._connect_signals()
        self._show_csv_hint()
        self._update_lidar_input_mode()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignTop)

        title = QLabel("LiDAR → IMU 简化运动约束标定")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        file_group = QGroupBox("轨迹输入")
        file_layout = QFormLayout(file_group)
        self.lidar_input_mode_combo = QComboBox()
        self.lidar_input_mode_combo.addItem("LiDAR 位姿 CSV", "csv")
        self.lidar_input_mode_combo.addItem("连续 PCD 文件夹", "pcd")
        self.lidar_path_edit = QLineEdit()
        self.imu_csv_edit = QLineEdit()
        file_layout.addRow("LiDAR 输入类型", self.lidar_input_mode_combo)
        file_layout.addRow("LiDAR 输入路径", self._line_with_button(self.lidar_path_edit, self._choose_lidar_input))
        file_layout.addRow("IMU/INS 位姿 CSV", self._line_with_button(self.imu_csv_edit, self._choose_imu_csv))
        layout.addWidget(file_group)

        pcd_group = QGroupBox("PCD 里程计参数")
        self.pcd_group = pcd_group
        pcd_layout = QFormLayout(pcd_group)
        self.pcd_frame_interval_spin = self._make_double_spin(0.001, 10.0, 0.1, 0.01, 4)
        self.voxel_size_spin = self._make_double_spin(0.0, 5.0, 0.5, 0.05, 3)
        self.max_corr_spin = self._make_double_spin(0.05, 20.0, 1.5, 0.1, 3)
        self.icp_iterations_spin = QSpinBox()
        self.icp_iterations_spin.setRange(1, 500)
        self.icp_iterations_spin.setValue(50)
        self.icp_method_combo = QComboBox()
        self.icp_method_combo.addItem("点到点 ICP", "point_to_point")
        self.icp_method_combo.addItem("点到面 ICP", "point_to_plane")
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(1000, 2000000)
        self.max_points_spin.setValue(80000)
        self.max_points_spin.setSingleStep(10000)
        pcd_layout.addRow("PCD 帧间隔(s)", self.pcd_frame_interval_spin)
        pcd_layout.addRow("体素降采样(m)", self.voxel_size_spin)
        pcd_layout.addRow("ICP 最大对应距离(m)", self.max_corr_spin)
        pcd_layout.addRow("ICP 迭代次数", self.icp_iterations_spin)
        pcd_layout.addRow("ICP 类型", self.icp_method_combo)
        pcd_layout.addRow("单帧最大点数", self.max_points_spin)
        layout.addWidget(pcd_group)

        param_group = QGroupBox("标定参数")
        param_layout = QFormLayout(param_group)
        self.interval_spin = self._make_double_spin(0.02, 20.0, 1.0, 0.1, 3)
        self.imu_time_offset_spin = self._make_double_spin(-5.0, 5.0, 0.0, 0.001, 6)
        self.min_rotation_spin = self._make_double_spin(0.0, 45.0, 1.0, 0.5, 3)
        self.max_pairs_spin = QSpinBox()
        self.max_pairs_spin.setRange(3, 5000)
        self.max_pairs_spin.setValue(400)
        self.max_pairs_spin.setSingleStep(50)
        param_layout.addRow("相对运动间隔(s)", self.interval_spin)
        param_layout.addRow("IMU 时间偏移(s)", self.imu_time_offset_spin)
        param_layout.addRow("最小旋转量(deg)", self.min_rotation_spin)
        param_layout.addRow("最大运动片段数", self.max_pairs_spin)
        layout.addWidget(param_group)

        action_row = QHBoxLayout()
        self.run_button = QPushButton("运行 LiDAR-IMU 标定")
        self.save_button = QPushButton("保存结果 JSON")
        self.run_button.setMinimumHeight(42)
        self.save_button.setMinimumHeight(42)
        self.run_button.setStyleSheet("QPushButton { font-weight: bold; }")
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.save_button)
        layout.addLayout(action_row)

        output_group = QGroupBox("结果")
        output_layout = QVBoxLayout(output_group)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        output_layout.addWidget(self.output)
        layout.addWidget(output_group, 1)

    def _connect_signals(self) -> None:
        self.run_button.clicked.connect(self._run_calibration)
        self.save_button.clicked.connect(self._save_result)
        self.lidar_input_mode_combo.currentIndexChanged.connect(self._update_lidar_input_mode)

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

    def _choose_lidar_input(self) -> None:
        current = self.lidar_path_edit.text() or str(Path.home())
        if self._lidar_mode() == "pcd":
            path = QFileDialog.getExistingDirectory(self, "选择连续 PCD 文件夹", current)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择 LiDAR 位姿 CSV", current, "CSV (*.csv);;All Files (*)")
        if path:
            self.lidar_path_edit.setText(path)

    def _choose_imu_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 IMU/INS 位姿 CSV", self.imu_csv_edit.text() or str(Path.home()), "CSV (*.csv);;All Files (*)")
        if path:
            self.imu_csv_edit.setText(path)

    def _show_csv_hint(self) -> None:
        self.output.setPlainText(
            "输入说明:\n"
            "  这是一个简化版 LiDAR-IMU 标定器。\n"
            "  LiDAR 输入可以是位姿 CSV，也可以是连续 PCD 文件夹。\n"
            "  PCD 模式会用相邻帧 ICP 自动生成 LiDAR odometry，第一帧点云坐标作为 LiDAR 世界系。\n"
            "  文件名如果像时间戳，会使用文件名时间；否则按 PCD 帧间隔生成时间。\n"
            "  IMU/INS 仍需要一条位姿 CSV；只有原始加速度/角速度还不能用这个简化模块直接求外参。\n\n"
            "  CSV 支持表头格式:\n"
            "    timestamp,tx,ty,tz,qx,qy,qz,qw\n"
            "  或:\n"
            "    timestamp,tx,ty,tz,roll_deg,pitch_deg,yaw_deg\n"
            "  无表头格式默认按 timestamp,tx,ty,tz,qx,qy,qz,qw 解析。\n\n"
            "算法约定:\n"
            "  输入位姿应为传感器到世界坐标 T_world_sensor。\n"
            "  输出 T_imu_lidar，满足 P_imu = R_imu_lidar * P_lidar + t_imu_lidar。\n"
        )

    def _lidar_mode(self) -> str:
        return str(self.lidar_input_mode_combo.currentData())

    def _update_lidar_input_mode(self) -> None:
        is_pcd_mode = self._lidar_mode() == "pcd"
        self.pcd_group.setEnabled(is_pcd_mode)
        if is_pcd_mode:
            self.lidar_path_edit.setPlaceholderText("选择包含连续 .pcd 的文件夹")
        else:
            self.lidar_path_edit.setPlaceholderText("选择 LiDAR 位姿 CSV")

    def _append_progress(self, message: str) -> None:
        self.output.appendPlainText(message)
        QApplication.processEvents()

    def _run_calibration(self) -> None:
        lidar_input = self.lidar_path_edit.text().strip()
        imu_csv = self.imu_csv_edit.text().strip()
        if not lidar_input or not imu_csv:
            QMessageBox.warning(self, "缺少输入", "请先选择 LiDAR 输入和 IMU/INS 位姿 CSV。")
            return

        try:
            if self._lidar_mode() == "pcd":
                self.output.setPlainText("正在从 PCD 文件夹估计 LiDAR 里程计...\n")
                self.result = calibrate_lidar_imu_from_pcd_folder(
                    pcd_folder=lidar_input,
                    imu_csv=imu_csv,
                    interval_sec=self.interval_spin.value(),
                    min_rotation_deg=self.min_rotation_spin.value(),
                    max_pairs=self.max_pairs_spin.value(),
                    imu_time_offset_sec=self.imu_time_offset_spin.value(),
                    pcd_frame_interval_sec=self.pcd_frame_interval_spin.value(),
                    voxel_size=self.voxel_size_spin.value(),
                    max_correspondence_distance=self.max_corr_spin.value(),
                    icp_max_iteration=self.icp_iterations_spin.value(),
                    icp_method=str(self.icp_method_combo.currentData()),
                    max_points=self.max_points_spin.value(),
                    progress_callback=self._append_progress,
                )
            else:
                self.output.setPlainText("正在读取 LiDAR/IMU 位姿 CSV 并求解外参...\n")
                self.result = calibrate_lidar_imu(
                    lidar_csv=lidar_input,
                    imu_csv=imu_csv,
                    interval_sec=self.interval_spin.value(),
                    min_rotation_deg=self.min_rotation_spin.value(),
                    max_pairs=self.max_pairs_spin.value(),
                    imu_time_offset_sec=self.imu_time_offset_spin.value(),
                )
            self.output.setPlainText(result_to_json(self.result))
        except Exception as exc:
            QMessageBox.critical(self, "标定失败", str(exc))

    def _save_result(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "无结果", "请先运行标定。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存 LiDAR-IMU 外参", str(Path.home() / "lidar_to_imu_extrinsics.json"), "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(result_to_json(self.result), encoding="utf-8")
        self.output.appendPlainText(f"\n已保存: {path}")
