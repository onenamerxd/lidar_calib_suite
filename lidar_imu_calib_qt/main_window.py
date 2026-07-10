from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
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

from .calibrator import LidarImuCalibrationResult, calibrate_lidar_imu_open_calib, result_to_json


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LiDAR → IMU 外参标定")
        self.resize(980, 760)
        self.result: LidarImuCalibrationResult | None = None

        self._build_ui()
        self._connect_signals()
        self._show_csv_hint()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignTop)

        title = QLabel("LiDAR → IMU OpenCalib 自动标定")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        file_group = QGroupBox("OpenCalib 数据输入")
        file_layout = QFormLayout(file_group)
        self.pcd_folder_edit = QLineEdit()
        self.pose_file_edit = QLineEdit()
        self.extrinsic_json_edit = QLineEdit()
        file_layout.addRow("LiDAR PCD 文件夹", self._line_with_button(self.pcd_folder_edit, self._choose_pcd_folder))
        file_layout.addRow("IMU/INS pose 文件", self._line_with_button(self.pose_file_edit, self._choose_pose_file))
        file_layout.addRow("初始外参 JSON", self._line_with_button(self.extrinsic_json_edit, self._choose_extrinsic_json))
        layout.addWidget(file_group)

        param_group = QGroupBox("标定参数")
        param_layout = QFormLayout(param_group)
        self.turn_count_spin = QSpinBox()
        self.turn_count_spin.setRange(1, 50)
        self.turn_count_spin.setValue(20)
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(3, 50)
        self.window_size_spin.setValue(10)
        self.upper_bound_spin = QSpinBox()
        self.upper_bound_spin.setRange(20, 100000)
        self.upper_bound_spin.setValue(1000)
        self.voxel_size_spin = self._make_double_spin(0.1, 5.0, 1.0, 0.1, 3)
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 8)
        self.max_depth_spin.setValue(5)
        self.eigen_limit_spin = self._make_double_spin(1.0, 100.0, 16.0, 1.0, 3)
        self.max_residuals_spin = QSpinBox()
        self.max_residuals_spin.setRange(1000, 500000)
        self.max_residuals_spin.setValue(30000)
        self.max_residuals_spin.setSingleStep(5000)
        param_layout.addRow("优化轮数", self.turn_count_spin)
        param_layout.addRow("滑窗帧数", self.window_size_spin)
        param_layout.addRow("最多使用帧数", self.upper_bound_spin)
        param_layout.addRow("体素边长(m)", self.voxel_size_spin)
        param_layout.addRow("八叉树最大深度", self.max_depth_spin)
        param_layout.addRow("平面特征特征值比阈值", self.eigen_limit_spin)
        param_layout.addRow("每轮最大残差数", self.max_residuals_spin)
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

    def _choose_pcd_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 LiDAR PCD 文件夹", self.pcd_folder_edit.text() or str(Path.home()))
        if path:
            self.pcd_folder_edit.setText(path)

    def _choose_pose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 IMU/INS pose 文件", self.pose_file_edit.text() or str(Path.home()), "Text (*.txt *.csv);;All Files (*)")
        if path:
            self.pose_file_edit.setText(path)

    def _choose_extrinsic_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择初始外参 JSON", self.extrinsic_json_edit.text() or str(Path.home()), "JSON (*.json);;All Files (*)")
        if path:
            self.extrinsic_json_edit.setText(path)

    def _show_csv_hint(self) -> None:
        self.output.setPlainText(
            "输入说明:\n"
            "  本模块已替换为 OpenCalib lidar2imu 自动标定方案的 Python 复现。\n"
            "  LiDAR PCD 文件夹内的文件名必须与 pose 文件第一列时间戳一致，例如 xxx.pcd。\n"
            "  pose 文件每行格式: timestamp r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz。\n"
            "  初始外参 JSON 使用 OpenCalib root.param.sensor_calib.data 4x4 格式。\n\n"
            "核心流程:\n"
            "  1. 用初始 IMU->LiDAR 外参反算 LiDAR->IMU，并把 pose 转为初始 LiDAR 位姿。\n"
            "  2. 对每帧 PCD 提取 LOAM 风格面/边特征，按滑窗投到同一坐标系。\n"
            "  3. 八叉树细分体素，保留平面结构明显的体素。\n"
            "  4. 前半轮优化旋转增量，后半轮优化旋转和 XY 平移增量，Z 平移沿用初值。\n"
        )

    def _append_progress(self, message: str) -> None:
        self.output.appendPlainText(message)
        QApplication.processEvents()

    def _run_calibration(self) -> None:
        pcd_folder = self.pcd_folder_edit.text().strip()
        pose_file = self.pose_file_edit.text().strip()
        extrinsic_json = self.extrinsic_json_edit.text().strip()
        if not pcd_folder or not pose_file or not extrinsic_json:
            QMessageBox.warning(self, "缺少输入", "请先选择 LiDAR PCD 文件夹、pose 文件和初始外参 JSON。")
            return

        try:
            self.output.setPlainText("正在运行 OpenCalib LiDAR-IMU 自动标定...\n")
            self.result = calibrate_lidar_imu_open_calib(
                pcd_folder=pcd_folder,
                pose_file=pose_file,
                extrinsic_json=extrinsic_json,
                turn_count=self.turn_count_spin.value(),
                window_size=self.window_size_spin.value(),
                upper_bound=self.upper_bound_spin.value(),
                voxel_size=self.voxel_size_spin.value(),
                max_depth=self.max_depth_spin.value(),
                eigen_limit=self.eigen_limit_spin.value(),
                max_residuals=self.max_residuals_spin.value(),
                progress_callback=self._append_progress,
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
