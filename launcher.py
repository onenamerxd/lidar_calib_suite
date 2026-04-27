from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from camera_intrinsic_calib.main_window import MainWindow as IntrinsicCalibWindow
from lidar_camera_calib.main_window import MainWindow as CameraCalibWindow
from lidar_extrinsic_calib_qt.main_window import MainWindow as LidarCalibWindow
from lidar_imu_calib_qt.main_window import MainWindow as LidarImuCalibWindow


class LauncherWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LiDAR 标定工具箱")
        self.resize(480, 460)

        self.intrinsic_window: IntrinsicCalibWindow | None = None
        self.camera_window: CameraCalibWindow | None = None
        self.lidar_window: LidarCalibWindow | None = None
        self.lidar_imu_window: LidarImuCalibWindow | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        title = QLabel("选择标定工具")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.btn_intrinsic = QPushButton("Camera 内参标定")
        self.btn_intrinsic.setMinimumHeight(60)
        self.btn_intrinsic.setStyleSheet("font-size: 16px;")
        self.btn_intrinsic.clicked.connect(self._open_intrinsic_calib)
        layout.addWidget(self.btn_intrinsic)

        self.btn_camera = QPushButton("LiDAR → Camera 联合标定")
        self.btn_camera.setMinimumHeight(60)
        self.btn_camera.setStyleSheet("font-size: 16px;")
        self.btn_camera.clicked.connect(self._open_camera_calib)
        layout.addWidget(self.btn_camera)

        self.btn_lidar = QPushButton("LiDAR → LiDAR 外参标定")
        self.btn_lidar.setMinimumHeight(60)
        self.btn_lidar.setStyleSheet("font-size: 16px;")
        self.btn_lidar.clicked.connect(self._open_lidar_calib)
        layout.addWidget(self.btn_lidar)

        self.btn_lidar_imu = QPushButton("LiDAR → IMU 外参标定")
        self.btn_lidar_imu.setMinimumHeight(60)
        self.btn_lidar_imu.setStyleSheet("font-size: 16px;")
        self.btn_lidar_imu.clicked.connect(self._open_lidar_imu_calib)
        layout.addWidget(self.btn_lidar_imu)

        hint = QLabel("提示: 四个工具可以独立运行，不会互相干扰。")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)

    def _open_intrinsic_calib(self) -> None:
        if self.intrinsic_window is None:
            self.intrinsic_window = IntrinsicCalibWindow()
        self.intrinsic_window.show()
        self.intrinsic_window.raise_()
        self.intrinsic_window.activateWindow()

    def _open_camera_calib(self) -> None:
        if self.camera_window is None:
            self.camera_window = CameraCalibWindow()
        self.camera_window.show()
        self.camera_window.raise_()
        self.camera_window.activateWindow()

    def _open_lidar_calib(self) -> None:
        if self.lidar_window is None:
            self.lidar_window = LidarCalibWindow()
        self.lidar_window.show()
        self.lidar_window.raise_()
        self.lidar_window.activateWindow()

    def _open_lidar_imu_calib(self) -> None:
        if self.lidar_imu_window is None:
            self.lidar_imu_window = LidarImuCalibWindow()
        self.lidar_imu_window.show()
        self.lidar_imu_window.raise_()
        self.lidar_imu_window.activateWindow()


def main() -> int:
    app = QApplication(sys.argv)
    launcher = LauncherWindow()
    launcher.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
