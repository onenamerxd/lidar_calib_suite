# LiDAR 标定工具箱 (LiDAR Calibration Suite)

统一的启动入口，将两个标定工具整合在一起：

- **LiDAR → Camera 联合标定** (`lidar_camera_calib`)
- **LiDAR → LiDAR 外参标定** (`lidar_extrinsic_calib_qt`)

## 运行方式

```bash
cd /home/rxd/lidar_calib_suite
./run_tool.sh
```

或者直接用 Python 启动：

```bash
/home/rxd/.ChenYX/QtWs/bin/python3 launcher.py
```

## 项目结构

```
lidar_calib_suite/
├── launcher.py                 # 统一启动器（选择界面）
├── run_tool.sh                 # 启动脚本
├── README.md
├── lidar_camera_calib/         # 激光雷达-相机联合标定
│   ├── main_window.py
│   ├── widgets.py
│   ├── io_utils.py
│   ├── math_utils.py
│   ├── models.py
│   └── settings_store.py
└── lidar_extrinsic_calib_qt/   # 激光雷达-激光雷达外参标定
    ├── main_window.py
    ├── widgets.py
    ├── calibrator.py
    └── math_utils.py
```

## 使用说明

1. 运行后会出现一个选择窗口，点击对应按钮即可打开相应的标定工具。
2. 两个工具可以同时打开，互不干扰。
3. 每个工具的使用方式与原项目完全一致。
