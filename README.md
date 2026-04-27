# LiDAR 标定工具箱 (LiDAR Calibration Suite)

统一的启动入口，将多个标定工具整合在一起：

- **LiDAR → Camera 联合标定** (`lidar_camera_calib`)
- **LiDAR → LiDAR 外参标定** (`lidar_extrinsic_calib_qt`)
- **LiDAR → IMU 外参标定** (`lidar_imu_calib_qt`)

## 运行方式

首次使用先创建本地 Python 环境并安装依赖：

```bash
cd /path/to/lidar_calib_suite
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

如果系统提示 `ensurepip` 不可用，Ubuntu/Debian 用户需要先安装：

```bash
sudo apt install python3 python3-venv
```

```bash
cd /path/to/lidar_calib_suite
source .venv/bin/activate
./run_tool.sh
```

`run_tool.sh` 只是项目启动入口，会优先使用已激活的 Conda / venv 环境，也兼容旧安装器生成的 `../.miniconda3/envs/calib`。

或者直接用当前环境的 Python 启动：

```bash
.venv/bin/python launcher.py
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
├── lidar_extrinsic_calib_qt/   # 激光雷达-激光雷达外参标定
    ├── main_window.py
    ├── widgets.py
    ├── calibrator.py
    └── math_utils.py
└── lidar_imu_calib_qt/         # 激光雷达-IMU 外参标定
    ├── main_window.py
    └── calibrator.py
```

## 使用说明

1. 运行后会出现一个选择窗口，点击对应按钮即可打开相应的标定工具。
2. 多个工具可以同时打开，互不干扰。
3. 每个工具的使用方式与原项目完全一致。

## LiDAR-IMU 标定输入

LiDAR-IMU 模块是简化版运动约束标定器。LiDAR 输入支持两种方式：

- LiDAR odometry 位姿 CSV。
- 连续 PCD 文件夹。程序会用相邻帧 ICP 自动估计 LiDAR odometry，并以第一帧点云坐标作为 LiDAR 世界系。

IMU/INS 仍需要输入一条位姿 CSV。只有原始 IMU 加速度/角速度时，当前简化模块不能直接完成外参标定，需要先用 INS/VIO/LIO 等方法得到 IMU 轨迹，或进一步实现完整的 raw IMU 优化模型。

支持 CSV 表头：

```csv
timestamp,tx,ty,tz,qx,qy,qz,qw
```

或：

```csv
timestamp,tx,ty,tz,roll_deg,pitch_deg,yaw_deg
```

位姿约定为 `T_world_sensor`，输出 `T_imu_lidar`，满足 `P_imu = R_imu_lidar * P_lidar + t_imu_lidar`。

PCD 文件夹模式会优先从文件名推断时间戳。例如 `1700000000.123.pcd` 会按秒级时间戳读取；如果文件名只是 `000001.pcd`、`000002.pcd` 这类序号，则按界面里的 `PCD 帧间隔(s)` 生成时间。
