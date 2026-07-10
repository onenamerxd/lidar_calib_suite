# 标定工具箱 (Calibration Toolbox)

统一的启动入口，将多个标定工具整合在一起：

- **Camera 内参标定** (`camera_intrinsic_calib`)
- **LiDAR → Camera 联合标定** (`lidar_camera_calib`)
- **LiDAR → LiDAR 外参标定** (`lidar_extrinsic_calib_qt`)
- **LiDAR → IMU 外参标定** (`lidar_imu_calib_qt`)

## 运行方式

首次使用先创建本地 Python 环境并安装依赖：

```bash
cd /path/to/calibration_toolbox
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

如果系统提示 `ensurepip` 不可用，Ubuntu/Debian 用户需要先安装：

```bash
sudo apt install python3 python3-venv
```

```bash
cd /path/to/calibration_toolbox
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
calibration_toolbox/
├── launcher.py                 # 统一启动器（选择界面）
├── run_tool.sh                 # 启动脚本
├── README.md
├── camera_intrinsic_calib/      # 相机内参标定
│   ├── main_window.py
│   ├── widgets.py
│   └── calibrator.py
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

## LiDAR-IMU 自动标定

LiDAR-IMU 模块已替换为 OpenCalib `lidar2imu/auto_calib` 的自动标定方案复现，不再使用旧的 ICP 里程计 + 手眼标定简化流程。

### 输入数据要求

需要三类输入。

1. LiDAR PCD 文件夹

   每帧一个 `.pcd` 文件，文件名必须与 pose 文件第一列时间戳一致。

   ```text
   top_center_lidar/
   ├── 2021-10-26-16-21-29-468.pcd
   ├── 2021-10-26-16-21-29-568.pcd
   └── 2021-10-26-16-21-29-668.pcd
   ```

   推荐 PCD 字段包含：

   ```text
   x y z intensity ring
   ```

   `ring` 是激光线束编号，`intensity` 是反射强度。程序会优先使用这两个字段提取 LOAM 风格特征；如果缺少 `ring`，会按垂直角估计线束，但效果不如真实 `ring` 稳定。

2. IMU/INS pose 文件

   每行格式：

   ```text
   timestamp r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
   ```

   也就是时间戳加 3x4 位姿矩阵。第一列 `timestamp` 必须能对应到 PCD 文件名：

   ```text
   2021-10-26-16-21-29-468 -> 2021-10-26-16-21-29-468.pcd
   ```

   示例：

   ```text
   2021-10-26-16-21-29-468 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
   ```

3. 初始外参 JSON

   使用 OpenCalib 格式，核心字段为 `root.param.sensor_calib.data` 4x4 矩阵，表示初始 `IMU -> LiDAR` 外参。

   ```json
   {
     "gnss-to-top_center_lidar-extrinsic": {
       "param": {
         "sensor_calib": {
           "data": [
             [1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]
           ]
         }
       }
     }
   }
   ```

### 采集要求

- 路面尽量平坦。
- 周围需要有足够静态结构特征，例如墙面、车道线、杆、路沿、静止车辆等。
- 车辆需要有足够运动激励，OpenCalib 原说明建议按闭环轨迹行驶数圈，速度约 10 km/h。
- 尽量减少动态物体，尤其是旁车、行人、大面积移动目标。
- 时间同步要可靠；PCD 时间戳、pose 时间戳和外参初值坐标系必须一致。
- 初始外参不能偏差过大，否则体素平面约束可能收敛到错误结果。

### 界面参数含义

`优化轮数`：默认 `20`。对应 OpenCalib 原流程的多轮迭代。前半轮偏旋转粗优化，后半轮优化旋转和 XY 平移。

`滑窗帧数`：默认 `10`。每轮抽取多少帧点云一起构建体素约束。太小约束弱，太大运行慢。

`最多使用帧数`：默认 `1000`。从 pose/PCD 序列中最多使用多少帧参与抽样。数据很长时可限制运行时间。

`体素边长(m)`：默认 `1.0`。构建根体素的边长。场景稀疏时可以适当增大，特征丰富时可以适当减小。

`八叉树最大深度`：默认 `5`。体素递归细分深度。越大划分越细，但计算量越大。

`平面特征特征值比阈值`：默认 `16.0`。用于判断体素内点云是否足够接近平面。阈值越大，筛选越严格。

`每轮最大残差数`：默认 `30000`。限制每轮优化使用的点到平面残差数量，避免界面长时间卡死。机器性能较好时可以调高。

### 输出结果

保存 JSON 后重点关注：

```text
transform_imu_lidar
transform_lidar_imu
delta_transform
metrics.residual_rmse_m
```

- `transform_imu_lidar`：最终 `IMU -> LiDAR` 外参，方向与 OpenCalib 输出的 `refined_calib_imu_to_lidar.txt` 一致。
- `transform_lidar_imu`：`transform_imu_lidar` 的逆矩阵，表示 `LiDAR -> IMU`。
- `delta_transform`：在初始 `LiDAR -> IMU` 外参上优化出来的增量。
- `metrics.residual_rmse_m`：体素平面残差，越小越好。偏大通常说明时间同步、pose 质量、初始外参或点云特征存在问题。

### 可视化界面说明

当前工具箱有 PySide6 图形界面：可以选择 PCD 文件夹、pose 文件、初始外参 JSON，设置优化参数，运行后查看和保存 JSON 结果。

当前 LiDAR-IMU 自动标定模块没有点云三维交互可视化窗口，不会显示拼接前后的点云地图；它主要是参数输入、进度日志和结果输出界面。

OpenCalib 原开源项目中，`lidar2imu/manual_calib` 手动标定工具有 Pangolin 点云可视化面板，可以通过键盘或按钮调外参并观察点云对齐效果。`lidar2imu/auto_calib` 自动标定工具本身是命令行程序，主要输出优化结果和对比图片说明，不是交互式可视化软件。

核心逻辑与 OpenCalib 保持一致：

- 将初始 IMU -> LiDAR 外参取逆，得到 LiDAR -> IMU 初值，并与 pose 组合得到初始 LiDAR 位姿。
- 对每帧 PCD 提取 LOAM 风格特征，优先使用 PCD 中的 `ring` 和 `intensity` 字段；缺少 `ring` 时会按垂直角估计线束。
- 在滑窗内将特征点投到当前估计坐标系，按体素和八叉树细分构造局部平面结构。
- 前半轮只优化旋转增量，后半轮优化旋转和 XY 平移增量；Z 平移沿用初始值，这与原 C++ 残差模型一致。
- 输出 refined IMU -> LiDAR 和 LiDAR -> IMU 矩阵、每轮残差和优化增量。
