# LiDAR-IMU OpenCalib 自动标定方案

本模块替换掉原来的 ICP 里程计 + AX=XB 手眼标定简化方案，迁移 OpenCalib `lidar2imu/auto_calib` 的核心逻辑。

## 输入数据

1. LiDAR PCD 文件夹

   文件名必须与 pose 文件第一列一致：

   ```text
   top_center_lidar/
   ├── 2021-10-26-16-21-29-468.pcd
   ├── 2021-10-26-16-21-29-568.pcd
   └── ...
   ```

   优先使用 PCD 中的 `x y z intensity ring` 字段。缺少 `ring` 时会按垂直角估计线束；缺少 `intensity` 时按高强度处理。

2. pose 文件

   每行格式：

   ```text
   timestamp r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
   ```

   矩阵部分是 3x4 位姿。迁移逻辑沿用 OpenCalib：读取该矩阵后右乘初始 `T_lidar_to_imu`，作为优化中使用的初始 LiDAR 位姿。

3. 初始外参 JSON

   OpenCalib 格式：

   ```json
   {
     "gnss-to-top_center_lidar-extrinsic": {
       "param": {
         "sensor_calib": {
           "data": [[...], [...], [...], [0, 0, 0, 1]]
         }
       }
     }
   }
   ```

   `sensor_calib.data` 被视为初始 `T_imu_to_lidar`，内部会取逆得到 `T_lidar_to_imu`。

## 核心流程

1. 加载初始外参和 pose。
2. 选择若干轮滑窗。默认 20 轮，每轮 10 帧，最多使用前 1000 帧。
3. 读取每个滑窗中的 PCD，并提取 LOAM 风格特征：
   - 按 64 线 ring 分组。
   - 通过前后 5 点计算曲率。
   - 每条线切成 30 段。
   - 选高曲率角点和低曲率面点。
   - 对 less-flat 面点做 0.2 m 体素降采样。
4. 用当前外参增量把特征点投到 pose 坐标系。
5. 构建 1 m 根体素，并递归八叉树细分到最大 5 层。
6. 对每个叶子体素计算协方差特征值，只保留平面性明显且法向接近 Z 轴的体素。
7. 用体素内平面残差优化外参增量：
   - 前半轮只优化旋转增量。
   - 后半轮优化旋转和 X/Y 平移增量。
   - Z 平移保持初值，和 OpenCalib C++ 残差中注释掉 `t[2]` 的行为一致。
8. 输出 refined `T_imu_to_lidar`、`T_lidar_to_imu`、每轮 RMSE、体素数、残差数和最终增量。

## 与 OpenCalib 原代码的对应关系

- `calibration.cpp::LoadTimeAndPoes` -> `load_open_calib_pose_file`
- `gen_BALM_feature.hpp::genPcdFeature` -> `extract_loam_features`
- `BALM.hpp::cut_voxel/recut/calc_eigen` -> `build_voxel_leaves`
- `BALM.hpp::optimizeDeltaTrans` -> `_calibrate_round`
- `run_lidar2imu.cpp` -> `calibrate_lidar_imu_open_calib`

## 复现注意事项

- 原 OpenCalib 样例 PCD 是 `binary_compressed`，需要运行环境安装 `open3d`。
- 场景需要平坦路面和足够静态特征；动态车辆、时间不同步、初始外参偏差过大都会导致体素平面残差偏大。
- Python 版为了 GUI 可用性限制了每轮最大残差数，默认 30000。数据很大且机器性能允许时可以调高。
