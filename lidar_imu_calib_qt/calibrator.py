from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class PoseSample:
    timestamp: float
    position: np.ndarray
    rotation: Rotation


@dataclass
class MotionPair:
    start_time: float
    end_time: float
    imu_motion: np.ndarray
    lidar_motion: np.ndarray


@dataclass
class LidarImuCalibrationResult:
    transform_imu_lidar: np.ndarray
    rotation_xyzw: list[float]
    euler_deg: list[float]
    translation: list[float]
    pair_count: int
    rotation_rmse_deg: float
    translation_rmse_m: float
    time_offset_sec: float
    interval_sec: float
    min_rotation_deg: float
    warnings: list[str]
    lidar_source: str = "csv"
    lidar_frame_count: int = 0
    lidar_registration_mean_fitness: float | None = None
    lidar_registration_mean_rmse_m: float | None = None
    lidar_registration_failed_pairs: int = 0


TIME_COLUMNS = ("timestamp", "time", "t", "stamp", "sec")
TX_COLUMNS = ("tx", "x", "px", "pos_x", "position_x")
TY_COLUMNS = ("ty", "y", "py", "pos_y", "position_y")
TZ_COLUMNS = ("tz", "z", "pz", "pos_z", "position_z")
QX_COLUMNS = ("qx", "quat_x", "orientation_x")
QY_COLUMNS = ("qy", "quat_y", "orientation_y")
QZ_COLUMNS = ("qz", "quat_z", "orientation_z")
QW_COLUMNS = ("qw", "quat_w", "orientation_w")
ROLL_COLUMNS = ("roll_deg", "roll", "rx")
PITCH_COLUMNS = ("pitch_deg", "pitch", "ry")
YAW_COLUMNS = ("yaw_deg", "yaw", "rz")
PCD_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)")


def _pick(row: dict[str, str], names: Iterable[str]) -> str | None:
    lowered = {key.strip().lower(): value for key, value in row.items() if key is not None}
    for name in names:
        if name in lowered and lowered[name] != "":
            return lowered[name]
    return None


def _float_from(row: dict[str, str], names: Iterable[str], label: str) -> float:
    raw = _pick(row, names)
    if raw is None:
        raise ValueError(f"CSV 缺少字段: {label}")
    return float(raw)


def _rotation_from_row(row: dict[str, str]) -> Rotation:
    qx = _pick(row, QX_COLUMNS)
    qy = _pick(row, QY_COLUMNS)
    qz = _pick(row, QZ_COLUMNS)
    qw = _pick(row, QW_COLUMNS)
    if None not in (qx, qy, qz, qw):
        return Rotation.from_quat([float(qx), float(qy), float(qz), float(qw)])

    roll = _pick(row, ROLL_COLUMNS)
    pitch = _pick(row, PITCH_COLUMNS)
    yaw = _pick(row, YAW_COLUMNS)
    if None not in (roll, pitch, yaw):
        return Rotation.from_euler("xyz", [float(roll), float(pitch), float(yaw)], degrees=True)

    raise ValueError("CSV 需要 qx,qy,qz,qw 或 roll_deg,pitch_deg,yaw_deg")


def _has_header(first_line: str) -> bool:
    tokens = [token.strip() for token in first_line.split(",")]
    for token in tokens:
        try:
            float(token)
        except ValueError:
            return True
    return False


def load_pose_csv(path: str | Path, time_offset_sec: float = 0.0) -> list[PoseSample]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    text = csv_path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"CSV 为空: {csv_path}")

    samples: list[PoseSample] = []
    lines = text.splitlines()
    if _has_header(lines[0]):
        for row in csv.DictReader(lines):
            timestamp = _float_from(row, TIME_COLUMNS, "timestamp/time/t") + time_offset_sec
            position = np.array(
                [
                    _float_from(row, TX_COLUMNS, "tx/x"),
                    _float_from(row, TY_COLUMNS, "ty/y"),
                    _float_from(row, TZ_COLUMNS, "tz/z"),
                ],
                dtype=np.float64,
            )
            samples.append(PoseSample(timestamp=timestamp, position=position, rotation=_rotation_from_row(row)))
    else:
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 8:
            raise ValueError("无表头 CSV 至少需要 8 列: timestamp,tx,ty,tz,qx,qy,qz,qw")
        for row in data:
            samples.append(
                PoseSample(
                    timestamp=float(row[0]) + time_offset_sec,
                    position=np.asarray(row[1:4], dtype=np.float64),
                    rotation=Rotation.from_quat(row[4:8]),
                )
            )

    samples.sort(key=lambda sample: sample.timestamp)
    deduped: list[PoseSample] = []
    for sample in samples:
        if deduped and abs(sample.timestamp - deduped[-1].timestamp) < 1e-9:
            continue
        deduped.append(sample)
    if len(deduped) < 3:
        raise ValueError(f"位姿数量不足: {csv_path}")
    return deduped


def _load_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("PCD 文件夹模式需要安装 open3d，请先执行: pip install -r requirements.txt") from exc
    return o3d


def _last_number_token(text: str) -> str | None:
    matches = PCD_NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1]


def _pcd_sort_key(path: Path) -> tuple[int, float | str]:
    token = _last_number_token(path.stem)
    if token is None:
        return (1, path.name)
    return (0, float(token))


def collect_pcd_files(folder: str | Path) -> list[Path]:
    pcd_folder = Path(folder)
    if not pcd_folder.exists():
        raise FileNotFoundError(pcd_folder)
    if not pcd_folder.is_dir():
        raise ValueError(f"PCD 输入必须是文件夹: {pcd_folder}")

    files = [path for path in pcd_folder.iterdir() if path.is_file() and path.suffix.lower() == ".pcd"]
    files.sort(key=_pcd_sort_key)
    if len(files) < 3:
        raise ValueError(f"PCD 文件数量不足，至少需要 3 帧: {pcd_folder}")
    return files


def _timestamp_from_numeric_token(token: str) -> float:
    value = float(token)
    if "." in token:
        return value

    digits = token.lstrip("+-")
    if len(digits) >= 18:
        return value / 1e9
    if len(digits) >= 15:
        return value / 1e6
    if len(digits) >= 12:
        return value / 1e3
    return value


def infer_pcd_timestamps(files: list[Path], frame_interval_sec: float) -> list[float]:
    if frame_interval_sec <= 0.0:
        raise ValueError("PCD 帧间隔必须大于 0")

    tokens = [_last_number_token(path.stem) for path in files]
    if all(token is not None for token in tokens):
        numeric_tokens = [token for token in tokens if token is not None]
        looks_like_timestamp = any("." in token for token in numeric_tokens) or max(len(token.lstrip("+-")) for token in numeric_tokens) >= 10
        if looks_like_timestamp:
            timestamps = [_timestamp_from_numeric_token(token) for token in numeric_tokens]
            if len(set(timestamps)) == len(timestamps):
                return timestamps

    return [index * frame_interval_sec for index in range(len(files))]


def load_registration_cloud(
    path: Path,
    voxel_size: float,
    max_points: int,
    estimate_normals: bool,
    normal_radius: float,
):
    o3d = _load_open3d()
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"点云为空: {path}")

    cloud = cloud.remove_non_finite_points()
    if voxel_size > 0.0:
        cloud = cloud.voxel_down_sample(voxel_size)
    point_count = len(cloud.points)
    if point_count < 20:
        raise ValueError(f"点云有效点太少: {path}")
    if max_points > 0 and point_count > max_points:
        indices = np.linspace(0, point_count - 1, max_points, dtype=np.int64)
        cloud = cloud.select_by_index(indices.tolist())

    if estimate_normals:
        search_radius = max(normal_radius, voxel_size * 3.0, 0.5)
        cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30))
    return cloud


def estimate_lidar_odometry_from_pcd_folder(
    folder: str | Path,
    frame_interval_sec: float,
    voxel_size: float,
    max_correspondence_distance: float,
    icp_max_iteration: int,
    icp_method: str = "point_to_point",
    max_points: int = 80000,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[PoseSample], dict[str, float | int | str | list[str]]]:
    if max_correspondence_distance <= 0.0:
        raise ValueError("ICP 最大对应距离必须大于 0")
    if icp_max_iteration <= 0:
        raise ValueError("ICP 最大迭代次数必须大于 0")

    o3d = _load_open3d()
    files = collect_pcd_files(folder)
    timestamps = infer_pcd_timestamps(files, frame_interval_sec)
    if progress_callback is not None:
        progress_callback(f"发现 {len(files)} 帧 PCD，开始相邻帧 ICP 里程计。")

    use_point_to_plane = icp_method == "point_to_plane"
    if use_point_to_plane:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(icp_max_iteration))
    normal_radius = max(max_correspondence_distance * 2.0, voxel_size * 3.0, 0.5)

    current_pose = np.eye(4, dtype=np.float64)
    samples = [PoseSample(timestamp=float(timestamps[0]), position=current_pose[:3, 3].copy(), rotation=Rotation.identity())]
    fitness_values: list[float] = []
    rmse_values: list[float] = []
    warnings: list[str] = []

    target = load_registration_cloud(files[0], voxel_size, max_points, use_point_to_plane, normal_radius)
    for index, source_path in enumerate(files[1:], start=1):
        if progress_callback is not None:
            progress_callback(f"ICP {index}/{len(files) - 1}: {source_path.name}")

        source = load_registration_cloud(source_path, voxel_size, max_points, use_point_to_plane, normal_radius)
        registration = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance,
            np.eye(4, dtype=np.float64),
            estimation,
            criteria,
        )
        transform_prev_curr = np.asarray(registration.transformation, dtype=np.float64)
        current_pose = current_pose @ transform_prev_curr

        try:
            rotation = Rotation.from_matrix(current_pose[:3, :3])
        except ValueError as exc:
            raise ValueError(f"ICP 结果旋转矩阵异常: {source_path}") from exc

        samples.append(
            PoseSample(
                timestamp=float(timestamps[index]),
                position=current_pose[:3, 3].copy(),
                rotation=rotation,
            )
        )
        fitness_values.append(float(registration.fitness))
        rmse_values.append(float(registration.inlier_rmse))
        if registration.fitness < 0.05:
            warnings.append(f"{source_path.name} ICP fitness 过低({registration.fitness:.3f})，该段里程计可能不可靠。")
        target = source

    failed_pairs = sum(1 for value in fitness_values if value < 0.05)
    summary: dict[str, float | int | str | list[str]] = {
        "source": str(Path(folder)),
        "frame_count": len(files),
        "relative_count": max(0, len(files) - 1),
        "mean_fitness": float(np.mean(fitness_values)) if fitness_values else 0.0,
        "mean_rmse_m": float(np.mean(rmse_values)) if rmse_values else 0.0,
        "failed_pairs": int(failed_pairs),
        "warnings": warnings,
    }
    return samples, summary


def pose_to_matrix(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def interpolate_pose(samples: list[PoseSample], timestamp: float) -> np.ndarray:
    times = np.array([sample.timestamp for sample in samples], dtype=np.float64)
    if timestamp < times[0] or timestamp > times[-1]:
        raise ValueError(f"时间 {timestamp:.6f} 超出轨迹范围 [{times[0]:.6f}, {times[-1]:.6f}]")

    positions = np.vstack([sample.position for sample in samples])
    position = np.array([np.interp(timestamp, times, positions[:, axis]) for axis in range(3)], dtype=np.float64)
    rotations = Rotation.concatenate([sample.rotation for sample in samples])
    rotation = Slerp(times, rotations)([timestamp])[0]
    return pose_to_matrix(position, rotation)


def relative_motion(start_pose: np.ndarray, end_pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(start_pose) @ end_pose


def build_motion_pairs(
    lidar_samples: list[PoseSample],
    imu_samples: list[PoseSample],
    interval_sec: float,
    min_rotation_deg: float,
    max_pairs: int,
) -> list[MotionPair]:
    if interval_sec <= 0.0:
        raise ValueError("运动间隔必须大于 0")

    lidar_start, lidar_end = lidar_samples[0].timestamp, lidar_samples[-1].timestamp
    imu_start, imu_end = imu_samples[0].timestamp, imu_samples[-1].timestamp
    start = max(lidar_start, imu_start)
    end = min(lidar_end, imu_end) - interval_sec
    if end <= start:
        raise ValueError("LiDAR 和 IMU 轨迹时间没有足够重叠区间")

    lidar_times = np.array([sample.timestamp for sample in lidar_samples], dtype=np.float64)
    candidate_times = lidar_times[(lidar_times >= start) & (lidar_times <= end)]
    if candidate_times.size == 0:
        candidate_times = np.linspace(start, end, max(2, min(max_pairs, 200)))
    if candidate_times.size > max_pairs:
        indices = np.linspace(0, candidate_times.size - 1, max_pairs).astype(int)
        candidate_times = candidate_times[indices]

    pairs: list[MotionPair] = []
    min_rot_rad = np.deg2rad(min_rotation_deg)
    for time in candidate_times:
        next_time = float(time + interval_sec)
        lidar_motion = relative_motion(interpolate_pose(lidar_samples, float(time)), interpolate_pose(lidar_samples, next_time))
        imu_motion = relative_motion(interpolate_pose(imu_samples, float(time)), interpolate_pose(imu_samples, next_time))
        lidar_angle = np.linalg.norm(Rotation.from_matrix(lidar_motion[:3, :3]).as_rotvec())
        imu_angle = np.linalg.norm(Rotation.from_matrix(imu_motion[:3, :3]).as_rotvec())
        if max(lidar_angle, imu_angle) < min_rot_rad:
            continue
        pairs.append(MotionPair(start_time=float(time), end_time=next_time, imu_motion=imu_motion, lidar_motion=lidar_motion))

    if len(pairs) < 3:
        raise ValueError("有效运动片段不足。请增大运动幅度、减小最小旋转阈值，或调整运动间隔。")
    return pairs


def solve_hand_eye(motion_pairs: list[MotionPair]) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []

    def rotation_residual(rotvec: np.ndarray) -> np.ndarray:
        rotation_x = Rotation.from_rotvec(rotvec).as_matrix()
        residuals = []
        for pair in motion_pairs:
            rotation_a = pair.imu_motion[:3, :3]
            rotation_b = pair.lidar_motion[:3, :3]
            error = rotation_a @ rotation_x @ rotation_b.T @ rotation_x.T
            residuals.append(Rotation.from_matrix(error).as_rotvec())
        return np.concatenate(residuals)

    rotation_result = least_squares(rotation_residual, x0=np.zeros(3), loss="soft_l1", f_scale=0.1, max_nfev=300)
    rotation_x = Rotation.from_rotvec(rotation_result.x).as_matrix()

    lhs_rows = []
    rhs_rows = []
    for pair in motion_pairs:
        rotation_a = pair.imu_motion[:3, :3]
        translation_a = pair.imu_motion[:3, 3]
        translation_b = pair.lidar_motion[:3, 3]
        lhs_rows.append(rotation_a - np.eye(3))
        rhs_rows.append(rotation_x @ translation_b - translation_a)

    lhs = np.vstack(lhs_rows)
    rhs = np.concatenate(rhs_rows)
    translation_x, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

    rank = np.linalg.matrix_rank(lhs)
    if rank < 3:
        warnings.append("平移约束矩阵秩不足，平移结果可能不可靠。需要更多非共线运动。")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_x
    transform[:3, 3] = translation_x
    return transform, warnings


def compute_motion_errors(transform_imu_lidar: np.ndarray, motion_pairs: list[MotionPair]) -> tuple[float, float]:
    rotation_errors = []
    translation_errors = []
    inv_left = None
    for pair in motion_pairs:
        left = pair.imu_motion @ transform_imu_lidar
        right = transform_imu_lidar @ pair.lidar_motion
        inv_left = np.linalg.inv(left)
        error = inv_left @ right
        rotation_errors.append(np.linalg.norm(Rotation.from_matrix(error[:3, :3]).as_rotvec()))
        translation_errors.append(np.linalg.norm(error[:3, 3]))

    if inv_left is None:
        return float("inf"), float("inf")
    return float(np.rad2deg(np.sqrt(np.mean(np.square(rotation_errors))))), float(np.sqrt(np.mean(np.square(translation_errors))))


def calibrate_lidar_imu(
    lidar_csv: str | Path,
    imu_csv: str | Path,
    interval_sec: float,
    min_rotation_deg: float,
    max_pairs: int,
    imu_time_offset_sec: float = 0.0,
) -> LidarImuCalibrationResult:
    lidar_samples = load_pose_csv(lidar_csv)
    return calibrate_lidar_imu_from_samples(
        lidar_samples=lidar_samples,
        imu_csv=imu_csv,
        interval_sec=interval_sec,
        min_rotation_deg=min_rotation_deg,
        max_pairs=max_pairs,
        imu_time_offset_sec=imu_time_offset_sec,
        lidar_source=f"csv:{Path(lidar_csv)}",
        lidar_frame_count=len(lidar_samples),
    )


def calibrate_lidar_imu_from_samples(
    lidar_samples: list[PoseSample],
    imu_csv: str | Path,
    interval_sec: float,
    min_rotation_deg: float,
    max_pairs: int,
    imu_time_offset_sec: float = 0.0,
    lidar_source: str = "samples",
    lidar_frame_count: int | None = None,
    lidar_registration_mean_fitness: float | None = None,
    lidar_registration_mean_rmse_m: float | None = None,
    lidar_registration_failed_pairs: int = 0,
    extra_warnings: list[str] | None = None,
) -> LidarImuCalibrationResult:
    imu_samples = load_pose_csv(imu_csv, time_offset_sec=imu_time_offset_sec)
    motion_pairs = build_motion_pairs(
        lidar_samples=lidar_samples,
        imu_samples=imu_samples,
        interval_sec=interval_sec,
        min_rotation_deg=min_rotation_deg,
        max_pairs=max_pairs,
    )
    transform, warnings = solve_hand_eye(motion_pairs)
    rotation_rmse_deg, translation_rmse_m = compute_motion_errors(transform, motion_pairs)
    rotation = Rotation.from_matrix(transform[:3, :3])
    quat = rotation.as_quat()
    euler = rotation.as_euler("xyz", degrees=True)

    if len(motion_pairs) < 10:
        warnings.append("有效运动片段少于 10 组，建议采集更长且运动更丰富的数据。")
    if rotation_rmse_deg > 2.0:
        warnings.append("旋转残差偏大，检查时间同步、坐标系约定和轨迹质量。")
    if not np.isfinite(translation_rmse_m) or translation_rmse_m > 0.5:
        warnings.append("平移残差偏大，平移外参可能不可靠。")
    if extra_warnings:
        warnings.extend(extra_warnings)

    return LidarImuCalibrationResult(
        transform_imu_lidar=transform,
        rotation_xyzw=[float(value) for value in quat],
        euler_deg=[float(value) for value in euler],
        translation=[float(value) for value in transform[:3, 3]],
        pair_count=len(motion_pairs),
        rotation_rmse_deg=rotation_rmse_deg,
        translation_rmse_m=translation_rmse_m,
        time_offset_sec=float(imu_time_offset_sec),
        interval_sec=float(interval_sec),
        min_rotation_deg=float(min_rotation_deg),
        warnings=warnings,
        lidar_source=lidar_source,
        lidar_frame_count=int(lidar_frame_count if lidar_frame_count is not None else len(lidar_samples)),
        lidar_registration_mean_fitness=lidar_registration_mean_fitness,
        lidar_registration_mean_rmse_m=lidar_registration_mean_rmse_m,
        lidar_registration_failed_pairs=int(lidar_registration_failed_pairs),
    )


def calibrate_lidar_imu_from_pcd_folder(
    pcd_folder: str | Path,
    imu_csv: str | Path,
    interval_sec: float,
    min_rotation_deg: float,
    max_pairs: int,
    imu_time_offset_sec: float = 0.0,
    pcd_frame_interval_sec: float = 0.1,
    voxel_size: float = 0.5,
    max_correspondence_distance: float = 1.5,
    icp_max_iteration: int = 50,
    icp_method: str = "point_to_point",
    max_points: int = 80000,
    progress_callback: Callable[[str], None] | None = None,
) -> LidarImuCalibrationResult:
    lidar_samples, summary = estimate_lidar_odometry_from_pcd_folder(
        folder=pcd_folder,
        frame_interval_sec=pcd_frame_interval_sec,
        voxel_size=voxel_size,
        max_correspondence_distance=max_correspondence_distance,
        icp_max_iteration=icp_max_iteration,
        icp_method=icp_method,
        max_points=max_points,
        progress_callback=progress_callback,
    )
    warnings = list(summary.get("warnings", []))
    mean_fitness = float(summary["mean_fitness"])
    if mean_fitness < 0.2:
        warnings.append("PCD ICP 平均 fitness 偏低，建议增大最大对应距离、减小运动间隔或检查点云重叠。")

    return calibrate_lidar_imu_from_samples(
        lidar_samples=lidar_samples,
        imu_csv=imu_csv,
        interval_sec=interval_sec,
        min_rotation_deg=min_rotation_deg,
        max_pairs=max_pairs,
        imu_time_offset_sec=imu_time_offset_sec,
        lidar_source=f"pcd_folder:{Path(pcd_folder)}",
        lidar_frame_count=int(summary["frame_count"]),
        lidar_registration_mean_fitness=mean_fitness,
        lidar_registration_mean_rmse_m=float(summary["mean_rmse_m"]),
        lidar_registration_failed_pairs=int(summary["failed_pairs"]),
        extra_warnings=warnings,
    )


def result_to_json(result: LidarImuCalibrationResult) -> str:
    payload = {
        "convention": "P_imu = R_imu_lidar * P_lidar + t_imu_lidar",
        "transform_imu_lidar": result.transform_imu_lidar.tolist(),
        "translation": result.translation,
        "rotation_xyzw": result.rotation_xyzw,
        "euler_deg": {
            "roll": result.euler_deg[0],
            "pitch": result.euler_deg[1],
            "yaw": result.euler_deg[2],
        },
        "metrics": {
            "pair_count": result.pair_count,
            "rotation_rmse_deg": result.rotation_rmse_deg,
            "translation_rmse_m": result.translation_rmse_m,
            "time_offset_sec": result.time_offset_sec,
            "interval_sec": result.interval_sec,
            "min_rotation_deg": result.min_rotation_deg,
            "lidar_source": result.lidar_source,
            "lidar_frame_count": result.lidar_frame_count,
            "lidar_registration_mean_fitness": result.lidar_registration_mean_fitness,
            "lidar_registration_mean_rmse_m": result.lidar_registration_mean_rmse_m,
            "lidar_registration_failed_pairs": result.lidar_registration_failed_pairs,
        },
        "warnings": result.warnings,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
