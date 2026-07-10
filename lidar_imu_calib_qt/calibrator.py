from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ProgressCallback = Callable[[str], None]

LIDAR_LASER_NUM = 64
SCAN_LINE_CUT = 30
INTENSITY_THRESHOLD = 35.0
MIN_POINTS_PER_VOXEL = 7


@dataclass
class OpenCalibPoseFrame:
    stamp: str
    pose: np.ndarray


@dataclass
class PcdFrame:
    points: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray


@dataclass
class VoxelLeaf:
    points_orig: list[np.ndarray]
    points_tran: list[np.ndarray]
    center: np.ndarray
    normal: np.ndarray
    center_zero: np.ndarray
    normal_zero: np.ndarray
    center_orig: np.ndarray
    normal_orig: np.ndarray
    eigen_ratio: float


@dataclass
class CalibrationRoundInfo:
    round_index: int
    start_index: int
    step: int
    frame_count: int
    feature_points: int
    voxel_count: int
    residual_count: int
    cost_before: float
    cost_after: float
    delta_rpy_deg: list[float]
    delta_t: list[float]


@dataclass
class LidarImuCalibrationResult:
    transform_imu_lidar: np.ndarray
    transform_lidar_imu: np.ndarray
    initial_transform_imu_lidar: np.ndarray
    initial_transform_lidar_imu: np.ndarray
    delta_transform: np.ndarray
    rotation_xyzw: list[float]
    euler_deg: list[float]
    translation: list[float]
    refined_lidar_to_imu_euler_deg: list[float]
    refined_lidar_to_imu_translation: list[float]
    pose_count: int
    used_frame_count: int
    pcd_frame_count: int
    round_count: int
    residual_rmse_m: float
    delta_rpy_deg: list[float]
    delta_translation: list[float]
    warnings: list[str]
    rounds: list[CalibrationRoundInfo]
    lidar_source: str = "open_calib_pcd"
    lidar_frame_count: int = 0
    pair_count: int = 0
    rotation_rmse_deg: float = 0.0
    translation_rmse_m: float = 0.0
    time_offset_sec: float = 0.0
    interval_sec: float = 0.0
    min_rotation_deg: float = 0.0
    lidar_registration_mean_fitness: float | None = None
    lidar_registration_mean_rmse_m: float | None = None
    lidar_registration_failed_pairs: int = 0


def _load_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("LiDAR-IMU 自动标定读取 PCD 需要 open3d，请先安装 requirements.txt。") from exc
    return o3d


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def make_delta_transform(rotvec: np.ndarray, trans_xy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    transform[0, 3] = float(trans_xy[0])
    transform[1, 3] = float(trans_xy[1])
    return transform


def _floor_voxel(points: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor(points / voxel_size).astype(np.int64)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0 or voxel_size <= 0:
        return points
    keys = _floor_voxel(points, voxel_size)
    buckets: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for key, point in zip(map(tuple, keys), points, strict=False):
        buckets.setdefault(key, []).append(point)
    return np.asarray([np.mean(bucket, axis=0) for bucket in buckets.values()], dtype=np.float64)


def load_open_calib_pose_file(path: str | Path, initial_lidar_to_imu: np.ndarray) -> list[OpenCalibPoseFrame]:
    pose_path = Path(path)
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    frames: list[OpenCalibPoseFrame] = []
    for line_no, line in enumerate(pose_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 13:
            raise ValueError(f"pose 文件第 {line_no} 行字段不足，需要 timestamp + 12 个矩阵数值。")
        pose = np.eye(4, dtype=np.float64)
        values = [float(value) for value in parts[1:13]]
        pose[:3, :] = np.asarray(values, dtype=np.float64).reshape(3, 4)
        frames.append(OpenCalibPoseFrame(stamp=parts[0], pose=pose @ initial_lidar_to_imu))
    if len(frames) < 10:
        raise ValueError("pose 数量不足，OpenCalib 自动标定建议至少几十帧，并覆盖多段运动。")
    return frames


def load_open_calib_extrinsic_json(path: str | Path) -> np.ndarray:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    if "transform_imu_lidar" in payload:
        return np.asarray(payload["transform_imu_lidar"], dtype=np.float64)
    if "param" in payload and "sensor_calib" in payload["param"]:
        data = payload["param"]["sensor_calib"]["data"]
        return np.asarray(data, dtype=np.float64)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("初始外参 JSON 为空或格式不正确。")
    first_key = next(iter(payload))
    try:
        data = payload[first_key]["param"]["sensor_calib"]["data"]
    except (KeyError, TypeError) as exc:
        raise ValueError("初始外参 JSON 需要包含 root.param.sensor_calib.data 4x4 矩阵。") from exc
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("初始外参矩阵必须是 4x4。")
    return matrix


def _parse_pcd_header(raw: bytes) -> tuple[dict[str, list[str] | int], int]:
    marker = b"DATA "
    pos = raw.find(marker)
    if pos < 0:
        raise ValueError("PCD 文件缺少 DATA 头。")
    line_end = raw.find(b"\n", pos)
    if line_end < 0:
        raise ValueError("PCD DATA 头不完整。")
    header_text = raw[: line_end + 1].decode("latin1")
    header: dict[str, list[str] | int] = {}
    for line in header_text.splitlines():
        parts = line.strip().split()
        if not parts or parts[0].startswith("#"):
            continue
        key = parts[0].upper()
        if key in {"FIELDS", "SIZE", "TYPE", "COUNT"}:
            header[key] = parts[1:]
        elif key in {"WIDTH", "HEIGHT", "POINTS"}:
            header[key] = int(parts[1])
        elif key == "DATA":
            header[key] = [parts[1].lower()]
    return header, line_end + 1


def _read_pcd_with_fields(path: Path) -> PcdFrame | None:
    raw = path.read_bytes()
    header, data_offset = _parse_pcd_header(raw)
    fields = list(header.get("FIELDS", []))
    if not fields or "x" not in fields or "y" not in fields or "z" not in fields:
        return None
    sizes = [int(value) for value in header.get("SIZE", [])]
    types = list(header.get("TYPE", []))
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    point_count = int(header.get("POINTS", header.get("WIDTH", 0)))
    data_kind = str(header.get("DATA", [""])[0])
    if data_kind == "ascii":
        rows = np.loadtxt(path, comments="#", skiprows=len(raw[:data_offset].decode("latin1").splitlines()))
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        point_count = rows.shape[0]
        field_index = {name: idx for idx, name in enumerate(fields)}
        points = rows[:, [field_index["x"], field_index["y"], field_index["z"]]].astype(np.float64)
        intensity = rows[:, field_index["intensity"]].astype(np.float64) if "intensity" in field_index else np.full(point_count, 255.0)
        ring_name = "ring" if "ring" in field_index else "r" if "r" in field_index else None
        ring = rows[:, field_index[ring_name]].astype(np.int64) if ring_name else estimate_rings(points)
        return PcdFrame(points=points, intensity=intensity, ring=ring)
    if data_kind != "binary":
        return None

    fmt_parts: list[str] = []
    expanded_fields: list[str] = []
    for name, size, typ, count in zip(fields, sizes, types, counts, strict=False):
        if typ == "F" and size == 4:
            code = "f"
        elif typ == "F" and size == 8:
            code = "d"
        elif typ == "U" and size == 1:
            code = "B"
        elif typ == "U" and size == 2:
            code = "H"
        elif typ == "U" and size == 4:
            code = "I"
        elif typ == "I" and size == 1:
            code = "b"
        elif typ == "I" and size == 2:
            code = "h"
        elif typ == "I" and size == 4:
            code = "i"
        else:
            return None
        fmt_parts.extend([code] * count)
        expanded_fields.extend([name] * count)
    point_struct = struct.Struct("<" + "".join(fmt_parts))
    needed = data_offset + point_struct.size * point_count
    if len(raw) < needed:
        raise ValueError(f"PCD 二进制数据长度不足: {path}")
    field_index: dict[str, int] = {}
    for idx, name in enumerate(expanded_fields):
        field_index.setdefault(name, idx)
    values = np.empty((point_count, len(expanded_fields)), dtype=np.float64)
    for idx in range(point_count):
        values[idx] = point_struct.unpack_from(raw, data_offset + idx * point_struct.size)
    points = values[:, [field_index["x"], field_index["y"], field_index["z"]]]
    intensity = values[:, field_index["intensity"]] if "intensity" in field_index else np.full(point_count, 255.0)
    ring_name = "ring" if "ring" in field_index else "r" if "r" in field_index else None
    ring = values[:, field_index[ring_name]].astype(np.int64) if ring_name else estimate_rings(points)
    return PcdFrame(points=points.astype(np.float64), intensity=intensity.astype(np.float64), ring=ring)


def estimate_rings(points: np.ndarray, ring_count: int = LIDAR_LASER_NUM) -> np.ndarray:
    distance_xy = np.linalg.norm(points[:, :2], axis=1)
    angles = np.arctan2(points[:, 2], np.maximum(distance_xy, 1e-9))
    low, high = np.percentile(angles, [1.0, 99.0])
    if abs(high - low) < 1e-6:
        return np.zeros(points.shape[0], dtype=np.int64)
    ring = np.round((angles - low) / (high - low) * (ring_count - 1)).astype(np.int64)
    return np.clip(ring, 0, ring_count - 1)


def load_pcd_frame(path: str | Path) -> PcdFrame:
    pcd_path = Path(path)
    try:
        parsed = _read_pcd_with_fields(pcd_path)
    except Exception:
        parsed = None
    if parsed is not None:
        mask = np.isfinite(parsed.points).all(axis=1)
        if mask.sum() < 20:
            raise ValueError(f"点云有效点太少: {pcd_path}")
        return PcdFrame(points=parsed.points[mask], intensity=parsed.intensity[mask], ring=parsed.ring[mask])

    o3d = _load_open3d()
    cloud = o3d.io.read_point_cloud(str(pcd_path)).remove_non_finite_points()
    points = np.asarray(cloud.points, dtype=np.float64)
    if points.shape[0] < 20:
        raise ValueError(f"点云有效点太少: {pcd_path}")
    return PcdFrame(points=points, intensity=np.full(points.shape[0], 255.0), ring=estimate_rings(points))


def collect_open_calib_pcd_files(folder: str | Path, pose_frames: list[OpenCalibPoseFrame]) -> list[Path]:
    pcd_dir = Path(folder)
    if not pcd_dir.exists() or not pcd_dir.is_dir():
        raise ValueError(f"PCD 输入必须是文件夹: {pcd_dir}")
    files = [pcd_dir / f"{frame.stamp}.pcd" for frame in pose_frames]
    found = [path for path in files if path.exists()]
    if len(found) < 3:
        raise ValueError("PCD 文件名需要与 pose 第一列时间戳一致，例如 2021-...-468.pcd。")
    return found


def extract_loam_features(frame: PcdFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = frame.points
    intensity = frame.intensity
    rings = frame.ring
    surf_less_flat: list[np.ndarray] = []
    surf_flat: list[np.ndarray] = []
    corner_sharp: list[np.ndarray] = []

    for ring_id in range(LIDAR_LASER_NUM):
        indices = np.flatnonzero(rings == ring_id)
        if indices.size < 11:
            continue
        scan_points = points[indices]
        scan_intensity = intensity[indices]
        curvature = np.zeros(indices.size, dtype=np.float64)
        for local_idx in range(5, indices.size - 5):
            diff = scan_points[local_idx - 5 : local_idx].sum(axis=0)
            diff += scan_points[local_idx + 1 : local_idx + 6].sum(axis=0)
            diff -= 10.0 * scan_points[local_idx]
            curvature[local_idx] = float(diff @ diff)
        picked = np.zeros(indices.size, dtype=bool)
        labels = np.zeros(indices.size, dtype=np.int8)
        valid_start, valid_end = 5, indices.size - 6
        valid_count = valid_end - valid_start
        if valid_count < 6:
            continue
        for segment in range(SCAN_LINE_CUT):
            start = valid_start + valid_count * segment // SCAN_LINE_CUT
            end = valid_start + valid_count * (segment + 1) // SCAN_LINE_CUT - 1
            if end <= start:
                continue
            order = np.argsort(curvature[start : end + 1]) + start
            largest = 0
            for local_idx in reversed(order):
                if scan_intensity[local_idx] < INTENSITY_THRESHOLD:
                    continue
                if picked[local_idx] or curvature[local_idx] <= 10.0:
                    continue
                largest += 1
                if largest <= 2:
                    labels[local_idx] = 2
                    corner_sharp.append(scan_points[local_idx])
                elif largest <= 20:
                    labels[local_idx] = 1
                else:
                    break
                picked[local_idx] = True
                _mark_neighbors(scan_points, picked, local_idx)
            smallest = 0
            for local_idx in order:
                if scan_intensity[local_idx] < INTENSITY_THRESHOLD:
                    continue
                if picked[local_idx] or curvature[local_idx] >= 10.0:
                    continue
                labels[local_idx] = -1
                surf_flat.append(scan_points[local_idx])
                smallest += 1
                picked[local_idx] = True
                _mark_neighbors(scan_points, picked, local_idx)
                if smallest >= 4:
                    break
            for local_idx in range(start, end + 1):
                if scan_intensity[local_idx] >= INTENSITY_THRESHOLD and labels[local_idx] <= 0:
                    surf_less_flat.append(scan_points[local_idx])

    surf = voxel_downsample(np.asarray(surf_less_flat, dtype=np.float64), 0.2) if surf_less_flat else np.empty((0, 3))
    surf_sharp = np.asarray(surf_flat, dtype=np.float64) if surf_flat else np.empty((0, 3))
    corn = np.asarray(corner_sharp, dtype=np.float64) if corner_sharp else np.empty((0, 3))
    return surf, surf_sharp, corn


def _mark_neighbors(scan_points: np.ndarray, picked: np.ndarray, local_idx: int) -> None:
    for offset in range(1, 6):
        nxt = local_idx + offset
        if nxt >= scan_points.shape[0]:
            break
        if float(np.sum((scan_points[nxt] - scan_points[nxt - 1]) ** 2)) > 0.05:
            break
        picked[nxt] = True
    for offset in range(-1, -6, -1):
        prv = local_idx + offset
        if prv < 0:
            break
        if float(np.sum((scan_points[prv] - scan_points[prv + 1]) ** 2)) > 0.05:
            break
        picked[prv] = True


def _fit_leaf(points_orig: list[np.ndarray], points_tran: list[np.ndarray]) -> VoxelLeaf | None:
    all_tran = np.vstack([pts for pts in points_tran if pts.size])
    all_orig = np.vstack([pts for pts in points_orig if pts.size])
    if all_tran.shape[0] < MIN_POINTS_PER_VOXEL or points_orig[0].shape[0] == 0:
        return None
    center = all_tran.mean(axis=0)
    cov = np.cov(all_tran.T, bias=True)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals[0] <= 1e-12:
        return None
    normal = eigvecs[:, 0]
    eigen_ratio = float(eigvals[2] / eigvals[0])
    angle = math.degrees(math.acos(float(np.clip(abs(normal @ np.array([0.0, 0.0, 1.0])), -1.0, 1.0))))
    if angle > 20.0:
        return None

    zero = points_tran[0]
    orig0 = points_orig[0]
    center_zero, normal_zero = _center_and_plane_normal(zero)
    center_orig, normal_orig = _center_and_plane_normal(orig0)
    return VoxelLeaf(
        points_orig=points_orig,
        points_tran=points_tran,
        center=center,
        normal=normal,
        center_zero=center_zero,
        normal_zero=normal_zero,
        center_orig=center_orig,
        normal_orig=normal_orig,
        eigen_ratio=eigen_ratio,
    )


def _center_and_plane_normal(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    cov = np.cov(points.T, bias=True)
    _, eigvecs = np.linalg.eigh(cov)
    return center, eigvecs[:, 0]


def _subdivide_leaf(
    points_orig: list[np.ndarray],
    points_tran: list[np.ndarray],
    voxel_center: np.ndarray,
    quarter_length: float,
    depth: int,
    max_depth: int,
    eigen_limit: float,
) -> list[VoxelLeaf]:
    leaf = _fit_leaf(points_orig, points_tran)
    if leaf is None:
        return []
    if leaf.eigen_ratio >= eigen_limit:
        return [leaf]
    if depth >= max_depth:
        return []
    children_orig: list[list[list[np.ndarray]]] = [[[] for _ in points_orig] for _ in range(8)]
    children_tran: list[list[list[np.ndarray]]] = [[[] for _ in points_tran] for _ in range(8)]
    for frame_idx, tran_points in enumerate(points_tran):
        orig_points = points_orig[frame_idx]
        for orig_point, tran_point in zip(orig_points, tran_points, strict=False):
            xyz = (tran_point > voxel_center).astype(np.int64)
            child = int(4 * xyz[0] + 2 * xyz[1] + xyz[2])
            children_orig[child][frame_idx].append(orig_point)
            children_tran[child][frame_idx].append(tran_point)
    result: list[VoxelLeaf] = []
    for child in range(8):
        child_orig = [np.asarray(bucket, dtype=np.float64).reshape(-1, 3) for bucket in children_orig[child]]
        child_tran = [np.asarray(bucket, dtype=np.float64).reshape(-1, 3) for bucket in children_tran[child]]
        if sum(len(bucket) for bucket in child_tran) < MIN_POINTS_PER_VOXEL:
            continue
        bits = np.array([(child >> 2) & 1, (child >> 1) & 1, child & 1], dtype=np.float64)
        child_center = voxel_center + (2.0 * bits - 1.0) * quarter_length
        result.extend(_subdivide_leaf(child_orig, child_tran, child_center, quarter_length / 2.0, depth + 1, max_depth, eigen_limit))
    return result


def build_voxel_leaves(
    feature_frames: list[np.ndarray],
    poses: list[np.ndarray],
    delta_transform: np.ndarray,
    voxel_size: float,
    max_depth: int,
    eigen_limit: float,
) -> tuple[list[VoxelLeaf], int]:
    buckets: dict[tuple[int, int, int], tuple[list[list[np.ndarray]], list[list[np.ndarray]]]] = {}
    total_points = 0
    for frame_idx, points in enumerate(feature_frames):
        if points.size == 0:
            continue
        transformed = transform_points(poses[frame_idx] @ delta_transform, points)
        total_points += points.shape[0]
        keys = map(tuple, _floor_voxel(transformed, voxel_size))
        for key, orig_point, tran_point in zip(keys, points, transformed, strict=False):
            if key not in buckets:
                buckets[key] = ([[] for _ in feature_frames], [[] for _ in feature_frames])
            buckets[key][0][frame_idx].append(orig_point)
            buckets[key][1][frame_idx].append(tran_point)

    leaves: list[VoxelLeaf] = []
    for key, (orig_lists, tran_lists) in buckets.items():
        points_orig = [np.asarray(bucket, dtype=np.float64).reshape(-1, 3) for bucket in orig_lists]
        points_tran = [np.asarray(bucket, dtype=np.float64).reshape(-1, 3) for bucket in tran_lists]
        voxel_center = (np.asarray(key, dtype=np.float64) + 0.5) * voxel_size
        leaves.extend(_subdivide_leaf(points_orig, points_tran, voxel_center, voxel_size / 4.0, 0, max_depth, eigen_limit))
    return leaves, total_points


def residuals_from_leaves(params: np.ndarray, leaves: list[VoxelLeaf], poses: list[np.ndarray], method: int) -> np.ndarray:
    rotvec = params[:3]
    trans_xy = params[3:5] if method == 2 else np.zeros(2, dtype=np.float64)
    delta = make_delta_transform(rotvec, trans_xy)
    residual_chunks: list[np.ndarray] = []
    for leaf in leaves:
        if leaf.points_orig[0].shape[0] == 0:
            continue
        if method == 4:
            point_average = leaf.center_orig
            normal = leaf.normal_orig
            pose0 = poses[0]
        else:
            point_average = leaf.center_orig
            normal = leaf.normal_orig
            pose0 = poses[0]
        avg_imu = transform_points(pose0 @ delta, point_average.reshape(1, 3))[0]
        normal_imu = (pose0[:3, :3] @ delta[:3, :3] @ normal)
        norm = np.linalg.norm(normal_imu)
        if norm < 1e-9:
            continue
        normal_imu = normal_imu / norm
        for frame_idx, points in enumerate(leaf.points_orig):
            if points.size == 0:
                continue
            transformed = transform_points(poses[frame_idx] @ delta, points)
            residual_chunks.append((transformed - avg_imu) @ normal_imu)
    if not residual_chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(residual_chunks)


def _calibrate_round(
    round_index: int,
    pcd_dir: Path,
    frames: list[OpenCalibPoseFrame],
    start: int,
    step: int,
    frame_count: int,
    current_params: np.ndarray,
    method: int,
    voxel_size: float,
    max_depth: int,
    eigen_limit: float,
    max_residuals: int,
    progress_callback: ProgressCallback | None,
) -> tuple[np.ndarray, CalibrationRoundInfo]:
    feature_frames: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    for local_idx in range(frame_count):
        frame_idx = start + local_idx * step
        pose_frame = frames[frame_idx]
        pcd_path = pcd_dir / f"{pose_frame.stamp}.pcd"
        if not pcd_path.exists():
            raise FileNotFoundError(pcd_path)
        if progress_callback:
            progress_callback(f"第 {round_index + 1} 轮: 读取/提取特征 {local_idx + 1}/{frame_count} {pcd_path.name}")
        surf, surf_sharp, _corn = extract_loam_features(load_pcd_frame(pcd_path))
        features = surf_sharp if method == 4 else surf
        feature_frames.append(features)
        poses.append(pose_frame.pose)

    delta = make_delta_transform(current_params[:3], current_params[3:5])
    leaves, feature_points = build_voxel_leaves(feature_frames, poses, delta, voxel_size, max_depth, eigen_limit)
    if not leaves:
        raise ValueError("没有可优化的体素平面特征，请检查 PCD ring/intensity 字段、场景纹理和初始外参。")

    x0 = current_params[:3] if method == 4 else current_params[:5]

    def fun(x: np.ndarray) -> np.ndarray:
        params = current_params.copy()
        if method == 4:
            params[:3] = x
        else:
            params[:5] = x
        residuals = residuals_from_leaves(params, leaves, poses, method)
        if max_residuals > 0 and residuals.size > max_residuals:
            index = np.linspace(0, residuals.size - 1, max_residuals, dtype=np.int64)
            residuals = residuals[index]
        return residuals

    before = fun(x0)
    result = least_squares(fun, x0=x0, loss="soft_l1", f_scale=0.2, max_nfev=60, xtol=1e-6, ftol=1e-6, gtol=1e-6)
    updated = current_params.copy()
    if method == 4:
        updated[:3] = result.x
    else:
        updated[:5] = result.x
    after = fun(result.x)
    info = CalibrationRoundInfo(
        round_index=round_index,
        start_index=start,
        step=step,
        frame_count=frame_count,
        feature_points=feature_points,
        voxel_count=len(leaves),
        residual_count=int(after.size),
        cost_before=float(np.sqrt(np.mean(before**2))) if before.size else float("inf"),
        cost_after=float(np.sqrt(np.mean(after**2))) if after.size else float("inf"),
        delta_rpy_deg=[float(v) for v in np.rad2deg(updated[:3])],
        delta_t=[float(updated[3]), float(updated[4]), 0.0],
    )
    return updated, info


def calibrate_lidar_imu_open_calib(
    pcd_folder: str | Path,
    pose_file: str | Path,
    extrinsic_json: str | Path,
    turn_count: int = 20,
    window_size: int = 10,
    upper_bound: int = 1000,
    voxel_size: float = 1.0,
    max_depth: int = 5,
    eigen_limit: float = 16.0,
    max_residuals: int = 30000,
    progress_callback: ProgressCallback | None = None,
) -> LidarImuCalibrationResult:
    if turn_count < 1:
        raise ValueError("优化轮数必须大于 0。")
    if window_size < 3:
        raise ValueError("滑窗帧数至少为 3。")
    if voxel_size <= 0:
        raise ValueError("体素边长必须大于 0。")

    initial_imu_lidar = load_open_calib_extrinsic_json(extrinsic_json)
    initial_lidar_imu = np.linalg.inv(initial_imu_lidar)
    frames = load_open_calib_pose_file(pose_file, initial_lidar_imu)
    pcd_dir = Path(pcd_folder)
    found = collect_open_calib_pcd_files(pcd_dir, frames)

    usable_upper = min(len(frames), len(found), int(upper_bound))
    if usable_upper <= window_size + 1:
        raise ValueError("可用 PCD/pose 帧数不足，无法组成 OpenCalib 滑窗。")
    start_step = max(1, usable_upper // 2 // turn_count - 1)
    params = np.zeros(5, dtype=np.float64)
    rounds: list[CalibrationRoundInfo] = []

    for round_index in range(turn_count):
        start = max(0, usable_upper // 2 - round_index * start_step - 1)
        step = max(1, (usable_upper - start) // window_size - 1)
        if start + (window_size - 1) * step >= usable_upper:
            step = max(1, (usable_upper - start - 1) // max(1, window_size - 1))
        method = 4 if round_index < turn_count // 2 else 2
        if progress_callback:
            stage = "旋转粗优化" if method == 4 else "旋转+XY平移精优化"
            progress_callback(f"开始第 {round_index + 1}/{turn_count} 轮 {stage}: start={start}, step={step}, window={window_size}")
        params, info = _calibrate_round(
            round_index=round_index,
            pcd_dir=pcd_dir,
            frames=frames,
            start=start,
            step=step,
            frame_count=window_size,
            current_params=params,
            method=method,
            voxel_size=voxel_size,
            max_depth=max_depth,
            eigen_limit=eigen_limit,
            max_residuals=max_residuals,
            progress_callback=progress_callback,
        )
        rounds.append(info)
        if progress_callback:
            progress_callback(
                f"第 {round_index + 1} 轮完成: RMSE {info.cost_before:.4f} -> {info.cost_after:.4f}, "
                f"delta_rpy={info.delta_rpy_deg}, delta_t={info.delta_t}"
            )

    delta_transform = make_delta_transform(params[:3], params[3:5])
    refined_lidar_imu = initial_lidar_imu @ delta_transform
    refined_imu_lidar = np.linalg.inv(refined_lidar_imu)
    rotation = Rotation.from_matrix(refined_imu_lidar[:3, :3])
    residual_rmse = rounds[-1].cost_after if rounds else float("inf")
    warnings: list[str] = []
    if len(found) < len(frames):
        warnings.append(f"pose 中有 {len(frames)} 帧，实际匹配到 {len(found)} 个 PCD；缺失帧已限制可用范围。")
    if residual_rmse > 0.3:
        warnings.append("最终体素平面残差偏大，建议检查时间同步、初始外参、点云 ring/intensity 字段和采集轨迹。")
    if abs(params[4]) < 1e-12 and abs(params[3]) < 1e-12:
        warnings.append("平移只按 OpenCalib 原实现优化 X/Y 增量，Z 平移保持初始值。")

    return LidarImuCalibrationResult(
        transform_imu_lidar=refined_imu_lidar,
        transform_lidar_imu=refined_lidar_imu,
        initial_transform_imu_lidar=initial_imu_lidar,
        initial_transform_lidar_imu=initial_lidar_imu,
        delta_transform=delta_transform,
        rotation_xyzw=[float(v) for v in rotation.as_quat()],
        euler_deg=[float(v) for v in rotation.as_euler("xyz", degrees=True)],
        translation=[float(v) for v in refined_imu_lidar[:3, 3]],
        refined_lidar_to_imu_euler_deg=[float(v) for v in Rotation.from_matrix(refined_lidar_imu[:3, :3]).as_euler("xyz", degrees=True)],
        refined_lidar_to_imu_translation=[float(v) for v in refined_lidar_imu[:3, 3]],
        pose_count=len(frames),
        used_frame_count=usable_upper,
        pcd_frame_count=len(found),
        round_count=len(rounds),
        residual_rmse_m=float(residual_rmse),
        delta_rpy_deg=[float(v) for v in np.rad2deg(params[:3])],
        delta_translation=[float(params[3]), float(params[4]), 0.0],
        warnings=warnings,
        rounds=rounds,
        lidar_source=f"open_calib_pcd:{pcd_dir}",
        lidar_frame_count=len(found),
        pair_count=len(rounds),
        translation_rmse_m=float(residual_rmse),
    )


def calibrate_lidar_imu_from_pcd_folder(
    pcd_folder: str | Path,
    imu_csv: str | Path,
    interval_sec: float = 1.0,
    min_rotation_deg: float = 1.0,
    max_pairs: int = 400,
    imu_time_offset_sec: float = 0.0,
    pcd_frame_interval_sec: float = 0.1,
    voxel_size: float = 1.0,
    max_correspondence_distance: float = 1.5,
    icp_max_iteration: int = 50,
    icp_method: str = "point_to_point",
    max_points: int = 80000,
    progress_callback: ProgressCallback | None = None,
) -> LidarImuCalibrationResult:
    raise RuntimeError("旧版 PCD+ICP 手眼方案已替换。请调用 calibrate_lidar_imu_open_calib(pcd_folder, pose_file, extrinsic_json)。")


def calibrate_lidar_imu(*args, **kwargs) -> LidarImuCalibrationResult:
    raise RuntimeError("旧版 CSV 手眼方案已替换。请使用 OpenCalib 自动方案: PCD 文件夹 + pose 矩阵文件 + 初始外参 JSON。")


def result_to_json(result: LidarImuCalibrationResult) -> str:
    payload = {
        "algorithm": "OpenCalib lidar2imu auto calibration (Python BALM-style port)",
        "convention": {
            "transform_imu_lidar": "Refined IMU -> LiDAR matrix, same direction as OpenCalib refined_calib_imu_to_lidar.txt.",
            "transform_lidar_imu": "Inverse matrix. It maps LiDAR points into IMU frame and is used internally during optimization.",
            "delta_transform": "Increment optimized on top of initial T_lidar_to_imu; rotation + x/y translation, z kept from initial value like OpenCalib code.",
        },
        "transform_imu_lidar": result.transform_imu_lidar.tolist(),
        "transform_lidar_imu": result.transform_lidar_imu.tolist(),
        "initial_transform_imu_lidar": result.initial_transform_imu_lidar.tolist(),
        "delta_transform": result.delta_transform.tolist(),
        "translation": result.translation,
        "rotation_xyzw": result.rotation_xyzw,
        "euler_deg": {
            "roll": result.euler_deg[0],
            "pitch": result.euler_deg[1],
            "yaw": result.euler_deg[2],
        },
        "refined_lidar_to_imu": {
            "translation": result.refined_lidar_to_imu_translation,
            "euler_deg": {
                "roll": result.refined_lidar_to_imu_euler_deg[0],
                "pitch": result.refined_lidar_to_imu_euler_deg[1],
                "yaw": result.refined_lidar_to_imu_euler_deg[2],
            },
            "matrix": result.transform_lidar_imu.tolist(),
        },
        "metrics": {
            "pose_count": result.pose_count,
            "pcd_frame_count": result.pcd_frame_count,
            "used_frame_count": result.used_frame_count,
            "round_count": result.round_count,
            "residual_rmse_m": result.residual_rmse_m,
            "delta_rpy_deg": result.delta_rpy_deg,
            "delta_translation": result.delta_translation,
            "lidar_source": result.lidar_source,
        },
        "rounds": [
            {
                "round_index": item.round_index,
                "start_index": item.start_index,
                "step": item.step,
                "frame_count": item.frame_count,
                "feature_points": item.feature_points,
                "voxel_count": item.voxel_count,
                "residual_count": item.residual_count,
                "cost_before": item.cost_before,
                "cost_after": item.cost_after,
                "delta_rpy_deg": item.delta_rpy_deg,
                "delta_t": item.delta_t,
            }
            for item in result.rounds
        ],
        "warnings": result.warnings,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
