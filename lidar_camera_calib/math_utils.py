from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from .models import CalibrationCorrespondence, CameraIntrinsics, Extrinsics


def euler_deg_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def matrix_to_euler_deg(matrix: np.ndarray) -> tuple[float, float, float]:
    sy = -matrix[2, 0]
    pitch = math.asin(max(-1.0, min(1.0, sy)))

    if abs(math.cos(pitch)) < 1e-8:
        roll = 0.0
        yaw = math.atan2(-matrix[0, 1], matrix[1, 1])
    else:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def quaternion_to_matrix(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quaternion_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def euler_deg_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    roll = math.radians(roll_deg) * 0.5
    pitch = math.radians(pitch_deg) * 0.5
    yaw = math.radians(yaw_deg) * 0.5

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


def extrinsics_to_rt(extrinsics: Extrinsics) -> tuple[np.ndarray, np.ndarray]:
    rotation = euler_deg_to_matrix(extrinsics.roll_deg, extrinsics.pitch_deg, extrinsics.yaw_deg)
    translation = np.array([extrinsics.tx, extrinsics.ty, extrinsics.tz], dtype=np.float64)
    return rotation, translation


def transform_lidar_to_camera(points_xyz: np.ndarray, extrinsics: Extrinsics) -> np.ndarray:
    rotation, translation = extrinsics_to_rt(extrinsics)
    return points_xyz @ rotation.T + translation


def apply_distortion(normalized_xy: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    x = normalized_xy[:, 0]
    y = normalized_xy[:, 1]
    k1, k2, p1, p2, k3, k4, k5, k6 = intrinsics.clipped_distortion()
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial_num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    radial_den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    radial = radial_num / np.where(np.abs(radial_den) < 1e-12, 1.0, radial_den)

    xy = x * y
    x2 = x * x
    y2 = y * y
    x_dist = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
    y_dist = y * radial + p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy
    return np.column_stack([x_dist, y_dist])


def project_lidar_to_image(
    points_xyz: np.ndarray,
    intrinsics: CameraIntrinsics,
    extrinsics: Extrinsics,
    use_distortion: bool = True,
    flip_mode: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = transform_lidar_to_camera(points_xyz, extrinsics)
    depth = camera_points[:, 2]
    valid_depth = depth > 1e-4

    projected = np.zeros((points_xyz.shape[0], 2), dtype=np.float64)
    if np.any(valid_depth):
        normalized = camera_points[valid_depth, :2] / depth[valid_depth, None]
        if use_distortion:
            normalized = apply_distortion(normalized, intrinsics)
        projected[valid_depth, 0] = intrinsics.fx * normalized[:, 0] + intrinsics.cx
        projected[valid_depth, 1] = intrinsics.fy * normalized[:, 1] + intrinsics.cy

    # Apply image flip/rotation to projected coordinates
    if flip_mode == 1:  # horizontal
        projected[:, 0] = intrinsics.width - 1 - projected[:, 0]
    elif flip_mode == 2:  # vertical
        projected[:, 1] = intrinsics.height - 1 - projected[:, 1]
    elif flip_mode == 3:  # rotate 180
        projected[:, 0] = intrinsics.width - 1 - projected[:, 0]
        projected[:, 1] = intrinsics.height - 1 - projected[:, 1]

    in_bounds = (
        valid_depth
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] < intrinsics.width)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < intrinsics.height)
    )
    return projected, depth, in_bounds


def camera_pose_in_lidar(extrinsics: Extrinsics) -> tuple[np.ndarray, np.ndarray]:
    rotation, translation = extrinsics_to_rt(extrinsics)
    camera_rotation_in_lidar = rotation.T
    camera_origin_in_lidar = -camera_rotation_in_lidar @ translation
    return camera_origin_in_lidar, camera_rotation_in_lidar


def camera_frustum_in_lidar(
    intrinsics: CameraIntrinsics,
    extrinsics: Extrinsics,
    depth_m: float = 25.0,
) -> list[np.ndarray]:
    camera_origin, camera_rotation = camera_pose_in_lidar(extrinsics)
    corners_px = np.array(
        [
            [0.0, 0.0],
            [intrinsics.width, 0.0],
            [intrinsics.width, intrinsics.height],
            [0.0, intrinsics.height],
        ],
        dtype=np.float64,
    )
    rays_cam = np.column_stack(
        [
            (corners_px[:, 0] - intrinsics.cx) / intrinsics.fx,
            (corners_px[:, 1] - intrinsics.cy) / intrinsics.fy,
            np.ones(4, dtype=np.float64),
        ]
    )
    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
    rays_lidar = rays_cam @ camera_rotation.T
    return [camera_origin, camera_origin + rays_lidar[0] * depth_m, camera_origin + rays_lidar[1] * depth_m, camera_origin + rays_lidar[2] * depth_m, camera_origin + rays_lidar[3] * depth_m]


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    if depth.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    depth = np.asarray(depth, dtype=np.float64)
    d_min = float(np.min(depth))
    d_max = float(np.max(depth))
    if abs(d_max - d_min) < 1e-9:
        scaled = np.zeros_like(depth)
    else:
        scaled = (depth - d_min) / (d_max - d_min)

    red = np.clip(255.0 * (1.5 - np.abs(4.0 * scaled - 3.0)), 0, 255)
    green = np.clip(255.0 * (1.5 - np.abs(4.0 * scaled - 2.0)), 0, 255)
    blue = np.clip(255.0 * (1.5 - np.abs(4.0 * scaled - 1.0)), 0, 255)
    return np.column_stack([red, green, blue]).astype(np.uint8)


def solve_extrinsics_from_correspondences(
    correspondences: list[CalibrationCorrespondence],
    intrinsics: CameraIntrinsics,
    initial: Extrinsics,
    use_distortion: bool = True,
    flip_mode: int = 0,
) -> tuple[bool, Extrinsics, float, str]:
    if len(correspondences) < 4:
        return False, initial, 0.0, "至少需要 4 组 2D-3D 对应点才能求解外参。"

    lidar_points = np.array([[c.lidar_x, c.lidar_y, c.lidar_z] for c in correspondences], dtype=np.float64)
    image_points = np.array([[c.image_u, c.image_v] for c in correspondences], dtype=np.float64)

    def residuals(params: np.ndarray) -> np.ndarray:
        candidate = Extrinsics(
            tx=float(params[0]),
            ty=float(params[1]),
            tz=float(params[2]),
            roll_deg=float(params[3]),
            pitch_deg=float(params[4]),
            yaw_deg=float(params[5]),
        )
        projected, depth, mask = project_lidar_to_image(lidar_points, intrinsics, candidate, use_distortion=use_distortion, flip_mode=flip_mode)
        residual = projected - image_points
        penalty = np.zeros((lidar_points.shape[0], 2), dtype=np.float64)
        invalid = (~mask) | (depth <= 1e-4)
        penalty[invalid, :] = 500.0
        return (residual + penalty).reshape(-1)

    x0 = np.array(
        [
            initial.tx,
            initial.ty,
            initial.tz,
            initial.roll_deg,
            initial.pitch_deg,
            initial.yaw_deg,
        ],
        dtype=np.float64,
    )

    result = least_squares(residuals, x0=x0, loss="soft_l1", f_scale=10.0, max_nfev=300)
    solved = Extrinsics(
        tx=float(result.x[0]),
        ty=float(result.x[1]),
        tz=float(result.x[2]),
        roll_deg=float(result.x[3]),
        pitch_deg=float(result.x[4]),
        yaw_deg=float(result.x[5]),
    )

    projected, depth, mask = project_lidar_to_image(lidar_points, intrinsics, solved, use_distortion=use_distortion, flip_mode=flip_mode)
    valid = mask & (depth > 1e-4)
    if np.any(valid):
        rmse = float(np.sqrt(np.mean(np.sum((projected[valid] - image_points[valid]) ** 2, axis=1))))
    else:
        rmse = float("inf")
    message = "求解成功。" if result.success else f"优化结束，但求解器提示: {result.message}"
    return bool(result.success), solved, rmse, message
