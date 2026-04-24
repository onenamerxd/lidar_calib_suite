from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d


@dataclass
class RegistrationStage:
    voxel_size: float
    max_correspondence_distance: float
    max_iterations: int


@dataclass
class RegistrationResult:
    transform: np.ndarray
    fitness: float
    inlier_rmse: float
    converged: bool
    stage_metrics: List[dict]


def parse_csv_floats(raw: str) -> List[float]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return [float(token) for token in values]


def parse_csv_ints(raw: str) -> List[int]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return [int(token) for token in values]


def build_registration_stages(
    voxel_sizes_raw: str, max_corr_raw: str, max_iters_raw: str
) -> List[RegistrationStage]:
    voxel_sizes = parse_csv_floats(voxel_sizes_raw)
    max_corrs = parse_csv_floats(max_corr_raw)
    max_iters = parse_csv_ints(max_iters_raw)

    if not (len(voxel_sizes) == len(max_corrs) == len(max_iters)):
        raise ValueError("voxel sizes, max correspondence distances, and max iterations must have the same length")

    stages = []
    for voxel_size, max_corr, max_iter in zip(voxel_sizes, max_corrs, max_iters):
        if voxel_size <= 0.0:
            raise ValueError("voxel sizes must be positive")
        if max_corr <= 0.0:
            raise ValueError("max correspondence distances must be positive")
        if max_iter <= 0:
            raise ValueError("max iterations must be positive")
        stages.append(
            RegistrationStage(
                voxel_size=float(voxel_size),
                max_correspondence_distance=float(max_corr),
                max_iterations=int(max_iter),
            )
        )
    return stages


def load_transform_matrix(path: Path) -> np.ndarray:
    matrix = np.asarray(json.loads(Path(path).read_text(encoding="utf-8")), dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform matrix in {path}, got shape {matrix.shape}")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError("Transform matrix must be homogeneous with last row [0, 0, 0, 1]")
    return matrix


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Point cloud is empty or unreadable: {path}")
    points = np.asarray(cloud.points)
    valid_mask = np.isfinite(points).all(axis=1)
    if not np.all(valid_mask):
        cloud = cloud.select_by_index(np.flatnonzero(valid_mask))
    if cloud.is_empty():
        raise ValueError(f"Point cloud became empty after removing invalid points: {path}")
    return cloud


def crop_point_cloud(
    cloud: o3d.geometry.PointCloud, crop_range: Optional[float], z_range: Optional[Tuple[Optional[float], Optional[float]]]
) -> o3d.geometry.PointCloud:
    if crop_range is None and z_range is None:
        return cloud

    points = np.asarray(cloud.points)
    mask = np.ones(points.shape[0], dtype=bool)

    if crop_range is not None:
        mask &= np.linalg.norm(points[:, :2], axis=1) <= crop_range

    if z_range is not None:
        z_min, z_max = z_range
        if z_min is not None:
            mask &= points[:, 2] >= z_min
        if z_max is not None:
            mask &= points[:, 2] <= z_max

    cropped = cloud.select_by_index(np.flatnonzero(mask))
    if cropped.is_empty():
        raise ValueError("Point cloud became empty after cropping. Relax crop-range or z bounds.")
    return cropped


def estimate_normals_if_needed(cloud: o3d.geometry.PointCloud, voxel_size: float) -> None:
    radius = max(voxel_size * 2.5, 0.3)
    max_nn = max(30, int(radius / max(voxel_size, 1e-3) * 20))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))


def preprocess_cloud(
    cloud: o3d.geometry.PointCloud,
    voxel_size: float,
    crop_range: Optional[float],
    z_range: Optional[Tuple[Optional[float], Optional[float]]],
    estimation_method: str,
) -> o3d.geometry.PointCloud:
    cropped = crop_point_cloud(cloud, crop_range, z_range)
    downsampled = cropped.voxel_down_sample(voxel_size)
    if downsampled.is_empty():
        raise ValueError(f"Point cloud became empty after voxel downsampling at voxel_size={voxel_size}")
    if estimation_method == "point_to_plane":
        estimate_normals_if_needed(downsampled, voxel_size)
    return downsampled


def get_estimation_method(estimation_method: str):
    if estimation_method == "point_to_plane":
        return o3d.pipelines.registration.TransformationEstimationPointToPlane()
    if estimation_method == "point_to_point":
        return o3d.pipelines.registration.TransformationEstimationPointToPoint()
    raise ValueError(f"Unsupported estimation method: {estimation_method}")


def register_multiscale(
    target_cloud: o3d.geometry.PointCloud,
    source_cloud: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    stages: Sequence[RegistrationStage],
    crop_range: Optional[float],
    z_range: Optional[Tuple[Optional[float], Optional[float]]],
    estimation_method: str,
) -> RegistrationResult:
    transform = np.asarray(init_transform, dtype=float).copy()
    metrics = []
    estimation = get_estimation_method(estimation_method)

    for index, stage in enumerate(stages, start=1):
        target_level = preprocess_cloud(target_cloud, stage.voxel_size, crop_range, z_range, estimation_method)
        source_level = preprocess_cloud(source_cloud, stage.voxel_size, crop_range, z_range, estimation_method)

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=stage.max_iterations)
        reg = o3d.pipelines.registration.registration_icp(
            source=source_level,
            target=target_level,
            max_correspondence_distance=stage.max_correspondence_distance,
            init=transform,
            estimation_method=estimation,
            criteria=criteria,
        )

        transform = reg.transformation
        metrics.append(
            {
                "stage": index,
                "voxel_size": stage.voxel_size,
                "max_correspondence_distance": stage.max_correspondence_distance,
                "max_iterations": stage.max_iterations,
                "fitness": float(reg.fitness),
                "inlier_rmse": float(reg.inlier_rmse),
            }
        )

    return RegistrationResult(
        transform=transform,
        fitness=metrics[-1]["fitness"] if metrics else 0.0,
        inlier_rmse=metrics[-1]["inlier_rmse"] if metrics else 0.0,
        converged=bool(metrics and metrics[-1]["fitness"] > 0.0),
        stage_metrics=metrics,
    )


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rotation[2, 1] - rotation[1, 2]) * s
        qy = (rotation[0, 2] - rotation[2, 0]) * s
        qz = (rotation[1, 0] - rotation[0, 1]) * s
    else:
        if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=float)
    quat /= np.linalg.norm(quat)
    if quat[3] < 0.0:
        quat *= -1.0
    return quat


def matrix_to_xyz_quat(transform: np.ndarray) -> List[float]:
    translation = transform[:3, 3]
    quat = rotation_matrix_to_quaternion(transform[:3, :3])
    return [float(translation[0]), float(translation[1]), float(translation[2]), *[float(v) for v in quat]]


def merge_point_clouds(
    target_cloud: o3d.geometry.PointCloud, source_cloud: o3d.geometry.PointCloud, transform: np.ndarray
) -> o3d.geometry.PointCloud:
    aligned_source = o3d.geometry.PointCloud(source_cloud)
    aligned_source.transform(transform)

    merged = o3d.geometry.PointCloud(target_cloud)
    merged += aligned_source
    return merged


def save_transform_matrix(path: Path, transform: np.ndarray) -> None:
    Path(path).write_text(json.dumps(transform.tolist(), indent=2), encoding="utf-8")


def format_stage_metrics(stage_metrics: Iterable[dict]) -> str:
    lines = []
    for metric in stage_metrics:
        lines.append(
            "stage={stage} voxel={voxel_size:.3f} max_corr={max_correspondence_distance:.3f} "
            "iters={max_iterations} fitness={fitness:.6f} rmse={inlier_rmse:.6f}".format(**metric)
        )
    return "\n".join(lines)
