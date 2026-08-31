from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


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


@dataclass
class MultiFrameRegistrationResult:
    transform: np.ndarray
    fitness: float
    inlier_rmse: float
    converged: bool
    stage_metrics: List[dict]
    selected_pair_indices: List[int]
    overlap_scores: List[float]
    iterations: int
    correspondence_count: int


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


def xyz_rpy_to_matrix(
    x: float,
    y: float,
    z: float,
    roll_degrees: float,
    pitch_degrees: float,
    yaw_degrees: float,
) -> np.ndarray:
    """Build a Source -> Target transform from XYZ and fixed-axis RPY angles."""
    roll, pitch, yaw = np.deg2rad([roll_degrees, pitch_degrees, yaw_degrees])
    sin_roll, cos_roll = np.sin(roll), np.cos(roll)
    sin_pitch, cos_pitch = np.sin(pitch), np.cos(pitch)
    sin_yaw, cos_yaw = np.sin(yaw), np.cos(yaw)

    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_roll, -sin_roll], [0.0, sin_roll, cos_roll]],
        dtype=float,
    )
    rotation_y = np.array(
        [[cos_pitch, 0.0, sin_pitch], [0.0, 1.0, 0.0], [-sin_pitch, 0.0, cos_pitch]],
        dtype=float,
    )
    rotation_z = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = [x, y, z]
    return transform


def matrix_to_xyz_rpy(transform: np.ndarray) -> List[float]:
    """Return XYZ and fixed-axis RPY degrees from a homogeneous transform."""
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform matrix, got shape {matrix.shape}")

    rotation = matrix[:3, :3]
    horizontal_norm = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    if horizontal_norm > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], horizontal_norm)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        # At pitch +/-90 degrees roll and yaw are coupled. Keep yaw at zero and
        # put the observable rotation into roll so rebuilding stays equivalent.
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], horizontal_norm)
        yaw = 0.0

    roll_degrees, pitch_degrees, yaw_degrees = np.rad2deg([roll, pitch, yaw])
    return [
        float(matrix[0, 3]),
        float(matrix[1, 3]),
        float(matrix[2, 3]),
        float(roll_degrees),
        float(pitch_degrees),
        float(yaw_degrees),
    ]


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


def transform_delta(reference: np.ndarray, candidate: np.ndarray) -> Tuple[float, float]:
    """Return translation and rotation deltas between two rigid transforms."""
    translation_delta = float(np.linalg.norm(candidate[:3, 3] - reference[:3, 3]))
    relative_rotation = candidate[:3, :3] @ reference[:3, :3].T
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
    rotation_delta_degrees = float(np.degrees(np.arccos(cosine)))
    return translation_delta, rotation_delta_degrees


def register_multiscale(
    target_cloud: o3d.geometry.PointCloud,
    source_cloud: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    stages: Sequence[RegistrationStage],
    crop_range: Optional[float],
    z_range: Optional[Tuple[Optional[float], Optional[float]]],
    estimation_method: str,
    max_translation_delta: Optional[float] = 1.0,
    max_rotation_delta_degrees: Optional[float] = 15.0,
    min_fitness: float = 0.05,
) -> RegistrationResult:
    initial_transform = np.asarray(init_transform, dtype=float).copy()
    transform = initial_transform.copy()
    metrics = []
    estimation = get_estimation_method(estimation_method)

    if max_translation_delta is not None and max_translation_delta <= 0.0:
        raise ValueError("max_translation_delta must be positive or None")
    if max_rotation_delta_degrees is not None and max_rotation_delta_degrees <= 0.0:
        raise ValueError("max_rotation_delta_degrees must be positive or None")
    if not 0.0 <= min_fitness <= 1.0:
        raise ValueError("min_fitness must be between 0 and 1")

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

        candidate = np.asarray(reg.transformation, dtype=float)
        translation_delta, rotation_delta_degrees = transform_delta(initial_transform, candidate)
        rejection_reasons = []
        if max_translation_delta is not None and translation_delta > max_translation_delta:
            rejection_reasons.append(
                f"translation delta {translation_delta:.3f} m exceeds {max_translation_delta:.3f} m"
            )
        if max_rotation_delta_degrees is not None and rotation_delta_degrees > max_rotation_delta_degrees:
            rejection_reasons.append(
                f"rotation delta {rotation_delta_degrees:.3f} deg exceeds "
                f"{max_rotation_delta_degrees:.3f} deg"
            )

        accepted = not rejection_reasons
        if accepted:
            transform = candidate
            fitness = float(reg.fitness)
            inlier_rmse = float(reg.inlier_rmse)
        else:
            safe_evaluation = o3d.pipelines.registration.evaluate_registration(
                source_level,
                target_level,
                stage.max_correspondence_distance,
                transform,
            )
            fitness = float(safe_evaluation.fitness)
            inlier_rmse = float(safe_evaluation.inlier_rmse)

        metrics.append(
            {
                "stage": index,
                "voxel_size": stage.voxel_size,
                "max_correspondence_distance": stage.max_correspondence_distance,
                "max_iterations": stage.max_iterations,
                "fitness": fitness,
                "inlier_rmse": inlier_rmse,
                "accepted": accepted,
                "translation_delta_from_init": translation_delta,
                "rotation_delta_from_init_degrees": rotation_delta_degrees,
                "rejection_reason": "; ".join(rejection_reasons),
            }
        )

    final_metric = metrics[-1] if metrics else None
    converged = bool(
        final_metric
        and final_metric["accepted"]
        and np.isfinite(final_metric["inlier_rmse"])
        and final_metric["fitness"] >= min_fitness
    )
    return RegistrationResult(
        transform=transform,
        fitness=final_metric["fitness"] if final_metric else 0.0,
        inlier_rmse=final_metric["inlier_rmse"] if final_metric else 0.0,
        converged=converged,
        stage_metrics=metrics,
    )


def _prepare_multiframe_cloud(cloud_or_path, voxel_size: float, z_range):
    if isinstance(cloud_or_path, (str, Path)):
        cloud = load_point_cloud(Path(cloud_or_path))
    else:
        cloud = o3d.geometry.PointCloud(cloud_or_path)
    prepared = preprocess_cloud(cloud, voxel_size, None, z_range, "point_to_plane")
    return np.asarray(prepared.points).copy(), np.asarray(prepared.normals).copy()


def register_multiframe_translation(
    cloud_pairs: Sequence[Tuple[object, object]],
    init_transform: np.ndarray,
    voxel_size: float = 0.1,
    max_correspondence_distance: float = 0.25,
    top_overlap_fraction: float = 0.1,
    min_selected_pairs: int = 5,
    max_selected_pairs: int = 30,
    normal_angle_threshold_degrees: float = 30.0,
    huber_delta: float = 0.04,
    max_iterations: int = 15,
    max_points_per_pair: int = 3000,
    max_translation_step: float = 0.05,
    max_translation_delta: float = 1.0,
    z_range: Optional[Tuple[Optional[float], Optional[float]]] = (-5.0, 5.0),
    random_seed: int = 42,
) -> MultiFrameRegistrationResult:
    """Jointly refine translation from the highest-overlap synchronized pairs.

    Rotation is deliberately held at the supplied initial value. With partially
    overlapping LiDAR fields of view, unconstrained 6-DoF nearest-neighbor ICP
    couples weakly observable rotation into large translation errors. Ranking
    pairs at the initial transform also prevents a wrong ICP solution from
    selecting its own apparently high-fitness frames.
    """
    if not cloud_pairs:
        raise ValueError("cloud_pairs must not be empty")
    if voxel_size <= 0.0 or max_correspondence_distance <= 0.0:
        raise ValueError("voxel_size and max_correspondence_distance must be positive")
    if not 0.0 < top_overlap_fraction <= 1.0:
        raise ValueError("top_overlap_fraction must be in (0, 1]")
    if min_selected_pairs <= 0 or max_selected_pairs < min_selected_pairs:
        raise ValueError("invalid selected-pair limits")
    if not 0.0 < normal_angle_threshold_degrees < 90.0:
        raise ValueError("normal_angle_threshold_degrees must be in (0, 90)")
    if huber_delta <= 0.0 or max_iterations <= 0 or max_points_per_pair <= 0:
        raise ValueError("Huber delta, iteration count, and point limit must be positive")

    initial = np.asarray(init_transform, dtype=float).copy()
    rotation = initial[:3, :3].copy()
    initial_translation = initial[:3, 3].copy()
    normal_cosine_threshold = float(np.cos(np.deg2rad(normal_angle_threshold_degrees)))

    prepared_pairs = []
    overlap_scores = []
    for target_input, source_input in cloud_pairs:
        target_points, target_normals = _prepare_multiframe_cloud(target_input, voxel_size, z_range)
        source_points, source_normals = _prepare_multiframe_cloud(source_input, voxel_size, z_range)
        target_tree = cKDTree(target_points)
        source_initial = source_points @ rotation.T + initial_translation
        distances, target_indices = target_tree.query(source_initial, workers=-1)
        transformed_source_normals = source_normals @ rotation.T
        normal_compatibility = np.abs(
            np.sum(transformed_source_normals * target_normals[target_indices], axis=1)
        ) >= normal_cosine_threshold
        overlap = (distances < max_correspondence_distance) & normal_compatibility
        score = float(np.count_nonzero(overlap) / max(len(source_points), 1))
        overlap_scores.append(score)
        prepared_pairs.append(
            {
                "target_points": target_points,
                "target_normals": target_normals,
                "target_tree": target_tree,
                "source_points": source_points,
            }
        )

    desired_count = int(np.ceil(len(prepared_pairs) * top_overlap_fraction))
    selected_count = min(len(prepared_pairs), max_selected_pairs, max(min_selected_pairs, desired_count))
    selected_pair_indices = sorted(
        range(len(prepared_pairs)), key=lambda index: overlap_scores[index], reverse=True
    )[:selected_count]
    selected_pairs = [prepared_pairs[index] for index in selected_pair_indices]

    translation = initial_translation.copy()
    rng = np.random.default_rng(random_seed)
    correspondence_count = 0
    plane_rmse = float("inf")
    condition_number = float("inf")
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        normal_blocks = []
        residual_blocks = []
        for pair in selected_pairs:
            transformed_source = pair["source_points"] @ rotation.T + translation
            distances, target_indices = pair["target_tree"].query(transformed_source, workers=-1)
            correspondence_indices = np.flatnonzero(distances < max_correspondence_distance)
            if len(correspondence_indices) > max_points_per_pair:
                correspondence_indices = rng.choice(
                    correspondence_indices, max_points_per_pair, replace=False
                )
            if not len(correspondence_indices):
                continue
            target_indices = target_indices[correspondence_indices]
            normals = pair["target_normals"][target_indices]
            residuals = np.sum(
                normals
                * (
                    transformed_source[correspondence_indices]
                    - pair["target_points"][target_indices]
                ),
                axis=1,
            )
            normal_blocks.append(normals)
            residual_blocks.append(residuals)

        if not normal_blocks:
            break
        normals = np.concatenate(normal_blocks)
        residuals = np.concatenate(residual_blocks)
        correspondence_count = len(residuals)
        plane_rmse = float(np.sqrt(np.mean(np.square(residuals))))

        weights = np.ones_like(residuals)
        outliers = np.abs(residuals) > huber_delta
        weights[outliers] = huber_delta / np.abs(residuals[outliers])
        sqrt_weights = np.sqrt(weights)
        weighted_normals = normals * sqrt_weights[:, None]
        weighted_residuals = residuals * sqrt_weights
        normal_matrix = weighted_normals.T @ weighted_normals
        condition_number = float(np.linalg.cond(normal_matrix))
        translation_step = np.linalg.lstsq(
            weighted_normals, -weighted_residuals, rcond=None
        )[0]
        step_norm = float(np.linalg.norm(translation_step))
        if step_norm > max_translation_step:
            translation_step *= max_translation_step / step_norm
        translation += translation_step
        iterations = iteration

        if np.linalg.norm(translation - initial_translation) > max_translation_delta:
            raise RuntimeError("joint translation optimization exceeded the safety delta")
        if np.linalg.norm(translation_step) < 1e-5:
            break

    final_fitness_values = []
    for pair in selected_pairs:
        transformed_source = pair["source_points"] @ rotation.T + translation
        distances, _ = pair["target_tree"].query(transformed_source, workers=-1)
        final_fitness_values.append(
            float(np.count_nonzero(distances < max_correspondence_distance) / len(distances))
        )

    result_transform = initial.copy()
    result_transform[:3, 3] = translation
    converged = bool(
        iterations
        and correspondence_count >= 100
        and np.isfinite(plane_rmse)
        and np.isfinite(condition_number)
        and condition_number < 1e6
    )
    fitness = float(np.median(final_fitness_values)) if final_fitness_values else 0.0
    stage_metrics = [
        {
            "stage": 1,
            "voxel_size": voxel_size,
            "max_correspondence_distance": max_correspondence_distance,
            "max_iterations": max_iterations,
            "fitness": fitness,
            "inlier_rmse": plane_rmse,
            "accepted": converged,
            "translation_delta_from_init": float(np.linalg.norm(translation - initial_translation)),
            "rotation_delta_from_init_degrees": 0.0,
            "rejection_reason": "" if converged else "joint translation optimization was ill-conditioned",
            "selected_pair_count": selected_count,
            "correspondence_count": correspondence_count,
            "normal_matrix_condition_number": condition_number,
        }
    ]
    return MultiFrameRegistrationResult(
        transform=result_transform,
        fitness=fitness,
        inlier_rmse=plane_rmse,
        converged=converged,
        stage_metrics=stage_metrics,
        selected_pair_indices=selected_pair_indices,
        overlap_scores=overlap_scores,
        iterations=iterations,
        correspondence_count=correspondence_count,
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
        line = (
            "stage={stage} voxel={voxel_size:.3f} max_corr={max_correspondence_distance:.3f} "
            "iters={max_iterations} fitness={fitness:.6f} rmse={inlier_rmse:.6f}".format(**metric)
        )
        line += f" accepted={metric.get('accepted', True)}"
        if metric.get("rejection_reason"):
            line += f" reason={metric['rejection_reason']}"
        lines.append(line)
    return "\n".join(lines)
