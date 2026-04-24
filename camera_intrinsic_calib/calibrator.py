from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


@dataclass
class DetectionResult:
    image_path: Path
    success: bool
    corners: np.ndarray | None = None
    reprojection_error: float = 0.0
    message: str = ""


@dataclass
class CalibrationResult:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: list[float]
    rms_error: float
    per_image_errors: list[tuple[str, float]] = field(default_factory=list)
    camera_matrix: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "distortion": self.distortion,
            "rms_error": self.rms_error,
        }


def _load_image_gray(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img


def detect_chessboard_corners(
    image_path: Path,
    pattern_size: tuple[int, int],  # (rows, cols)
) -> DetectionResult:
    img = _load_image_gray(image_path)
    if img is None:
        return DetectionResult(image_path=image_path, success=False, message="无法读取图像")

    # OpenCV expects (cols, rows)
    cv_pattern_size = (pattern_size[1], pattern_size[0])
    found, corners = cv2.findChessboardCorners(img, cv_pattern_size, None)
    if not found:
        return DetectionResult(image_path=image_path, success=False, message="未检测到棋盘格角点")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)

    return DetectionResult(
        image_path=image_path,
        success=True,
        corners=corners_refined,
        message=f"检测到 {corners_refined.shape[0]} 个角点",
    )


def detect_circle_grid_corners(
    image_path: Path,
    pattern_size: tuple[int, int],  # (rows, cols)
    symmetric: bool = True,
) -> DetectionResult:
    img = _load_image_gray(image_path)
    if img is None:
        return DetectionResult(image_path=image_path, success=False, message="无法读取图像")

    # OpenCV expects (cols, rows)
    cv_pattern_size = (pattern_size[1], pattern_size[0])
    if symmetric:
        found, corners = cv2.findCirclesGrid(img, cv_pattern_size, None)
    else:
        found, corners = cv2.findCirclesGrid(
            img, cv_pattern_size, None, cv2.CALIB_CB_ASYMMETRIC_GRID
        )

    if not found:
        pattern_name = "对称圆点" if symmetric else "非对称圆点"
        return DetectionResult(
            image_path=image_path, success=False, message=f"未检测到{pattern_name}格角点"
        )

    return DetectionResult(
        image_path=image_path,
        success=True,
        corners=corners,
        message=f"检测到 {corners.shape[0]} 个圆点",
    )


def build_object_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    return objp


def calibrate_camera(
    image_paths: list[Path],
    pattern_size: tuple[int, int],
    square_size: float,
    pattern_type: Literal["chessboard", "symmetric_circles", "asymmetric_circles"] = "chessboard",
) -> tuple[CalibrationResult, list[DetectionResult]]:
    object_points = build_object_points(pattern_size, square_size)
    obj_points_list: list[np.ndarray] = []
    img_points_list: list[np.ndarray] = []
    all_results: list[DetectionResult] = []

    for path in image_paths:
        if pattern_type == "chessboard":
            result = detect_chessboard_corners(path, pattern_size)
        elif pattern_type == "symmetric_circles":
            result = detect_circle_grid_corners(path, pattern_size, symmetric=True)
        else:
            result = detect_circle_grid_corners(path, pattern_size, symmetric=False)

        all_results.append(result)
        if result.success and result.corners is not None:
            obj_points_list.append(object_points)
            img_points_list.append(result.corners)

    if len(obj_points_list) < 2:
        raise ValueError(f"成功检测的图片不足（仅 {len(obj_points_list)} 张），至少需要 2 张才能标定。")

    first_img = _load_image_gray(image_paths[0])
    if first_img is None:
        raise ValueError("无法读取第一张图像以获取图像尺寸。")

    image_size = (first_img.shape[1], first_img.shape[0])

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_list,
        img_points_list,
        image_size,
        None,
        None,
    )

    dist = dist_coeffs.flatten().tolist()
    while len(dist) < 8:
        dist.append(0.0)

    # Compute per-image reprojection errors
    per_image_errors: list[tuple[str, float]] = []
    total_error = 0.0
    for i, (objp, imgp, rvec, tvec) in enumerate(zip(obj_points_list, img_points_list, rvecs, tvecs)):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = float(np.sqrt(np.mean((imgp - projected) ** 2)))
        per_image_errors.append((str(image_paths[i].name), error))
        total_error += error

    mean_error = total_error / len(obj_points_list) if obj_points_list else 0.0

    return CalibrationResult(
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
        cx=float(camera_matrix[0, 2]),
        cy=float(camera_matrix[1, 2]),
        width=image_size[0],
        height=image_size[1],
        distortion=dist,
        rms_error=ret,
        per_image_errors=per_image_errors,
        camera_matrix=camera_matrix,
    ), all_results


def draw_detected_corners(image: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    img = image.copy()
    if corners is not None:
        cv2.drawChessboardCorners(img, (0, 0), corners, True)
        for i, corner in enumerate(corners):
            x, y = int(corner[0][0]), int(corner[0][1])
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            if i < 10 or i == len(corners) - 1:
                cv2.putText(img, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return img


def undistort_image(image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]
    return undistorted


def save_calibration_json(path: Path, result: CalibrationResult) -> None:
    payload = {
        "camera_matrix": result.camera_matrix.tolist() if result.camera_matrix is not None else None,
        "distortion_coefficients": {
            "k1": result.distortion[0] if len(result.distortion) > 0 else 0.0,
            "k2": result.distortion[1] if len(result.distortion) > 1 else 0.0,
            "p1": result.distortion[2] if len(result.distortion) > 2 else 0.0,
            "p2": result.distortion[3] if len(result.distortion) > 3 else 0.0,
            "k3": result.distortion[4] if len(result.distortion) > 4 else 0.0,
            "k4": result.distortion[5] if len(result.distortion) > 5 else 0.0,
            "k5": result.distortion[6] if len(result.distortion) > 6 else 0.0,
            "k6": result.distortion[7] if len(result.distortion) > 7 else 0.0,
        },
        "image_width": result.width,
        "image_height": result.height,
        "fx": result.fx,
        "fy": result.fy,
        "cx": result.cx,
        "cy": result.cy,
        "rms_reprojection_error": result.rms_error,
        "model": "pinhole",
        "distortion_model": "radial-tangential",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
