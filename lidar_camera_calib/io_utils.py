from __future__ import annotations

import bisect
import io
import json
import re
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage

from .math_utils import matrix_to_euler_deg, quaternion_to_matrix
from .models import CameraIntrinsics, Extrinsics, FramePair, PointCloudData


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
PCD_EXTENSIONS = {".pcd"}


def parse_timestamp(path: Path) -> float | None:
    stem = path.stem
    try:
        return float(stem)
    except ValueError:
        # Extract the last number group to avoid matching "camera_4_1920_1080_down_0001"
        matches = list(re.finditer(r"(\d+\.\d+|\d+)", stem))
        if not matches:
            return None
        try:
            return float(matches[-1].group(1))
        except ValueError:
            return None


def _collect_files(folder: Path, suffixes: set[str]) -> list[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
    return sorted(files)


def load_camera_json(json_path: str | Path) -> tuple[CameraIntrinsics | None, Extrinsics | None, dict]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    intrinsics = None
    extrinsics = None

    if "K" in payload and payload["K"]:
        matrix_k = payload["K"]
        intrinsics = CameraIntrinsics(
            fx=float(matrix_k[0][0]),
            fy=float(matrix_k[1][1]),
            cx=float(matrix_k[0][2]),
            cy=float(matrix_k[1][2]),
            width=int(payload.get("width", 1920)),
            height=int(payload.get("height", 1080)),
            distortion=[float(v) for v in payload.get("D", [])],
            flip_mode=int(payload.get("flip_mode", 0)),
        )

    if "rotation" in payload and "translation" in payload:
        rotation = quaternion_to_matrix(payload["rotation"])
        roll, pitch, yaw = matrix_to_euler_deg(rotation)
        translation = payload["translation"]
        extrinsics = Extrinsics(
            tx=float(translation[0]),
            ty=float(translation[1]),
            tz=float(translation[2]),
            roll_deg=roll,
            pitch_deg=pitch,
            yaw_deg=yaw,
        )

    return intrinsics, extrinsics, payload


def build_frame_pairs(image_dir: str | Path, lidar_dir: str | Path, time_offset_sec: float = 0.0) -> list[FramePair]:
    image_root = Path(image_dir)
    lidar_root = Path(lidar_dir)
    image_files = _collect_files(image_root, IMAGE_EXTENSIONS)
    lidar_files = _collect_files(lidar_root, PCD_EXTENSIONS)
    if not image_files or not lidar_files:
        return []

    image_ts = [parse_timestamp(p) for p in image_files]
    lidar_ts = [parse_timestamp(p) for p in lidar_files]

    if all(ts is not None for ts in image_ts) and all(ts is not None for ts in lidar_ts):
        lidar_keys = [float(ts) + time_offset_sec for ts in lidar_ts]
        pairs: list[FramePair] = []
        for img_path, img_ts in zip(image_files, image_ts):
            target = float(img_ts)
            insert_at = bisect.bisect_left(lidar_keys, target)
            candidates = []
            if insert_at < len(lidar_keys):
                candidates.append(insert_at)
            if insert_at > 0:
                candidates.append(insert_at - 1)
            best_index = min(candidates, key=lambda idx: abs(lidar_keys[idx] - target))
            delta = float(target - lidar_keys[best_index])
            pairs.append(
                FramePair(
                    image_path=img_path,
                    lidar_path=lidar_files[best_index],
                    image_timestamp=float(img_ts),
                    lidar_timestamp=float(lidar_ts[best_index]),
                    delta_seconds=delta,
                )
            )
        return pairs

    pairs = []
    total = min(len(image_files), len(lidar_files))
    for index in range(total):
        pairs.append(
            FramePair(
                image_path=image_files[index],
                lidar_path=lidar_files[index],
                image_timestamp=image_ts[index],
                lidar_timestamp=lidar_ts[index],
                delta_seconds=None,
            )
        )
    return pairs


def load_qimage(image_path: str | Path) -> QImage:
    image = QImage(str(image_path))
    if image.isNull():
        raise ValueError(f"无法读取图片: {image_path}")
    return image


def _parse_pcd_header(raw: bytes) -> tuple[dict[str, list[str]], int]:
    offset = 0
    header: dict[str, list[str]] = {}
    while True:
        newline = raw.find(b"\n", offset)
        if newline < 0:
            raise ValueError("PCD 头不完整")
        line = raw[offset:newline].decode("ascii", errors="ignore").strip()
        offset = newline + 1
        if not line or line.startswith("#"):
            continue
        key, *values = line.split()
        header[key.upper()] = values
        if key.upper() == "DATA":
            break
    return header, offset


def load_pcd(pcd_path: str | Path) -> PointCloudData:
    path = Path(pcd_path)
    raw = path.read_bytes()
    header, data_offset = _parse_pcd_header(raw)

    fields = header.get("FIELDS", [])
    sizes = [int(v) for v in header.get("SIZE", ["4"] * len(fields))]
    types = header.get("TYPE", ["F"] * len(fields))
    counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
    points = int(header.get("POINTS", ["0"])[0])
    data_kind = header.get("DATA", ["ascii"])[0].lower()

    if data_kind == "ascii":
        text = raw[data_offset:].decode("ascii", errors="ignore").strip()
        if not text:
            matrix = np.zeros((0, len(fields)), dtype=np.float32)
        else:
            matrix = np.loadtxt(io.StringIO(text), dtype=np.float32)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
    elif data_kind == "binary":
        dtype_fields = []
        for name, size, typ, count in zip(fields, sizes, types, counts):
            if typ == "F" and size == 4:
                base_dtype = np.float32
            elif typ == "F" and size == 8:
                base_dtype = np.float64
            elif typ == "U" and size == 1:
                base_dtype = np.uint8
            elif typ == "U" and size == 2:
                base_dtype = np.uint16
            elif typ == "U" and size == 4:
                base_dtype = np.uint32
            elif typ == "I" and size == 1:
                base_dtype = np.int8
            elif typ == "I" and size == 2:
                base_dtype = np.int16
            elif typ == "I" and size == 4:
                base_dtype = np.int32
            else:
                raise ValueError(f"暂不支持的 PCD 字段类型: {typ}{size}")
            if count == 1:
                dtype_fields.append((name, base_dtype))
            else:
                dtype_fields.append((name, base_dtype, (count,)))

        dtype = np.dtype(dtype_fields)
        records = np.frombuffer(raw, dtype=dtype, count=points, offset=data_offset)
        matrix = np.zeros((points, len(fields)), dtype=np.float32)
        for column_index, field_name in enumerate(fields):
            column = records[field_name]
            matrix[:, column_index] = column[:, 0] if column.ndim > 1 else column
    else:
        raise ValueError(f"暂不支持的 PCD DATA 类型: {data_kind}")

    if matrix.size == 0:
        xyz = np.zeros((0, 3), dtype=np.float32)
        intensity = np.zeros((0,), dtype=np.float32)
        return PointCloudData(points_xyz=xyz, intensity=intensity, fields=list(fields))

    def column(field_name: str, fallback: float = 0.0) -> np.ndarray:
        if field_name in fields:
            return matrix[:, fields.index(field_name)].astype(np.float32)
        return np.full((matrix.shape[0],), fallback, dtype=np.float32)

    xyz = np.column_stack([column("x"), column("y"), column("z")]).astype(np.float32)
    intensity = column("intensity")
    return PointCloudData(points_xyz=xyz, intensity=intensity, fields=list(fields))
