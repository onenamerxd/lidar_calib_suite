from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CameraIntrinsics:
    fx: float = 1000.0
    fy: float = 1000.0
    cx: float = 960.0
    cy: float = 540.0
    width: int = 1920
    height: int = 1080
    distortion: list[float] = field(default_factory=lambda: [0.0] * 8)
    flip_mode: int = 0  # 0=none, 1=horizontal, 2=vertical, 3=rotate180

    def clipped_distortion(self) -> list[float]:
        values = list(self.distortion[:8])
        while len(values) < 8:
            values.append(0.0)
        return values

    def to_dict(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "distortion": list(self.distortion),
            "flip_mode": self.flip_mode,
        }


@dataclass
class Extrinsics:
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tx": self.tx,
            "ty": self.ty,
            "tz": self.tz,
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
        }


@dataclass
class FramePair:
    image_path: Path
    lidar_path: Path
    image_timestamp: Optional[float]
    lidar_timestamp: Optional[float]
    delta_seconds: Optional[float]


@dataclass
class PointCloudData:
    points_xyz: "np.ndarray"
    intensity: "np.ndarray"
    fields: list[str]


@dataclass
class CalibrationCorrespondence:
    image_u: float
    image_v: float
    lidar_x: float
    lidar_y: float
    lidar_z: float
