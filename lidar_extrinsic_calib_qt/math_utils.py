from __future__ import annotations

import numpy as np


def depth_to_rgb(z_values: np.ndarray) -> np.ndarray:
    """Map float depths to RGB colors using a simple jet-like colormap."""
    if z_values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    if abs(z_max - z_min) < 1e-6:
        return np.full((z_values.shape[0], 3), 128, dtype=np.uint8)
    t = (z_values - z_min) / (z_max - z_min)
    t = np.clip(t, 0.0, 1.0)
    colors = np.zeros((t.shape[0], 3), dtype=np.uint8)
    for i, val in enumerate(t):
        colors[i] = _jet_color(float(val))
    return colors


def _jet_color(t: float) -> tuple[int, int, int]:
    """Simple jet colormap: t in [0,1] -> RGB."""
    t = max(0.0, min(1.0, t))
    if t < 0.25:
        r = 0
        g = int(255 * (4 * t))
        b = 255
    elif t < 0.5:
        r = 0
        g = 255
        b = int(255 * (1 - 4 * (t - 0.25)))
    elif t < 0.75:
        r = int(255 * (4 * (t - 0.5)))
        g = 255
        b = 0
    else:
        r = 255
        g = int(255 * (1 - 4 * (t - 0.75)))
        b = 0
    return r, g, b


def compute_bounding_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return min, max corners of the axis-aligned bounding box."""
    if points.shape[0] == 0:
        return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    return np.min(points, axis=0), np.max(points, axis=0)
