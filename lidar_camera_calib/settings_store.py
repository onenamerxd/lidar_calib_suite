from __future__ import annotations

import json
from pathlib import Path


SETTINGS_FILENAME = "lidar_camera_calib_settings.json"


def _settings_path() -> Path:
    return Path(__file__).parent.parent / SETTINGS_FILENAME


def load_settings() -> dict:
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(settings: dict) -> None:
    path = _settings_path()
    try:
        path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
