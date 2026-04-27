#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
PARENT_VENV_PYTHON="$SCRIPT_DIR/../.venv/bin/python"
LEGACY_BUNDLE_PYTHON="$SCRIPT_DIR/../.miniconda3/envs/calib/bin/python"

python_can_launch_project() {
    local python_bin="$1"

    if [ ! -x "$python_bin" ]; then
        return 1
    fi

    "$python_bin" - <<'PY' >/dev/null 2>&1
import importlib

required = ["PySide6", "numpy", "cv2", "scipy", "open3d"]
for name in required:
    importlib.import_module(name)
PY
}

find_runtime_python() {
    local current_python=""
    local path_python=""

    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        current_python="${CONDA_PREFIX}/bin/python"
        if python_can_launch_project "$current_python"; then
            printf '%s\n' "$current_python"
            return 0
        fi
    fi

    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        current_python="${VIRTUAL_ENV}/bin/python"
        if python_can_launch_project "$current_python"; then
            printf '%s\n' "$current_python"
            return 0
        fi
    fi

    if python_can_launch_project "$LOCAL_VENV_PYTHON"; then
        printf '%s\n' "$LOCAL_VENV_PYTHON"
        return 0
    fi

    if python_can_launch_project "$PARENT_VENV_PYTHON"; then
        printf '%s\n' "$PARENT_VENV_PYTHON"
        return 0
    fi

    if python_can_launch_project "$LEGACY_BUNDLE_PYTHON"; then
        printf '%s\n' "$LEGACY_BUNDLE_PYTHON"
        return 0
    fi

    path_python="$(command -v python3 2>/dev/null || true)"
    if [ -n "$path_python" ] && python_can_launch_project "$path_python"; then
        printf '%s\n' "$path_python"
        return 0
    fi

    path_python="$(command -v python 2>/dev/null || true)"
    if [ -n "$path_python" ] && python_can_launch_project "$path_python"; then
        printf '%s\n' "$path_python"
        return 0
    fi

    return 1
}

PYTHON_BIN="$(find_runtime_python || true)"

if [ -z "$PYTHON_BIN" ]; then
    printf '%s\n' "未找到可用的 Python 环境。" >&2
    printf '%s\n' "请先创建 .venv 并安装 requirements.txt，或激活一个已安装 PySide6/numpy/opencv-python/scipy/open3d 的环境。" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" launcher.py "$@"
