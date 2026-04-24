#!/usr/bin/env python3
"""将相机标定 TXT 文件转换为 lidar_calib_suite 可读的 JSON 格式."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_txt(txt_path: Path) -> dict:
    """解析 TXT 标定文件，提取内参和畸变系数."""
    content = txt_path.read_text(encoding="utf-8")

    data = {}
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(?:\d+\s+)?([^:]+):\s*(.*)$", line)
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        data[key] = value

    return data


def _f(v: float) -> str:
    """格式化浮点数，保留最多 10 位小数，不使用科学计数法."""
    if isinstance(v, float) and v.is_integer():
        return f"{int(v)}.0"
    s = f"{v:.10f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _row(values: list[float]) -> str:
    return ", ".join(_f(v) for v in values)


def build_json_content(data: dict, width: int = 1920, height: int = 1080) -> str:
    """根据解析的数据构建 lidar_calib_suite 所需 JSON 文本."""
    fx = float(data["FX"])
    fy = float(data["FY"])
    cx = float(data["CX"])
    cy = float(data["CY"])

    k1 = float(data["K1"])
    k2 = float(data["K2"])
    p1 = float(data["P1"])
    p2 = float(data["P2"])
    k3 = float(data["K3"])
    k4 = float(data["K4"])
    k5 = float(data["K5"])
    k6 = float(data["K6"])

    sn = data.get("SN码", "")
    rms_line = f'    "rms": {_f(float(data["RMS"]))},\n' if "RMS" in data else ""

    return (
        "{\n"
        f'    "sn": "{sn}",\n'
        f'    "width": {width},\n'
        f'    "height": {height},\n'
        f'    "K": [\n'
        f'        [{_row([fx, 0.0, cx])}],\n'
        f'        [{_row([0.0, fy, cy])}],\n'
        f'        [{_row([0.0, 0.0, 1.0])}]\n'
        f'    ],\n'
        f'    "D": [{_row([k1, k2, p1, p2, k3, k4, k5, k6])}],\n'
        f'    "R": [\n'
        f'        [{_row([1.0, 0.0, 0.0])}],\n'
        f'        [{_row([0.0, 1.0, 0.0])}],\n'
        f'        [{_row([0.0, 0.0, 1.0])}]\n'
        f'    ],\n'
        f'    "P": [\n'
        f'        [{_row([fx, 0.0, cx, 0.0])}],\n'
        f'        [{_row([0.0, fy, cy, 0.0])}],\n'
        f'        [{_row([0.0, 0.0, 1.0, 0.0])}]\n'
        f'    ]\n'
        "}\n"
    )


def convert_file(txt_path: Path, out_dir: Path | None = None, width: int = 1920, height: int = 1080) -> Path:
    """转换单个 TXT 文件为 JSON."""
    data = parse_txt(txt_path)

    if "SN码" not in data:
        raise ValueError(f"文件中未找到 SN码: {txt_path}")

    sn = data["SN码"]
    json_name = f"camera_{sn}.json"
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / json_name
    else:
        json_path = txt_path.parent / json_name

    json_path.write_text(build_json_content(data, width=width, height=height), encoding="utf-8")
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将相机标定 TXT 转换为 lidar_calib_suite 可读的 JSON"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=".",
        help="输入 TXT 文件或目录（默认当前目录）",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="输出目录（默认与输入文件同级）",
    )
    parser.add_argument(
        "-W", "--width",
        type=int,
        default=1920,
        help="图像宽度（默认 1920）",
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=1080,
        help="图像高度（默认 1080）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else None

    if input_path.is_file():
        txt_files = [input_path]
    else:
        txt_files = sorted(input_path.glob("*.txt"))

    if not txt_files:
        print(f"未找到 .txt 文件: {input_path}")
        return 1

    for txt_file in txt_files:
        try:
            json_path = convert_file(txt_file, out_dir=out_dir, width=args.width, height=args.height)
            print(f"转换成功: {txt_file} -> {json_path}")
        except Exception as e:
            print(f"转换失败: {txt_file} -> {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
