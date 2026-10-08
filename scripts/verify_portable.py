#!/usr/bin/env python3
"""Check bundled assets, installed dependencies, tests and one AI sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_DIR / "data/portable_manifest.json"


def check_assets() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = manifest["files"]
    for item in files:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"非法清单路径：{relative}")
        path = PROJECT_DIR / relative
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise FileNotFoundError(f"缺失或大小不符：{relative}")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"SHA256 不符：{relative}")
    print(f"资源完整：{len(files)} 个文件")


def check_tests() -> None:
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                    "-p", "test_*.py", "-q"], cwd=PROJECT_DIR, check=True)


def check_vision() -> None:
    import cv2
    from box_hole_inspection.vision_ai import EngineBoreDetector

    sample = sorted((PROJECT_DIR / "data/images/engine_bore_seed").glob("*.jpg"))[0]
    image = cv2.imread(str(sample))
    if image is None:
        raise RuntimeError(f"无法读取验收照片：{sample}")
    result = EngineBoreDetector().detect(image)
    if result.part_id != "ENGINE_001" or len(result.holes) != 4:
        raise RuntimeError("视觉样本未得到 ENGINE_001 的 4 个可靠圆心")
    print("视觉推理通过：ENGINE_001，4 个可靠像素圆心")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(PROJECT_DIR))
    check_assets()
    check_tests()
    if not args.core_only:
        check_vision()
    print("工程复用检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
