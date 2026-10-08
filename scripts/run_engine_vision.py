#!/usr/bin/env python3
"""Run the supplied trained detector on photos without moving either robot.

    python3 scripts/run_engine_vision.py --images data/images/engine_bore_seed
    python3 scripts/run_engine_vision.py --images path/to/photo.jpg --out /tmp/vision.json
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection import schemas
from box_hole_inspection.vision_ai import EngineBoreDetector


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path,
                        default=PROJECT_DIR / "data/images/engine_bore_seed")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.images.is_file():
        images = [args.images]
    elif args.images.is_dir():
        images = sorted(p for p in args.images.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    else:
        parser.error(f"image path not found: {args.images}")
    if not images:
        parser.error(f"no images found: {args.images}")

    detector = EngineBoreDetector()
    rows = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            rows.append({"image": str(path), "error": "image read failed"})
            continue
        try:
            detection = detector.detect(image)
            row = {"image": str(path),
                   "detection": schemas.to_dict(detection),
                   "diagnostics": detector.last_diagnostics}
            print(f"{path.name}: {len(detection.holes)}/4 reliable bores")
        except Exception as exc:
            row = {"image": str(path), "error": str(exc)}
            print(f"{path.name}: ERROR {exc}")
        rows.append(row)

    out = args.out or (PROJECT_DIR / "data/runs"
                       / datetime.now().strftime("%Y-%m-%d")
                       / "engine_vision_detections.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results: {out}")
    return 0 if all("error" not in row for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
