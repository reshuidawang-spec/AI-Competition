#!/usr/bin/env python3
"""合成训练数据集生成器：sim_camera 渲染 + 实例掩膜 -> YOLO-Seg 格式。

用途：真实箱体照片未拍/未标注前，先用渲染图预训练 YOLO（箱体+圆孔是简单
几何，合成->真实迁移效果好）；真实数据到位后在此基础上微调或重训。

输出（YOLO-Seg 惯例，可直接交给 ultralytics）：

    <out_dir>/images/train|val/*.png
    <out_dir>/labels/train|val/*.txt   每行: class cx cy w h x1 y1 x2 y2 ...
    <out_dir>/dataset.yaml

用法：

    python3 scripts/gen_synthetic_dataset.py --out data/synthetic_box \
        --n-train 300 --n-val 60 --seed 0

训练（数据生成后）：

    yolo segment train model=yolo11n-seg.pt data=data/synthetic_box/dataset.yaml \
        epochs=100 imgsz=1280 batch=8

注意：合成数据集的摆放变化范围要覆盖比赛场景（实验2 的 ±50mm / ±20°），
训练出的模型才具备对箱体位置变化的鲁棒性。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from interfaces.camera import CameraIntrinsics
from sim.sim_camera import SimCamera
from sim.sim_world import SimWorld, default_box
from box_hole_inspection.utils import transforms as tf

CLASS_NAMES = {0: "box", 1: "hole"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/synthetic_box", help="输出目录")
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--n-val", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--box-xy-jitter-mm", type=float, default=50.0,
                   help="箱体 x/y 相对基准点的抖动幅度（覆盖实验2范围）")
    p.add_argument("--rz-range-deg", type=float, default=20.0)
    p.add_argument("--cam-tilt-range-deg", type=float, default=10.0)
    return p.parse_args(argv)


def render_sample(world: SimWorld, cam: SimCamera, rng: np.random.Generator,
                  box_xy_jitter: float, rz_range: float, tilt_range: float):
    """随机摆放箱体 + 随机相机俯仰角，渲染一帧。"""
    x = 900.0 + rng.uniform(-box_xy_jitter, box_xy_jitter)
    y = 0.0 + rng.uniform(-box_xy_jitter, box_xy_jitter)
    rz = rng.uniform(-rz_range, rz_range)
    world.set_box_pose(x, y, rz)

    ry = rng.uniform(-tilt_range, tilt_range)
    rx = 180.0 + rng.uniform(-tilt_range * 0.2, tilt_range * 0.2)
    t_w_c = tf.pose_to_T({"x": x, "y": y, "z": 480.0,
                          "rx": rx, "ry": ry, "rz": 0.0})
    cam.T_W_C = t_w_c
    cam.T_C_W = tf.invert_T(t_w_c)
    cam.noise_sigma = rng.uniform(1.5, 4.0)
    cam.brightness_jitter = rng.uniform(2.0, 10.0)
    return cam.render(world)


def mask_to_yolo_lines(mask: np.ndarray, class_id: int, w: int, h: int) -> str:
    """bool 掩膜 -> YOLO-Seg 一行（bbox + 多边形，归一化坐标）。"""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""
    c = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)
    cx, cy = (x + cw / 2.0) / w, (y + ch / 2.0) / h
    bw, bh = cw / w, ch / h
    pts = [f"{px / w:.6f} {py / h:.6f}" for px, py in c.reshape(-1, 2)]
    return (f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
            + " ".join(pts) + "\n")


def generate_split(out_dir: Path, split: str, n: int, rng: np.random.Generator,
                   args) -> int:
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    world = SimWorld(default_box())
    intrinsics = CameraIntrinsics(fx=1000, fy=1000, cx=640, cy=512,
                                  width=1280, height=1024)
    t_w_c = np.eye(4)
    cam = SimCamera(intrinsics, t_w_c, seed=int(rng.integers(1 << 31)))

    written = 0
    for i in range(n):
        render = render_sample(world, cam, rng, args.box_xy_jitter_mm,
                               args.rz_range_deg, args.cam_tilt_range_deg)
        if not all(render.hole_visible.values()):
            continue                       # 孔不全可见的样本不进入训练集
        name = f"{split}_{i:05d}"
        cv2.imwrite(str(img_dir / f"{name}.png"), render.image)
        lines = []
        if "box" in render.masks:
            line = mask_to_yolo_lines(render.masks["box"], 0,
                                      intrinsics.width, intrinsics.height)
            if line:
                lines.append(line)
        for hid in sorted(render.masks):
            if hid == "box":
                continue
            line = mask_to_yolo_lines(render.masks[hid], 1,
                                      intrinsics.width, intrinsics.height)
            if line:
                lines.append(line)
        (lbl_dir / f"{name}.txt").write_text("".join(lines), encoding="utf-8")
        written += 1
    return written


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out)
    rng = np.random.default_rng(args.seed)

    n_train = generate_split(out_dir, "train", args.n_train, rng, args)
    n_val = generate_split(out_dir, "val", args.n_val, rng, args)

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"# 合成数据集（sim_camera 渲染生成），可被 ultralytics 直接使用\n"
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: box\n  1: hole\n",
        encoding="utf-8")

    print(f"生成完成: train={n_train} val={n_val} -> {out_dir.resolve()}")
    print(f"训练命令: yolo segment train model=yolo11n-seg.pt "
          f"data={yaml_path} epochs=100 imgsz=1280 batch=8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
