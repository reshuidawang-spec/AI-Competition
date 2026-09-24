"""Simulated camera: pinhole rendering of the SimWorld.

Renders the box top face (holes as projected circles -> ellipses) from a
known camera pose T_W_C with realistic noise. The SAME T_W_C convention as
coordinate_transform.py means:

    render(T_W_C, world)  ->  image  ->  vision_ai  ->  pixels
        ->  CoordinateTransform.pixel_to_world  ->  ≈  world.hole_centers_w()

is a closed, verifiable loop in simulation (locked by tests). Instance masks
are exported alongside the image, so renders double as FREE labeled
training data for the YOLO-Seg pipeline (synthetic pretraining before real
photos arrive).

Rendering is geometric (no GPU, no 3D engine): polygons projected through
the pinhole and rasterised with cv2.fillPoly. Primitives with any vertex
behind the camera are skipped (not clipped) and flagged not visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from box_hole_inspection.utils import transforms as tf
from interfaces.camera import CameraIntrinsics
from sim.sim_world import SimWorld

# 渲染配色（BGR）：台面 / 箱体顶面 / 顶面描边 / 孔
COLOR_TABLE = (72, 72, 72)
COLOR_BOX_TOP = (186, 186, 186)
COLOR_BOX_EDGE = (120, 120, 120)
COLOR_HOLE = (22, 22, 22)

CIRCLE_SAMPLES = 72          # 圆采样点数（渲染近似椭圆精度）
NEAR_Z_MM = 1.0              # 相机前方最近可渲染深度


@dataclass
class SimRender:
    """One rendered frame + ground-truth metadata."""
    image: np.ndarray                                  # BGR uint8
    hole_pixel_gt: dict = field(default_factory=dict)  # id -> (u, v) 3D孔心投影像素
    hole_visible: dict = field(default_factory=dict)   # id -> bool
    box_bbox_px: Optional[tuple] = None                # [x0, y0, x1, y1]
    masks: dict = field(default_factory=dict)          # id -> bool mask（含 "box"）


class SimCamera:
    """Pinhole camera at a fixed pose. Deterministic given the seed."""

    def __init__(self, intrinsics: CameraIntrinsics, t_w_c: np.ndarray,
                 seed: int = 0, noise_sigma: float = 2.5,
                 brightness_jitter: float = 6.0):
        self.intrinsics = intrinsics
        self.T_W_C = np.asarray(t_w_c, dtype=float)
        self.T_C_W = tf.invert_T(self.T_W_C)
        self.rng = np.random.default_rng(seed)
        self.noise_sigma = noise_sigma
        self.brightness_jitter = brightness_jitter
        self.K = np.array([[intrinsics.fx, 0.0, intrinsics.cx],
                           [0.0, intrinsics.fy, intrinsics.cy],
                           [0.0, 0.0, 1.0]])

    @classmethod
    def from_robot_a(cls, robot, t_ea_c: np.ndarray,
                     capture_joints_deg: list, **kwargs) -> "SimCamera":
        """Build the camera from vision-arm state: T_W_C = base_w @ FK(q) @ T_EA_C."""
        T_W_EA = tf.compose(robot.base_pose_w, robot.fk(capture_joints_deg))
        T_W_C = tf.compose(T_W_EA, np.asarray(t_ea_c, dtype=float))
        intrinsics = kwargs.pop("intrinsics", None)
        if intrinsics is None:
            intrinsics = CameraIntrinsics()
        return cls(intrinsics, T_W_C, **kwargs)

    # -- projection ---------------------------------------------------------

    def project(self, p_w) -> Optional[tuple]:
        """World point -> (u, v) pixel; None if behind the camera."""
        p_c = tf.apply_T(self.T_C_W, p_w)
        if p_c[2] < NEAR_Z_MM:
            return None
        uv = self.K @ (p_c / p_c[2])
        return float(uv[0]), float(uv[1])

    def _project_many(self, pts_w: np.ndarray):
        """(N,3) world points -> (N,2) pixels + visibility mask."""
        p_c = tf.apply_T(self.T_C_W, pts_w)
        vis = p_c[:, 2] >= NEAR_Z_MM
        uv = np.full((len(pts_w), 2), np.nan)
        # 先除以深度归一化：K @ (3,N) -> (3,N)，取前两行 -> (2,N)，转置 -> (N,2)
        uv[vis] = (self.K @ (p_c[vis] / p_c[vis, 2:3]).T)[:2].T
        return uv, vis

    # -- rendering ----------------------------------------------------------

    def render(self, world: SimWorld) -> SimRender:
        h, w = self.intrinsics.height, self.intrinsics.width
        img = np.full((h, w, 3), COLOR_TABLE, dtype=np.float32)
        masks: dict = {}
        result = SimRender(image=img.astype(np.uint8))

        # 箱体顶面多边形
        corners = world.top_face_corners_w()
        uv, vis = self._project_many(corners)
        if not vis.all():
            return result                      # 顶面部分在相机后：整面不可见
        poly = uv.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(img, [poly], COLOR_BOX_TOP)
        cv2.polylines(img, [poly], True, COLOR_BOX_EDGE, 2, cv2.LINE_AA)
        mask_box = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_box, [poly], 1)
        masks["box"] = mask_box.astype(bool)
        x0 = int(np.nanmin(uv[:, 0])); x1 = int(np.nanmax(uv[:, 0]))
        y0 = int(np.nanmin(uv[:, 1])); y1 = int(np.nanmax(uv[:, 1]))
        result.box_bbox_px = [x0, y0, x1, y1]

        # 各孔：3D 圆采样 -> 投影 -> 填充
        centers_w = world.hole_centers_w()
        x, y, rz = world.box.pose
        R = tf.euler_to_R(0.0, 0.0, rz)
        for hole in world.box.holes:
            c_w = centers_w[hole.id]
            r = hole.diameter_mm / 2.0
            theta = np.linspace(0.0, 2.0 * np.pi, CIRCLE_SAMPLES, endpoint=False)
            ring_o = np.stack([r * np.cos(theta), r * np.sin(theta),
                               np.zeros_like(theta)], axis=1)
            ring_w = c_w + (R @ ring_o.T).T
            uv, vis = self._project_many(ring_w)
            if not vis.all():
                result.hole_visible[hole.id] = False
                continue
            poly = uv.astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(img, [poly], COLOR_HOLE)
            mask_h = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_h, [poly], 1)
            masks[hole.id] = mask_h.astype(bool)
            result.hole_visible[hole.id] = True
            gt = self.project(c_w)
            if gt is not None:
                result.hole_pixel_gt[hole.id] = gt

        # 噪声 + 亮度抖动（种子确定，可复现）
        img += self.rng.normal(0.0, self.noise_sigma, img.shape)
        img += float(self.rng.uniform(-self.brightness_jitter,
                                      self.brightness_jitter))
        result.image = np.clip(img, 0, 255).astype(np.uint8)
        result.masks = masks
        return result
