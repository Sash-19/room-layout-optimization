from __future__ import annotations

import numpy as np

from src.geometry.camera import CameraIntrinsics
from src.schemas import DepthMap, Detection, StructureMasks

WIDTH = 640
HEIGHT = 480
FOCAL = 500.0
CAMERA_HEIGHT = 1.5


def make_camera() -> CameraIntrinsics:
    return CameraIntrinsics(FOCAL, FOCAL, WIDTH / 2.0, HEIGHT / 2.0, WIDTH, HEIGHT, "test")


def make_depth(camera: CameraIntrinsics, height: float = CAMERA_HEIGHT) -> np.ndarray:
    depth = np.zeros((camera.height, camera.width), dtype=np.float32)
    for v in range(camera.height):
        ray_y = (v - camera.cy) / camera.fy
        if ray_y > 1e-6:
            depth[v, :] = height / ray_y
    return depth


def make_floor_mask(rect: tuple[int, int, int, int] = (220, 320, 420, 460)) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    x1, y1, x2, y2 = rect
    mask[y1:y2, x1:x2] = True
    return mask


def make_structures(mask: np.ndarray | None = None) -> StructureMasks:
    floor = make_floor_mask() if mask is None else mask
    return StructureMasks(floor=floor, wall=np.zeros_like(floor), source="test")


def make_bed_detection(conf: float = 0.9) -> Detection:
    return Detection("bed", conf, (250.0, 330.0, 390.0, 430.0))


def make_door_detection(conf: float = 0.8) -> Detection:
    return Detection("door", conf, (300.0, 280.0, 350.0, 450.0))


def make_perception(
    detections: list[Detection] | None = None,
    depth: np.ndarray | None = None,
    metric: bool = True,
) -> tuple[DepthMap, StructureMasks]:
    depth = make_depth(make_camera()) if depth is None else depth
    return DepthMap(depth, metric=metric, source="synthetic"), make_structures()
