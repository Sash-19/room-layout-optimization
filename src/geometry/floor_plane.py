from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .camera import CameraIntrinsics


@dataclass(slots=True)
class FloorPlane:
    normal: np.ndarray
    offset: float
    inliers: int = 0
    total: int = 0
    threshold: float = 0.0

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            raise ValueError("plane normal must be non-zero")
        self.normal = normal / norm
        self.offset = float(self.offset)

    @property
    def camera_height(self) -> float:
        return float(self.offset)

    @property
    def inlier_ratio(self) -> float:
        return float(self.inliers) / float(self.total) if self.total else 0.0

    def distance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return points @ self.normal + self.offset

    def scale(self, factor: float) -> FloorPlane:
        if factor <= 0:
            raise ValueError("scale factor must be positive")
        return FloorPlane(
            normal=self.normal.copy(),
            offset=self.offset * float(factor),
            inliers=self.inliers,
            total=self.total,
            threshold=self.threshold * float(factor),
        )

    def intersect_rays(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
        denom = rays @ self.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -self.offset / denom
        valid = np.isfinite(t) & (t > 0) & (np.abs(denom) > 1e-9)
        points = rays * np.where(valid, t, 0.0)[:, None]
        return points, valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal": [round(float(value), 6) for value in self.normal],
            "offset": round(float(self.offset), 6),
            "camera_height": round(float(self.camera_height), 6),
            "inliers": int(self.inliers),
            "total": int(self.total),
            "inlier_ratio": round(self.inlier_ratio, 4),
            "threshold": round(float(self.threshold), 6),
        }


def _plane_from_three_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float] | None:
    normal = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    normal = normal / norm
    offset = -float(normal @ a)
    return normal, offset


def _refit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("degenerate floor points")
    normal = normal / norm
    offset = -float(normal @ centroid)
    return normal, offset


def fit_floor_plane(
    depth: np.ndarray,
    camera: CameraIntrinsics,
    floor_mask: np.ndarray | None = None,
    stride: int = 4,
    max_points: int = 20000,
    threshold: float = 0.03,
    iterations: int = 300,
    seed: int = 0,
) -> FloorPlane:
    depth = np.asarray(depth, dtype=np.float64)
    if depth.shape != (camera.height, camera.width):
        raise ValueError(f"depth shape {depth.shape} does not match camera image size")
    if floor_mask is None:
        mask = np.ones_like(depth, dtype=bool)
    else:
        mask = np.asarray(floor_mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError("floor mask shape must match depth")
    valid = np.isfinite(depth) & (depth > 0) & mask
    ys, xs = np.nonzero(valid)
    if len(xs) < 3:
        raise ValueError("not enough valid floor pixels to fit a plane")
    step = max(1, int(stride))
    ys, xs = ys[::step], xs[::step]
    if len(xs) > max_points:
        rng = np.random.default_rng(seed)
        selection = rng.choice(len(xs), size=max_points, replace=False)
        ys, xs = ys[selection], xs[selection]
    points = camera.backproject(np.stack([xs, ys], axis=1), depth)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        raise ValueError("not enough finite depth points to fit a plane")

    rng = np.random.default_rng(seed)
    best = None
    best_count = -1
    for _ in range(max(1, int(iterations))):
        indices = rng.choice(len(points), size=3, replace=False)
        candidate = _plane_from_three_points(points[indices[0]], points[indices[1]], points[indices[2]])
        if candidate is None:
            continue
        normal, offset = candidate
        distances = np.abs(points @ normal + offset)
        count = int(np.sum(distances <= threshold))
        if count > best_count:
            best_count = count
            best = (normal, offset, distances <= threshold)
    if best is None or best_count < 3:
        raise ValueError("RANSAC failed to find a floor plane")
    normal, offset, inliers = best
    normal, offset = _refit_plane(points[inliers])
    if normal[1] > 0:
        normal = -normal
        offset = -offset
    if offset < 0:
        raise ValueError("fitted floor plane lies above the camera; floor fit failed")
    return FloorPlane(normal, offset, inliers=int(best_count), total=len(points), threshold=float(threshold))


def calibrate_inverse_depth(
    depth: np.ndarray,
    camera: CameraIntrinsics,
    floor_mask: np.ndarray | None = None,
    stride: int = 2,
    max_points: int = 20000,
    clip_sigma: float = 2.5,
    iterations: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth = np.asarray(depth, dtype=np.float64)
    if depth.shape != (camera.height, camera.width):
        raise ValueError(f"depth shape {depth.shape} does not match camera image size")

    positive = np.isfinite(depth) & (depth > 0)
    inverted = np.zeros_like(depth)
    inverted[positive] = 1.0 / depth[positive]
    fallback = {"method": "reciprocal", "num_points": 0}

    if floor_mask is None:
        return inverted, fallback
    mask = np.asarray(floor_mask, dtype=bool)
    if mask.shape != depth.shape:
        raise ValueError("floor mask shape must match depth")

    ys, xs = np.nonzero(mask & positive)
    if len(xs) < 30:
        return inverted, fallback
    step = max(1, int(stride))
    ys, xs = ys[::step], xs[::step]
    if len(xs) > max_points:
        rng = np.random.default_rng(seed)
        selection = rng.choice(len(xs), size=max_points, replace=False)
        ys, xs = ys[selection], xs[selection]

    values = depth[ys, xs]
    rays = camera.pixel_rays(np.stack([xs, ys], axis=1))
    design = rays[:, [0, 1]]
    count = len(values)
    keep = np.ones(count, dtype=bool)
    coefficients = None
    for _ in range(max(1, int(iterations))):
        if int(keep.sum()) < 30:
            return inverted, fallback
        matrix = np.column_stack([design[keep], np.ones(int(keep.sum()))])
        coefficients, *_ = np.linalg.lstsq(matrix, values[keep], rcond=None)
        residual = values - (design @ coefficients[:2] + coefficients[2])
        scale = float(np.std(residual[keep]))
        if scale <= 1e-12:
            break
        keep = np.abs(residual) <= float(clip_sigma) * scale
        count = int(keep.sum())

    a, b, c = (float(value) for value in coefficients)
    alpha = float(np.hypot(a, b))
    if alpha <= 1e-9:
        return inverted, fallback

    denominator = depth - c
    valid = positive & (denominator > 1e-6)
    calibrated = np.zeros_like(depth)
    calibrated[valid] = alpha / denominator[valid]
    residual = values - (design @ np.array([a, b]) + c)
    params = {
        "method": "affine-inverse",
        "num_points": count,
        "coefficients": [round(a, 6), round(b, 6), round(c, 6)],
        "alpha": round(alpha, 6),
        "residual_median": round(float(np.median(np.abs(residual))), 6),
        "relative_residual": round(float(np.median(np.abs(residual)) / max(np.median(values), 1e-9)), 4),
    }
    return calibrated, params


def floor_points_from_mask(
    camera: CameraIntrinsics,
    plane: FloorPlane,
    depth: np.ndarray,
    floor_mask: np.ndarray | None = None,
    stride: int = 4,
    max_points: int = 20000,
    seed: int = 0,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    if depth.shape != (camera.height, camera.width):
        raise ValueError(f"depth shape {depth.shape} does not match camera image size")
    if floor_mask is None:
        mask = np.ones_like(depth, dtype=bool)
    else:
        mask = np.asarray(floor_mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError("floor mask shape must match depth")
    valid = np.isfinite(depth) & (depth > 0) & mask
    ys, xs = np.nonzero(valid)
    if not len(xs):
        return np.zeros((0, 3), dtype=np.float64)
    step = max(1, int(stride))
    ys, xs = ys[::step], xs[::step]
    if len(xs) > max_points:
        rng = np.random.default_rng(seed)
        selection = rng.choice(len(xs), size=max_points, replace=False)
        ys, xs = ys[selection], xs[selection]
    rays = camera.pixel_rays(np.stack([xs, ys], axis=1))
    points, hits = plane.intersect_rays(rays)
    return points[hits]
