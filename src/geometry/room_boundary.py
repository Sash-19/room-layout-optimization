from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import Polygon, box

from .floor_plane import FloorPlane


@dataclass(slots=True)
class FloorFrame:
    normal: np.ndarray
    u: np.ndarray
    v: np.ndarray
    origin: np.ndarray

    def __post_init__(self) -> None:
        self.normal = np.asarray(self.normal, dtype=np.float64).reshape(3)
        self.u = np.asarray(self.u, dtype=np.float64).reshape(3)
        self.v = np.asarray(self.v, dtype=np.float64).reshape(3)
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)

    @classmethod
    def from_plane(cls, plane: FloorPlane) -> FloorFrame:
        normal = plane.normal / np.linalg.norm(plane.normal)
        candidate = np.array([1.0, 0.0, 0.0])
        u = candidate - float(candidate @ normal) * normal
        if np.linalg.norm(u) <= 1e-9:
            candidate = np.array([0.0, 0.0, 1.0])
            u = candidate - float(candidate @ normal) * normal
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        origin = -plane.offset * normal
        return cls(normal, u, v, origin)

    def to_2d(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        relative = points - self.origin
        return np.stack([relative @ self.u, relative @ self.v], axis=1)

    def to_3d(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
        return self.origin + coords[:, 0:1] * self.u + coords[:, 1:2] * self.v

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal": [round(float(value), 6) for value in self.normal],
            "u": [round(float(value), 6) for value in self.u],
            "v": [round(float(value), 6) for value in self.v],
            "origin": [round(float(value), 6) for value in self.origin],
        }


@dataclass(slots=True)
class RoomEstimate:
    polygon: Polygon
    frame: FloorFrame
    width: float
    depth: float
    bounds: tuple[float, float, float, float]
    num_points: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "polygon": [[round(x, 4), round(y, 4)] for x, y in self.polygon.exterior.coords],
            "width": round(float(self.width), 4),
            "depth": round(float(self.depth), 4),
            "bounds": [round(float(value), 4) for value in self.bounds],
            "num_points": int(self.num_points),
            "frame": self.frame.to_dict(),
        }


def estimate_room_boundary(
    floor_points: np.ndarray,
    frame: FloorFrame,
    percentile: float = 2.0,
    width_m: float | None = None,
    depth_m: float | None = None,
    min_size_m: float = 1.5,
    max_size_m: float = 30.0,
) -> RoomEstimate:
    points = np.asarray(floor_points, dtype=np.float64).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        raise ValueError("not enough floor points to estimate the room boundary")
    coords = frame.to_2d(points)
    low = float(np.clip(percentile, 0.0, 45.0))
    min_x, min_y = np.percentile(coords, low, axis=0)
    max_x, max_y = np.percentile(coords, 100.0 - low, axis=0)

    if max_x - min_x < min_size_m:
        center = (min_x + max_x) / 2.0
        min_x, max_x = center - min_size_m / 2.0, center + min_size_m / 2.0
    if max_y - min_y < min_size_m:
        center = (min_y + max_y) / 2.0
        min_y, max_y = center - min_size_m / 2.0, center + min_size_m / 2.0

    if width_m is not None:
        if width_m <= 0:
            raise ValueError("width_m must be positive")
        target = float(min(max_size_m, max(min_size_m, width_m)))
        center = (min_x + max_x) / 2.0
        min_x, max_x = center - target / 2.0, center + target / 2.0
    if depth_m is not None:
        if depth_m <= 0:
            raise ValueError("depth_m must be positive")
        target = float(min(max_size_m, max(min_size_m, depth_m)))
        center = (min_y + max_y) / 2.0
        min_y, max_y = center - target / 2.0, center + target / 2.0

    if max_x - min_x > max_size_m:
        center = (min_x + max_x) / 2.0
        min_x, max_x = center - max_size_m / 2.0, center + max_size_m / 2.0
    if max_y - min_y > max_size_m:
        center = (min_y + max_y) / 2.0
        min_y, max_y = center - max_size_m / 2.0, center + max_size_m / 2.0

    polygon = box(min_x, min_y, max_x, max_y)
    return RoomEstimate(
        polygon=polygon,
        frame=frame,
        width=float(max_x - min_x),
        depth=float(max_y - min_y),
        bounds=(float(min_x), float(min_y), float(max_x), float(max_y)),
        num_points=len(points),
    )
