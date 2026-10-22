from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import Polygon

from ..floorplan.representation import Room

EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


@dataclass(slots=True)
class OccupancyGrid:
    room: Room
    resolution: float = 0.05
    x0: float = 0.0
    y0: float = 0.0
    width: int = 0
    height: int = 0
    inside: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    occupied: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    free: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    distance: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    nearest_free: np.ndarray = field(default_factory=lambda: np.zeros((2, 0, 0), dtype=int))

    def __post_init__(self) -> None:
        self.resolution = float(self.resolution)
        if self.resolution <= 0:
            raise ValueError("grid resolution must be positive")
        self.x0, self.y0, max_x, max_y = self.room.bounds
        self.width = max(1, int(math.ceil((max_x - self.x0) / self.resolution)))
        self.height = max(1, int(math.ceil((max_y - self.y0) / self.resolution)))
        xs = self.x0 + (np.arange(self.width) + 0.5) * self.resolution
        ys = self.y0 + (np.arange(self.height) + 0.5) * self.resolution
        grid_x, grid_y = np.meshgrid(xs, ys)
        self.inside = shapely.contains_xy(self.room.polygon.buffer(1e-9), grid_x, grid_y)
        occupied = np.zeros_like(self.inside)
        for item in self.room.furniture:
            occupied |= shapely.contains_xy(item.footprint.buffer(1e-9), grid_x, grid_y)
        self.occupied = occupied
        self.free = self.inside & ~occupied
        padded_free = np.pad(self.free, 1, constant_values=False)
        self.distance = ndimage.distance_transform_edt(padded_free, sampling=self.resolution)[1:-1, 1:-1]
        _, indices = ndimage.distance_transform_edt(~padded_free, return_indices=True)
        self.nearest_free = indices[:, 1:-1, 1:-1] - 1

    @classmethod
    def from_room(cls, room: Room, resolution: float = 0.05) -> OccupancyGrid:
        return cls(room=room, resolution=resolution)

    @property
    def cell_area(self) -> float:
        return self.resolution * self.resolution

    @property
    def free_area(self) -> float:
        return float(self.free.sum()) * self.cell_area

    @property
    def occupied_area(self) -> float:
        return float(self.occupied.sum()) * self.cell_area

    def walkable(self, min_width_m: float) -> np.ndarray:
        return self.free & (self.distance >= float(min_width_m) / 2.0)

    def walkable_area(self, min_width_m: float) -> float:
        return float(self.walkable(min_width_m).sum()) * self.cell_area

    def index_of(self, x: float, y: float) -> tuple[int, int]:
        col = int(round((float(x) - self.x0) / self.resolution - 0.5))
        row = int(round((float(y) - self.y0) / self.resolution - 0.5))
        return min(max(row, 0), self.height - 1), min(max(col, 0), self.width - 1)

    def world_of(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.x0 + (float(col) + 0.5) * self.resolution,
            self.y0 + (float(row) + 0.5) * self.resolution,
        )

    def nearest_free_index(self, x: float, y: float) -> tuple[int, int] | None:
        row, col = self.index_of(x, y)
        if self.free[row, col]:
            return row, col
        nearest_row, nearest_col = self.nearest_free[:, row, col]
        if not self.free[nearest_row, nearest_col]:
            return None
        return int(nearest_row), int(nearest_col)

    def rasterize(self, polygon: Polygon) -> np.ndarray:
        xs = self.x0 + (np.arange(self.width) + 0.5) * self.resolution
        ys = self.y0 + (np.arange(self.height) + 0.5) * self.resolution
        grid_x, grid_y = np.meshgrid(xs, ys)
        return shapely.contains_xy(polygon.buffer(1e-9), grid_x, grid_y)

    def shortest_path_length(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        mask: np.ndarray | None = None,
    ) -> float | None:
        mask = self.free if mask is None else (mask & self.free)
        start_index = self.nearest_free_index(*start)
        goal_index = self.nearest_free_index(*goal)
        if start_index is None or goal_index is None:
            return None
        if not mask[start_index] or not mask[goal_index]:
            return None
        if start_index == goal_index:
            return 0.0

        rows, cols = mask.shape
        distances = np.full((rows, cols), np.inf, dtype=float)
        distances[start_index] = 0.0
        queue = [(0.0, start_index[0], start_index[1])]
        neighbors = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
        while queue:
            current, row, col = heapq.heappop(queue)
            if current > distances[row, col]:
                continue
            if (row, col) == goal_index:
                return float(current * self.resolution)
            for delta_row, delta_col, weight in neighbors:
                next_row = row + delta_row
                next_col = col + delta_col
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                if not mask[next_row, next_col]:
                    continue
                candidate = current + weight
                if candidate < distances[next_row, next_col]:
                    distances[next_row, next_col] = candidate
                    heapq.heappush(queue, (candidate, next_row, next_col))
        return None

    def components(self, mask: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        mask = self.free if mask is None else mask
        return ndimage.label(mask, structure=EIGHT_CONNECTED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution": round(self.resolution, 4),
            "width": self.width,
            "height": self.height,
            "free_area": round(self.free_area, 4),
            "occupied_area": round(self.occupied_area, 4),
        }
