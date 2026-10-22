from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shapely import affinity
from shapely.geometry import Polygon, box

WALL_NAMES = ("min_y", "max_x", "max_y", "min_x")


@dataclass(slots=True)
class Opening:
    kind: str
    wall_index: int
    offset: float
    width: float
    swing_radius: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"door", "window"}:
            raise ValueError(f"unsupported opening kind: {self.kind!r}")
        if self.wall_index not in range(4):
            raise ValueError("wall_index must be 0..3")
        if self.width <= 0:
            raise ValueError("opening width must be positive")
        if self.swing_radius < 0:
            raise ValueError("swing_radius must be non-negative")
        self.offset = float(self.offset)
        self.width = float(self.width)
        self.swing_radius = float(self.swing_radius)

    @property
    def wall_name(self) -> str:
        return WALL_NAMES[self.wall_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "wall_index": self.wall_index,
            "wall_name": self.wall_name,
            "offset": round(self.offset, 4),
            "width": round(self.width, 4),
            "swing_radius": round(self.swing_radius, 4),
        }


@dataclass(slots=True)
class Furniture:
    cls: str
    w: float
    d: float
    x: float
    y: float
    theta: float = 0.0
    front_clearance: float = 0.0
    wall_affinity: bool = False
    confidence: float = 1.0
    locked: bool = False

    def __post_init__(self) -> None:
        if self.w <= 0 or self.d <= 0:
            raise ValueError("furniture dimensions must be positive")
        self.w = float(self.w)
        self.d = float(self.d)
        self.x = float(self.x)
        self.y = float(self.y)
        self.theta = float(self.theta)
        self.front_clearance = float(self.front_clearance)
        self.confidence = float(self.confidence)

    @property
    def footprint(self) -> Polygon:
        rectangle = box(-self.w / 2.0, -self.d / 2.0, self.w / 2.0, self.d / 2.0)
        rotated = affinity.rotate(rectangle, self.theta, origin=(0.0, 0.0), use_radians=True)
        return affinity.translate(rotated, xoff=self.x, yoff=self.y)

    @property
    def area(self) -> float:
        return self.w * self.d

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.cls,
            "w": round(self.w, 4),
            "d": round(self.d, 4),
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "theta_deg": round(self.theta * 180.0 / 3.141592653589793, 2),
            "front_clearance": round(self.front_clearance, 4),
            "wall_affinity": self.wall_affinity,
            "confidence": round(self.confidence, 4),
            "locked": self.locked,
            "footprint": [[round(x, 4), round(y, 4)] for x, y in self.footprint.exterior.coords],
        }


@dataclass(slots=True)
class Room:
    polygon: Polygon
    openings: list[Opening] = field(default_factory=list)
    furniture: list[Furniture] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.polygon.is_valid:
            self.polygon = self.polygon.buffer(0)
        if self.polygon.is_empty or self.polygon.area <= 0:
            raise ValueError("room polygon must have positive area")

    @property
    def area(self) -> float:
        return float(self.polygon.area)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return tuple(float(value) for value in self.polygon.bounds)

    @property
    def width(self) -> float:
        min_x, _, max_x, _ = self.bounds
        return max_x - min_x

    @property
    def depth(self) -> float:
        _, min_y, _, max_y = self.bounds
        return max_y - min_y

    def to_dict(self) -> dict[str, Any]:
        return {
            "polygon": [[round(x, 4), round(y, 4)] for x, y in self.polygon.exterior.coords],
            "area": round(self.area, 4),
            "width": round(self.width, 4),
            "depth": round(self.depth, 4),
            "bounds": [round(value, 4) for value in self.bounds],
            "openings": [opening.to_dict() for opening in self.openings],
            "furniture": [item.to_dict() for item in self.furniture],
        }


@dataclass(slots=True)
class FloorPlan:
    room: Room
    scale: float = 1.0
    scale_mode: str = "unscaled"
    camera_height: float = 0.0
    camera: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "room": self.room.to_dict(),
            "scale": round(float(self.scale), 6),
            "scale_mode": self.scale_mode,
            "camera_height": round(float(self.camera_height), 4),
            "camera": self.camera,
            "metadata": self.metadata,
        }
