from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class Detection:
    cls_name: str
    conf: float
    bbox: BBox
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cls_name, str) or not self.cls_name:
            raise ValueError("cls_name must be a non-empty string")
        x1, y1, x2, y2 = (float(v) for v in self.bbox)
        if not bool(np.all(np.isfinite([x1, y1, x2, y2]))):
            raise ValueError("bbox coordinates must be finite")
        if x2 < x1 or y2 < y1:
            raise ValueError(f"invalid bbox: {self.bbox}")
        conf = float(self.conf)
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {conf}")
        self.bbox = (x1, y1, x2, y2)
        self.conf = conf
        if self.mask is not None:
            mask = np.asarray(self.mask)
            if mask.ndim != 2:
                raise ValueError("mask must be 2D")
            self.mask = mask.astype(bool, copy=False)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    def clip(self, image_width: int, image_height: int) -> Detection:
        x1, y1, x2, y2 = self.bbox
        bbox = (
            float(np.clip(x1, 0, image_width)),
            float(np.clip(y1, 0, image_height)),
            float(np.clip(x2, 0, image_width)),
            float(np.clip(y2, 0, image_height)),
        )
        return Detection(self.cls_name, self.conf, bbox, self.mask)

    def iou(self, other: Detection) -> float:
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other.bbox
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return 0.0 if union <= 0 else inter / union

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.cls_name,
            "confidence": round(self.conf, 4),
            "bbox_xyxy": [round(v, 2) for v in self.bbox],
            "has_mask": self.mask is not None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Detection:
        name = data.get("class", data.get("cls_name"))
        bbox = data["bbox_xyxy"]
        return cls(str(name), float(data["confidence"]), (bbox[0], bbox[1], bbox[2], bbox[3]))


@dataclass(slots=True)
class DepthMap:
    depth: np.ndarray
    metric: bool = False
    source: str = ""

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"depth must be 2D, got shape {depth.shape}")
        self.depth = depth

    @property
    def height(self) -> int:
        return int(self.depth.shape[0])

    @property
    def width(self) -> int:
        return int(self.depth.shape[1])

    def to_dict(self) -> dict[str, Any]:
        finite = self.depth[np.isfinite(self.depth)]
        stats = None
        if finite.size:
            stats = {
                "min": round(float(finite.min()), 4),
                "max": round(float(finite.max()), 4),
                "mean": round(float(finite.mean()), 4),
            }
        return {
            "metric": self.metric,
            "source": self.source,
            "height": self.height,
            "width": self.width,
            "stats": stats,
        }


@dataclass(slots=True)
class StructureMasks:
    floor: np.ndarray
    wall: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        floor = np.asarray(self.floor, dtype=bool)
        wall = np.asarray(self.wall, dtype=bool)
        if floor.ndim != 2 or wall.ndim != 2:
            raise ValueError("structure masks must be 2D")
        if floor.shape != wall.shape:
            raise ValueError("floor and wall masks must have the same shape")
        self.floor = floor
        self.wall = wall

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.floor.shape[0]), int(self.floor.shape[1]))

    def to_dict(self) -> dict[str, Any]:
        pixels = self.floor.size
        return {
            "source": self.source,
            "height": self.shape[0],
            "width": self.shape[1],
            "floor_fraction": round(float(self.floor.sum()) / pixels, 4) if pixels else 0.0,
            "wall_fraction": round(float(self.wall.sum()) / pixels, 4) if pixels else 0.0,
        }
