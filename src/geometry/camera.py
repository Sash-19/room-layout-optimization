from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

EXIF_FOCAL_LENGTH = 37386
EXIF_FOCAL_LENGTH_35MM = 41486


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image size must be positive")
        self.fx = float(self.fx)
        self.fy = float(self.fy)
        self.cx = float(self.cx)
        self.cy = float(self.cy)
        self.width = int(self.width)
        self.height = int(self.height)

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        fov_deg: float = 65.0,
        source: str = "fov",
    ) -> CameraIntrinsics:
        if not 0.0 < fov_deg < 180.0:
            raise ValueError("fov_deg must be between 0 and 180")
        fx = width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
        return cls(fx, fx, width / 2.0, height / 2.0, width, height, source)

    @classmethod
    def from_image(
        cls,
        image: Image.Image,
        fov_deg: float = 65.0,
        sensor_width_mm: float = 36.0,
    ) -> CameraIntrinsics:
        width, height = image.size
        fx = None
        source = "fov"
        try:
            exif = image.getexif()
        except Exception:
            exif = None
        if exif is not None:
            focal_35 = exif.get(EXIF_FOCAL_LENGTH_35MM)
            focal_mm = exif.get(EXIF_FOCAL_LENGTH)
            if focal_35 is not None:
                try:
                    fx = float(focal_35) / sensor_width_mm * width
                    source = "exif-35mm"
                except (TypeError, ValueError, ZeroDivisionError):
                    fx = None
            if fx is None and focal_mm is not None:
                try:
                    fx = float(focal_mm) / sensor_width_mm * width
                    source = "exif-focal"
                except (TypeError, ValueError, ZeroDivisionError):
                    fx = None
        if fx is None or not np.isfinite(fx) or fx <= 0:
            intrinsic = cls.from_fov(width, height, fov_deg)
            return cls(intrinsic.fx, intrinsic.fy, intrinsic.cx, intrinsic.cy, width, height, "fov")
        return cls(fx, fx, width / 2.0, height / 2.0, width, height, source)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        fov_deg: float = 65.0,
        sensor_width_mm: float = 36.0,
    ) -> CameraIntrinsics:
        with Image.open(path) as image:
            return cls.from_image(image.convert("RGB"), fov_deg=fov_deg, sensor_width_mm=sensor_width_mm)

    def pixel_rays(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        u = pixels[:, 0]
        v = pixels[:, 1]
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        return np.stack([x, y, np.ones_like(x)], axis=1)

    def backproject(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float64)
        if depth.shape != (self.height, self.width):
            raise ValueError(f"depth shape {depth.shape} does not match camera image size")
        pixels = np.asarray(pixels, dtype=np.int64).reshape(-1, 2)
        us = np.clip(pixels[:, 0], 0, self.width - 1)
        vs = np.clip(pixels[:, 1], 0, self.height - 1)
        z = depth[vs, us]
        rays = self.pixel_rays(np.stack([us, vs], axis=1))
        return rays * z[:, None]

    def project(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        z = points[:, 2]
        safe = np.where(np.abs(z) < 1e-12, np.nan, z)
        u = self.fx * points[:, 0] / safe + self.cx
        v = self.fy * points[:, 1] / safe + self.cy
        return np.stack([u, v], axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx": round(self.fx, 4),
            "fy": round(self.fy, 4),
            "cx": round(self.cx, 4),
            "cy": round(self.cy, 4),
            "width": self.width,
            "height": self.height,
            "source": self.source,
        }
