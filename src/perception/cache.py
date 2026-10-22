from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..schemas import DepthMap, Detection, StructureMasks
from ..utils import load_image, load_json
from .pipeline import PerceptionResult


def load_cached_perception(path: str | Path) -> tuple[Image.Image, PerceptionResult]:
    path = Path(path)
    image = load_image(path / "input.jpg")
    payload = load_json(path / "result.json")
    detections = [Detection.from_dict(item) for item in payload.get("detections", [])]

    depth = None
    depth_path = path / "depth.npy"
    if depth_path.exists():
        depth_info = payload.get("depth") or {}
        depth = DepthMap(
            np.load(depth_path),
            metric=bool(depth_info.get("metric", False)),
            source=str(depth_info.get("source", "cached")),
        )

    structures = None
    floor_path = path / "floor_mask.png"
    wall_path = path / "wall_mask.png"
    if floor_path.exists() and wall_path.exists():
        floor = np.asarray(Image.open(floor_path).convert("L")) > 127
        wall = np.asarray(Image.open(wall_path).convert("L")) > 127
        structures = StructureMasks(floor, wall, source="cached")

    return image, PerceptionResult(
        detections=detections,
        image_size=image.size,
        depth=depth,
        structures=structures,
    )
