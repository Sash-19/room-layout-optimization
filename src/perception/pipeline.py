from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from ..schemas import DepthMap, Detection, StructureMasks
from ..utils import load_image
from .depth import DepthEstimator
from .detector import Detector
from .structure import FloorWallSegmenter


@dataclass(slots=True)
class PerceptionResult:
    detections: list[Detection]
    image_size: tuple[int, int]
    depth: DepthMap | None = None
    structures: StructureMasks | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def class_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(item.cls_name for item in self.detections).items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "num_detections": len(self.detections),
            "class_counts": self.class_counts(),
            "detections": [item.to_dict() for item in self.detections],
            "depth": self.depth.to_dict() if self.depth is not None else None,
            "structures": self.structures.to_dict() if self.structures is not None else None,
            "timings": {key: round(value, 4) for key, value in self.timings.items()},
        }


class PerceptionPipeline:
    def __init__(
        self,
        detector: Detector,
        depth_estimator: DepthEstimator | None = None,
        segmenter: FloorWallSegmenter | None = None,
    ) -> None:
        self.detector = detector
        self.depth_estimator = depth_estimator
        self.segmenter = segmenter

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        include_depth: bool = True,
        include_structure: bool = True,
    ) -> PerceptionPipeline:
        detector = Detector.from_config(config.get("detector", {}))
        depth_config = config.get("depth", {})
        structure_config = config.get("structure", {})
        depth_estimator = None
        segmenter = None
        if include_depth and depth_config.get("enabled", True):
            depth_estimator = DepthEstimator.from_config(depth_config)
        if include_structure and structure_config.get("enabled", True):
            segmenter = FloorWallSegmenter.from_config(structure_config)
        return cls(detector, depth_estimator, segmenter)

    def run(
        self,
        image: str | Path | Image.Image,
        want_depth: bool = True,
        want_structure: bool = True,
    ) -> PerceptionResult:
        if not isinstance(image, Image.Image):
            image = load_image(image)
        image_size = (image.size[0], image.size[1])
        timings: dict[str, float] = {}

        start = time.perf_counter()
        detections = self.detector.detect(image)
        timings["detection_s"] = time.perf_counter() - start

        depth = None
        if want_depth and self.depth_estimator is not None:
            start = time.perf_counter()
            depth = self.depth_estimator.predict(image)
            timings["depth_s"] = time.perf_counter() - start

        structures = None
        if want_structure and self.segmenter is not None:
            start = time.perf_counter()
            structures = self.segmenter.predict(image)
            timings["structure_s"] = time.perf_counter() - start

        return PerceptionResult(
            detections=detections,
            image_size=image_size,
            depth=depth,
            structures=structures,
            timings=timings,
        )
