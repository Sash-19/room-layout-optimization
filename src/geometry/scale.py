from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..floorplan.priors import get_prior, opening_prior
from ..schemas import Detection
from .camera import CameraIntrinsics
from .floor_plane import FloorPlane
from .lift import bottom_edge_floor_points
from .room_boundary import FloorFrame, estimate_room_boundary


@dataclass(slots=True)
class ScaleEstimate:
    scale: float
    mode: str
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": round(float(self.scale), 6),
            "mode": self.mode,
            "details": self.details,
            "warnings": list(self.warnings),
        }


def _valid_scale(scale: float, warnings: list[str]) -> float | None:
    if not np.isfinite(scale) or scale <= 0:
        warnings.append("computed scale was invalid; falling back to 1.0")
        return None
    if scale < 1e-3 or scale > 1e3:
        warnings.append(f"computed scale {scale:.4g} is extreme; check the inputs")
    return float(scale)


def scale_from_extents(
    observed_width: float,
    observed_depth: float,
    target_width: float | None = None,
    target_depth: float | None = None,
) -> float:
    ratios = []
    if target_width is not None:
        if observed_width <= 0:
            raise ValueError("observed_width must be positive")
        ratios.append(float(target_width) / float(observed_width))
    if target_depth is not None:
        if observed_depth <= 0:
            raise ValueError("observed_depth must be positive")
        ratios.append(float(target_depth) / float(observed_depth))
    if not ratios:
        raise ValueError("provide target_width and/or target_depth")
    return float(np.mean(ratios))


def scale_from_reference(measured_units: float, real_width_m: float) -> float:
    if measured_units <= 0 or real_width_m <= 0:
        raise ValueError("reference measurements must be positive")
    return float(real_width_m) / float(measured_units)


def _reference_detection(
    detections: Iterable[Detection],
    reference_class: str,
) -> Detection | None:
    candidates = [item for item in detections if item.cls_name == reference_class]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.conf)


def _reference_width(reference_class: str, configured: float | None) -> float | None:
    if configured is not None:
        return float(configured)
    opening = opening_prior(reference_class)
    if opening is not None:
        return float(opening.width)
    prior = get_prior(reference_class)
    if prior is not None:
        return float(prior.width)
    return None


def estimate_scale(
    camera: CameraIntrinsics,
    plane: FloorPlane,
    frame: FloorFrame,
    floor_points: np.ndarray,
    detections: Iterable[Detection],
    depth_metric: bool,
    config: dict | None = None,
) -> ScaleEstimate:
    config = config or {}
    detections = list(detections)
    requested = str(config.get("mode", "auto")).lower()
    warnings: list[str] = []
    mode = requested

    if mode == "auto":
        if depth_metric:
            mode = "metric"
        elif config.get("room_width_m") or config.get("room_depth_m"):
            mode = "user"
        else:
            reference_class = str(config.get("reference_class", "door"))
            mode = "reference" if _reference_detection(detections, reference_class) else "unscaled"

    if mode == "metric":
        if not depth_metric:
            warnings.append("metric scale requested but the depth map is relative")
        return ScaleEstimate(1.0, "metric", {"requested_mode": requested}, warnings)

    if mode == "reference":
        reference_class = str(config.get("reference_class", "door"))
        detection = _reference_detection(detections, reference_class)
        if detection is None:
            warnings.append(f"no {reference_class!r} detection found for reference scaling")
            return ScaleEstimate(1.0, "unscaled", {"requested_mode": requested}, warnings)
        target = _reference_width(reference_class, config.get("reference_width_m"))
        if target is None:
            warnings.append(f"no reference width prior for class {reference_class!r}")
            return ScaleEstimate(1.0, "unscaled", {"requested_mode": requested}, warnings)
        points = bottom_edge_floor_points(detection, camera, plane)
        if len(points) < 2:
            warnings.append("reference object did not project onto the floor")
            return ScaleEstimate(1.0, "unscaled", {"requested_mode": requested}, warnings)
        measured = float(np.linalg.norm(points[-1] - points[0]))
        if measured <= 1e-9:
            warnings.append("reference object had zero measured width")
            return ScaleEstimate(1.0, "unscaled", {"requested_mode": requested}, warnings)
        scale = scale_from_reference(measured, target)
        valid = _valid_scale(scale, warnings)
        if valid is None:
            return ScaleEstimate(1.0, "unscaled", {"requested_mode": requested}, warnings)
        details = {
            "requested_mode": requested,
            "reference_class": reference_class,
            "reference_width_m": round(target, 4),
            "measured_width_units": round(measured, 4),
            "confidence": round(detection.conf, 4),
        }
        return ScaleEstimate(valid, "reference", details, warnings)

    if mode == "user":
        target_width = config.get("room_width_m")
        target_depth = config.get("room_depth_m")
        if target_width is None and target_depth is None:
            warnings.append("user scaling requested but no room_width_m/room_depth_m provided")
            return ScaleEstimate(1.0, "unscaled", {"requested_mode": requested}, warnings)
        try:
            estimate = estimate_room_boundary(
                floor_points,
                frame,
                percentile=float(config.get("boundary_percentile", 2.0)),
                min_size_m=float(config.get("room_min_size_m", 1.5)),
                max_size_m=float(config.get("room_max_size_m", 30.0)),
            )
        except ValueError as error:
            warnings.append(f"room boundary estimation failed during scaling: {error}")
            return ScaleEstimate(1.0, "unscaled", {"requested_mode": requested}, warnings)
        scale = scale_from_extents(
            estimate.width,
            estimate.depth,
            target_width=float(target_width) if target_width is not None else None,
            target_depth=float(target_depth) if target_depth is not None else None,
        )
        valid = _valid_scale(scale, warnings)
        if valid is None:
            return ScaleEstimate(1.0, "unscaled", {"requested_mode": requested}, warnings)
        details = {
            "requested_mode": requested,
            "observed_width_units": round(estimate.width, 4),
            "observed_depth_units": round(estimate.depth, 4),
            "room_width_m": target_width,
            "room_depth_m": target_depth,
        }
        return ScaleEstimate(valid, "user", details, warnings)

    warnings.append("depth is relative and no scale reference was available; floor plan is unscaled")
    return ScaleEstimate(1.0, "unscaled", {"requested_mode": requested}, warnings)
