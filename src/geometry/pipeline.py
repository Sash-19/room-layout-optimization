from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..floorplan.representation import FloorPlan, Room
from ..perception.pipeline import PerceptionResult
from ..utils import load_image
from .camera import CameraIntrinsics
from .floor_plane import calibrate_inverse_depth, floor_points_from_mask, fit_floor_plane
from .lift import lift_detections, lift_openings
from .room_boundary import FloorFrame, estimate_room_boundary
from .scale import estimate_scale


class GeometryPipeline:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> GeometryPipeline:
        return cls(config)

    def _camera(self, image: Image.Image) -> CameraIntrinsics:
        config = self.config.get("camera", {})
        width, height = image.size
        fx = config.get("fx")
        fy = config.get("fy")
        if fx is not None and fy is not None:
            return CameraIntrinsics(
                fx=float(fx),
                fy=float(fy),
                cx=float(config.get("cx", width / 2.0)),
                cy=float(config.get("cy", height / 2.0)),
                width=width,
                height=height,
                source="config",
            )
        return CameraIntrinsics.from_image(
            image,
            fov_deg=float(config.get("fov_deg", 65.0)),
            sensor_width_mm=float(config.get("sensor_width_mm", 36.0)),
        )

    def run(
        self,
        image: str | Path | Image.Image,
        perception: PerceptionResult,
        room_width_m: float | None = None,
        room_depth_m: float | None = None,
        scale_mode: str | None = None,
    ) -> FloorPlan:
        if not isinstance(image, Image.Image):
            image = load_image(image)
        if image.size != perception.image_size:
            raise ValueError(f"image size {image.size} does not match perception size {perception.image_size}")
        if perception.depth is None:
            raise ValueError("geometry requires a depth map; enable depth in the perception stage")

        depth = perception.depth
        if depth.depth.shape != (image.size[1], image.size[0]):
            raise ValueError("depth map shape does not match the image size")

        warnings: list[str] = []
        timings: dict[str, float] = {}
        camera = self._camera(image)
        floor_config = self.config.get("floor", {})
        scale_config = dict(self.config.get("scale", {}))
        lift_config = self.config.get("lift", {})
        room_config = self.config.get("room", {})

        floor_mask = None
        if perception.structures is not None:
            fraction = float(perception.structures.floor.mean())
            minimum = float(floor_config.get("min_floor_fraction", 0.02))
            if fraction >= minimum:
                floor_mask = perception.structures.floor
            else:
                warnings.append(f"floor mask coverage {fraction:.4f} below minimum {minimum:.4f}; fitting on depth")
        else:
            warnings.append("no floor mask available; fitting floor plane on all depth pixels")

        start = time.perf_counter()
        depth_array = depth.depth
        calibration = None
        fit_options = {
            "floor_mask": floor_mask,
            "stride": int(floor_config.get("sample_stride", 4)),
            "max_points": int(floor_config.get("max_points", 20000)),
            "threshold": float(floor_config.get("inlier_threshold", 0.03)),
            "iterations": int(floor_config.get("ransac_iterations", 300)),
            "seed": int(floor_config.get("seed", 0)),
        }
        try:
            plane = fit_floor_plane(depth_array, camera, **fit_options)
        except ValueError as error:
            if depth.metric:
                raise
            calibrated, calibration = calibrate_inverse_depth(
                depth_array,
                camera,
                floor_mask=floor_mask,
                stride=int(floor_config.get("sample_stride", 2)),
                max_points=int(floor_config.get("max_points", 20000)),
                seed=int(floor_config.get("seed", 0)),
            )
            try:
                plane = fit_floor_plane(calibrated, camera, **fit_options)
            except ValueError:
                if calibration.get("method") != "affine-inverse":
                    raise error
                calibrated, calibration = calibrate_inverse_depth(depth_array, camera)
                try:
                    plane = fit_floor_plane(calibrated, camera, **fit_options)
                except ValueError:
                    raise error
            depth_array = calibrated
            detail = "calibrated it against the floor plane" if calibration["method"] == "affine-inverse" else "inverted it"
            warnings.append(f"relative depth is inverse-depth (larger values closer); {detail} for floor-plane fitting")
        timings["floor_plane_s"] = time.perf_counter() - start
        frame = FloorFrame.from_plane(plane)

        start = time.perf_counter()
        boundary_mask = floor_mask
        if boundary_mask is None:
            rows = np.arange(camera.height)[:, None] >= int(0.55 * camera.height)
            boundary_mask = np.broadcast_to(rows, (camera.height, camera.width)).copy()
            warnings.append("room boundary uses the lower image region because no floor mask was available")
        floor_points = floor_points_from_mask(
            camera,
            plane,
            depth_array,
            floor_mask=boundary_mask,
            stride=int(floor_config.get("sample_stride", 4)),
            max_points=int(floor_config.get("max_points", 20000)),
        )
        if len(floor_points) < 3:
            raise ValueError("not enough floor points after ray-plane intersection")
        timings["floor_points_s"] = time.perf_counter() - start

        if room_width_m is not None:
            scale_config["room_width_m"] = room_width_m
        if room_depth_m is not None:
            scale_config["room_depth_m"] = room_depth_m
        if scale_mode is not None:
            scale_config["mode"] = scale_mode

        start = time.perf_counter()
        scale = estimate_scale(
            camera,
            plane,
            frame,
            floor_points,
            perception.detections,
            depth_metric=depth.metric,
            config=scale_config,
        )
        timings["scale_s"] = time.perf_counter() - start
        warnings.extend(scale.warnings)

        scaled_plane = plane.scale(scale.scale)
        scaled_frame = FloorFrame.from_plane(scaled_plane)
        scaled_points = floor_points * scale.scale

        start = time.perf_counter()
        room_estimate = estimate_room_boundary(
            scaled_points,
            scaled_frame,
            percentile=float(room_config.get("boundary_percentile", 2.0)),
            width_m=scale_config.get("room_width_m"),
            depth_m=scale_config.get("room_depth_m"),
            min_size_m=float(room_config.get("min_size_m", 1.5)),
            max_size_m=float(room_config.get("max_size_m", 30.0)),
        )
        timings["room_boundary_s"] = time.perf_counter() - start

        start = time.perf_counter()
        furniture = lift_detections(perception.detections, camera, scaled_plane, scaled_frame, lift_config)
        openings = lift_openings(perception.detections, camera, scaled_plane, scaled_frame, room_estimate)
        room = Room(polygon=room_estimate.polygon, openings=openings, furniture=furniture)
        outside = [item for item in furniture if not item.footprint.intersects(room_estimate.polygon)]
        if outside:
            warnings.append(
                f"{len(outside)} furniture footprint(s) fall outside the estimated room; "
                "a single photo may not show the whole room, so verify the room dimensions/scale"
            )
        timings["lift_s"] = time.perf_counter() - start

        movable = sum(1 for item in perception.detections if item.cls_name in {"bed", "desk", "chair", "couch", "table", "cabinet"})
        if movable > len(furniture):
            warnings.append(f"{movable - len(furniture)} movable detections could not be lifted onto the floor")

        metadata = {
            "warnings": warnings,
            "depth": depth.to_dict(),
            "plane": scaled_plane.to_dict(),
            "scale": scale.to_dict(),
            "room_estimate": room_estimate.to_dict(),
            "timings": {key: round(value, 4) for key, value in timings.items()},
        }
        if calibration is not None:
            metadata["depth_calibration"] = calibration
        return FloorPlan(
            room=room,
            scale=float(scale.scale),
            scale_mode=scale.mode,
            camera_height=float(scaled_plane.camera_height),
            camera=camera.to_dict(),
            metadata=metadata,
        )
