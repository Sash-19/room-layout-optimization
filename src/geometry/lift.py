from __future__ import annotations

from typing import Iterable

import numpy as np

from ..floorplan.priors import clamp_width, furniture_from_detection, get_prior, opening_from_dimensions, opening_prior
from ..floorplan.representation import Furniture, Opening
from ..schemas import Detection
from .camera import CameraIntrinsics
from .floor_plane import FloorPlane
from .room_boundary import FloorFrame, RoomEstimate

MOVABLE_CLASSES = ("bed", "desk", "chair", "couch", "table", "cabinet")
FIXED_CLASSES = ("door", "window")


def bottom_edge_pixels(
    detection: Detection,
    camera: CameraIntrinsics,
    num: int = 9,
) -> np.ndarray:
    x1, _, x2, y2 = detection.bbox
    x1 = float(np.clip(x1, 0.0, camera.width - 1))
    x2 = float(np.clip(x2, 1.0, camera.width - 1))
    if x2 <= x1:
        x2 = min(camera.width - 1.0, x1 + 1.0)
    v = float(np.clip(y2 - 0.5, 0.0, camera.height - 1))
    us = np.linspace(x1, x2, num=max(2, int(num)))
    vs = np.full_like(us, v)
    return np.stack([us, vs], axis=1)


def bottom_edge_floor_points(
    detection: Detection,
    camera: CameraIntrinsics,
    plane: FloorPlane,
    num: int = 9,
) -> np.ndarray:
    rays = camera.pixel_rays(bottom_edge_pixels(detection, camera, num=num))
    points, valid = plane.intersect_rays(rays)
    return points[valid]


def _snap_angle(theta: float, step_deg: float, enabled: bool) -> float:
    if not enabled:
        return float(theta) % (2.0 * np.pi)
    step = np.deg2rad(float(step_deg))
    if step <= 0:
        return float(theta) % (2.0 * np.pi)
    snapped = np.round(theta / step) * step
    return float(snapped % (2.0 * np.pi))


def lift_detections(
    detections: Iterable[Detection],
    camera: CameraIntrinsics,
    plane: FloorPlane,
    frame: FloorFrame,
    config: dict | None = None,
) -> list[Furniture]:
    config = config or {}
    snap = bool(config.get("snap_orientation", True))
    step_deg = float(config.get("rotation_step_deg", 90.0))
    furniture: list[Furniture] = []

    for detection in detections:
        if detection.cls_name not in MOVABLE_CLASSES:
            continue
        points = bottom_edge_floor_points(detection, camera, plane)
        if len(points) < 2:
            continue
        left = points[0]
        right = points[-1]
        if np.linalg.norm(right - left) < 1e-6:
            continue
        floor_points = frame.to_2d(np.stack([left, right], axis=0))
        midpoint = floor_points.mean(axis=0)
        width_vector = floor_points[1] - floor_points[0]
        theta = _snap_angle(float(np.arctan2(width_vector[1], width_vector[0])), step_deg, snap)
        measured_width = float(np.linalg.norm(right - left))
        prior = get_prior(detection.cls_name)
        width = clamp_width(detection.cls_name, measured_width) if prior else measured_width
        depth = prior.depth if prior else width * 0.6

        away = np.asarray(midpoint, dtype=np.float64)
        norm = float(np.linalg.norm(away))
        if norm < 0.1:
            away = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
            if away[1] < 0:
                away = -away
        else:
            away = away / norm
        center = midpoint + 0.5 * depth * away

        item = furniture_from_detection(
            detection.cls_name,
            width=width,
            x=float(center[0]),
            y=float(center[1]),
            theta=theta,
            confidence=detection.conf,
        )
        item.d = depth
        furniture.append(item)
    return furniture


def lift_openings(
    detections: Iterable[Detection],
    camera: CameraIntrinsics,
    plane: FloorPlane,
    frame: FloorFrame,
    room: RoomEstimate,
) -> list[Opening]:
    min_x, min_y, max_x, max_y = room.bounds
    width_x = max_x - min_x
    depth_y = max_y - min_y
    openings: list[Opening] = []

    for detection in detections:
        if detection.cls_name not in FIXED_CLASSES:
            continue
        prior = opening_prior(detection.cls_name)
        if prior is None:
            continue
        points = bottom_edge_floor_points(detection, camera, plane)
        if len(points) < 2:
            continue
        coords = frame.to_2d(points)
        midpoint = coords.mean(axis=0)
        measured = float(np.linalg.norm(coords[-1] - coords[0]))
        width = measured if measured > 0.2 else prior.width
        width = float(np.clip(width, 0.5 * prior.width, 2.0 * prior.width))

        distances = (
            abs(midpoint[1] - min_y),
            abs(midpoint[0] - max_x),
            abs(midpoint[1] - max_y),
            abs(midpoint[0] - min_x),
        )
        wall_index = int(np.argmin(distances))
        if wall_index in (0, 2):
            wall_length = width_x
            start = min_x
            coordinate = midpoint[0]
        else:
            wall_length = depth_y
            start = min_y
            coordinate = midpoint[1]
        if wall_length <= 0:
            continue
        offset = float(np.clip(coordinate - start, 0.0, max(0.0, wall_length - width)))
        openings.append(opening_from_dimensions(detection.cls_name, wall_index, offset, width))
    return openings
