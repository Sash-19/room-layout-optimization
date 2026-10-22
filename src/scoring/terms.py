from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..floorplan.priors import get_prior
from ..floorplan.representation import Room
from .grid import OccupancyGrid
from .zones import front_zone, inward_point, opening_zone, zone_free_fraction


@dataclass(slots=True)
class TermResult:
    score: float
    details: dict[str, Any] = field(default_factory=dict)


def _settings(config: dict[str, Any] | None) -> dict[str, Any]:
    return (config or {}).get("terms", {})


def _mean(values: list[float], default: float = 1.0) -> float:
    return float(np.mean(values)) if values else float(default)


def term_overlap(room: Room, grid: OccupancyGrid, config: dict[str, Any] | None = None) -> TermResult:
    total_area = sum(item.area for item in room.furniture)
    pairs = []
    overlap_area = 0.0
    for first in range(len(room.furniture)):
        for second in range(first + 1, len(room.furniture)):
            area = float(room.furniture[first].footprint.intersection(room.furniture[second].footprint).area)
            if area > 0:
                overlap_area += area
                pairs.append(
                    {
                        "items": [first, second],
                        "area": round(area, 4),
                    }
                )
    score = 1.0 if total_area <= 0 else max(0.0, 1.0 - overlap_area / total_area)
    return TermResult(score, {"overlap_area": round(overlap_area, 4), "pairs": pairs})


def term_walk(room: Room, grid: OccupancyGrid, config: dict[str, Any] | None = None) -> TermResult:
    settings = _settings(config)
    width = float(settings.get("walkway_width_m", 0.6))
    free_area = grid.free_area
    walkable_area = grid.walkable_area(width)
    score = 0.0 if free_area <= 0 else max(0.0, min(1.0, walkable_area / free_area))
    return TermResult(
        score,
        {
            "walkway_width_m": round(width, 4),
            "free_area": round(free_area, 4),
            "walkable_area": round(walkable_area, 4),
        },
    )


def term_util(room: Room, grid: OccupancyGrid, config: dict[str, Any] | None = None) -> TermResult:
    labels, count = grid.components(grid.free)
    free_area = grid.free_area
    if count <= 0 or free_area <= 0:
        return TermResult(0.0, {"components": 0, "largest_component_area": 0.0})
    sizes = np.bincount(labels.ravel())
    largest = float(sizes[1:].max()) * grid.cell_area if len(sizes) > 1 else 0.0
    score = max(0.0, min(1.0, largest / free_area))
    return TermResult(
        score,
        {
            "components": int(count),
            "largest_component_area": round(largest, 4),
            "free_area": round(free_area, 4),
        },
    )


def term_path(room: Room, grid: OccupancyGrid, config: dict[str, Any] | None = None) -> TermResult:
    settings = _settings(config)
    length_factor = float(settings.get("path_length_factor", 1.5))
    doors = [opening for opening in room.openings if opening.kind == "door"]
    targets: list[tuple[str, tuple[float, float]]] = [(item.cls, (item.x, item.y)) for item in room.furniture]
    targets.extend(
        ("window", inward_point(room, opening, 0.3))
        for opening in room.openings
        if opening.kind == "window"
    )
    if not doors or not targets:
        return TermResult(1.0, {"paths": [], "reachable_fraction": 1.0})

    max_length = length_factor * math.hypot(room.width, room.depth) + 1e-9
    paths = []
    scores = []
    for door_index, door in enumerate(doors):
        start = inward_point(room, door, 0.3)
        for label, goal in targets:
            length = grid.shortest_path_length(start, goal, mask=grid.free)
            if length is None:
                scores.append(0.0)
                paths.append({"door": door_index, "target": label, "length": None})
            else:
                quality = max(0.0, min(1.0, 1.0 - length / max_length))
                scores.append(quality)
                paths.append({"door": door_index, "target": label, "length": round(length, 3)})
    return TermResult(
        _mean(scores, default=1.0),
        {
            "max_length": round(max_length, 3),
            "reachable_fraction": round(_mean([1.0 if path["length"] is not None else 0.0 for path in paths]), 4),
            "paths": paths,
        },
    )


def term_wall(room: Room, grid: OccupancyGrid, config: dict[str, Any] | None = None) -> TermResult:
    settings = _settings(config)
    ideal = float(settings.get("wall_gap_ideal_m", 0.05))
    tolerance = max(1e-6, float(settings.get("wall_gap_tolerance_m", 0.5)))
    scores = []
    items = []
    for index, item in enumerate(room.furniture):
        distance = float(item.footprint.distance(room.polygon.boundary))
        if item.wall_affinity:
            score = 1.0 if distance <= ideal else max(0.0, 1.0 - (distance - ideal) / tolerance)
        else:
            score = 1.0 if distance >= ideal else max(0.0, distance / ideal)
        scores.append(score)
        items.append(
            {
                "index": index,
                "class": item.cls,
                "distance": round(distance, 4),
                "wall_affinity": item.wall_affinity,
                "score": round(score, 4),
            }
        )
    return TermResult(_mean(scores), {"items": items})


def term_door(room: Room, grid: OccupancyGrid, config: dict[str, Any] | None = None) -> TermResult:
    settings = _settings(config)
    depth = float(settings.get("door_clear_depth_m", 0.9))
    margin = float(settings.get("door_clear_margin_m", 0.1))
    doors = [opening for opening in room.openings if opening.kind == "door"]
    if not doors:
        return TermResult(1.0, {"doors": []})
    zones = []
    scores = []
    for door in doors:
        zone = opening_zone(room, door, max(depth, door.swing_radius), margin)
        fraction = zone_free_fraction(zone, room.furniture)
        scores.append(fraction)
        zones.append(
            {
                "wall_index": door.wall_index,
                "offset": round(door.offset, 4),
                "free_fraction": round(fraction, 4),
            }
        )
    return TermResult(_mean(scores), {"doors": zones})


def term_window(room: Room, grid: OccupancyGrid, config: dict[str, Any] | None = None) -> TermResult:
    settings = _settings(config)
    depth = float(settings.get("window_clear_depth_m", 0.3))
    tall_height = float(settings.get("tall_item_height_m", 1.0))
    windows = [opening for opening in room.openings if opening.kind == "window"]
    if not windows:
        return TermResult(1.0, {"windows": []})
    tall = [item for item in room.furniture if get_prior(item.cls) is not None and get_prior(item.cls).height >= tall_height]
    scores = []
    zones = []
    for window in windows:
        zone = opening_zone(room, window, depth)
        fraction = zone_free_fraction(zone, tall)
        scores.append(fraction)
        zones.append(
            {
                "wall_index": window.wall_index,
                "offset": round(window.offset, 4),
                "free_fraction": round(fraction, 4),
            }
        )
    return TermResult(_mean(scores), {"windows": zones})


def term_clear(room: Room, grid: OccupancyGrid, config: dict[str, Any] | None = None) -> TermResult:
    scores = []
    items = []
    for index, item in enumerate(room.furniture):
        zone = front_zone(item)
        if zone is None:
            continue
        fraction = zone_free_fraction(zone, [other for other in room.furniture if other is not item])
        scores.append(fraction)
        items.append({"index": index, "class": item.cls, "free_fraction": round(fraction, 4)})
    return TermResult(_mean(scores), {"items": items})


def term_pair(room: Room, grid: OccupancyGrid, config: dict[str, Any] | None = None) -> TermResult:
    settings = _settings(config)
    pairs = settings.get("pairs", [])
    scores = []
    details = []
    for specification in pairs:
        classes = tuple(specification.get("classes", ()))
        target = float(specification.get("target_m", 0.8))
        sigma = max(1e-6, float(specification.get("sigma_m", 0.3)))
        for first in range(len(room.furniture)):
            for second in range(first + 1, len(room.furniture)):
                a = room.furniture[first]
                b = room.furniture[second]
                if {a.cls, b.cls} != set(classes):
                    continue
                distance = float(a.footprint.distance(b.footprint))
                score = math.exp(-((distance - target) ** 2) / (2.0 * sigma * sigma))
                scores.append(score)
                details.append(
                    {
                        "items": [first, second],
                        "classes": [a.cls, b.cls],
                        "distance": round(distance, 4),
                        "score": round(score, 4),
                    }
                )
    return TermResult(_mean(scores), {"pairs": details})


TERMS: dict[str, Callable[[Room, OccupancyGrid, dict[str, Any] | None], TermResult]] = {
    "overlap": term_overlap,
    "walk": term_walk,
    "path": term_path,
    "wall": term_wall,
    "door": term_door,
    "window": term_window,
    "util": term_util,
    "clear": term_clear,
    "pair": term_pair,
}
