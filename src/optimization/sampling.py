from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from random import Random

from ..floorplan.representation import FloorPlan, Furniture, Room
from ..scoring.constraints import check_constraints


def optimizer_settings(config: dict | None) -> dict:
    return (config or {}).get("optimizer", config or {})


def scoring_settings(config: dict | None) -> dict:
    return (config or {}).get("scoring", config or {})


def rotation_options(step_deg: float) -> list[float]:
    step = max(1e-6, float(step_deg))
    count = max(1, int(round(360.0 / step)))
    return [math.radians(index * step) for index in range(count)]


def _half_extents(item: Furniture, theta: float) -> tuple[float, float]:
    cos = abs(math.cos(theta))
    sin = abs(math.sin(theta))
    return (
        (item.w * cos + item.d * sin) / 2.0,
        (item.w * sin + item.d * cos) / 2.0,
    )


def _wall_pose(item: Furniture, room: Room, wall: int, rng: Random) -> Furniture | None:
    min_x, min_y, max_x, max_y = room.bounds
    width = max_x - min_x
    depth = max_y - min_y
    if wall == 0:
        if width < item.w:
            return None
        return replace(
            item,
            x=rng.uniform(min_x + item.w / 2.0, max_x - item.w / 2.0),
            y=min_y + item.d / 2.0,
            theta=0.0,
        )
    if wall == 1:
        if depth < item.w:
            return None
        return replace(
            item,
            x=max_x - item.d / 2.0,
            y=rng.uniform(min_y + item.w / 2.0, max_y - item.w / 2.0),
            theta=math.pi / 2.0,
        )
    if wall == 2:
        if width < item.w:
            return None
        return replace(
            item,
            x=rng.uniform(min_x + item.w / 2.0, max_x - item.w / 2.0),
            y=max_y - item.d / 2.0,
            theta=math.pi,
        )
    if depth < item.w:
        return None
    return replace(
        item,
        x=min_x + item.d / 2.0,
        y=rng.uniform(min_y + item.w / 2.0, max_y - item.w / 2.0),
        theta=-math.pi / 2.0,
    )


def _random_pose(item: Furniture, room: Room, rng: Random, rotations: list[float]) -> Furniture | None:
    min_x, min_y, max_x, max_y = room.bounds
    theta = rng.choice(rotations)
    half_x, half_y = _half_extents(item, theta)
    if max_x - min_x < 2.0 * half_x or max_y - min_y < 2.0 * half_y:
        return None
    return replace(
        item,
        x=rng.uniform(min_x + half_x, max_x - half_x),
        y=rng.uniform(min_y + half_y, max_y - half_y),
        theta=theta,
    )


def _candidate_ok(room: Room, placed: list[Furniture], candidate: Furniture, config: dict | None) -> bool:
    partial = Room(room.polygon, list(room.openings), [*placed, candidate])
    return not check_constraints(partial, scoring_settings(config))


def candidate_is_valid(
    room: Room,
    placed: list[Furniture],
    candidate: Furniture,
    config: dict | None = None,
) -> bool:
    return _candidate_ok(room, placed, candidate, config)


def sample_pose(
    room: Room,
    item: Furniture,
    placed: list[Furniture],
    rng: Random,
    config: dict | None = None,
    wall_snap_probability: float | None = None,
) -> Furniture | None:
    settings = optimizer_settings(config)
    sampling = settings.get("sampling", {})
    attempts = int(sampling.get("pose_attempts", 80))
    probability = float(
        sampling.get("wall_snap_probability", 0.7)
        if wall_snap_probability is None
        else wall_snap_probability
    )
    rotations = rotation_options(settings.get("rotation_step_deg", 90.0))
    for _ in range(max(1, attempts)):
        if item.wall_affinity and rng.random() < probability:
            candidate = _wall_pose(item, room, rng.randrange(4), rng)
        else:
            candidate = _random_pose(item, room, rng, rotations)
        if candidate is not None and _candidate_ok(room, placed, candidate, config):
            return candidate
    return None


def sample_layout(plan: FloorPlan, rng: Random, config: dict | None = None) -> FloorPlan | None:
    room = plan.room
    placed: dict[int, Furniture] = {
        index: deepcopy(item) for index, item in enumerate(room.furniture) if item.locked
    }
    movable = [index for index, item in enumerate(room.furniture) if not item.locked]
    for index in movable:
        candidate = sample_pose(room, room.furniture[index], list(placed.values()), rng, config)
        if candidate is None:
            return None
        placed[index] = candidate
    furniture = [placed[index] for index in range(len(room.furniture))]
    return FloorPlan(
        room=Room(room.polygon, list(room.openings), furniture),
        scale=plan.scale,
        scale_mode=plan.scale_mode,
        camera_height=plan.camera_height,
        camera=deepcopy(plan.camera),
        metadata=deepcopy(plan.metadata),
    )
