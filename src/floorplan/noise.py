from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from random import Random

from .priors import clamp_footprint
from .representation import FloorPlan, Room


def jitter_plan(
    plan: FloorPlan,
    position_sigma_m: float = 0.15,
    size_sigma_ratio: float = 0.08,
    rotation_probability: float = 0.1,
    seed: int = 0,
) -> FloorPlan:
    """Return a copy of the plan with simulated perception/lifting noise.

    Used by the E6 robustness experiment: geometry errors are injected as position,
    footprint-size, and orientation jitter so the optimizer can be tested against
    imperfect detected plans rather than only clean ones.
    """
    rng = Random(int(seed))
    furniture = []
    for item in plan.room.furniture:
        if item.locked:
            furniture.append(deepcopy(item))
            continue
        x = item.x + rng.gauss(0.0, float(position_sigma_m))
        y = item.y + rng.gauss(0.0, float(position_sigma_m))
        scale = math.exp(rng.gauss(0.0, float(size_sigma_ratio)))
        width = item.w * scale
        depth = item.d * scale
        width, depth = clamp_footprint(item.cls, width, depth)
        theta = item.theta
        if rng.random() < float(rotation_probability):
            theta += rng.choice((-math.pi / 2.0, math.pi / 2.0))
        furniture.append(replace(item, x=x, y=y, w=width, d=depth, theta=theta))
    return FloorPlan(
        room=Room(plan.room.polygon, list(plan.room.openings), furniture),
        scale=plan.scale,
        scale_mode=plan.scale_mode,
        camera_height=plan.camera_height,
        camera=deepcopy(plan.camera),
        metadata=deepcopy(plan.metadata),
    )


__all__ = ["jitter_plan"]