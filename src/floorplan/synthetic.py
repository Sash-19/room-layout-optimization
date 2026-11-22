from __future__ import annotations

from shapely.geometry import box

from .priors import get_prior
from .representation import FloorPlan, Furniture, Opening, Room


def furniture(cls: str, x: float, y: float, theta: float = 0.0, **overrides) -> Furniture:
    prior = get_prior(cls)
    if prior is None:
        raise ValueError(f"unknown furniture class {cls!r}")
    values = {
        "cls": cls,
        "w": prior.width,
        "d": prior.depth,
        "x": x,
        "y": y,
        "theta": theta,
        "front_clearance": prior.front_clearance,
        "wall_affinity": prior.wall_affinity,
    }
    values.update(overrides)
    return Furniture(**values)


def plan(
    size: tuple[float, float],
    furniture_items: list[Furniture] | None = None,
    openings: list[Opening] | None = None,
    name: str = "synthetic",
) -> FloorPlan:
    width, depth = size
    return FloorPlan(
        room=Room(box(0.0, 0.0, float(width), float(depth)), list(openings or []), list(furniture_items or [])),
        scale=1.0,
        scale_mode="synthetic",
        metadata={"name": name},
    )


def good_bedroom() -> FloorPlan:
    return plan(
        (4.0, 3.5),
        [
            furniture("bed", 1.0, 2.0),
            furniture("desk", 3.3, 1.0),
            furniture("chair", 3.3, 1.75, w=0.5, d=0.5),
            furniture("cabinet", 0.5, 0.3),
        ],
        [Opening("door", 0, 1.6, 0.8, 0.8), Opening("window", 2, 1.2, 1.2)],
        "good_bedroom",
    )


def good_office() -> FloorPlan:
    return plan(
        (4.0, 3.0),
        [
            furniture("desk", 2.6, 1.2),
            furniture("chair", 2.6, 0.55, w=0.5, d=0.5),
            furniture("cabinet", 0.5, 2.6),
        ],
        [Opening("door", 3, 1.0, 0.8, 0.8), Opening("window", 2, 1.4, 1.2)],
        "good_office",
    )


def bad_door_blocked() -> FloorPlan:
    return plan(
        (4.0, 4.0),
        [
            furniture("cabinet", 2.0, 0.3, w=1.0, d=0.5),
            furniture("chair", 3.2, 2.0, w=0.5, d=0.5),
        ],
        [Opening("door", 0, 1.6, 0.8, 0.8)],
        "bad_door_blocked",
    )


def bad_window_blocked() -> FloorPlan:
    return plan(
        (4.0, 3.0),
        [furniture("cabinet", 2.0, 0.25, w=1.4, d=0.5)],
        [Opening("window", 0, 1.4, 1.2)],
        "bad_window_blocked",
    )


def bad_overlap() -> FloorPlan:
    return plan(
        (4.0, 3.0),
        [
            furniture("bed", 1.5, 1.5),
            furniture("cabinet", 1.5, 1.5, w=1.0, d=0.5),
        ],
        [Opening("door", 0, 1.6, 0.8, 0.8)],
        "bad_overlap",
    )


def bad_outside() -> FloorPlan:
    return plan(
        (4.0, 3.0),
        [furniture("bed", 0.4, 1.5)],
        [Opening("door", 0, 2.2, 0.8, 0.8)],
        "bad_outside",
    )


def synthetic_suite() -> dict[str, FloorPlan]:
    return {
        "good_bedroom": good_bedroom(),
        "good_office": good_office(),
        "bad_door_blocked": bad_door_blocked(),
        "bad_window_blocked": bad_window_blocked(),
        "bad_overlap": bad_overlap(),
        "bad_outside": bad_outside(),
    }


__all__ = [
    "bad_door_blocked",
    "bad_outside",
    "bad_overlap",
    "bad_window_blocked",
    "furniture",
    "good_bedroom",
    "good_office",
    "plan",
    "synthetic_suite",
]