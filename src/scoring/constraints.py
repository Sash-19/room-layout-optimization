from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..floorplan.priors import get_prior
from ..floorplan.representation import Room
from .zones import opening_zone


@dataclass(frozen=True, slots=True)
class ConstraintViolation:
    kind: str
    message: str
    items: tuple[int, ...] = ()
    area: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "items": list(self.items),
            "area": round(float(self.area), 4),
            "details": self.details,
        }


def check_constraints(room: Room, config: dict[str, Any] | None = None) -> list[ConstraintViolation]:
    settings = (config or {}).get("constraints", {})
    inside_tolerance = float(settings.get("inside_tolerance_m", 1e-3))
    overlap_tolerance = float(settings.get("overlap_tolerance_m2", 1e-3))
    door_depth = float(settings.get("door_clear_depth_m", 0.9))
    door_margin = float(settings.get("door_clear_margin_m", 0.1))
    window_depth = float(settings.get("window_clear_depth_m", 0.3))
    tall_height = float(settings.get("tall_item_height_m", 1.0))
    allow_outside = bool(settings.get("allow_outside_room", False))
    allow_overlap = bool(settings.get("allow_overlap", False))
    allow_door_blocked = bool(settings.get("allow_door_blocked", False))
    allow_window_blocked = bool(settings.get("allow_window_blocked", False))

    violations: list[ConstraintViolation] = []
    allowed = room.polygon.buffer(inside_tolerance)
    if not allow_outside:
        for index, item in enumerate(room.furniture):
            if not allowed.covers(item.footprint):
                violations.append(
                    ConstraintViolation(
                        "outside_room",
                        f"{item.cls} {index} is not fully inside the room",
                        (index,),
                        float(item.footprint.difference(room.polygon).area),
                    )
                )

    if not allow_overlap:
        for first in range(len(room.furniture)):
            for second in range(first + 1, len(room.furniture)):
                area = float(room.furniture[first].footprint.intersection(room.furniture[second].footprint).area)
                if area > overlap_tolerance:
                    violations.append(
                        ConstraintViolation(
                            "overlap",
                            f"{room.furniture[first].cls} {first} overlaps {room.furniture[second].cls} {second}",
                            (first, second),
                            area,
                        )
                    )

    for door in (opening for opening in room.openings if opening.kind == "door"):
        if allow_door_blocked:
            continue
        zone = opening_zone(room, door, max(door_depth, door.swing_radius), door_margin)
        if zone is None:
            continue
        for index, item in enumerate(room.furniture):
            area = float(zone.intersection(item.footprint).area)
            if area > overlap_tolerance:
                violations.append(
                    ConstraintViolation(
                        "door_blocked",
                        f"{item.cls} {index} blocks the door clear zone",
                        (index,),
                        area,
                        {"wall_index": door.wall_index, "offset": round(door.offset, 4)},
                    )
                )

    if allow_window_blocked:
        return violations

    for window in (opening for opening in room.openings if opening.kind == "window"):
        zone = opening_zone(room, window, window_depth)
        if zone is None:
            continue
        tall = [
            (index, item)
            for index, item in enumerate(room.furniture)
            if (get_prior(item.cls) is not None and get_prior(item.cls).height >= tall_height)
        ]
        if not tall:
            continue
        covered = sum(float(zone.intersection(item.footprint).area) for _, item in tall)
        if covered >= float(zone.area) - overlap_tolerance:
            violations.append(
                ConstraintViolation(
                    "window_blocked",
                    "a tall item fully blocks a window",
                    tuple(index for index, _ in tall),
                    covered,
                    {"wall_index": window.wall_index, "offset": round(window.offset, 4)},
                )
            )

    return violations


def is_feasible(room: Room, config: dict[str, Any] | None = None) -> bool:
    return not check_constraints(room, config)
