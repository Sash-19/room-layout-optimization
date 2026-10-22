from __future__ import annotations

from shapely import affinity
from shapely.geometry import Polygon, box

from ..floorplan.representation import Furniture, Opening, Room

INWARD = {
    0: (0.0, 1.0),
    1: (-1.0, 0.0),
    2: (0.0, -1.0),
    3: (1.0, 0.0),
}


def opening_center(room: Room, opening: Opening) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = room.bounds
    if opening.wall_index == 0:
        return (min_x + opening.offset + opening.width / 2.0, min_y)
    if opening.wall_index == 1:
        return (max_x, min_y + opening.offset + opening.width / 2.0)
    if opening.wall_index == 2:
        return (min_x + opening.offset + opening.width / 2.0, max_y)
    return (min_x, min_y + opening.offset + opening.width / 2.0)


def inward_point(room: Room, opening: Opening, distance: float) -> tuple[float, float]:
    x, y = opening_center(room, opening)
    dx, dy = INWARD[opening.wall_index]
    return x + dx * float(distance), y + dy * float(distance)


def opening_zone(
    room: Room,
    opening: Opening,
    depth: float,
    margin: float = 0.0,
) -> Polygon | None:
    min_x, min_y, max_x, max_y = room.bounds
    depth = max(0.0, float(depth))
    margin = max(0.0, float(margin))
    start = opening.offset - margin
    end = opening.offset + opening.width + margin
    if opening.wall_index == 0:
        rectangle = box(min_x + start, min_y, min_x + end, min_y + depth)
    elif opening.wall_index == 1:
        rectangle = box(max_x - depth, min_y + start, max_x, min_y + end)
    elif opening.wall_index == 2:
        rectangle = box(min_x + start, max_y - depth, min_x + end, max_y)
    else:
        rectangle = box(min_x, min_y + start, min_x + depth, min_y + end)
    zone = rectangle.intersection(room.polygon)
    return zone if not zone.is_empty and zone.area > 0 else None


def front_zone(item: Furniture, clearance: float | None = None) -> Polygon | None:
    clearance = item.front_clearance if clearance is None else float(clearance)
    if clearance <= 0:
        return None
    rectangle = box(-item.w / 2.0, item.d / 2.0, item.w / 2.0, item.d / 2.0 + clearance)
    rotated = affinity.rotate(rectangle, item.theta, origin=(0.0, 0.0), use_radians=True)
    return affinity.translate(rotated, xoff=item.x, yoff=item.y)


def overlap_area(a: Furniture, b: Furniture) -> float:
    return float(a.footprint.intersection(b.footprint).area)


def zone_free_fraction(zone: Polygon | None, footprints: list[Furniture]) -> float:
    if zone is None:
        return 1.0
    area = float(zone.area)
    if area <= 0:
        return 1.0
    covered = sum(float(zone.intersection(item.footprint).area) for item in footprints)
    return float(max(0.0, min(1.0, 1.0 - covered / area)))
