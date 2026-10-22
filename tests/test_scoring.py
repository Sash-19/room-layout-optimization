import json

import pytest
from shapely.geometry import box

from src.floorplan.representation import Furniture, Opening, Room
from src.scoring.constraints import check_constraints, is_feasible
from src.scoring.grid import OccupancyGrid
from src.scoring.score import LayoutScorer
from src.scoring.terms import (
    term_clear,
    term_door,
    term_overlap,
    term_pair,
    term_path,
    term_util,
    term_walk,
    term_wall,
    term_window,
)


def make_room(furniture=None, openings=None, size=(4.0, 3.0)):
    return Room(box(0, 0, size[0], size[1]), openings or [], furniture or [])


def test_grid_rasterization_and_geometry():
    room = make_room([Furniture("bed", 1.6, 2.0, 1.0, 1.5)])
    grid = OccupancyGrid.from_room(room, resolution=0.05)
    assert grid.free_area == pytest.approx(room.area - 3.2, abs=0.3)
    assert not grid.free[grid.index_of(1.0, 1.5)]
    assert grid.free[grid.index_of(3.5, 2.5)]
    assert grid.distance[grid.index_of(2.0, 1.5)] > grid.distance[grid.index_of(3.95, 1.5)]
    assert grid.walkable_area(0.6) > 0
    assert grid.shortest_path_length((0.3, 0.3), (3.7, 2.7)) is not None
    assert grid.nearest_free_index(1.0, 1.5) is not None


def test_hard_constraints_accept_clean_layout():
    room = make_room(
        [Furniture("desk", 1.2, 0.6, 2.0, 2.6), Furniture("chair", 0.5, 0.5, 2.0, 1.8)],
        [Opening("door", 0, 1.5, 0.8, 0.8)],
    )
    assert check_constraints(room) == []
    assert is_feasible(room)


def test_hard_constraints_outside_overlap_and_door():
    outside = make_room([Furniture("bed", 1.6, 2.0, 0.2, 1.5)])
    assert any(item.kind == "outside_room" for item in check_constraints(outside))

    overlapping = make_room(
        [Furniture("bed", 1.6, 2.0, 1.0, 1.5), Furniture("chair", 0.5, 0.5, 1.0, 1.5)]
    )
    assert any(item.kind == "overlap" for item in check_constraints(overlapping))

    blocked = make_room(
        [Furniture("cabinet", 1.0, 0.5, 2.0, 0.3)],
        [Opening("door", 0, 1.5, 0.8, 0.8)],
    )
    assert any(item.kind == "door_blocked" for item in check_constraints(blocked))


def test_hard_constraints_window_fully_blocked():
    room = make_room(
        [Furniture("cabinet", 1.4, 0.5, 2.5, 0.25)],
        [Opening("window", 0, 1.9, 1.2)],
    )
    violations = check_constraints(room)
    assert any(item.kind == "window_blocked" for item in violations)


def test_soft_terms_empty_room():
    room = make_room()
    grid = OccupancyGrid.from_room(room)
    assert term_overlap(room, grid).score == 1.0
    assert term_walk(room, grid).score > 0.5
    assert term_util(room, grid).score == pytest.approx(1.0)
    assert term_path(room, grid).score == 1.0
    assert term_wall(room, grid).score == 1.0
    assert term_door(room, grid).score == 1.0
    assert term_window(room, grid).score == 1.0
    assert term_clear(room, grid).score == 1.0
    assert term_pair(room, grid).score == 1.0


def test_wall_and_overlap_terms_react_to_layout():
    near_wall = make_room([Furniture("bed", 1.6, 2.0, 0.8, 1.5, wall_affinity=True)])
    middle = make_room([Furniture("bed", 1.6, 2.0, 2.0, 1.5, wall_affinity=True)])
    grid_near = OccupancyGrid.from_room(near_wall)
    grid_middle = OccupancyGrid.from_room(middle)
    assert term_wall(near_wall, grid_near).score > term_wall(middle, grid_middle).score

    overlapping = make_room(
        [Furniture("bed", 1.6, 2.0, 1.0, 1.5), Furniture("chair", 0.5, 0.5, 1.0, 1.5)]
    )
    assert term_overlap(overlapping, OccupancyGrid.from_room(overlapping)).score < 1.0


def test_walk_door_window_and_clearance_terms():
    empty = make_room(openings=[Opening("door", 0, 1.5, 0.8, 0.8)])
    blocked = make_room(
        [Furniture("cabinet", 1.0, 0.5, 2.0, 0.3)],
        [Opening("door", 0, 1.5, 0.8, 0.8)],
    )
    assert term_walk(empty, OccupancyGrid.from_room(empty)).score > term_walk(blocked, OccupancyGrid.from_room(blocked)).score
    assert term_door(blocked, OccupancyGrid.from_room(blocked)).score < 1.0

    window_room = make_room(
        [Furniture("cabinet", 1.0, 0.5, 2.0, 0.3)],
        [Opening("window", 0, 1.5, 1.2)],
    )
    assert term_window(window_room, OccupancyGrid.from_room(window_room)).score < 1.0

    desk = Furniture("desk", 1.2, 0.6, 2.0, 1.6, front_clearance=0.75)
    clear = make_room([desk])
    obstructed = make_room([desk, Furniture("chair", 0.5, 0.5, 2.0, 2.3)])
    assert term_clear(clear, OccupancyGrid.from_room(clear)).score > term_clear(
        obstructed, OccupancyGrid.from_room(obstructed)
    ).score


def test_utilization_and_pair_terms():
    cluttered = make_room(
        [
            Furniture("cabinet", 1.0, 3.0, 1.0, 1.5),
            Furniture("cabinet", 1.0, 3.0, 2.0, 1.5),
            Furniture("cabinet", 1.0, 3.0, 3.0, 1.5),
        ]
    )
    open_room = make_room()
    assert term_util(open_room, OccupancyGrid.from_room(open_room)).score >= term_util(
        cluttered, OccupancyGrid.from_room(cluttered)
    ).score

    config = {"terms": {"pairs": [{"classes": ["chair", "desk"], "target_m": 0.6, "sigma_m": 0.2}]}}
    close = make_room(
        [Furniture("desk", 1.2, 0.6, 2.0, 2.2), Furniture("chair", 0.5, 0.5, 2.0, 1.4)]
    )
    far = make_room(
        [Furniture("desk", 1.2, 0.6, 2.0, 2.6), Furniture("chair", 0.5, 0.5, 0.5, 0.5)]
    )
    assert term_pair(close, OccupancyGrid.from_room(close), config).score > term_pair(
        far, OccupancyGrid.from_room(far), config
    ).score


def test_scorer_end_to_end_and_json():
    room = make_room(
        [Furniture("bed", 1.6, 2.0, 0.8, 1.5, wall_affinity=True), Furniture("desk", 1.2, 0.6, 3.0, 2.5)],
        [Opening("door", 0, 2.5, 0.8, 0.8), Opening("window", 1, 0.8, 1.2)],
    )
    scorer = LayoutScorer.from_config({"terms": {"walkway_width_m": 0.6}})
    result = scorer.evaluate(room)
    assert 0.0 <= result.total <= 1.0
    assert set(result.terms) == {"overlap", "walk", "path", "wall", "door", "window", "util", "clear", "pair"}
    assert result.feasible
    assert result.metrics["room_area_m2"] == pytest.approx(12.0)
    assert result.metrics["free_area_m2"] < 12.0
    json.dumps(result.to_dict())

    bad = make_room([Furniture("bed", 1.6, 2.0, 0.2, 1.5)])
    bad_result = scorer.evaluate(bad)
    assert not bad_result.feasible
    assert bad_result.violations


def test_scorer_weight_override():
    room = make_room([Furniture("bed", 1.6, 2.0, 1.0, 1.5)])
    scorer = LayoutScorer.from_config({"weights": {"walk": 1.0, "overlap": 0.0, "path": 0.0, "wall": 0.0, "door": 0.0, "window": 0.0, "util": 0.0, "clear": 0.0, "pair": 0.0}})
    result = scorer.evaluate(room)
    assert set(result.terms) == {"walk"}
    assert result.total == pytest.approx(result.terms["walk"])
