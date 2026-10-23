import math

import pytest
from shapely.geometry import box

from src.floorplan.representation import FloorPlan, Furniture, Opening, Room
from src.optimization import greedy_search, random_search
from src.scoring.score import LayoutScorer


def make_plan(size=(5.0, 4.0), furniture=None, openings=None):
    return FloorPlan(
        Room(
            box(0.0, 0.0, size[0], size[1]),
            openings or [Opening("door", 0, 0.5, 0.8, 0.8)],
            furniture or [],
        ),
        scale=1.0,
        scale_mode="test",
    )


def make_config(**overrides):
    optimizer = {
        "seed": 3,
        "rotation_step_deg": 90,
        "sampling": {"pose_attempts": 60, "wall_snap_probability": 0.8},
        "random": {"iterations": 80, "max_layout_attempts": 30},
        "greedy": {"candidates_per_item": 80},
    }
    optimizer.update(overrides)
    return {"optimizer": optimizer, "scoring": {"grid": {"resolution": 0.1}}}


def test_random_search_finds_feasible_improvement():
    plan = make_plan(
        furniture=[
            Furniture("bed", 1.6, 2.0, 0.8, 1.0, wall_affinity=True),
            Furniture("desk", 1.2, 0.6, 4.0, 2.0),
            Furniture("chair", 0.5, 0.5, 4.0, 2.0),
        ]
    )
    scorer = LayoutScorer.from_config({"grid": {"resolution": 0.1}})
    initial = scorer.evaluate(plan)
    assert not initial.feasible

    result = random_search(plan, scorer, make_config())

    assert result.best_score.feasible
    assert result.best_score.total >= initial.total
    assert len(result.best_plan.room.furniture) == 3
    assert result.evaluations > 1


def test_greedy_search_places_items_feasibly():
    plan = make_plan(
        furniture=[
            Furniture("bed", 1.6, 2.0, 2.5, 2.0, wall_affinity=True),
            Furniture("desk", 1.2, 0.6, 2.5, 2.0),
            Furniture("chair", 0.5, 0.5, 2.5, 2.0),
        ]
    )
    scorer = LayoutScorer.from_config({"grid": {"resolution": 0.1}})
    initial = scorer.evaluate(plan)
    assert not initial.feasible

    result = greedy_search(plan, scorer, make_config())

    assert result.best_score.feasible
    assert result.best_score.total >= initial.total
    assert result.best_plan.room.furniture[0].cls == "bed"


def test_locked_items_are_not_moved():
    bed = Furniture("bed", 1.6, 2.0, 0.8, 1.0, theta=0.3, wall_affinity=True, locked=True)
    plan = make_plan(furniture=[bed, Furniture("chair", 0.5, 0.5, 3.0, 2.0)])
    scorer = LayoutScorer.from_config({"grid": {"resolution": 0.1}})

    result = greedy_search(plan, scorer, make_config())

    locked = result.best_plan.room.furniture[0]
    assert locked.x == pytest.approx(bed.x)
    assert locked.y == pytest.approx(bed.y)
    assert locked.theta == pytest.approx(bed.theta)


def test_wall_affinity_item_snaps_to_a_wall():
    plan = make_plan(furniture=[Furniture("couch", 1.8, 0.9, 2.5, 2.0, wall_affinity=True)])
    scorer = LayoutScorer.from_config({"grid": {"resolution": 0.1}})

    result = greedy_search(plan, scorer, make_config(greedy={"candidates_per_item": 120}))
    couch = result.best_plan.room.furniture[0]

    assert couch.footprint.distance(result.best_plan.room.polygon.boundary) <= 0.11
    assert min(abs(couch.theta - angle) for angle in (0.0, math.pi / 2, math.pi, -math.pi / 2)) < 1e-6
