from src.floorplan.synthetic import (
    bad_door_blocked,
    bad_outside,
    bad_overlap,
    bad_window_blocked,
    good_bedroom,
    good_office,
    synthetic_suite,
)
from src.optimization import greedy_search
from src.scoring.score import LayoutScorer

CONFIG = {
    "optimizer": {
        "seed": 0,
        "rotation_step_deg": 90,
        "sampling": {"pose_attempts": 80, "wall_snap_probability": 0.8},
        "greedy": {"candidates_per_item": 120},
    },
    "scoring": {"grid": {"resolution": 0.1}},
}


def kinds(score):
    return {violation["kind"] for violation in score.violations}


def test_suite_has_good_and_bad_layouts():
    suite = synthetic_suite()
    assert set(suite) == {
        "good_bedroom",
        "good_office",
        "bad_door_blocked",
        "bad_window_blocked",
        "bad_overlap",
        "bad_outside",
    }


def test_good_layouts_are_feasible():
    scorer = LayoutScorer()
    assert scorer.evaluate(good_bedroom()).feasible
    assert scorer.evaluate(good_office()).feasible


def test_bad_layouts_trigger_expected_violations():
    scorer = LayoutScorer()
    assert "door_blocked" in kinds(scorer.evaluate(bad_door_blocked()))
    assert "window_blocked" in kinds(scorer.evaluate(bad_window_blocked()))
    assert "overlap" in kinds(scorer.evaluate(bad_overlap()))
    assert "outside_room" in kinds(scorer.evaluate(bad_outside()))


def test_optimizer_repairs_bad_layouts():
    scorer = LayoutScorer.from_config(CONFIG["scoring"])
    for factory in (bad_door_blocked, bad_window_blocked, bad_overlap, bad_outside):
        plan = factory()
        initial = scorer.evaluate(plan)
        assert not initial.feasible
        result = greedy_search(plan, scorer, CONFIG)
        assert result.best_score.feasible
        assert result.best_score.total >= initial.total
