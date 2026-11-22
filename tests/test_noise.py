import pytest

from src.floorplan.noise import jitter_plan
from src.floorplan.synthetic import good_bedroom


def test_jitter_preserves_structure_and_is_deterministic():
    plan = good_bedroom()
    first = jitter_plan(plan, position_sigma_m=0.1, seed=1)
    second = jitter_plan(plan, position_sigma_m=0.1, seed=1)
    assert len(first.room.furniture) == len(plan.room.furniture)
    assert first.scale_mode == plan.scale_mode
    assert first.metadata == plan.metadata
    for original, jittered, repeat in zip(plan.room.furniture, first.room.furniture, second.room.furniture):
        assert jittered.cls == original.cls
        assert jittered.x == pytest.approx(repeat.x)
        assert jittered.y == pytest.approx(repeat.y)
        assert jittered.w == pytest.approx(repeat.w)


def test_jitter_moves_unlocked_but_not_locked_items():
    plan = good_bedroom()
    locked = plan.room.furniture[0]
    locked.locked = True
    jittered = jitter_plan(plan, position_sigma_m=0.2, seed=4)
    assert jittered.room.furniture[0].x == pytest.approx(locked.x)
    assert jittered.room.furniture[0].y == pytest.approx(locked.y)


def test_jitter_does_not_mutate_original():
    plan = good_bedroom()
    before = (plan.room.furniture[1].x, plan.room.furniture[1].y)
    jitter_plan(plan, position_sigma_m=0.5, seed=2)
    after = (plan.room.furniture[1].x, plan.room.furniture[1].y)
    assert before == after


def test_zero_sigma_is_near_identity():
    plan = good_bedroom()
    jittered = jitter_plan(plan, position_sigma_m=0.0, size_sigma_ratio=0.0, rotation_probability=0.0, seed=0)
    for original, item in zip(plan.room.furniture, jittered.room.furniture):
        assert item.x == pytest.approx(original.x)
        assert item.y == pytest.approx(original.y)
        assert item.w == pytest.approx(original.w)
        assert item.d == pytest.approx(original.d)
