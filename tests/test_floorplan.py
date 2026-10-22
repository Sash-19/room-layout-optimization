import json

import pytest
from shapely.geometry import box

from src.floorplan.priors import clamp_footprint, get_prior, opening_prior
from src.floorplan.representation import FloorPlan, Furniture, Opening, Room


def test_furniture_footprint_geometry():
    item = Furniture("bed", 1.6, 2.0, 3.0, 4.0, theta=0.0)
    assert item.footprint.area == pytest.approx(3.2)
    assert item.footprint.bounds == pytest.approx((2.2, 3.0, 3.8, 5.0))

    rotated = Furniture("bed", 2.0, 1.0, 0.0, 0.0, theta=1.5707963267948966)
    min_x, min_y, max_x, max_y = rotated.footprint.bounds
    assert max_x - min_x == pytest.approx(1.0, abs=1e-6)
    assert max_y - min_y == pytest.approx(2.0, abs=1e-6)


def test_room_and_opening_serialization():
    room = Room(
        polygon=box(0, 0, 4, 3),
        openings=[Opening("door", 0, 1.0, 0.8, 0.8), Opening("window", 2, 0.5, 1.2)],
        furniture=[Furniture("desk", 1.2, 0.6, 2.0, 2.5)],
    )
    payload = room.to_dict()
    assert payload["area"] == pytest.approx(12.0)
    assert payload["width"] == pytest.approx(4.0)
    assert payload["openings"][0]["wall_name"] == "min_y"
    assert payload["furniture"][0]["class"] == "desk"
    assert len(payload["furniture"][0]["footprint"]) == 5


def test_floorplan_json_serializable():
    plan = FloorPlan(
        room=Room(box(0, 0, 4, 3), [Opening("door", 0, 0.5, 0.8, 0.8)], [Furniture("chair", 0.5, 0.5, 1.0, 1.0)]),
        scale=1.25,
        scale_mode="user",
        camera_height=1.6,
        camera={"fx": 500},
        metadata={"warnings": []},
    )
    encoded = json.dumps(plan.to_dict())
    decoded = json.loads(encoded)
    assert decoded["scale"] == 1.25
    assert decoded["room"]["openings"][0]["kind"] == "door"

    restored = FloorPlan.from_dict(decoded)
    assert restored.room.area == pytest.approx(12.0)
    assert restored.room.openings[0].wall_name == "min_y"
    assert restored.room.furniture[0].footprint.equals(plan.room.furniture[0].footprint)
    assert restored.metadata["warnings"] == []


def test_room_rejects_invalid_polygon():
    with pytest.raises(ValueError):
        Room(polygon=box(0, 0, 0, 0))
    with pytest.raises(ValueError):
        Opening("arch", 0, 0.0, 1.0)
    with pytest.raises(ValueError):
        Furniture("bed", -1.0, 2.0, 0.0, 0.0)


def test_priors_clamp_and_lookup():
    prior = get_prior("bed")
    assert prior is not None and prior.depth == pytest.approx(2.0)
    assert clamp_footprint("bed", 0.1, 10.0) == pytest.approx((prior.min_width, prior.max_depth))
    assert opening_prior("door").width == pytest.approx(0.8)
    assert get_prior("toaster") is None
