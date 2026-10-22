import numpy as np
import pytest

from src.geometry.floor_plane import fit_floor_plane, floor_points_from_mask
from src.geometry.lift import bottom_edge_floor_points
from src.geometry.room_boundary import FloorFrame
from src.geometry.scale import estimate_scale, scale_from_extents, scale_from_reference
from tests.synthetic_geometry import make_bed_detection, make_camera, make_depth, make_door_detection, make_floor_mask


def setup_geometry():
    camera = make_camera()
    depth = make_depth(camera)
    mask = make_floor_mask()
    plane = fit_floor_plane(depth, camera, floor_mask=mask, stride=2, threshold=0.005, iterations=200)
    frame = FloorFrame.from_plane(plane)
    points = floor_points_from_mask(camera, plane, depth, mask, stride=4, max_points=5000)
    return camera, plane, frame, points


def test_scale_helpers():
    assert scale_from_extents(4.0, 5.0, target_width=8.0, target_depth=10.0) == pytest.approx(2.0)
    assert scale_from_extents(4.0, 5.0, target_width=8.0) == pytest.approx(2.0)
    assert scale_from_reference(1.6, 0.8) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        scale_from_extents(0.0, 1.0, target_width=2.0)
    with pytest.raises(ValueError):
        scale_from_reference(0.0, 1.0)


def test_metric_depth_auto_scale_is_one():
    camera, plane, frame, points = setup_geometry()
    result = estimate_scale(
        camera,
        plane,
        frame,
        points,
        [make_bed_detection()],
        depth_metric=True,
        config={"mode": "auto"},
    )
    assert result.scale == 1.0
    assert result.mode == "metric"
    assert not result.warnings


def test_user_dimension_scale():
    camera, plane, frame, points = setup_geometry()
    result = estimate_scale(
        camera,
        plane,
        frame,
        points,
        [make_bed_detection()],
        depth_metric=False,
        config={"mode": "user", "room_width_m": 4.0},
    )
    assert result.mode == "user"
    assert result.scale > 0
    assert result.details["room_width_m"] == 4.0
    assert result.details["observed_width_units"] > 0


def test_reference_scale_uses_door_width():
    camera, plane, frame, points = setup_geometry()
    door = make_door_detection()
    door_points = bottom_edge_floor_points(door, camera, plane)
    measured = float(np.linalg.norm(door_points[-1] - door_points[0]))
    result = estimate_scale(
        camera,
        plane,
        frame,
        points,
        [make_bed_detection(), door],
        depth_metric=False,
        config={"mode": "reference", "reference_class": "door", "reference_width_m": 0.8},
    )
    assert result.mode == "reference"
    assert result.scale == pytest.approx(0.8 / measured, rel=1e-6)
    assert result.details["measured_width_units"] == pytest.approx(measured, rel=1e-4)


def test_unscaled_relative_depth_and_missing_reference():
    camera, plane, frame, points = setup_geometry()
    result = estimate_scale(
        camera,
        plane,
        frame,
        points,
        [make_bed_detection()],
        depth_metric=False,
        config={"mode": "auto"},
    )
    assert result.mode == "unscaled"
    assert result.scale == 1.0
    assert result.warnings

    missing = estimate_scale(
        camera,
        plane,
        frame,
        points,
        [make_bed_detection()],
        depth_metric=False,
        config={"mode": "reference", "reference_class": "door"},
    )
    assert missing.mode == "unscaled"
    assert any("reference" in warning for warning in missing.warnings)
