import numpy as np
import pytest

from src.geometry.floor_plane import fit_floor_plane, floor_points_from_mask
from src.geometry.room_boundary import FloorFrame, estimate_room_boundary
from tests.synthetic_geometry import CAMERA_HEIGHT, make_camera, make_depth, make_floor_mask


@pytest.fixture
def floor_data():
    camera = make_camera()
    depth = make_depth(camera)
    mask = make_floor_mask()
    plane = fit_floor_plane(depth, camera, floor_mask=mask, stride=2, threshold=0.005, iterations=200)
    frame = FloorFrame.from_plane(plane)
    points = floor_points_from_mask(camera, plane, depth, mask, stride=4, max_points=5000)
    return plane, frame, points


def test_floor_frame_coordinates(floor_data):
    plane, frame, _ = floor_data
    np.testing.assert_allclose(frame.origin, [0, CAMERA_HEIGHT, 0], atol=1e-6)
    np.testing.assert_allclose(frame.to_2d([[0, CAMERA_HEIGHT, 3.0]]), [[0.0, 3.0]], atol=1e-6)
    np.testing.assert_allclose(frame.to_3d([[0.0, 3.0]]), [[0.0, CAMERA_HEIGHT, 3.0]], atol=1e-6)
    assert frame.to_dict()["normal"][1] == pytest.approx(-1.0)


def test_room_boundary_from_points(floor_data):
    _, frame, points = floor_data
    estimate = estimate_room_boundary(points, frame, percentile=2.0)
    assert estimate.width > 2.0
    assert estimate.depth > 2.0
    assert estimate.polygon.area == pytest.approx(estimate.width * estimate.depth)
    min_x, min_y, max_x, max_y = estimate.bounds
    assert estimate.polygon.bounds == pytest.approx((min_x, min_y, max_x, max_y))


def test_room_boundary_user_dimensions(floor_data):
    _, frame, points = floor_data
    estimate = estimate_room_boundary(points, frame, width_m=4.0, depth_m=5.0)
    assert estimate.width == pytest.approx(4.0)
    assert estimate.depth == pytest.approx(5.0)


def test_room_boundary_minimum_size_and_validation(floor_data):
    _, frame, _ = floor_data
    few = np.array([[0, CAMERA_HEIGHT, 3.0], [0, CAMERA_HEIGHT, 4.0], [1, CAMERA_HEIGHT, 3.0]])
    estimate = estimate_room_boundary(few, frame, min_size_m=2.0)
    assert estimate.width >= 2.0
    assert estimate.depth >= 2.0
    with pytest.raises(ValueError):
        estimate_room_boundary(np.zeros((0, 3)), frame)
