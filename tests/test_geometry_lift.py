import numpy as np
import pytest

from src.geometry.floor_plane import fit_floor_plane, floor_points_from_mask
from src.geometry.lift import bottom_edge_floor_points, bottom_edge_pixels, lift_detections, lift_openings
from src.geometry.room_boundary import FloorFrame, estimate_room_boundary
from src.schemas import Detection
from tests.synthetic_geometry import make_bed_detection, make_camera, make_depth, make_door_detection, make_floor_mask


@pytest.fixture
def geometry():
    camera = make_camera()
    depth = make_depth(camera)
    mask = make_floor_mask()
    plane = fit_floor_plane(depth, camera, floor_mask=mask, stride=2, threshold=0.005, iterations=200)
    frame = FloorFrame.from_plane(plane)
    points = floor_points_from_mask(camera, plane, depth, mask, stride=4, max_points=5000)
    room = estimate_room_boundary(points, frame, percentile=2.0)
    return camera, plane, frame, room


def test_bottom_edge_pixels_and_projection(geometry):
    camera, plane, frame, _ = geometry
    detection = make_bed_detection()
    pixels = bottom_edge_pixels(detection, camera, num=5)
    assert pixels.shape == (5, 2)
    assert np.all(pixels[:, 1] == pytest.approx(429.5))
    points = bottom_edge_floor_points(detection, camera, plane, num=5)
    assert len(points) == 5
    assert np.allclose(points[:, 1], 1.5, atol=1e-4)


def test_lift_bed_to_floor(geometry):
    camera, plane, frame, _ = geometry
    items = lift_detections([make_bed_detection()], camera, plane, frame, {"snap_orientation": True})
    assert len(items) == 1
    bed = items[0]
    assert bed.cls == "bed"
    assert 0.9 <= bed.w <= 2.4
    assert bed.d == pytest.approx(2.0)
    assert bed.theta == pytest.approx(0.0, abs=1e-6)
    assert bed.front_clearance == pytest.approx(0.6)
    assert bed.wall_affinity is True
    center = frame.to_3d([[bed.x, bed.y]])
    assert plane.distance(center)[0] == pytest.approx(0.0, abs=1e-6)
    midpoint = bottom_edge_floor_points(make_bed_detection(), camera, plane)
    midpoint_2d = frame.to_2d(midpoint).mean(axis=0)
    assert bed.y > midpoint_2d[1]


def test_lift_skips_unprojectable_detection(geometry):
    camera, plane, frame, _ = geometry
    above_horizon = Detection("chair", 0.9, (250, 100, 350, 180))
    assert lift_detections([above_horizon], camera, plane, frame) == []


def test_lift_openings_assigns_walls(geometry):
    camera, plane, frame, room = geometry
    door = make_door_detection()
    window = Detection("window", 0.7, (430, 300, 500, 360))
    openings = lift_openings([door, window], camera, plane, frame, room)
    assert [item.kind for item in openings] == ["door", "window"]
    for opening in openings:
        assert 0 <= opening.wall_index <= 3
        assert opening.offset >= 0
        assert 0.4 <= opening.width <= 1.6
    assert openings[0].swing_radius == pytest.approx(openings[0].width)
    assert openings[1].swing_radius == 0.0
