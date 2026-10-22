import numpy as np
import pytest

from src.geometry.floor_plane import calibrate_inverse_depth, fit_floor_plane, floor_points_from_mask
from tests.synthetic_geometry import CAMERA_HEIGHT, make_camera, make_depth, make_floor_mask


@pytest.fixture
def fitted_plane():
    camera = make_camera()
    depth = make_depth(camera)
    mask = make_floor_mask()
    plane = fit_floor_plane(
        depth,
        camera,
        floor_mask=mask,
        stride=2,
        threshold=0.005,
        iterations=200,
        max_points=5000,
        seed=0,
    )
    return camera, depth, mask, plane


def test_plane_fit_recovers_known_floor(fitted_plane):
    camera, depth, mask, plane = fitted_plane
    np.testing.assert_allclose(plane.normal, [0, -1, 0], atol=0.02)
    assert plane.camera_height == pytest.approx(CAMERA_HEIGHT, abs=0.02)
    assert plane.inlier_ratio > 0.9
    assert plane.distance(np.array([[0, CAMERA_HEIGHT, 3.0]]))[0] == pytest.approx(0.0, abs=1e-6)


def test_ray_plane_intersection(fitted_plane):
    camera, _, _, plane = fitted_plane
    rays = camera.pixel_rays([[320, 400], [320, 240], [320, 100]])
    points, valid = plane.intersect_rays(rays)
    assert valid.tolist() == [True, False, False]
    assert points[0][1] == pytest.approx(CAMERA_HEIGHT, abs=1e-4)


def test_plane_scaling_and_floor_point_extraction(fitted_plane):
    camera, depth, mask, plane = fitted_plane
    scaled = plane.scale(2.0)
    assert scaled.camera_height == pytest.approx(2 * CAMERA_HEIGHT)
    assert scaled.offset == pytest.approx(2 * plane.offset)

    points = floor_points_from_mask(camera, plane, depth, mask, stride=4, max_points=2000)
    assert len(points) > 100
    assert np.allclose(points[:, 1], CAMERA_HEIGHT, atol=1e-4)
    assert np.isfinite(points).all()


def test_plane_fit_requires_valid_pixels():
    camera = make_camera()
    depth = np.zeros((camera.height, camera.width), dtype=np.float32)
    with pytest.raises(ValueError):
        fit_floor_plane(depth, camera)


def test_calibrate_inverse_depth_recovers_floor():
    camera = make_camera()
    depth = make_depth(camera)
    inverse = np.zeros_like(depth)
    positive = depth > 0
    inverse[positive] = 1.0 / depth[positive]

    calibrated, params = calibrate_inverse_depth(inverse, camera, floor_mask=make_floor_mask())
    assert params["method"] == "affine-inverse"
    assert params["relative_residual"] < 0.02

    plane = fit_floor_plane(calibrated, camera, floor_mask=make_floor_mask(), threshold=0.02, iterations=200)
    assert plane.camera_height == pytest.approx(1.0, abs=0.05)
    np.testing.assert_allclose(plane.normal, [0, -1, 0], atol=0.05)

    reciprocal, fallback = calibrate_inverse_depth(inverse, camera)
    assert fallback["method"] == "reciprocal"
    np.testing.assert_allclose(reciprocal[positive], depth[positive], rtol=1e-6)
