import numpy as np
import pytest
from PIL import Image

from src.geometry.camera import CameraIntrinsics
from tests.synthetic_geometry import FOCAL, HEIGHT, WIDTH, make_camera


def test_intrinsics_from_fov():
    camera = CameraIntrinsics.from_fov(640, 480, fov_deg=65.0)
    expected = 640 / (2 * np.tan(np.deg2rad(65.0) / 2))
    assert camera.fx == pytest.approx(expected)
    assert camera.cx == 320
    assert camera.cy == 240
    assert camera.source == "fov"


def test_pixel_rays_and_backprojection_roundtrip():
    camera = make_camera()
    rays = camera.pixel_rays([[320, 240], [320, 340]])
    np.testing.assert_allclose(rays[0], [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(rays[1], [0, 0.2, 1], atol=1e-9)

    depth = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
    points = camera.backproject([[320, 240]], depth)
    np.testing.assert_allclose(points[0], [0, 0, 2], atol=1e-9)

    projected = camera.project(points)
    np.testing.assert_allclose(projected[0], [320, 240], atol=1e-6)


def test_camera_from_image_falls_back_to_fov():
    image = Image.new("RGB", (WIDTH, HEIGHT))
    camera = CameraIntrinsics.from_image(image, fov_deg=65.0)
    assert camera.source == "fov"
    assert camera.fx == pytest.approx(640 / (2 * np.tan(np.deg2rad(65.0) / 2)))


def test_intrinsics_validation_and_serialization():
    with pytest.raises(ValueError):
        CameraIntrinsics(0, 1, 0, 0, 10, 10)
    camera = make_camera()
    payload = camera.to_dict()
    assert payload["width"] == WIDTH
    assert payload["height"] == HEIGHT
    assert payload["fx"] == FOCAL
