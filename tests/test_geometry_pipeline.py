import json

import numpy as np
import pytest
from PIL import Image

from src.geometry.pipeline import GeometryPipeline
from src.perception.pipeline import PerceptionResult
from src.schemas import DepthMap, Detection
from tests.synthetic_geometry import (
    HEIGHT,
    WIDTH,
    make_bed_detection,
    make_camera,
    make_depth,
    make_door_detection,
    make_structures,
)


def make_result(detections=None, metric=True, depth_map=None) -> PerceptionResult:
    camera = make_camera()
    depth = depth_map or DepthMap(make_depth(camera), metric=metric, source="synthetic")
    return PerceptionResult(
        detections=detections or [make_bed_detection(), make_door_detection()],
        image_size=(WIDTH, HEIGHT),
        depth=depth,
        structures=make_structures(),
    )


def make_config(**overrides) -> dict:
    config = {
        "camera": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
        "floor": {"threshold": 0.005, "inlier_threshold": 0.005, "sample_stride": 2, "ransac_iterations": 200},
        "scale": {"mode": "metric"},
        "lift": {"snap_orientation": True, "rotation_step_deg": 90},
        "room": {"boundary_percentile": 2.0, "min_size_m": 1.5, "max_size_m": 30.0},
    }
    for key, value in overrides.items():
        config[key] = {**config.get(key, {}), **value} if isinstance(value, dict) else value
    return config


def test_geometry_pipeline_metric():
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    plan = GeometryPipeline.from_config(make_config()).run(image, make_result())

    assert plan.scale == pytest.approx(1.0)
    assert plan.scale_mode == "metric"
    assert plan.camera_height == pytest.approx(1.5, abs=0.02)
    assert plan.room.width > 2.0 and plan.room.depth > 2.0
    assert len(plan.room.furniture) == 1
    assert plan.room.furniture[0].cls == "bed"
    assert len(plan.room.openings) == 1
    assert plan.room.openings[0].kind == "door"
    assert not plan.metadata["warnings"]
    json.dumps(plan.to_dict())


def test_geometry_pipeline_user_scale():
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    config = make_config(scale={"mode": "user", "room_width_m": 4.0, "room_depth_m": 5.0})
    plan = GeometryPipeline.from_config(config).run(image, make_result())
    assert plan.scale_mode == "user"
    assert plan.scale > 0
    assert plan.room.width == pytest.approx(4.0)
    assert plan.room.depth == pytest.approx(5.0)
    assert plan.metadata["scale"]["details"]["observed_width_units"] > 0


def test_geometry_pipeline_unscaled_relative_depth():
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    result = make_result(detections=[make_bed_detection()], metric=False)
    plan = GeometryPipeline.from_config(make_config(scale={"mode": "auto"})).run(image, result)
    assert plan.scale_mode == "unscaled"
    assert plan.scale == pytest.approx(1.0)
    assert any("relative" in warning for warning in plan.metadata["warnings"])


def test_geometry_pipeline_inverts_inverse_relative_depth():
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    camera = make_camera()
    true_depth = make_depth(camera)
    inverse = np.zeros_like(true_depth)
    positive = true_depth > 0
    inverse[positive] = 1.0 / true_depth[positive]
    depth_map = DepthMap(inverse, metric=False, source="synthetic-inverse")
    result = make_result(detections=[make_bed_detection()], metric=False, depth_map=depth_map)
    plan = GeometryPipeline.from_config(make_config(scale={"mode": "auto"})).run(image, result)
    assert plan.scale_mode == "unscaled"
    assert plan.camera_height > 0
    assert len(plan.room.furniture) == 1
    assert any("inverse-depth" in warning for warning in plan.metadata["warnings"])


def test_geometry_pipeline_requires_depth():
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    result = PerceptionResult(detections=[Detection("bed", 0.9, (10, 10, 20, 20))], image_size=(WIDTH, HEIGHT))
    with pytest.raises(ValueError):
        GeometryPipeline.from_config(make_config()).run(image, result)


def test_geometry_pipeline_rejects_size_mismatch():
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    with pytest.raises(ValueError):
        GeometryPipeline.from_config(make_config()).run(image, make_result())
