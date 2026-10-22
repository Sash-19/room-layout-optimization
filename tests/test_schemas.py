import numpy as np
import pytest

from src.schemas import DepthMap, Detection, StructureMasks


def test_detection_properties():
    detection = Detection("bed", 0.9, (10, 20, 30, 60))
    assert detection.width == 20
    assert detection.height == 40
    assert detection.area == 800
    assert detection.center == (20, 40)


def test_detection_rejects_invalid_input():
    with pytest.raises(ValueError):
        Detection("bed", 1.2, (0, 0, 10, 10))
    with pytest.raises(ValueError):
        Detection("bed", 0.5, (10, 0, 0, 10))
    with pytest.raises(ValueError):
        Detection("bed", 0.5, (0, 0, float("nan"), 10))


def test_detection_clip_and_iou():
    detection = Detection("chair", 0.8, (-5, -5, 50, 50))
    clipped = detection.clip(40, 40)
    assert clipped.bbox == (0.0, 0.0, 40.0, 40.0)
    other = Detection("chair", 0.7, (20, 20, 60, 60))
    assert detection.iou(other) == pytest.approx(900 / 3725, rel=1e-6)


def test_detection_roundtrip():
    detection = Detection("window", 0.55, (1, 2, 3, 4))
    restored = Detection.from_dict(detection.to_dict())
    assert restored.cls_name == "window"
    assert restored.bbox == detection.bbox
    assert restored.conf == detection.conf


def test_depth_map_summary():
    depth = DepthMap(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), metric=True, source="test")
    summary = depth.to_dict()
    assert summary["metric"] is True
    assert summary["width"] == 2
    assert summary["stats"]["max"] == 4.0


def test_structure_masks_validation():
    floor = np.zeros((4, 4), dtype=bool)
    wall = np.ones((4, 4), dtype=bool)
    masks = StructureMasks(floor, wall, source="test")
    assert masks.to_dict()["wall_fraction"] == 1.0
    with pytest.raises(ValueError):
        StructureMasks(np.zeros((4, 4)), np.zeros((3, 4)))
