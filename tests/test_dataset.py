import json

import pytest
import yaml
from PIL import Image

from src.dataset import (
    LabeledSample,
    assign_splits,
    coco_xywh_to_yolo,
    collect_coco_source,
    remap_yolo_line,
    split_counts,
    write_dataset,
)


def test_coco_xywh_to_yolo_and_clipping():
    assert coco_xywh_to_yolo([10, 20, 30, 40], 100, 200) == pytest.approx([0.25, 0.2, 0.3, 0.2])
    assert coco_xywh_to_yolo([90, 190, 30, 40], 100, 200) == pytest.approx([0.95, 0.975, 0.1, 0.05])
    assert coco_xywh_to_yolo([0, 0, 0, 10], 100, 200) is None
    assert coco_xywh_to_yolo([-20, -20, 10, 10], 100, 200) is None


def test_remap_yolo_line_handles_boxes_polygons_and_unknowns():
    names = ["Sofa", "Cupboard", "Car"]
    class_map = {"Sofa": "couch", "Cupboard": "cabinet"}
    assert remap_yolo_line(["0", "0.5", "0.5", "0.2", "0.2"], names, class_map) == (
        "couch",
        [0.5, 0.5, 0.2, 0.2],
    )
    polygon = remap_yolo_line(["1", "0.1", "0.1", "0.4", "0.1", "0.4", "0.4"], names, class_map)
    assert polygon is not None and polygon[0] == "cabinet" and len(polygon[1]) == 6
    assert remap_yolo_line(["2", "0.5", "0.5", "0.2", "0.2"], names, class_map) is None
    assert remap_yolo_line(["0", "0.5", "0.5", "0", "0.2"], names, class_map) is None
    assert remap_yolo_line(["0", "0.5", "0.5"], names, class_map) is None


def test_split_counts_are_deterministic_and_complete():
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    for count in range(1, 25):
        counts = split_counts(count, ratios)
        assert sum(counts.values()) == count
        assert counts["train"] >= 1
    assert split_counts(10, ratios) == {"train": 7, "val": 2, "test": 1}
    assert split_counts(0, ratios) == {"train": 0, "val": 0, "test": 0}


def test_assign_splits_is_seeded():
    samples = [LabeledSample(None, [("bed", [0.5, 0.5, 0.2, 0.2])], "x") for _ in range(12)]
    first = assign_splits(samples, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=3)
    second = assign_splits(samples, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=3)
    assert [len(first[split]) for split in ("train", "val", "test")] == [8, 2, 2]
    assert [id(item) for item in first["train"]] == [id(item) for item in second["train"]]


def test_write_dataset_produces_yolo_structure(tmp_path):
    images = tmp_path / "src"
    images.mkdir()
    for index, color in enumerate([(255, 0, 0), (0, 255, 0)]):
        Image.new("RGB", (20, 10), color).save(images / f"room{index}.jpg")
    samples = {
        "train": [
            LabeledSample(images / "room0.jpg", [("bed", [0.5, 0.5, 0.4, 0.4])], "test"),
            LabeledSample(images / "room1.jpg", [("window", [0.5, 0.5, 0.2, 0.2])], "test"),
        ],
        "val": [],
        "test": [],
    }
    output = tmp_path / "processed"
    manifest = write_dataset(samples, output)

    assert manifest["num_images"] == 2
    assert manifest["instances"] == {"train": 2, "val": 0, "test": 0}
    data_yaml = yaml.safe_load((output / "data.yaml").read_text())
    assert data_yaml["nc"] == 8
    assert data_yaml["names"][0] == "bed"
    assert data_yaml["names"][7] == "window"
    assert len(list((output / "images" / "train").iterdir())) == 2
    label_files = list((output / "labels" / "train").glob("*.txt"))
    assert len(label_files) == 2
    classes = {path.read_text().split()[0] for path in label_files}
    assert classes == {"0", "7"}


def test_collect_coco_source(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (10, 10), (0, 0, 0)).save(images / "a.jpg")
    payload = {
        "categories": [{"id": 1, "name": "sofa"}, {"id": 2, "name": "car"}],
        "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 5, 5]},
        ],
    }
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(payload))
    samples = collect_coco_source(images, annotations)
    assert len(samples) == 1
    assert samples[0].labels[0][0] == "couch"
    assert samples[0].labels[0][1] == pytest.approx([0.3, 0.3, 0.4, 0.4])
