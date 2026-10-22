import numpy as np
import pytest
import yaml
from PIL import Image

from src.perception.evaluation import (
    average_precision,
    evaluate_detector,
    match_image,
    precision_recall_f1,
    read_yolo_labels,
)
from src.schemas import Detection


def test_match_image_greedy_matching():
    gt = [Detection("bed", 1.0, (0, 0, 10, 10))]
    detections = [
        Detection("bed", 0.9, (0, 0, 10, 10)),
        Detection("bed", 0.8, (50, 50, 60, 60)),
    ]
    records = match_image(gt, detections, iou_threshold=0.5)
    assert records["bed"] == [(0.9, True), (0.8, False)]


def test_match_image_never_reuses_ground_truth():
    gt = [Detection("chair", 1.0, (0, 0, 10, 10))]
    detections = [Detection("chair", 0.9, (0, 0, 10, 10)), Detection("chair", 0.85, (0, 0, 10, 10))]
    records = match_image(gt, detections, iou_threshold=0.5)
    assert [matched for _, matched in records["chair"]] == [True, False]


def test_average_precision_perfect_and_with_false_positive():
    assert average_precision([(0.9, True), (0.8, True)], 2) == pytest.approx(1.0)
    assert average_precision([(0.95, False), (0.9, True)], 1) == pytest.approx(0.5)
    assert average_precision([], 2) == 0.0
    assert average_precision([(0.5, True)], 0) is None


def test_precision_recall_f1():
    records = [(0.9, True), (0.8, True), (0.7, False)]
    precision, recall, f1 = precision_recall_f1(records, 3, conf_threshold=0.5)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)
    assert precision_recall_f1(records, 3, conf_threshold=0.75) == pytest.approx((1.0, 2 / 3, 0.8))


def test_read_yolo_labels_boxes_and_polygons(tmp_path):
    label_path = tmp_path / "labels.txt"
    label_path.write_text(
        "0 0.5 0.5 0.4 0.2\n"
        "1 0.1 0.1 0.4 0.1 0.4 0.4\n"
        "2 0.5 0.5 0.0 0.2\n",
        encoding="utf-8",
    )
    labels = read_yolo_labels(label_path, ["bed", "chair", "couch"], 100, 200)
    assert len(labels) == 2
    assert labels[0].bbox == pytest.approx((30, 80, 70, 120))
    assert labels[1].bbox == pytest.approx((10, 20, 40, 80))


class ConstantDetector:
    def __init__(self, box):
        self.box = box

    def detect(self, image_path):
        return [Detection("bed", 0.9, self.box)]


def test_evaluate_detector_end_to_end(tmp_path):
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (20, 20), (0, 0, 0)).save(root / "images" / "val" / f"{index}.jpg")
        (root / "labels" / "val" / f"{index}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 8,
        "names": ["bed", "desk", "chair", "couch", "table", "cabinet", "door", "window"],
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    result = evaluate_detector(ConstantDetector((5, 5, 15, 15)), root, split="val", conf=0.001)
    assert result.num_images == 2
    assert result.num_ground_truth == 2
    assert result.map50 == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.per_class["bed"]["ap50"] == pytest.approx(1.0)
    assert result.per_class["desk"]["ap50"] is None
