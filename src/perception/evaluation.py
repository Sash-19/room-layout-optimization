from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
import yaml
from PIL import Image

from ..schemas import Detection

if TYPE_CHECKING:
    from .detector import Detector

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(slots=True)
class EvalResult:
    split: str
    num_images: int
    num_ground_truth: int
    num_detections: int
    map50: float
    map50_95: float
    precision: float
    recall: float
    f1: float
    per_class: dict[str, dict[str, Any]] = field(default_factory=dict)
    iou_thresholds: list[float] = field(default_factory=list)
    report_conf: float = 0.25
    data_root: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "data_root": self.data_root,
            "num_images": self.num_images,
            "num_ground_truth": self.num_ground_truth,
            "num_detections": self.num_detections,
            "map50": round(self.map50, 4),
            "map50_95": round(self.map50_95, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "report_conf": self.report_conf,
            "iou_thresholds": [round(v, 3) for v in self.iou_thresholds],
            "per_class": self.per_class,
        }


def _names_from_data(data: dict[str, Any]) -> list[str]:
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError("data.yaml must contain `names` as a list or mapping")


def load_data_yaml(data_root: str | Path) -> tuple[list[str], dict[str, Any]]:
    path = Path(data_root) / "data.yaml"
    if not path.exists():
        raise FileNotFoundError(f"data.yaml not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _names_from_data(data), data


def dataset_base(data_root: str | Path, data: dict[str, Any]) -> Path:
    root = Path(data_root)
    value = data.get("path")
    if value is None:
        return root
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_split_dir(data_root: str | Path, data: dict[str, Any], split: str, kind: str) -> Path:
    root = Path(data_root)
    base = dataset_base(root, data)
    value = data.get(split, f"images/{split}")
    images_path = Path(value)
    if not images_path.is_absolute():
        images_path = base / images_path
    if kind == "images":
        return images_path
    try:
        relative = images_path.relative_to(base / "images")
        return base / "labels" / relative
    except ValueError:
        return base / "labels" / split


def iter_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    relative = image_path.relative_to(images_dir)
    return (labels_dir / relative).with_suffix(".txt")


def read_yolo_labels(
    label_path: str | Path,
    names: Sequence[str],
    width: int,
    height: int,
) -> list[Detection]:
    path = Path(label_path)
    if not path.exists():
        return []
    detections: list[Detection] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(float(fields[0]))
        if class_id < 0 or class_id >= len(names):
            continue
        coords = [float(value) for value in fields[1:]]
        if len(coords) == 4:
            cx, cy, w, h = coords
            x1, y1 = (cx - w / 2.0) * width, (cy - h / 2.0) * height
            x2, y2 = (cx + w / 2.0) * width, (cy + h / 2.0) * height
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs = [value * width for value in coords[0::2]]
            ys = [value * height for value in coords[1::2]]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        detection = Detection(names[class_id], 1.0, (x1, y1, x2, y2)).clip(width, height)
        if detection.area > 0:
            detections.append(detection)
    return detections


def match_image(
    ground_truth: Sequence[Detection],
    detections: Sequence[Detection],
    iou_threshold: float = 0.5,
) -> dict[str, list[tuple[float, bool]]]:
    classes = {item.cls_name for item in ground_truth} | {item.cls_name for item in detections}
    records: dict[str, list[tuple[float, bool]]] = {name: [] for name in classes}
    for name in classes:
        gts = [item for item in ground_truth if item.cls_name == name]
        dets = sorted((item for item in detections if item.cls_name == name), key=lambda item: -item.conf)
        used = [False] * len(gts)
        for detection in dets:
            best_iou = 0.0
            best_index = -1
            for index, gt in enumerate(gts):
                if used[index]:
                    continue
                iou = detection.iou(gt)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            matched = best_index >= 0 and best_iou >= iou_threshold
            if matched:
                used[best_index] = True
            records[name].append((detection.conf, matched))
    return records


def average_precision(records: Sequence[tuple[float, bool]], num_ground_truth: int) -> float | None:
    if num_ground_truth <= 0:
        return None
    if not records:
        return 0.0
    ordered = sorted(records, key=lambda item: -item[0])
    tp = np.cumsum([1.0 if item[1] else 0.0 for item in ordered])
    fp = np.cumsum([0.0 if item[1] else 1.0 for item in ordered])
    recall = tp / float(num_ground_truth)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    indices = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1]))


def precision_recall_f1(
    records: Sequence[tuple[float, bool]],
    num_ground_truth: int,
    conf_threshold: float,
) -> tuple[float, float, float]:
    true_positive = sum(1 for conf, matched in records if conf >= conf_threshold and matched)
    false_positive = sum(1 for conf, matched in records if conf >= conf_threshold and not matched)
    false_negative = max(0, num_ground_truth - true_positive)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def evaluate_detector(
    detector: Detector,
    data_root: str | Path,
    split: str = "test",
    conf: float = 0.001,
    report_conf: float = 0.25,
    iou_thresholds: Sequence[float] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
    limit: int | None = None,
) -> EvalResult:
    root = Path(data_root)
    names, data = load_data_yaml(root)
    images_dir = resolve_split_dir(root, data, split, "images")
    labels_dir = resolve_split_dir(root, data, split, "labels")
    images = iter_images(images_dir)
    if limit is not None:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"no images found for split '{split}' in {images_dir}")

    thresholds = tuple(sorted({float(value) for value in iou_thresholds}))
    reference_threshold = 0.5 if 0.5 in thresholds else thresholds[len(thresholds) // 2]
    records: dict[float, dict[str, list[tuple[float, bool]]]] = {
        threshold: {name: [] for name in names} for threshold in thresholds
    }
    ground_truth_counts = {name: 0 for name in names}
    detection_counts = {name: 0 for name in names}
    total_detections = 0

    for image_path in images:
        with Image.open(image_path) as handle:
            width, height = handle.size
        label_path = label_path_for(image_path, images_dir, labels_dir)
        ground_truth = read_yolo_labels(label_path, names, width, height)
        for item in ground_truth:
            ground_truth_counts[item.cls_name] = ground_truth_counts.get(item.cls_name, 0) + 1
        detections = [
            item
            for item in detector.detect(image_path)
            if item.conf >= conf and item.cls_name in ground_truth_counts
        ]
        for item in detections:
            if item.conf >= report_conf:
                detection_counts[item.cls_name] = detection_counts.get(item.cls_name, 0) + 1
                total_detections += 1
        for threshold in thresholds:
            matched = match_image(ground_truth, detections, threshold)
            for name, values in matched.items():
                records[threshold][name].extend(values)

    aps: dict[float, dict[str, float | None]] = {
        threshold: {
            name: average_precision(records[threshold][name], ground_truth_counts[name]) for name in names
        }
        for threshold in thresholds
    }

    per_class: dict[str, dict[str, Any]] = {}
    for name in names:
        ap_values = [aps[threshold][name] for threshold in thresholds if aps[threshold][name] is not None]
        ap50 = aps[0.5][name] if 0.5 in aps else aps[reference_threshold][name]
        precision, recall, f1 = precision_recall_f1(
            records[reference_threshold][name], ground_truth_counts[name], report_conf
        )
        per_class[name] = {
            "ap50": None if ap50 is None else round(float(ap50), 4),
            "ap50_95": round(_mean(ap_values), 4) if ap_values else None,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "num_ground_truth": ground_truth_counts[name],
            "num_detections": detection_counts[name],
        }

    map50 = _mean(
        value for value in (aps[0.5] if 0.5 in aps else aps[reference_threshold]).values() if value is not None
    )
    map50_95 = _mean(
        aps[threshold][name]
        for threshold in thresholds
        for name in names
        if aps[threshold][name] is not None
    )
    overall_precision, overall_recall, overall_f1 = precision_recall_f1(
        [item for name in names for item in records[reference_threshold][name]],
        sum(ground_truth_counts.values()),
        report_conf,
    )

    return EvalResult(
        split=split,
        num_images=len(images),
        num_ground_truth=int(sum(ground_truth_counts.values())),
        num_detections=int(total_detections),
        map50=map50,
        map50_95=map50_95,
        precision=overall_precision,
        recall=overall_recall,
        f1=overall_f1,
        per_class=per_class,
        iou_thresholds=list(thresholds),
        report_conf=float(report_conf),
        data_root=str(root),
    )


def format_markdown(result: EvalResult) -> str:
    lines = [
        f"# Detection evaluation ({result.split})",
        "",
        f"- Images: {result.num_images}",
        f"- Ground-truth boxes: {result.num_ground_truth}",
        f"- Detections at conf >= {result.report_conf}: {result.num_detections}",
        f"- mAP@0.5: {result.map50:.4f}",
        f"- mAP@0.5:0.95: {result.map50_95:.4f}",
        f"- Precision: {result.precision:.4f}",
        f"- Recall: {result.recall:.4f}",
        f"- F1: {result.f1:.4f}",
        "",
        "| Class | AP@0.5 | AP@0.5:0.95 | Precision | Recall | F1 | GT | Detections |",
        "|-------|--------|-------------|-----------|--------|----|----|------------|",
    ]
    for name, values in result.per_class.items():
        ap50 = "n/a" if values["ap50"] is None else f"{values['ap50']:.4f}"
        ap50_95 = "n/a" if values["ap50_95"] is None else f"{values['ap50_95']:.4f}"
        lines.append(
            f"| {name} | {ap50} | {ap50_95} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1']:.4f} | {values['num_ground_truth']} | {values['num_detections']} |"
        )
    return "\n".join(lines) + "\n"
