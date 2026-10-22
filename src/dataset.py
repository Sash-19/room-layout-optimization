from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from PIL import Image

from .perception.taxonomy import UNIFIED_CLASSES, resolve_class_name

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


@dataclass(slots=True)
class LabeledSample:
    image_path: Path
    labels: list[tuple[str, list[float]]]
    source: str = ""
    split: str | None = None


def coco_xywh_to_yolo(bbox: Sequence[float], width: int, height: int) -> list[float] | None:
    x, y, w, h = (float(value) for value in bbox)
    if width <= 0 or height <= 0 or not np.all(np.isfinite([x, y, w, h])) or w <= 0 or h <= 0:
        return None
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return [
        (x1 + x2) / 2.0 / width,
        (y1 + y2) / 2.0 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


def _clip01(values: Sequence[float]) -> list[float]:
    return [float(min(1.0, max(0.0, value))) for value in values]


def remap_yolo_line(
    fields: Sequence[str],
    source_names: Sequence[str],
    class_map: dict[str, str | None] | None,
) -> tuple[str, list[float]] | None:
    if len(fields) < 5:
        return None
    class_id = int(float(fields[0]))
    if class_id < 0 or class_id >= len(source_names):
        return None
    name = resolve_class_name(source_names[class_id], class_map)
    if name is None:
        return None
    try:
        coords = [float(value) for value in fields[1:]]
    except ValueError:
        return None
    if not np.all(np.isfinite(coords)):
        return None
    if len(coords) == 4:
        if coords[2] <= 0 or coords[3] <= 0:
            return None
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        if len(coords) // 2 < 3:
            return None
    else:
        return None
    return name, _clip01(coords)


def _names_list(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    return [str(name) for name in names]


def _read_root_data_yaml(root: Path) -> dict[str, Any]:
    path = root / "data.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def collect_coco_source(
    images_dir: str | Path,
    annotations: str | Path,
    class_map: dict[str, str | None] | None = None,
    source_name: str = "coco",
    verify_images: bool = False,
) -> list[LabeledSample]:
    images_dir = Path(images_dir)
    with Path(annotations).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
    images = {int(item["id"]): item for item in payload.get("images", [])}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        grouped[int(annotation["image_id"])].append(annotation)

    samples: list[LabeledSample] = []
    for image_id, annotations in grouped.items():
        info = images.get(image_id)
        if info is None:
            continue
        image_path = images_dir / str(info["file_name"])
        if not image_path.exists():
            continue
        width, height = int(info.get("width", 0)), int(info.get("height", 0))
        if width <= 0 or height <= 0:
            if verify_images:
                with Image.open(image_path) as handle:
                    width, height = handle.size
            else:
                continue
        labels: list[tuple[str, list[float]]] = []
        for annotation in annotations:
            category = categories.get(int(annotation.get("category_id", -1)))
            if category is None:
                continue
            name = resolve_class_name(category, class_map)
            if name is None:
                continue
            coords = coco_xywh_to_yolo(annotation.get("bbox", []), width, height)
            if coords is not None:
                labels.append((name, coords))
        if labels:
            samples.append(LabeledSample(image_path, labels, source_name))
    return samples


def collect_yolo_source(
    root: str | Path,
    class_map: dict[str, str | None] | None = None,
    source_name: str = "yolo",
    names: Sequence[str] | None = None,
    split: str | None = None,
) -> list[LabeledSample]:
    root = Path(root)
    root_data = _read_root_data_yaml(root)
    source_names = _names_list(names) if names else _names_list(root_data.get("names", []))
    if not source_names:
        raise ValueError(f"no class names found for YOLO source {root}; provide `names` in the config")

    images_dir = root / "images"
    labels_dir = root / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")

    samples: list[LabeledSample] = []
    image_paths = sorted(
        path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in image_paths:
        relative = image_path.relative_to(images_dir)
        label_path = (labels_dir / relative).with_suffix(".txt")
        if not label_path.exists():
            continue
        labels: list[tuple[str, list[float]]] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            remapped = remap_yolo_line(line.split(), source_names, class_map)
            if remapped is not None:
                labels.append(remapped)
        if labels:
            samples.append(LabeledSample(image_path, labels, source_name, split))
    return samples


def load_sources(config: dict[str, Any], verify_images: bool = False) -> list[LabeledSample]:
    samples: list[LabeledSample] = []
    for source in config.get("sources", []) or []:
        source_type = str(source.get("type", "")).lower()
        name = str(source.get("name", source_type or "source"))
        class_map = source.get("class_map") or None
        if source_type == "coco":
            samples.extend(
                collect_coco_source(
                    images_dir=source["images"],
                    annotations=source["annotations"],
                    class_map=class_map,
                    source_name=name,
                    verify_images=verify_images,
                )
            )
        elif source_type == "yolo":
            samples.extend(
                collect_yolo_source(
                    root=source["root"],
                    class_map=class_map,
                    source_name=name,
                    names=source.get("names"),
                    split=source.get("split"),
                )
            )
        else:
            raise ValueError(f"unsupported source type: {source_type!r}")
    return samples


def split_counts(count: int, ratios: dict[str, float]) -> dict[str, int]:
    if count <= 0:
        return {split: 0 for split in SPLITS}
    train_ratio = float(ratios.get("train", 0.7))
    val_ratio = float(ratios.get("val", 0.15))
    test_ratio = float(ratios.get("test", 0.15))
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total
    if count == 1:
        return {"train": 1, "val": 0, "test": 0}
    train_count = max(1, min(count - 1, int(round(count * train_ratio))))
    remaining = count - train_count
    if remaining == 0:
        return {"train": count, "val": 0, "test": 0}
    if remaining == 1:
        return {"train": train_count, "val": 1, "test": 0}
    val_count = max(1, min(remaining - 1, int(round(remaining * val_ratio / (val_ratio + test_ratio)))))
    test_count = remaining - val_count
    return {"train": train_count, "val": val_count, "test": test_count}


def assign_splits(
    samples: Sequence[LabeledSample],
    ratios: dict[str, float],
    seed: int = 0,
) -> dict[str, list[LabeledSample]]:
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    counts = split_counts(len(ordered), ratios)
    splits: dict[str, list[LabeledSample]] = {}
    cursor = 0
    for split in SPLITS:
        splits[split] = ordered[cursor : cursor + counts[split]]
        cursor += counts[split]
    return splits


def keep_source_splits(samples: Sequence[LabeledSample]) -> dict[str, list[LabeledSample]]:
    splits: dict[str, list[LabeledSample]] = {split: [] for split in SPLITS}
    for sample in samples:
        split = sample.split if sample.split in SPLITS else "train"
        splits[split].append(sample)
    return splits


def _transfer_image(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if copy_mode == "link":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def write_dataset(
    splits: dict[str, list[LabeledSample]],
    output: str | Path,
    class_names: Sequence[str] = UNIFIED_CLASSES,
    copy_mode: str = "copy",
    verify_images: bool = False,
) -> dict[str, Any]:
    if copy_mode not in {"copy", "link"}:
        raise ValueError("copy_mode must be 'copy' or 'link'")
    output = Path(output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = {split: Counter() for split in SPLITS}
    index_lookup = {name: index for index, name in enumerate(class_names)}
    index = 0
    written = 0
    skipped = 0

    for split in SPLITS:
        for sample in splits.get(split, []):
            if verify_images:
                try:
                    with Image.open(sample.image_path) as handle:
                        handle.verify()
                except Exception:
                    skipped += 1
                    continue
            digest = hashlib.sha1(str(sample.image_path.resolve()).encode("utf-8")).hexdigest()[:8]
            suffix = sample.image_path.suffix.lower() or ".jpg"
            stem = f"{index:07d}_{digest}"
            index += 1
            destination = output / "images" / split / f"{stem}{suffix}"
            label_destination = output / "labels" / split / f"{stem}.txt"
            _transfer_image(sample.image_path, destination, copy_mode)
            lines = []
            sample_counts: Counter[str] = Counter()
            for name, coords in sample.labels:
                if name not in index_lookup:
                    continue
                lines.append(f"{index_lookup[name]} " + " ".join(f"{value:.6f}" for value in coords))
                sample_counts[name] += 1
            label_destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts[split].update(sample_counts)
            written += 1

    manifest = {
        "output": str(output.resolve()),
        "num_images": written,
        "num_skipped": skipped,
        "classes": list(class_names),
        "splits": {split: dict(counts[split]) for split in SPLITS},
    }
    means = [sum(counts[split].values()) for split in SPLITS]
    manifest["instances"] = dict(zip(SPLITS, means))

    data_yaml = {
        "path": str(output.resolve()),
        "train": f"images/{SPLITS[0]}",
        "val": f"images/{SPLITS[1]}",
        "test": f"images/{SPLITS[2]}",
        "nc": len(class_names),
        "names": list(class_names),
    }
    with (output / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest
