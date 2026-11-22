from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.taxonomy import UNIFIED_CLASSES
from src.utils import ensure_dir, save_json

ANNOTATIONS_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
CLASS_MAP = {
    "/m/03ssj5": "bed",
    "/m/01mzpv": "chair",
    "/m/02crq1": "couch",
    "/m/0642b4": "cabinet",
    "/m/01s105": "cabinet",
    "/m/01y9k5": "desk",
    "/m/02dgv": "door",
    "/m/04bcr3": "table",
    "/m/0d4v4": "window",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small Open Images room subset")
    parser.add_argument("--out", default="data/raw/openimages")
    parser.add_argument("--max-images", type=int, default=400)
    parser.add_argument("--min-per-class", type=int, default=40)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def download_file(url: str, path: Path, timeout: float, retries: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
            return
        except Exception:
            if path.exists():
                path.unlink()
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def load_annotations(
    path: Path,
    timeout: float,
    confidence: float,
) -> tuple[dict[str, list[tuple[str, list[float]]]], dict[str, set[str]]]:
    if not path.exists():
        print(f"downloading annotations -> {path}")
        download_file(ANNOTATIONS_URL, path, timeout)
    labels: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    classes: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label_id = row.get("LabelName", "")
            cls = CLASS_MAP.get(label_id)
            if cls is None:
                continue
            try:
                if float(row.get("Confidence", 0.0)) < confidence:
                    continue
                x_min = float(row["XMin"])
                x_max = float(row["XMax"])
                y_min = float(row["YMin"])
                y_max = float(row["YMax"])
            except (KeyError, TypeError, ValueError):
                continue
            width = x_max - x_min
            height = y_max - y_min
            if width <= 0 or height <= 0:
                continue
            image_id = row.get("ImageID", "")
            if not image_id:
                continue
            labels[image_id].append(
                (
                    cls,
                    [
                        (x_min + x_max) / 2.0,
                        (y_min + y_max) / 2.0,
                        width,
                        height,
                    ],
                )
            )
            classes[image_id].add(cls)
    return dict(labels), dict(classes)


def select_images(
    classes: dict[str, set[str]],
    max_images: int,
    min_per_class: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for image_id, image_classes in classes.items():
        for cls in image_classes:
            by_class[cls].append(image_id)
    for values in by_class.values():
        rng.shuffle(values)

    selected: list[str] = []
    chosen: set[str] = set()

    def add(image_id: str) -> None:
        if image_id not in chosen and len(selected) < max_images:
            chosen.add(image_id)
            selected.append(image_id)

    for cls in UNIFIED_CLASSES:
        candidates = list(by_class.get(cls, []))
        count = sum(1 for image_id in selected if cls in classes[image_id])
        for image_id in candidates:
            if count >= min_per_class or len(selected) >= max_images:
                break
            if image_id not in chosen:
                add(image_id)
                count += 1

    multi = list(classes)
    rng.shuffle(multi)
    for image_id in multi:
        if len(selected) >= max_images:
            break
        add(image_id)

    remaining = [image_id for image_id in classes if image_id not in chosen]
    rng.shuffle(remaining)
    for image_id in remaining:
        if len(selected) >= max_images:
            break
        add(image_id)
    return selected


def write_yolo_labels(
    image_id: str,
    labels: list[tuple[str, list[float]]],
    out_dir: Path,
) -> None:
    lines = []
    for cls, coords in labels:
        if cls not in UNIFIED_CLASSES:
            continue
        index = UNIFIED_CLASSES.index(cls)
        lines.append(f"{index} " + " ".join(f"{value:.6f}" for value in coords))
    (out_dir / "labels" / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out)
    annotations_path = out_dir / "annotations" / "validation-annotations-bbox.csv"
    labels, classes = load_annotations(annotations_path, args.timeout, args.confidence)
    if not labels:
        print("no matching Open Images annotations found", file=sys.stderr)
        return 1

    selected = select_images(classes, args.max_images, args.min_per_class, args.seed)
    counts = Counter(cls for image_id in selected for cls in classes[image_id])
    print(f"selected {len(selected)} images")
    print("class counts: " + ", ".join(f"{cls}={counts.get(cls, 0)}" for cls in UNIFIED_CLASSES))
    if args.dry_run:
        return 0

    ensure_dir(out_dir / "images")
    ensure_dir(out_dir / "labels")
    downloaded = 0
    failed: list[str] = []
    for index, image_id in enumerate(selected, start=1):
        image_path = out_dir / "images" / f"{image_id}.jpg"
        if not image_path.exists():
            try:
                download_file(IMAGE_URL.format(image_id=image_id), image_path, args.timeout)
            except Exception as error:
                failed.append(image_id)
                print(f"failed {image_id}: {error}", file=sys.stderr)
                continue
        write_yolo_labels(image_id, labels[image_id], out_dir)
        downloaded += 1
        if index % 25 == 0 or index == len(selected):
            print(f"  {index}/{len(selected)}")

    save_json(
        {
            "source": "Open Images V5 validation",
            "annotations": str(annotations_path),
            "selected_images": len(selected),
            "downloaded_images": downloaded,
            "failed_images": failed,
            "class_counts": dict(counts),
        },
        out_dir / "manifest.json",
    )
    (out_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images",
                "val: images",
                "test: images",
                "nc: 8",
                "names:",
                *[f"  - {name}" for name in UNIFIED_CLASSES],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {downloaded} images and labels to {out_dir}")
    if failed:
        print(f"{len(failed)} images failed; rerun to retry", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
