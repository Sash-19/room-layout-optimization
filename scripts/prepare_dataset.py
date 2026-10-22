from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import assign_splits, keep_source_splits, load_sources, write_dataset
from src.perception.taxonomy import UNIFIED_CLASSES
from src.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge COCO/YOLO sources into a unified YOLO dataset")
    parser.add_argument("--config", default="configs/detection.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--copy-mode", choices=("copy", "link"), default=None)
    parser.add_argument("--keep-splits", action="store_true")
    parser.add_argument("--verify-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    data_config = config.get("data", {})
    sources = data_config.get("sources") or []
    if not sources:
        print("No sources configured. Add entries under data.sources in configs/detection.yaml.", file=sys.stderr)
        return 2

    try:
        samples = load_sources(data_config, verify_images=bool(args.verify_images or data_config.get("verify_images")))
    except (FileNotFoundError, ValueError) as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        return 1
    if not samples:
        print("No labeled samples were found. Check the source paths and class maps.", file=sys.stderr)
        return 1

    keep_existing = bool(args.keep_splits or data_config.get("keep_source_splits", False))
    ratios = data_config.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    seed = int(args.seed if args.seed is not None else data_config.get("seed", 0))
    splits = keep_source_splits(samples) if keep_existing else assign_splits(samples, ratios, seed)

    output = Path(args.output or data_config.get("output", "data/processed"))
    copy_mode = args.copy_mode or data_config.get("copy_mode", "copy")
    manifest = write_dataset(
        splits,
        output,
        class_names=UNIFIED_CLASSES,
        copy_mode=copy_mode,
        verify_images=bool(args.verify_images or data_config.get("verify_images")),
    )

    print(f"Wrote {manifest['num_images']} images to {manifest['output']}")
    if manifest["num_skipped"]:
        print(f"Skipped {manifest['num_skipped']} unreadable images")
    for split in ("train", "val", "test"):
        instances = manifest["instances"][split]
        print(f"  {split}: {len(splits.get(split, []))} images, {instances} instances")
    print("Class counts:")
    for split in ("train", "val", "test"):
        counts = manifest["splits"][split]
        summary = ", ".join(f"{name}={counts[name]}" for name in UNIFIED_CLASSES if counts.get(name))
        print(f"  {split}: {summary or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
