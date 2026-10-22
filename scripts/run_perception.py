from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.pipeline import PerceptionPipeline
from src.perception.visualize import colorize_depth, draw_detections, draw_structure_overlay, save_mask
from src.utils import ensure_dir, load_image, load_yaml, parse_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the perception stage on room photographs")
    parser.add_argument("--config", default="configs/perception.yaml")
    parser.add_argument("--image", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-structure", action="store_true")
    return parser.parse_args()


def save_outputs(image: Image.Image, result, out_dir: Path, source: Path) -> None:
    ensure_dir(out_dir)
    image.save(out_dir / "input.jpg", quality=95)
    draw_detections(image, result.detections).save(out_dir / "detections.jpg", quality=95)
    save_json(
        {"image": str(source), "image_size": result.image_size, "detections": [d.to_dict() for d in result.detections]},
        out_dir / "detections.json",
    )
    if result.depth is not None:
        colorize_depth(result.depth.depth).save(out_dir / "depth.png")
        with (out_dir / "depth.npy").open("wb") as handle:
            np.save(handle, result.depth.depth)
    if result.structures is not None:
        save_mask(result.structures.floor, out_dir / "floor_mask.png")
        save_mask(result.structures.wall, out_dir / "wall_mask.png")
        draw_structure_overlay(image, result.structures).save(out_dir / "structures.jpg", quality=95)
    payload = result.to_dict()
    payload["image"] = str(source)
    save_json(payload, out_dir / "result.json")


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    detector_config = dict(config.get("detector", {}))
    depth_config = dict(config.get("depth", {}))
    structure_config = dict(config.get("structure", {}))
    if args.weights:
        detector_config["weights"] = args.weights
    if args.conf is not None:
        detector_config["conf"] = args.conf
    device = parse_device(args.device)
    if device is not None:
        detector_config["device"] = device
        depth_config["device"] = device
        structure_config["device"] = device
    config["detector"] = detector_config
    config["depth"] = depth_config
    config["structure"] = structure_config

    pipeline = PerceptionPipeline.from_config(
        config,
        include_depth=not args.no_depth,
        include_structure=not args.no_structure,
    )

    base_out = Path(args.out or config.get("output", {}).get("dir", "results/perception"))
    for image_arg in args.image:
        source = Path(image_arg)
        if not source.exists():
            print(f"Image not found: {source}", file=sys.stderr)
            continue
        image = load_image(source)
        result = pipeline.run(image)
        out_dir = base_out / source.stem
        save_outputs(image, result, out_dir, source)
        counts = result.class_counts()
        summary = ", ".join(f"{name}={count}" for name, count in counts.items()) or "none"
        print(f"{source.name}: {len(result.detections)} detections ({summary})")
        if result.depth is not None:
            print(f"  depth: {result.depth.width}x{result.depth.height} ({'metric' if result.depth.metric else 'relative'})")
        if result.structures is not None:
            print(
                f"  structure: floor={result.structures.to_dict()['floor_fraction']:.3f}, "
                f"wall={result.structures.to_dict()['wall_fraction']:.3f}"
            )
        print(f"  wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
