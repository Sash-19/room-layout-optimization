from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
from PIL import Image

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry.pipeline import GeometryPipeline
from src.geometry.visualize import save_floorplan
from src.perception.cache import load_cached_perception
from src.perception.pipeline import PerceptionPipeline, PerceptionResult
from src.utils import ensure_dir, load_image, load_yaml, parse_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lift detections to a metric 2D floor plan")
    parser.add_argument("--config", default="configs/geometry.yaml")
    parser.add_argument("--perception-config", default="configs/perception.yaml")
    parser.add_argument("--image", nargs="+", default=None)
    parser.add_argument("--perception-dir", nargs="+", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--scale-mode", choices=("auto", "metric", "reference", "user"), default=None)
    parser.add_argument("--room-width", type=float, default=None)
    parser.add_argument("--room-depth", type=float, default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-structure", action="store_true")
    return parser.parse_args()


def build_perception_pipeline(args: argparse.Namespace) -> PerceptionPipeline:
    config = load_yaml(args.perception_config)
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
    return PerceptionPipeline.from_config(
        config,
        include_depth=True,
        include_structure=not args.no_structure,
    )


def main() -> int:
    args = parse_args()
    if bool(args.image) == bool(args.perception_dir):
        print("Provide either --image or --perception-dir (not both).", file=sys.stderr)
        return 2

    geometry_config = load_yaml(args.config)
    pipeline = GeometryPipeline.from_config(geometry_config)
    perception_pipeline = build_perception_pipeline(args) if args.image else None
    base_out = Path(args.out or "results/geometry")

    items: list[tuple[str, Image.Image, PerceptionResult]] = []
    if args.image:
        for image_arg in args.image:
            source = Path(image_arg)
            if not source.exists():
                print(f"Image not found: {source}", file=sys.stderr)
                continue
            image = load_image(source)
            assert perception_pipeline is not None
            items.append((source.stem, image, perception_pipeline.run(image)))
    else:
        for directory in args.perception_dir or []:
            path = Path(directory)
            if not (path / "input.jpg").exists():
                print(f"Perception directory missing input.jpg: {path}", file=sys.stderr)
                continue
            image, result = load_cached_perception(path)
            items.append((path.name, image, result))

    if not items:
        print("No inputs processed.", file=sys.stderr)
        return 1

    for name, image, perception in items:
        if perception.depth is None:
            print(f"{name}: skipped, no depth map (geometry requires depth).", file=sys.stderr)
            continue
        floorplan = pipeline.run(
            image,
            perception,
            room_width_m=args.room_width,
            room_depth_m=args.room_depth,
            scale_mode=args.scale_mode,
        )
        out_dir = ensure_dir(base_out / name)
        save_json(floorplan.to_dict(), out_dir / "floorplan.json")
        save_floorplan(floorplan, out_dir / "floorplan.png")
        room = floorplan.room
        furniture_counts: dict[str, int] = {}
        for item in room.furniture:
            furniture_counts[item.cls] = furniture_counts.get(item.cls, 0) + 1
        opening_counts: dict[str, int] = {}
        for opening in room.openings:
            opening_counts[opening.kind] = opening_counts.get(opening.kind, 0) + 1
        print(
            f"{name}: room {room.width:.2f}x{room.depth:.2f}, area {room.area:.2f}, "
            f"scale={floorplan.scale:.3g} ({floorplan.scale_mode}), camera height {floorplan.camera_height:.2f}"
        )
        print(f"  furniture: {furniture_counts or 'none'}")
        print(f"  openings: {opening_counts or 'none'}")
        warnings = floorplan.metadata.get("warnings", [])
        for warning in warnings:
            print(f"  warning: {warning}")
        print(f"  wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
