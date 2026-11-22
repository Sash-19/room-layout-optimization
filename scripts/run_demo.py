from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import LayoutPipeline, load_config
from src.utils import ensure_dir, save_json
from src.viz.compare import save_comparison_grid

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full pipeline as a demo over a folder of images")
    parser.add_argument("--images", default="results/actual/perception")
    parser.add_argument("--perception-dir", default="results/actual/perception")
    parser.add_argument("--out", default="results/demo")
    parser.add_argument("--algorithm", choices=("random", "greedy", "annealing", "genetic"), default="greedy")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--room-width", type=float, default=None)
    parser.add_argument("--room-depth", type=float, default=None)
    parser.add_argument("--scale-mode", choices=("auto", "metric", "user", "reference"), default=None)
    parser.add_argument("--perception-config", default="configs/perception.yaml")
    parser.add_argument("--geometry-config", default="configs/geometry.yaml")
    parser.add_argument("--scoring-config", default="configs/scoring.yaml")
    parser.add_argument("--optimizer-config", default="configs/optimizer.yaml")
    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    candidates = [item for item in root.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES]
    inputs = [item for item in candidates if item.stem == "input"]
    return sorted(inputs or candidates)


def format_markdown(results: list[dict]) -> str:
    lines = [
        "# End-to-End Demo",
        "",
        "| Room | Objects | Detected score | Optimized score | Delta | Feasible | Output |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['room']} | {row['objects']} | {row['initial_score']:.4f} | {row['best_score']:.4f} | "
            f"{row['improvement']:+.4f} | {'yes' if row['feasible'] else 'no'} | "
            f"[{row['output_dir']}]({Path(row['output_dir']).as_posix()}) |"
        )
    if results:
        lines.extend(
            [
                "",
                f"Mean detected score: {mean(row['initial_score'] for row in results):.4f}",
                "",
                f"Mean optimized score: {mean(row['best_score'] for row in results):.4f}",
                "",
                f"Improved: {sum(1 for row in results if row['improvement'] > 1e-9)}/{len(results)}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    images = collect_images(Path(args.images))
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        print("no images found", file=sys.stderr)
        return 2
    config = load_config(
        args.perception_config,
        args.geometry_config,
        args.scoring_config,
        args.optimizer_config,
    )
    pipeline = LayoutPipeline.from_config(config, algorithm=args.algorithm)
    perception_root = Path(args.perception_dir)
    out_root = ensure_dir(args.out)

    records: list[dict] = []
    grid_items = []
    for image_path in images:
        room = image_path.parent.name if image_path.stem == "input" else image_path.stem
        cached = perception_root / room / "result.json"
        perception = None
        if cached.is_file():
            from src.perception.pipeline import PerceptionResult

            perception = PerceptionResult.from_cache(cached.parent)
        result = pipeline.run(
            image_path,
            output_dir=out_root / room,
            perception=perception,
            room_width_m=args.room_width,
            room_depth_m=args.room_depth,
            scale_mode=args.scale_mode,
        )
        record = {
            "room": room,
            "objects": len(result.floorplan.room.furniture),
            "initial_score": round(result.initial_score.total, 6),
            "best_score": round(result.best_score.total, 6),
            "improvement": round(result.improvement, 6),
            "feasible": result.best_score.feasible,
            "output_dir": str(result.output_dir),
        }
        records.append(record)
        grid_items.append((room, result.best_plan, result.best_score))
        print(
            f"{room}: {record['initial_score']:.4f} -> {record['best_score']:.4f} "
            f"({record['improvement']:+.4f}) feasible={record['feasible']}"
        )

    save_json({"algorithm": args.algorithm, "records": records}, out_root / "demo.json")
    markdown = format_markdown(records)
    (out_root / "demo.md").write_text(markdown, encoding="utf-8")
    save_comparison_grid(grid_items[:9], out_root / "demo_grid.png", columns=3)
    print()
    print(markdown)
    print(f"wrote {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())