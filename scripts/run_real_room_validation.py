from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.floorplan.representation import FloorPlan
from src.optimization import genetic_algorithm, greedy_search, random_search, simulated_annealing
from src.scoring.score import LayoutScorer
from src.utils import ensure_dir, load_yaml, save_json
from src.viz.compare import save_before_after, save_comparison_grid
from src.viz.overlay import can_overlay, save_overlay

ALGORITHMS = {
    "random": random_search,
    "greedy": greedy_search,
    "annealing": simulated_annealing,
    "genetic": genetic_algorithm,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-room end-to-end validation (supplementary)")
    parser.add_argument("--geometry", default="results/actual/geometry")
    parser.add_argument("--perception", default="results/actual/perception")
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), default="greedy")
    parser.add_argument("--config", default="configs/optimizer.yaml")
    parser.add_argument("--scoring-config", default="configs/scoring.yaml")
    parser.add_argument("--out", default="results/experiments/real_rooms")
    return parser.parse_args()


def format_markdown(records: list[dict], algorithm: str) -> str:
    lines = [
        "# Real-Room Validation",
        "",
        f"Optimizer: `{algorithm}`. Each detected room is optimized and rendered as a before/after plan, "
        "an optimizer comparison grid, and a back-projection overlay on the source image (when camera geometry is available).",
        "",
        "| Room | Objects | Initial | Best | Delta | Feasible | Walkable m2 | Door clearance (m) | Coverage | Outside m2 | Overlay |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['room']} | {record['objects']} | {record['initial_score']:.4f} | {record['best_score']:.4f} | "
            f"{record['delta']:+.4f} | {record['best_feasible']} | {record['walking_area_m2']:.2f} | "
            f"{record['door_clearance']:.2f} | {record['furniture_coverage_fraction']:.3f} | "
            f"{record['outside_furniture_area_m2']:.2f} | {'yes' if record['has_overlay'] else 'no'} |"
        )
    if records:
        lines.extend(
            [
                "",
                f"Mean initial score: {mean(row['initial_score'] for row in records):.4f}",
                "",
                f"Mean optimized score: {mean(row['best_score'] for row in records):.4f}",
                "",
                f"Rooms improved: {sum(1 for row in records if row['delta'] > 1e-9)}/{len(records)}",
                "",
                f"Feasible after optimization: {sum(1 for row in records if row['best_feasible'])}/{len(records)}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    geometry = Path(args.geometry)
    perception = Path(args.perception)
    if not geometry.is_dir():
        print(f"missing geometry directory: {geometry}", file=sys.stderr)
        return 2

    optimizer_config = load_yaml(args.config)
    scoring_config = load_yaml(args.scoring_config)
    config = {"optimizer": optimizer_config, "scoring": scoring_config}
    scorer = LayoutScorer.from_config(scoring_config)
    optimizer = ALGORITHMS[args.algorithm]
    out_dir = ensure_dir(args.out)

    records: list[dict] = []
    for floorplan_path in sorted(geometry.rglob("floorplan.json")):
        room = floorplan_path.parent.name
        plan = FloorPlan.from_dict(json.loads(floorplan_path.read_text(encoding="utf-8")))
        initial = scorer.evaluate(plan)
        result = optimizer(plan, scorer, config)
        best = result.best_score
        room_dir = ensure_dir(out_dir / room)
        save_json(plan.to_dict(), room_dir / "initial_floorplan.json")
        save_json(result.best_plan.to_dict(), room_dir / "optimized_floorplan.json")
        save_json(initial.to_dict(), room_dir / "initial_score.json")
        save_json(best.to_dict(), room_dir / "optimized_score.json")
        save_json(result.to_dict(), room_dir / "optimization_result.json")

        save_before_after(
            plan,
            initial,
            result.best_plan,
            best,
            room_dir / "before_after.png",
            title=f"{room}: {args.algorithm}",
        )
        save_comparison_grid(
            [
                ("Detected", plan, initial),
                ("Optimized", result.best_plan, best),
            ],
            room_dir / "comparison.png",
            columns=2,
        )

        image_path = perception / room / "input.jpg"
        has_overlay = False
        if image_path.is_file() and can_overlay(plan):
            save_overlay(image_path, plan, room_dir / "overlay_before.png")
            if can_overlay(result.best_plan):
                save_overlay(image_path, result.best_plan, room_dir / "overlay_after.png")
            has_overlay = True

        metrics = best.metrics
        record = {
            "room": room,
            "objects": len(plan.room.furniture),
            "initial_score": round(initial.total, 6),
            "best_score": round(best.total, 6),
            "delta": round(best.total - initial.total, 6),
            "initial_feasible": initial.feasible,
            "best_feasible": best.feasible,
            "walking_area_m2": metrics.get("walking_area_m2", 0.0),
            "door_clearance": metrics.get("door_clearance", 0.0),
            "furniture_coverage_fraction": metrics.get("furniture_coverage_fraction", 0.0),
            "outside_furniture_area_m2": metrics.get("outside_furniture_area_m2", 0.0),
            "evaluations": result.evaluations,
            "has_overlay": has_overlay,
        }
        records.append(record)
        print(
            f"{room}: {record['initial_score']:.4f} -> {record['best_score']:.4f} "
            f"({record['delta']:+.4f}) overlay={has_overlay}"
        )

    save_json(
        {
            "algorithm": args.algorithm,
            "records": records,
        },
        out_dir / "real_rooms.json",
    )
    markdown = format_markdown(records, args.algorithm)
    (out_dir / "real_rooms.md").write_text(markdown, encoding="utf-8")
    print()
    print(markdown)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())