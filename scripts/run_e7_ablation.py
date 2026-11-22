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
from src.optimization.sensitivity import ablate_weight
from src.scoring.score import LayoutScorer
from src.utils import ensure_dir, load_yaml, save_json

ALGORITHMS = {
    "random": random_search,
    "greedy": greedy_search,
    "annealing": simulated_annealing,
    "genetic": genetic_algorithm,
}

CONSTRAINT_FLAGS = {
    "overlap": "allow_overlap",
    "door": "allow_door_blocked",
    "window": "allow_window_blocked",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E7: scoring-term ablation")
    parser.add_argument("--floorplans", nargs="+", default=["results/geometry"])
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), default="greedy")
    parser.add_argument("--config", default="configs/optimizer.yaml")
    parser.add_argument("--scoring-config", default="configs/scoring.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="results/experiments/e7_ablation")
    return parser.parse_args()


def discover_floorplans(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            found.extend(sorted(path.rglob("floorplan.json")))
        elif path.is_file():
            found.append(path)
    return sorted(set(found))


def summarize(records: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for variant in sorted({record["variant"] for record in records}):
        rows = [record for record in records if record["variant"] == variant]
        summary[variant] = {
            "rooms": len(rows),
            "mean_full_score": round(mean(row["full_score"] for row in rows), 4),
            "feasible_rate": round(mean(1.0 if row["feasible"] else 0.0 for row in rows), 4),
            "outside_room": sum(row["violation_counts"].get("outside_room", 0) for row in rows),
            "overlap": sum(row["violation_counts"].get("overlap", 0) for row in rows),
            "door_blocked": sum(row["violation_counts"].get("door_blocked", 0) for row in rows),
            "window_blocked": sum(row["violation_counts"].get("window_blocked", 0) for row in rows),
            "mean_walkable_m2": round(mean(row["walking_area_m2"] for row in rows), 3),
            "mean_door_clearance": round(mean(row["door_clearance"] for row in rows), 3),
            "mean_outside_m2": round(mean(row["outside_furniture_area_m2"] for row in rows), 3),
            "mean_overlap_m2": round(mean(row["overlap_area_m2"] for row in rows), 3),
        }
    return summary


def format_markdown(summary: dict[str, dict], algorithm: str) -> str:
    lines = [
        "# E7: Scoring-Term Ablation",
        "",
        f"Optimizer: `{algorithm}`. Each row removes one term from the optimizer's objective "
        "(remaining weights renormalized), then evaluates the resulting layout with the **full** scorer. "
        "This shows which failure modes return when a term is ignored.",
        "",
        "| Removed term | Rooms | Full score | Feasible rate | Outside | Overlap | Door blocked | Window blocked | Walkable m2 | Door clearance | Outside m2 | Overlap m2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in summary.items():
        lines.append(
            f"| {variant} | {row['rooms']} | {row['mean_full_score']:.4f} | {row['feasible_rate']:.2f} | "
            f"{row['outside_room']} | {row['overlap']} | {row['door_blocked']} | {row['window_blocked']} | "
            f"{row['mean_walkable_m2']:.2f} | {row['mean_door_clearance']:.2f} | "
            f"{row['mean_outside_m2']:.2f} | {row['mean_overlap_m2']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    floorplans = discover_floorplans(args.floorplans)
    if args.limit is not None:
        floorplans = floorplans[: args.limit]
    if not floorplans:
        print("no floorplan.json files found", file=sys.stderr)
        return 2

    optimizer_config = load_yaml(args.config)
    scoring_config = load_yaml(args.scoring_config)
    base_weights = dict(scoring_config.get("weights", {}))
    if not base_weights:
        print("scoring config has no weights", file=sys.stderr)
        return 2
    full_scorer = LayoutScorer.from_config(scoring_config)
    optimizer = ALGORITHMS[args.algorithm]
    variants = {"baseline": base_weights}
    for key in base_weights:
        variants[f"without_{key}"] = ablate_weight(base_weights, key)

    plans = {
        path.parent.name: FloorPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for path in floorplans
    }
    out_dir = ensure_dir(args.out)
    records: list[dict] = []
    for variant, weights in variants.items():
        term = variant[len("without_") :] if variant.startswith("without_") else None
        constraints = dict(scoring_config.get("constraints", {}))
        if term in CONSTRAINT_FLAGS:
            constraints[CONSTRAINT_FLAGS[term]] = True
        variant_config = {**scoring_config, "weights": weights, "constraints": constraints}
        scorer = LayoutScorer.from_config(variant_config)
        config = {"optimizer": optimizer_config, "scoring": variant_config}
        for room, plan in plans.items():
            result = optimizer(plan, scorer, config)
            full = full_scorer.evaluate(result.best_plan)
            counts: dict[str, int] = {}
            for violation in full.violations:
                counts[violation["kind"]] = counts.get(violation["kind"], 0) + 1
            metrics = full.metrics
            records.append(
                {
                    "room": room,
                    "variant": variant,
                    "algorithm": args.algorithm,
                    "weights": {key: round(value, 6) for key, value in weights.items()},
                    "full_score": round(full.total, 6),
                    "feasible": full.feasible,
                    "violation_counts": counts,
                    "walking_area_m2": metrics.get("walking_area_m2", 0.0),
                    "door_clearance": metrics.get("door_clearance", 0.0),
                    "outside_furniture_area_m2": metrics.get("outside_furniture_area_m2", 0.0),
                    "overlap_area_m2": metrics.get("overlap_area_m2", 0.0),
                }
            )
            room_dir = ensure_dir(out_dir / variant / room)
            save_json(result.best_plan.to_dict(), room_dir / "floorplan.json")
            save_json(full.to_dict(), room_dir / "score_full.json")
        print(f"{variant}: optimized {len(plans)} room(s)")

    summary = summarize(records)
    save_json(
        {
            "floorplans": [str(path) for path in floorplans],
            "algorithm": args.algorithm,
            "records": records,
            "summary": summary,
        },
        out_dir / "e7_ablation.json",
    )
    markdown = format_markdown(summary, args.algorithm)
    (out_dir / "e7_ablation.md").write_text(markdown, encoding="utf-8")
    print()
    print(markdown)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())