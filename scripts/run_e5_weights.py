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
from src.optimization.sensitivity import weight_variants
from src.scoring.score import LayoutScorer
from src.utils import ensure_dir, load_yaml, save_json

ALGORITHMS = {
    "random": random_search,
    "greedy": greedy_search,
    "annealing": simulated_annealing,
    "genetic": genetic_algorithm,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E5: scoring-weight sensitivity")
    parser.add_argument("--floorplans", nargs="+", default=["results/geometry"])
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), default="greedy")
    parser.add_argument("--config", default="configs/optimizer.yaml")
    parser.add_argument("--scoring-config", default="configs/scoring.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="results/experiments/e5_weights")
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
            "mean_variant_score": round(mean(row["variant_score"] for row in rows), 4),
            "mean_baseline_score": round(mean(row["baseline_score"] for row in rows), 4),
            "feasible_best": sum(1 for row in rows if row["feasible"]),
            "mean_walkable_area_m2": round(mean(row["walking_area_m2"] for row in rows), 4),
            "mean_usable_floor_fraction": round(mean(row["usable_floor_fraction"] for row in rows), 4),
            "mean_door_clearance": round(mean(row["door_clearance"] for row in rows), 4),
            "mean_coverage_fraction": round(mean(row["furniture_coverage_fraction"] for row in rows), 4),
            "mean_outside_area_m2": round(mean(row["outside_furniture_area_m2"] for row in rows), 4),
        }
    return summary


def format_markdown(summary: dict[str, dict]) -> str:
    lines = [
        "# E5: Scoring-Weight Sensitivity",
        "",
        "Each variant re-optimizes every room with one weight scaled up/down and the rest renormalized.",
        "`baseline score` evaluates the resulting layout with the unmodified weights, so it is comparable across variants.",
        "",
        "| Variant | Rooms | Variant score | Baseline score | Feasible | Walkable m2 | Usable floor | Door clearance | Coverage | Outside m2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in summary.items():
        lines.append(
            f"| {variant} | {row['rooms']} | {row['mean_variant_score']:.4f} | "
            f"{row['mean_baseline_score']:.4f} | {row['feasible_best']}/{row['rooms']} | "
            f"{row['mean_walkable_area_m2']:.3f} | {row['mean_usable_floor_fraction']:.3f} | "
            f"{row['mean_door_clearance']:.3f} | {row['mean_coverage_fraction']:.3f} | "
            f"{row['mean_outside_area_m2']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def save_plot(summary: dict[str, dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(summary)
    figure, axes = plt.subplots(1, 2, figsize=(max(10.0, 0.55 * len(variants)), 4.5))
    axes[0].bar(variants, [summary[v]["mean_baseline_score"] for v in variants], color="#3478f6")
    axes[0].set_ylabel("mean baseline score of optimized layout")
    axes[0].set_title("Layout quality under the baseline scorer")
    axes[1].bar(variants, [summary[v]["mean_walkable_area_m2"] for v in variants], color="#2eb85c")
    axes[1].set_ylabel("mean walkable area (m2)")
    axes[1].set_title("Walkable area by weighting")
    for ax in axes:
        ax.tick_params(axis="x", rotation=55, labelsize=7)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


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
    variants = weight_variants(base_weights)
    baseline_scorer = LayoutScorer.from_config(scoring_config)
    optimizer = ALGORITHMS[args.algorithm]

    plans = {
        path.parent.name: FloorPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for path in floorplans
    }
    out_dir = ensure_dir(args.out)
    records: list[dict] = []
    for variant, weights in variants.items():
        variant_config = {**scoring_config, "weights": weights}
        scorer = LayoutScorer.from_config(variant_config)
        config = {"optimizer": optimizer_config, "scoring": variant_config}
        for room, plan in plans.items():
            result = optimizer(plan, scorer, config)
            baseline = baseline_scorer.evaluate(result.best_plan)
            metrics = baseline.metrics
            records.append(
                {
                    "room": room,
                    "variant": variant,
                    "algorithm": args.algorithm,
                    "weights": {key: round(value, 6) for key, value in weights.items()},
                    "variant_score": round(result.best_score.total, 6),
                    "baseline_score": round(baseline.total, 6),
                    "feasible": baseline.feasible,
                    "walking_area_m2": metrics.get("walking_area_m2", 0.0),
                    "usable_floor_fraction": metrics.get("usable_floor_fraction", 0.0),
                    "door_clearance": metrics.get("door_clearance", 0.0),
                    "furniture_coverage_fraction": metrics.get("furniture_coverage_fraction", 0.0),
                    "outside_furniture_area_m2": metrics.get("outside_furniture_area_m2", 0.0),
                }
            )
            room_dir = ensure_dir(out_dir / variant / room)
            save_json(result.best_plan.to_dict(), room_dir / "floorplan.json")
            save_json(baseline.to_dict(), room_dir / "score.json")
        print(f"{variant}: optimized {len(plans)} room(s)")

    summary = summarize(records)
    save_json(
        {
            "floorplans": [str(path) for path in floorplans],
            "algorithm": args.algorithm,
            "variants": {key: {name: round(value, 6) for name, value in weights.items()} for key, weights in variants.items()},
            "records": records,
            "summary": summary,
        },
        out_dir / "e5_weights.json",
    )
    markdown = format_markdown(summary)
    (out_dir / "e5_weights.md").write_text(markdown, encoding="utf-8")
    save_plot(summary, out_dir / "e5_weights.png")
    print()
    print(markdown)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())