from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.floorplan.representation import FloorPlan
from src.optimization import OptimizationResult, greedy_search, random_search, simulated_annealing
from src.scoring.score import LayoutScorer
from src.utils import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize furniture placement in a floor plan")
    parser.add_argument("--floorplan", required=True)
    parser.add_argument("--config", default="configs/optimizer.yaml")
    parser.add_argument("--scoring-config", default="configs/scoring.yaml")
    parser.add_argument(
        "--algorithm",
        choices=("random", "greedy", "annealing", "both", "all"),
        default=None,
    )
    parser.add_argument("--out", default="results/optimization")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def format_summary(results: list[OptimizationResult]) -> str:
    lines = [
        "# Layout Optimization",
        "",
        "| Algorithm | Initial | Best | Improvement | Feasible | Evaluations |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.algorithm} | {result.initial_score.total:.4f} | {result.best_score.total:.4f} | "
            f"{result.improvement:+.4f} | {'yes' if result.best_score.feasible else 'no'} | "
            f"{result.evaluations} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    optimizer_config = load_yaml(args.config)
    scoring_config = load_yaml(args.scoring_config)
    config = {"optimizer": optimizer_config, "scoring": scoring_config}
    plan = FloorPlan.from_dict(json.loads(Path(args.floorplan).read_text(encoding="utf-8")))
    scorer = LayoutScorer.from_config(scoring_config)

    algorithm = args.algorithm or optimizer_config.get("algorithm", "random")
    if algorithm == "both":
        algorithms = ["random", "greedy"]
    elif algorithm == "all":
        algorithms = ["random", "greedy", "annealing"]
    elif algorithm in {"random", "greedy", "annealing"}:
        algorithms = [algorithm]
    else:
        print(f"unknown algorithm: {algorithm}", file=sys.stderr)
        return 2

    out_dir = ensure_dir(args.out)
    stem = args.name or Path(args.floorplan).stem
    results: list[OptimizationResult] = []
    for name in algorithms:
        if name == "random":
            result = random_search(plan, scorer, config)
        elif name == "greedy":
            result = greedy_search(plan, scorer, config)
        else:
            result = simulated_annealing(plan, scorer, config)
        results.append(result)
        save_json(result.best_plan.to_dict(), out_dir / f"{stem}_{name}_floorplan.json")
        save_json(result.to_dict(), out_dir / f"{stem}_{name}_result.json")
        for index, (top_plan, top_score) in enumerate(zip(result.top_plans, result.top_scores), start=1):
            save_json(
                top_plan.to_dict(),
                out_dir / f"{stem}_{name}_top{index}_floorplan.json",
            )
            save_json(top_score.to_dict(), out_dir / f"{stem}_{name}_top{index}_score.json")

    save_json(
        {
            "floorplan": str(args.floorplan),
            "algorithms": [result.to_dict() for result in results],
        },
        out_dir / f"{stem}_summary.json",
    )
    summary = format_summary(results)
    (out_dir / f"{stem}_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
