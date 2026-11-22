from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.floorplan.synthetic import synthetic_suite
from src.optimization import genetic_algorithm, greedy_search, random_search, simulated_annealing
from src.scoring.score import LayoutScorer
from src.utils import ensure_dir, load_yaml, save_json
from src.viz.compare import save_comparison_grid

ALGORITHMS = {
    "random": random_search,
    "greedy": greedy_search,
    "annealing": simulated_annealing,
    "genetic": genetic_algorithm,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled synthetic stress tests")
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), default="greedy")
    parser.add_argument("--config", default="configs/optimizer.yaml")
    parser.add_argument("--scoring-config", default="configs/scoring.yaml")
    parser.add_argument("--out", default="results/experiments/synthetic_stress")
    return parser.parse_args()


def kinds(score) -> list[str]:
    return sorted({violation["kind"] for violation in score.violations})


def main() -> int:
    args = parse_args()
    optimizer_config = load_yaml(args.config)
    scoring_config = load_yaml(args.scoring_config)
    config = {"optimizer": optimizer_config, "scoring": scoring_config}
    scorer = LayoutScorer.from_config(scoring_config)
    optimizer = ALGORITHMS[args.algorithm]
    out_dir = ensure_dir(args.out)
    suite = synthetic_suite()

    records = []
    grid_items = []
    for name, plan in suite.items():
        initial = scorer.evaluate(plan)
        result = optimizer(plan, scorer, config)
        best = result.best_score
        record = {
            "name": name,
            "initial_score": round(initial.total, 6),
            "initial_feasible": initial.feasible,
            "initial_violations": kinds(initial),
            "best_score": round(best.total, 6),
            "best_feasible": best.feasible,
            "best_violations": kinds(best),
            "improvement": round(result.improvement, 6),
        }
        records.append(record)
        grid_items.extend(
            [
                (f"{name}\ndetected", plan, initial),
                (f"{name}\noptimized", result.best_plan, best),
            ]
        )
        room_dir = ensure_dir(out_dir / name)
        save_json(plan.to_dict(), room_dir / "initial_floorplan.json")
        save_json(initial.to_dict(), room_dir / "initial_score.json")
        save_json(result.best_plan.to_dict(), room_dir / "optimized_floorplan.json")
        save_json(best.to_dict(), room_dir / "optimized_score.json")
        print(
            f"{name}: {initial.total:.4f} ({', '.join(kinds(initial)) or 'ok'}) -> "
            f"{best.total:.4f} ({', '.join(kinds(best)) or 'ok'}) feasible={best.feasible}"
        )

    save_json({"algorithm": args.algorithm, "records": records}, out_dir / "synthetic_stress.json")
    lines = [
        "# Synthetic Stress Tests",
        "",
        "| Layout | Initial | Violations | Optimized | Violations | Feasible |",
        "|---|---:|---|---:|---|---|",
    ]
    for row in records:
        lines.append(
            f"| {row['name']} | {row['initial_score']:.4f} | "
            f"{', '.join(row['initial_violations']) or 'none'} | {row['best_score']:.4f} | "
            f"{', '.join(row['best_violations']) or 'none'} | {'yes' if row['best_feasible'] else 'no'} |"
        )
    markdown = "\n".join(lines) + "\n"
    (out_dir / "synthetic_stress.md").write_text(markdown, encoding="utf-8")
    save_comparison_grid(grid_items, out_dir / "synthetic_stress.png", columns=4)
    print()
    print(markdown)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())