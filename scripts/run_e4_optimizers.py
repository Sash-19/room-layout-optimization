from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.floorplan.representation import FloorPlan
from src.optimization import genetic_algorithm, greedy_search, random_search, simulated_annealing
from src.optimization.tracking import ScoreTracker
from src.scoring.score import LayoutScorer
from src.utils import ensure_dir, load_yaml, save_json
from src.viz.compare import save_optimizer_curves

ALGORITHMS = {
    "random": random_search,
    "greedy": greedy_search,
    "annealing": simulated_annealing,
    "genetic": genetic_algorithm,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E4: optimizer comparison with score-vs-evaluations curves")
    parser.add_argument("--floorplans", nargs="+", default=["results/geometry"])
    parser.add_argument("--algorithms", nargs="+", choices=tuple(ALGORITHMS), default=list(ALGORITHMS))
    parser.add_argument("--config", default="configs/optimizer.yaml")
    parser.add_argument("--scoring-config", default="configs/scoring.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="results/experiments/e4_optimizers")
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
    for algorithm in sorted({record["algorithm"] for record in records}):
        rows = [record for record in records if record["algorithm"] == algorithm]
        summary[algorithm] = {
            "rooms": len(rows),
            "mean_initial": round(mean(row["initial_score"] for row in rows), 4),
            "mean_best": round(mean(row["best_score"] for row in rows), 4),
            "mean_delta": round(mean(row["best_score"] - row["initial_score"] for row in rows), 4),
            "improved_rooms": sum(1 for row in rows if row["best_score"] > row["initial_score"] + 1e-9),
            "feasible_initial": sum(1 for row in rows if row["initial_feasible"]),
            "feasible_best": sum(1 for row in rows if row["best_feasible"]),
            "mean_evaluations": round(mean(row["evaluations"] for row in rows), 1),
            "mean_runtime_s": round(mean(row["runtime_s"] for row in rows), 4),
        }
    return summary


def format_markdown(records: list[dict], summary: dict[str, dict]) -> str:
    lines = [
        "# E4: Optimizer Comparison",
        "",
        "All optimizers start from the same detected layout and use the same scorer.",
        "",
        "## Aggregate",
        "",
        "| Algorithm | Rooms | Mean initial | Mean best | Mean delta | Improved | Feasible init -> best | Mean evaluations | Mean runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for algorithm, row in summary.items():
        lines.append(
            f"| {algorithm} | {row['rooms']} | {row['mean_initial']:.4f} | {row['mean_best']:.4f} | "
            f"{row['mean_delta']:+.4f} | {row['improved_rooms']}/{row['rooms']} | "
            f"{row['feasible_initial']} -> {row['feasible_best']} | {row['mean_evaluations']:.0f} | "
            f"{row['mean_runtime_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per room",
            "",
            "| Room | Algorithm | Initial | Best | Delta | Feasible | Evaluations | Runtime (s) |",
            "|---|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for record in sorted(records, key=lambda row: (row["room"], row["algorithm"])):
        lines.append(
            f"| {record['room']} | {record['algorithm']} | {record['initial_score']:.4f} | "
            f"{record['best_score']:.4f} | {record['best_score'] - record['initial_score']:+.4f} | "
            f"{'yes' if record['best_feasible'] else 'no'} | {record['evaluations']} | "
            f"{record['runtime_s']:.3f} |"
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
    config = {"optimizer": optimizer_config, "scoring": scoring_config}
    scorer = LayoutScorer.from_config(scoring_config)
    algorithms = list(dict.fromkeys(args.algorithms))

    out_dir = ensure_dir(args.out)
    records: list[dict] = []
    curves: list[dict] = []
    for floorplan_path in floorplans:
        room = floorplan_path.parent.name
        plan = FloorPlan.from_dict(json.loads(floorplan_path.read_text(encoding="utf-8")))
        for algorithm in algorithms:
            tracker = ScoreTracker(scorer)
            started = time.perf_counter()
            result = ALGORITHMS[algorithm](plan, tracker, config)
            runtime = time.perf_counter() - started
            record = {
                "room": room,
                "algorithm": algorithm,
                "initial_score": round(result.initial_score.total, 6),
                "best_score": round(result.best_score.total, 6),
                "initial_feasible": result.initial_score.feasible,
                "best_feasible": result.best_score.feasible,
                "improvement": round(result.improvement, 6),
                "evaluations": result.evaluations,
                "runtime_s": round(runtime, 4),
            }
            records.append(record)
            curves.append({"room": room, "algorithm": algorithm, "curve": tracker.curve()})
            room_dir = ensure_dir(out_dir / room)
            save_json(result.best_plan.to_dict(), room_dir / f"{algorithm}_floorplan.json")
            save_json(result.to_dict(), room_dir / f"{algorithm}_result.json")
            print(
                f"{room} {algorithm}: {record['initial_score']:.4f} -> {record['best_score']:.4f} "
                f"({record['improvement']:+.4f}) feasible={record['best_feasible']} "
                f"evals={record['evaluations']} runtime={runtime:.2f}s"
            )

    summary = summarize(records)
    save_json(
        {
            "floorplans": [str(path) for path in floorplans],
            "algorithms": algorithms,
            "records": records,
            "summary": summary,
            "curves": curves,
        },
        out_dir / "e4_optimizers.json",
    )
    markdown = format_markdown(records, summary)
    (out_dir / "e4_optimizers.md").write_text(markdown, encoding="utf-8")
    save_optimizer_curves(curves, out_dir / "e4_curves.png")
    print()
    print(markdown)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())