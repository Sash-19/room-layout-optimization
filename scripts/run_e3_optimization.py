from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.floorplan.representation import FloorPlan
from src.optimization import genetic_algorithm, greedy_search, random_search, simulated_annealing
from src.optimization.reporting import E3Record, format_e3_markdown, summarize_e3
from src.scoring.score import LayoutScorer
from src.utils import ensure_dir, load_yaml, save_json

ALGORITHMS = {
    "random": random_search,
    "greedy": greedy_search,
    "annealing": simulated_annealing,
    "genetic": genetic_algorithm,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the E3 optimization comparison")
    parser.add_argument("--floorplans", nargs="+", default=["results/geometry"])
    parser.add_argument("--algorithms", nargs="+", choices=tuple(ALGORITHMS), default=list(ALGORITHMS))
    parser.add_argument("--config", default="configs/optimizer.yaml")
    parser.add_argument("--scoring-config", default="configs/scoring.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="results/experiments/e3_optimization")
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
    records: list[E3Record] = []
    for floorplan_path in floorplans:
        room = floorplan_path.parent.name
        plan = FloorPlan.from_dict(json.loads(floorplan_path.read_text(encoding="utf-8")))
        for algorithm in algorithms:
            started = time.perf_counter()
            result = ALGORITHMS[algorithm](plan, scorer, config)
            runtime = time.perf_counter() - started
            records.append(
                E3Record(
                    room=room,
                    algorithm=algorithm,
                    initial_score=result.initial_score.total,
                    best_score=result.best_score.total,
                    initial_feasible=result.initial_score.feasible,
                    best_feasible=result.best_score.feasible,
                    evaluations=result.evaluations,
                    runtime_s=runtime,
                )
            )
            room_dir = ensure_dir(out_dir / room)
            save_json(result.best_plan.to_dict(), room_dir / f"{algorithm}_floorplan.json")
            save_json(result.to_dict(), room_dir / f"{algorithm}_result.json")
            print(
                f"{room} {algorithm}: {result.initial_score.total:.4f} -> "
                f"{result.best_score.total:.4f} ({result.improvement:+.4f}) "
                f"feasible={result.best_score.feasible} runtime={runtime:.2f}s"
            )

    summary = summarize_e3(records)
    save_json(
        {
            "floorplans": [str(path) for path in floorplans],
            "algorithms": algorithms,
            "records": [record.to_dict() for record in records],
            "summary": summary,
        },
        out_dir / "e3_optimization.json",
    )
    markdown = format_e3_markdown(records, summary)
    (out_dir / "e3_optimization.md").write_text(markdown, encoding="utf-8")
    print()
    print(markdown)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
