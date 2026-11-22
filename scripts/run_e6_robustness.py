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

from src.floorplan.noise import jitter_plan
from src.floorplan.representation import FloorPlan
from src.optimization import genetic_algorithm, greedy_search, random_search, simulated_annealing
from src.scoring.score import LayoutScorer
from src.utils import ensure_dir, load_yaml, save_json

ALGORITHMS = {
    "random": random_search,
    "greedy": greedy_search,
    "annealing": simulated_annealing,
    "genetic": genetic_algorithm,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E6: robustness to injected floor-plan noise")
    parser.add_argument("--floorplans", nargs="+", default=["results/geometry"])
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), default="greedy")
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.35])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--config", default="configs/optimizer.yaml")
    parser.add_argument("--scoring-config", default="configs/scoring.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="results/experiments/e6_robustness")
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
    for level in sorted({record["position_sigma_m"] for record in records}):
        rows = [record for record in records if record["position_sigma_m"] == level]
        summary[f"{level:.2f}"] = {
            "runs": len(rows),
            "mean_initial": round(mean(row["initial_score"] for row in rows), 4),
            "mean_best": round(mean(row["best_score"] for row in rows), 4),
            "mean_delta": round(mean(row["delta"] for row in rows), 4),
            "improved_rate": round(mean(1.0 if row["delta"] > 1e-9 else 0.0 for row in rows), 4),
            "feasible_initial_rate": round(mean(1.0 if row["initial_feasible"] else 0.0 for row in rows), 4),
            "feasible_best_rate": round(mean(1.0 if row["best_feasible"] else 0.0 for row in rows), 4),
            "mean_runtime_s": round(mean(row["runtime_s"] for row in rows), 4),
        }
    return summary


def format_markdown(summary: dict[str, dict], algorithm: str) -> str:
    lines = [
        "# E6: Robustness to Floor-Plan Noise",
        "",
        f"Optimizer: `{algorithm}`. Position jitter is injected into every unlocked footprint before optimization; "
        "size jitter is 8% lognormal and orientation flips with probability 0.1.",
        "",
        "| Position sigma (m) | Runs | Mean initial | Mean best | Mean delta | Improved rate | Feasible init | Feasible best | Mean runtime (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level, row in summary.items():
        lines.append(
            f"| {level} | {row['runs']} | {row['mean_initial']:.4f} | {row['mean_best']:.4f} | "
            f"{row['mean_delta']:+.4f} | {row['improved_rate']:.2f} | {row['feasible_initial_rate']:.2f} | "
            f"{row['feasible_best_rate']:.2f} | {row['mean_runtime_s']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def save_plot(summary: dict[str, dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [float(level) for level in summary]
    figure, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(
        levels,
        [summary[f"{level:.2f}"]["mean_delta"] for level in levels],
        marker="o",
        label="mean score delta",
    )
    ax.plot(
        levels,
        [summary[f"{level:.2f}"]["improved_rate"] for level in levels],
        marker="s",
        label="improvement rate",
    )
    ax.set_xlabel("injected position noise sigma (m)")
    ax.set_ylabel("value")
    ax.set_title("Optimizer robustness to floor-plan noise")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
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
    config = {"optimizer": optimizer_config, "scoring": scoring_config}
    scorer = LayoutScorer.from_config(scoring_config)
    optimizer = ALGORITHMS[args.algorithm]

    out_dir = ensure_dir(args.out)
    records: list[dict] = []
    for floorplan_path in floorplans:
        room = floorplan_path.parent.name
        plan = FloorPlan.from_dict(json.loads(floorplan_path.read_text(encoding="utf-8")))
        for level in args.noise_levels:
            for seed in args.seeds:
                noisy = jitter_plan(plan, position_sigma_m=level, seed=seed) if level > 0 else plan
                initial = scorer.evaluate(noisy)
                started = time.perf_counter()
                result = optimizer(noisy, scorer, config)
                runtime = time.perf_counter() - started
                record = {
                    "room": room,
                    "position_sigma_m": round(level, 4),
                    "seed": seed,
                    "initial_score": round(initial.total, 6),
                    "best_score": round(result.best_score.total, 6),
                    "delta": round(result.improvement, 6),
                    "initial_feasible": initial.feasible,
                    "best_feasible": result.best_score.feasible,
                    "runtime_s": round(runtime, 4),
                }
                records.append(record)
                print(
                    f"{room} sigma={level:.2f} seed={seed}: {record['initial_score']:.4f} -> "
                    f"{record['best_score']:.4f} ({record['delta']:+.4f})"
                )

    summary = summarize(records)
    save_json(
        {
            "floorplans": [str(path) for path in floorplans],
            "algorithm": args.algorithm,
            "noise_levels": args.noise_levels,
            "seeds": args.seeds,
            "records": records,
            "summary": summary,
        },
        out_dir / "e6_robustness.json",
    )
    markdown = format_markdown(summary, args.algorithm)
    (out_dir / "e6_robustness.md").write_text(markdown, encoding="utf-8")
    save_plot(summary, out_dir / "e6_robustness.png")
    print()
    print(markdown)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())