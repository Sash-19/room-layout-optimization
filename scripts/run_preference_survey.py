from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.floorplan.representation import FloorPlan
from src.scoring.score import LayoutScorer
from src.utils import ensure_dir, load_yaml, save_json
from src.viz.compare import save_before_after

CHOICES = ("detected", "optimized", "tie")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative preference survey (detected vs optimized)")
    parser.add_argument("--floorplans", nargs="+", default=["results/actual/geometry"])
    parser.add_argument("--optimized", default="results/experiments/e3_optimization")
    parser.add_argument("--algorithm", default="greedy")
    parser.add_argument("--scoring-config", default="configs/scoring.yaml")
    parser.add_argument("--out", default="results/survey")
    parser.add_argument("--aggregate", default=None, help="path to a filled ratings.csv to summarize")
    return parser.parse_args()


def build_pairs(args, out_dir: Path) -> list[dict]:
    scorer = LayoutScorer.from_config(load_yaml(args.scoring_config))
    pairs: list[dict] = []
    for floorplan_path in sorted(Path(path) for path in args.floorplans):
        if floorplan_path.is_dir():
            candidates = sorted(floorplan_path.rglob("floorplan.json"))
        else:
            candidates = [floorplan_path]
        for candidate in candidates:
            room = candidate.parent.name
            optimized_path = Path(args.optimized) / room / f"{args.algorithm}_floorplan.json"
            if not optimized_path.is_file():
                print(f"skipping {room}: no optimized layout at {optimized_path}", file=sys.stderr)
                continue
            detected = FloorPlan.from_dict(json.loads(candidate.read_text(encoding="utf-8")))
            optimized = FloorPlan.from_dict(json.loads(optimized_path.read_text(encoding="utf-8")))
            detected_score = scorer.evaluate(detected)
            optimized_score = scorer.evaluate(optimized)
            image = out_dir / room / "pair.png"
            save_before_after(
                detected,
                detected_score,
                optimized,
                optimized_score,
                image,
                title=f"{room}: A = detected, B = optimized",
            )
            pairs.append(
                {
                    "room": room,
                    "pair_image": str(image),
                    "detected_score": round(detected_score.total, 4),
                    "optimized_score": round(optimized_score.total, 4),
                }
            )
            print(f"{room}: wrote {image}")
    return pairs


def write_template(pairs: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["room", "choice"])
        for pair in pairs:
            writer.writerow([pair["room"], ""])


def aggregate(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter((row.get("choice") or "").strip().lower() for row in rows)
    counts.pop("", None)
    total = sum(counts.values())
    if not total:
        print("no ratings recorded")
        return 2
    lines = ["# Preference Survey Results", "", f"Rated pairs: {total}", ""]
    for choice, count in counts.most_common():
        lines.append(f"- {choice}: {count} ({count / total:.0%})")
    detected = counts.get("detected", 0)
    optimized = counts.get("optimized", 0)
    if detected + optimized:
        lines.extend(
            [
                "",
                f"Preference rate (excluding ties): optimized {optimized / (detected + optimized):.0%} "
                f"vs detected {detected / (detected + optimized):.0%}",
            ]
        )
    print("\n".join(lines))
    return 0


def main() -> int:
    args = parse_args()
    if args.aggregate:
        return aggregate(Path(args.aggregate))
    out_dir = ensure_dir(args.out)
    pairs = build_pairs(args, out_dir)
    if not pairs:
        print("no survey pairs built", file=sys.stderr)
        return 2
    save_json({"pairs": pairs}, out_dir / "pairs.json")
    write_template(pairs, out_dir / "ratings.csv")
    print(f"wrote {out_dir} (fill in ratings.csv, then rerun with --aggregate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
