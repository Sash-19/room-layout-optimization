from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.floorplan.representation import FloorPlan
from src.scoring.score import LayoutScorer
from src.utils import load_json, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a floor-plan JSON file")
    parser.add_argument("--floorplan", required=True, help="floorplan.json produced by run_geometry.py")
    parser.add_argument("--config", default="configs/scoring.yaml")
    parser.add_argument("--out", default=None, help="output score JSON (default: beside the floor plan)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    floorplan_path = Path(args.floorplan)
    floorplan = FloorPlan.from_dict(load_json(floorplan_path))
    scorer = LayoutScorer.from_config(load_yaml(args.config))
    result = scorer.evaluate(floorplan)

    output = Path(args.out) if args.out else floorplan_path.with_name("score.json")
    save_json(result.to_dict(), output)

    print(
        f"{floorplan_path.stem}: score={result.total:.4f} feasible={result.feasible} "
        f"furniture={len(floorplan.room.furniture)}"
    )
    print("terms: " + ", ".join(f"{key}={value:.3f}" for key, value in result.terms.items()))
    for violation in result.violations:
        print(f"  violation: {violation['message']}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
