from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.floorplan.representation import Room
from src.geometry.evaluation import evaluate_pair, format_markdown, summarize
from src.utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted floor plans against hand measurements")
    parser.add_argument("--predictions", required=True, help="floorplan.json file or a results/geometry directory")
    parser.add_argument("--ground-truth", required=True, help="ground-truth JSON file or data/floorplans_gt directory")
    parser.add_argument("--match-distance", type=float, default=1.5, help="max center distance for matching (m)")
    parser.add_argument("--out", default="results/geometry_eval")
    return parser.parse_args()


def load_room(path: Path) -> Room:
    payload = load_json(path)
    if isinstance(payload, dict) and "room" in payload:
        return Room.from_dict(payload["room"])
    return Room.from_dict(payload)


def find_ground_truth(root: Path, name: str) -> Path | None:
    candidates = (
        root / name / "floorplan_gt.json",
        root / f"{name}.json",
        root / name / "floorplan.json",
        root / name / "room.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_pairs(predictions: Path, ground_truth: Path) -> list[tuple[str, Path, Path]]:
    if predictions.is_file():
        if not ground_truth.is_file():
            raise ValueError("a single prediction requires a single ground-truth file")
        return [(predictions.parent.name, predictions, ground_truth)]

    pairs = []
    for prediction in sorted(predictions.glob("*/floorplan.json")):
        name = prediction.parent.name
        gt_path = find_ground_truth(ground_truth, name)
        if gt_path is None:
            print(f"skipping {name}: no ground truth found", file=sys.stderr)
            continue
        pairs.append((name, prediction, gt_path))
    return pairs


def main() -> int:
    args = parse_args()
    pairs = find_pairs(Path(args.predictions), Path(args.ground_truth))
    if not pairs:
        print("no prediction/ground-truth pairs found", file=sys.stderr)
        return 1

    records = []
    for name, prediction_path, gt_path in pairs:
        prediction = load_room(prediction_path)
        ground_truth = load_room(gt_path)
        record = evaluate_pair(prediction, ground_truth, match_distance_m=args.match_distance)
        record["name"] = name
        records.append(record)
        print(
            f"{name}: room width err {record['room']['width_pct_error']:.1f}% "
            f"depth err {record['room']['depth_pct_error']:.1f}% "
            f"furniture {record['summary']['matched_furniture']}/{record['summary']['gt_furniture']}"
        )

    summary = summarize(records)
    output = Path(args.out)
    save_json({"summary": summary, "records": records}, output / "metrics.json")
    (output / "metrics.md").write_text(format_markdown(summary), encoding="utf-8")

    print()
    print(format_markdown(summary))
    print(f"wrote {output / 'metrics.json'}")
    print(f"wrote {output / 'metrics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
