from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.floorplan.representation import FloorPlan, Room
from src.geometry.evaluation import evaluate_pair, summarize
from src.geometry.pipeline import GeometryPipeline
from src.geometry.visualize import save_floorplan
from src.perception.cache import load_cached_perception
from src.scoring.score import LayoutScorer
from src.utils import ensure_dir, load_json, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the E2 scale-mode comparison")
    parser.add_argument("--perception-dir", required=True)
    parser.add_argument("--config", default="configs/geometry.yaml")
    parser.add_argument("--scoring-config", default="configs/scoring.yaml")
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--modes", default="metric,reference,user")
    parser.add_argument("--room-width", type=float, default=None)
    parser.add_argument("--room-depth", type=float, default=None)
    parser.add_argument("--reference-class", default="door")
    parser.add_argument("--reference-width", type=float, default=None)
    parser.add_argument("--out", default="results/experiments/e2_scale")
    return parser.parse_args()


def load_room(path: str | Path | None) -> Room | None:
    if path is None:
        return None
    payload = load_json(path)
    if isinstance(payload, dict) and "room" in payload:
        return Room.from_dict(payload["room"])
    return Room.from_dict(payload)


def find_ground_truth(root: str | Path | None, name: str) -> Path | None:
    if root is None:
        return None
    root = Path(root)
    if root.is_file():
        return root
    for candidate in (
        root / name / "floorplan_gt.json",
        root / f"{name}.json",
        root / name / "floorplan.json",
    ):
        if candidate.exists():
            return candidate
    return None


def format_markdown(records: list[dict]) -> str:
    lines = [
        "# E2: Scale Calibration",
        "",
        "| Mode | Reported mode | Scale | Room (m) | Camera height (m) | Feasible | Score | Width err (%) | Depth err (%) | Furniture pos. err (m) |",
        "|---|---|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for record in records:
        geometry = record.get("geometry") or {}
        room = geometry.get("room") or {}
        summary = geometry.get("summary") or {}
        furniture = record.get("furniture") or {}
        position = furniture.get("position_error_m", {}).get("mean") if furniture else None
        lines.append(
            "| {mode} | {reported} | {scale:.3f} | {width:.2f} x {depth:.2f} | {camera:.2f} | {feasible} | "
            "{score:.3f} | {width_error} | {depth_error} | {position} |".format(
                mode=record["mode"],
                reported=record.get("scale_mode", record["mode"]),
                scale=record["scale"],
                width=record["room_width_m"],
                depth=record["room_depth_m"],
                camera=record["camera_height"],
                feasible="yes" if record["feasible"] else "no",
                score=record["score"],
                width_error="n/a" if room.get("width_pct_error") is None else f"{room['width_pct_error']:.2f}",
                depth_error="n/a" if room.get("depth_pct_error") is None else f"{room['depth_pct_error']:.2f}",
                position="n/a" if position is None else f"{position:.3f}",
            )
        )
    lines.extend(
        [
            "",
            "`n/a` means no hand-measured ground truth was provided. Relative depth cannot produce",
            "metric scale in `metric` mode; use a metric depth checkpoint for that row.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    image, perception = load_cached_perception(args.perception_dir)
    name = Path(args.perception_dir).name
    ground_truth = load_room(find_ground_truth(args.ground_truth, name))
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    base_config = load_yaml(args.config)
    scorer = LayoutScorer.from_config(load_yaml(args.scoring_config))
    output = ensure_dir(Path(args.out) / name)

    records = []
    for mode in modes:
        config = copy.deepcopy(base_config)
        config.setdefault("scale", {})
        config["scale"]["mode"] = mode
        if mode == "reference":
            config["scale"]["reference_class"] = args.reference_class
            if args.reference_width is not None:
                config["scale"]["reference_width_m"] = args.reference_width
        pipeline = GeometryPipeline.from_config(config)
        floorplan = pipeline.run(
            image,
            perception,
            room_width_m=args.room_width if mode == "user" else None,
            room_depth_m=args.room_depth if mode == "user" else None,
            scale_mode=mode,
        )
        score = scorer.evaluate(floorplan)
        record = {
            "mode": mode,
            "scale": round(float(floorplan.scale), 4),
            "scale_mode": floorplan.scale_mode,
            "room_width_m": round(floorplan.room.width, 4),
            "room_depth_m": round(floorplan.room.depth, 4),
            "room_area_m2": round(floorplan.room.area, 4),
            "camera_height": round(float(floorplan.camera_height), 4),
            "feasible": score.feasible,
            "score": round(score.total, 4),
            "terms": score.terms,
            "warnings": floorplan.metadata.get("warnings", []),
            "violations": score.violations,
            "furniture": score.metrics,
        }
        if ground_truth is not None:
            geometry = evaluate_pair(floorplan.room, ground_truth)
            record["geometry"] = geometry
            record["furniture"] = summarize([geometry])["furniture"]
        records.append(record)

        save_json(floorplan.to_dict(), output / f"{mode}_floorplan.json")
        save_floorplan(floorplan, output / f"{mode}_floorplan.png")
        save_json(score.to_dict(), output / f"{mode}_score.json")
        print(
            f"{mode}: reported={floorplan.scale_mode} scale={floorplan.scale:.3f} "
            f"room={floorplan.room.width:.2f}x{floorplan.room.depth:.2f} "
            f"score={score.total:.3f} feasible={score.feasible}"
        )

    save_json({"room": name, "modes": modes, "records": records}, output / "e2_scale.json")
    markdown = format_markdown(records)
    (output / "e2_scale.md").write_text(markdown, encoding="utf-8")
    print()
    print(markdown)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
