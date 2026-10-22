from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.detector import Detector
from src.perception.evaluation import evaluate_detector, format_markdown
from src.utils import ensure_dir, load_yaml, parse_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detections with unified-class mAP")
    parser.add_argument("--config", default="configs/detection.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--report-conf", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    data_config = config.get("data", {})
    model_config = dict(config.get("model", {}))
    eval_config = dict(config.get("eval", {}))

    if args.weights:
        model_config["weights"] = args.weights
    if args.device:
        model_config["device"] = parse_device(args.device)
    detector = Detector.from_config(model_config)

    conf = float(args.conf if args.conf is not None else eval_config.get("conf", 0.001))
    detector.conf = conf
    report_conf = float(args.report_conf if args.report_conf is not None else eval_config.get("report_conf", 0.25))
    split = args.split or eval_config.get("split", "test")
    data_root = args.data or data_config.get("output", "data/processed")
    limit = args.limit if args.limit is not None else eval_config.get("limit")
    thresholds = tuple(eval_config.get("iou_thresholds", (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)))

    print(f"Evaluating {detector.weights} on {data_root} [{split}]")
    result = evaluate_detector(
        detector,
        data_root,
        split=split,
        conf=conf,
        report_conf=report_conf,
        iou_thresholds=thresholds,
        limit=limit,
    )

    out_dir = Path(args.out or eval_config.get("out", "results/detection_eval"))
    ensure_dir(out_dir)
    stem = args.name or f"{Path(detector.weights).stem}_{split}"
    save_json(result.to_dict(), out_dir / f"{stem}.json")
    markdown = format_markdown(result)
    (out_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"Saved results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
