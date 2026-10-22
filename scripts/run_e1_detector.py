from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.detector import Detector
from src.perception.evaluation import EvalResult, evaluate_detector, format_markdown
from src.utils import ensure_dir, load_yaml, parse_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the E1 detector comparison")
    parser.add_argument("--config", default="configs/detection.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--weights", nargs="+", required=True, help="weights to compare, in order")
    parser.add_argument("--labels", nargs="+", default=None, help="optional names for the weights")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--report-conf", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="results/experiments/e1_detector")
    return parser.parse_args()


def format_comparison(results: list[tuple[str, EvalResult]]) -> str:
    lines = [
        "# E1: Detector Comparison",
        "",
        "| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, result in results:
        lines.append(
            f"| {label} | {result.map50:.4f} | {result.map50_95:.4f} | "
            f"{result.precision:.4f} | {result.recall:.4f} | {result.f1:.4f} |"
        )
    lines.extend(["", f"Split: `{results[0][1].split}` — {results[0][1].num_images} images", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    data_config = config.get("data", {})
    model_config = dict(config.get("model", {}))
    eval_config = dict(config.get("eval", {}))
    if args.device:
        model_config["device"] = parse_device(args.device)

    data_root = args.data or data_config.get("output", "data/processed")
    split = args.split or eval_config.get("split", "test")
    conf = float(args.conf if args.conf is not None else eval_config.get("conf", 0.001))
    report_conf = float(args.report_conf if args.report_conf is not None else eval_config.get("report_conf", 0.25))
    limit = args.limit if args.limit is not None else eval_config.get("limit")
    thresholds = tuple(eval_config.get("iou_thresholds", (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)))
    labels = args.labels or [Path(weight).stem for weight in args.weights]
    if len(labels) != len(args.weights):
        print("--labels must have the same length as --weights", file=sys.stderr)
        return 2

    out_dir = ensure_dir(args.out)
    results: list[tuple[str, EvalResult]] = []
    for label, weight in zip(labels, args.weights):
        detector = Detector.from_config({**model_config, "weights": weight})
        detector.conf = conf
        print(f"Evaluating {label} ({weight}) on {data_root} [{split}]")
        result = evaluate_detector(
            detector,
            data_root,
            split=split,
            conf=conf,
            report_conf=report_conf,
            iou_thresholds=thresholds,
            limit=limit,
        )
        results.append((label, result))
        save_json(result.to_dict(), out_dir / f"{label}_{split}.json")
        (out_dir / f"{label}_{split}.md").write_text(format_markdown(result), encoding="utf-8")

    summary = format_comparison(results)
    (out_dir / "e1_detector.md").write_text(summary, encoding="utf-8")
    save_json(
        {
            "split": split,
            "data_root": str(data_root),
            "models": [{"label": label, **result.to_dict()} for label, result in results],
        },
        out_dir / "e1_detector.json",
    )
    print()
    print(summary)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
