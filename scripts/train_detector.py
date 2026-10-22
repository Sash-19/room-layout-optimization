from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.detector import Detector
from src.utils import load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO detector on the unified room dataset")
    parser.add_argument("--config", default="configs/detection.yaml")
    parser.add_argument("--data", default=None, help="dataset yaml (defaults to <data.output>/data.yaml)")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    data_config = config.get("data", {})
    model_config = dict(config.get("model", {}))
    train_config = dict(config.get("train", {}))

    data_path = Path(args.data) if args.data else Path(data_config.get("output", "data/processed")) / "data.yaml"
    if not data_path.exists():
        print(f"Dataset yaml not found: {data_path}. Run scripts/prepare_dataset.py first.", file=sys.stderr)
        return 1

    if args.weights:
        model_config["weights"] = args.weights
    if args.device:
        model_config["device"] = args.device
    if args.epochs is not None:
        train_config["epochs"] = args.epochs
    if args.batch is not None:
        train_config["batch"] = args.batch
    if args.imgsz is not None:
        train_config["imgsz"] = args.imgsz
    if args.project:
        train_config["project"] = args.project
    if args.name:
        train_config["name"] = args.name
    if model_config.get("device") is not None:
        train_config.setdefault("device", model_config["device"])

    detector = Detector.from_config(model_config)
    print(f"Training {detector.weights} on {data_path}")
    print(f"Settings: {train_config}")
    result = detector.train(data_path, **train_config)

    save_dir = getattr(result, "save_dir", None)
    summary = {
        "weights": detector.weights,
        "data": str(data_path),
        "train_config": train_config,
        "save_dir": str(save_dir) if save_dir else None,
    }
    best_path = Path(save_dir) / "weights" / "best.pt" if save_dir else None
    if best_path is not None and best_path.exists():
        summary["best_weights"] = str(best_path)
        print(f"Best weights: {best_path}")
    save_json(summary, (Path(save_dir) if save_dir else Path(train_config.get("project", "runs/detect"))) / "train_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
