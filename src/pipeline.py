from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .floorplan.representation import FloorPlan
from .geometry.pipeline import GeometryPipeline
from .geometry.visualize import save_floorplan
from .optimization import genetic_algorithm, greedy_search, random_search, simulated_annealing
from .optimization.base import OptimizationResult
from .perception.pipeline import PerceptionPipeline, PerceptionResult
from .scoring.score import LayoutScorer, ScoreResult
from .utils import ensure_dir, load_image, load_yaml, save_json
from .viz.compare import save_before_after
from .viz.overlay import can_overlay, save_overlay

ALGORITHMS = {
    "random": random_search,
    "greedy": greedy_search,
    "annealing": simulated_annealing,
    "genetic": genetic_algorithm,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(slots=True)
class PipelineResult:
    image: Path
    output_dir: Path
    perception: PerceptionResult
    floorplan: FloorPlan
    initial_score: ScoreResult
    best_plan: FloorPlan
    best_score: ScoreResult
    optimization: OptimizationResult | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        return float(self.best_score.total - self.initial_score.total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": str(self.image),
            "output_dir": str(self.output_dir),
            "algorithm": self.optimization.algorithm if self.optimization is not None else None,
            "initial_score": self.initial_score.to_dict(),
            "best_score": self.best_score.to_dict(),
            "improvement": round(self.improvement, 4),
            "evaluations": self.optimization.evaluations if self.optimization is not None else 0,
            "geometry_reliable": self.floorplan.metadata.get("geometry_reliable"),
            "reliability_reasons": self.floorplan.metadata.get("reliability_reasons", []),
            "artifacts": self.artifacts,
        }


class LayoutPipeline:
    def __init__(
        self,
        geometry: GeometryPipeline,
        scorer: LayoutScorer,
        optimizer_config: dict[str, Any] | None = None,
        scoring_config: dict[str, Any] | None = None,
        perception_config: dict[str, Any] | None = None,
        algorithm: str = "greedy",
        perception: PerceptionPipeline | None = None,
    ) -> None:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {algorithm!r}; choose from {sorted(ALGORITHMS)}")
        self.geometry = geometry
        self.scorer = scorer
        self.optimizer_config = optimizer_config or {}
        self.scoring_config = scoring_config or {}
        self.perception_config = perception_config or {}
        self.algorithm = algorithm
        self._perception = perception

    @classmethod
    def from_config(cls, config: dict[str, Any], algorithm: str = "greedy") -> LayoutPipeline:
        perception_config = config.get("perception", {})
        geometry_config = config.get("geometry", {})
        scoring_config = config.get("scoring", {})
        optimizer_config = config.get("optimizer", {})
        return cls(
            geometry=GeometryPipeline.from_config(geometry_config),
            scorer=LayoutScorer.from_config(scoring_config),
            optimizer_config=optimizer_config,
            scoring_config=scoring_config,
            perception_config=perception_config,
            algorithm=algorithm,
        )

    @property
    def perception(self) -> PerceptionPipeline:
        if self._perception is None:
            self._perception = PerceptionPipeline.from_config(self.perception_config)
        return self._perception

    def run(
        self,
        image: str | Path,
        output_dir: str | Path | None = None,
        perception: PerceptionResult | None = None,
        room_width_m: float | None = None,
        room_depth_m: float | None = None,
        scale_mode: str | None = None,
        optimize: bool = True,
    ) -> PipelineResult:
        image_path = Path(image)
        image_obj = load_image(image_path)
        perception = perception or self.perception.run(image_obj)
        floorplan = self.geometry.run(
            image_obj,
            perception,
            room_width_m=room_width_m,
            room_depth_m=room_depth_m,
            scale_mode=scale_mode,
        )
        initial_score = self.scorer.evaluate(floorplan)
        optimization = None
        best_plan = floorplan
        best_score = initial_score
        if optimize:
            config = {"optimizer": self.optimizer_config, "scoring": self.scoring_config}
            optimization = ALGORITHMS[self.algorithm](floorplan, self.scorer, config)
            best_plan = optimization.best_plan
            best_score = optimization.best_score

        out_dir = ensure_dir(output_dir or Path("results/pipeline") / image_path.stem)
        artifacts: dict[str, str] = {}

        def record(name: str, path: Path) -> None:
            artifacts[name] = str(path)

        record("perception", save_json(perception.to_dict(), out_dir / "perception.json"))
        record("initial_floorplan", save_json(floorplan.to_dict(), out_dir / "floorplan_initial.json"))
        record("optimized_floorplan", save_json(best_plan.to_dict(), out_dir / "floorplan_optimized.json"))
        record("initial_score", save_json(initial_score.to_dict(), out_dir / "score_initial.json"))
        record("optimized_score", save_json(best_score.to_dict(), out_dir / "score_optimized.json"))
        record("initial_plot", save_floorplan(floorplan, out_dir / "floorplan_initial.png"))
        record("optimized_plot", save_floorplan(best_plan, out_dir / "floorplan_optimized.png"))
        record(
            "before_after",
            save_before_after(
                floorplan,
                initial_score,
                best_plan,
                best_score,
                out_dir / "before_after.png",
                title=f"{image_path.stem}: {self.algorithm}",
            ),
        )
        if optimization is not None:
            record("optimization", save_json(optimization.to_dict(), out_dir / "optimization.json"))
            top_layouts = []
            for rank, (plan, score) in enumerate(
                zip(optimization.top_plans, optimization.top_scores), start=1
            ):
                plot = save_floorplan(plan, out_dir / f"floorplan_top_{rank:02d}.png")
                top_layouts.append(
                    {
                        "rank": rank,
                        "score": round(float(score.total), 4),
                        "feasible": score.feasible,
                        "plot": str(plot),
                        "plan": plan.to_dict(),
                    }
                )
            if top_layouts:
                record("top_layouts", save_json({"layouts": top_layouts}, out_dir / "layouts_top.json"))
        if can_overlay(floorplan):
            record("overlay_initial", save_overlay(image_path, floorplan, out_dir / "overlay_initial.png"))
            record("overlay_optimized", save_overlay(image_path, best_plan, out_dir / "overlay_optimized.png"))

        result = PipelineResult(
            image=image_path,
            output_dir=out_dir,
            perception=perception,
            floorplan=floorplan,
            initial_score=initial_score,
            best_plan=best_plan,
            best_score=best_score,
            optimization=optimization,
            artifacts=artifacts,
        )
        record("run", save_json(result.to_dict(), out_dir / "run.json"))
        return result


def load_config(
    perception_config: str | Path = "configs/perception.yaml",
    geometry_config: str | Path = "configs/geometry.yaml",
    scoring_config: str | Path = "configs/scoring.yaml",
    optimizer_config: str | Path = "configs/optimizer.yaml",
) -> dict[str, Any]:
    return {
        "perception": load_yaml(perception_config),
        "geometry": load_yaml(geometry_config),
        "scoring": load_yaml(scoring_config),
        "optimizer": load_yaml(optimizer_config),
    }


def _cached_perception(root: Path | None, image_path: Path) -> PerceptionResult | None:
    if root is None:
        return None
    candidates = [root / image_path.stem, root]
    for candidate in candidates:
        if (candidate / "result.json").is_file():
            return PerceptionResult.from_cache(candidate)
    return None


def _collect_images(values: list[str]) -> list[Path]:
    images: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            images.extend(sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"image not found: {path}")
    return images


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: detections + depth -> floor plan -> score -> optimize -> render."
    )
    parser.add_argument("images", nargs="+", help="image file(s) or directory")
    parser.add_argument("--out", default="results/pipeline")
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), default="greedy")
    parser.add_argument("--perception-dir", default=None, help="cached perception root (e.g. results/actual/perception)")
    parser.add_argument("--room-width", type=float, default=None)
    parser.add_argument("--room-depth", type=float, default=None)
    parser.add_argument("--scale-mode", choices=("auto", "metric", "user", "reference"), default=None)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--config", default=None, help="single merged YAML config (optional)")
    parser.add_argument("--perception-config", default="configs/perception.yaml")
    parser.add_argument("--detector-weights", default=None, help="override perception.detector.weights")
    parser.add_argument("--geometry-config", default="configs/geometry.yaml")
    parser.add_argument("--scoring-config", default="configs/scoring.yaml")
    parser.add_argument("--optimizer-config", default="configs/optimizer.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = (
        load_yaml(args.config)
        if args.config
        else load_config(
            args.perception_config,
            args.geometry_config,
            args.scoring_config,
            args.optimizer_config,
        )
    )
    pipeline = LayoutPipeline.from_config(config, algorithm=args.algorithm)
    if args.detector_weights:
        pipeline.perception_config.setdefault("detector", {})["weights"] = args.detector_weights
    images = _collect_images(args.images)
    if not images:
        print("no images found", file=sys.stderr)
        return 2
    perception_root = Path(args.perception_dir) if args.perception_dir else None
    failures = 0
    for image_path in images:
        out_dir = Path(args.out) / image_path.stem if len(images) > 1 else Path(args.out)
        try:
            cached = _cached_perception(perception_root, image_path)
            if cached is not None:
                print(f"{image_path.name}: using cached perception")
            result = pipeline.run(
                image_path,
                output_dir=out_dir,
                perception=cached,
                room_width_m=args.room_width,
                room_depth_m=args.room_depth,
                scale_mode=args.scale_mode,
                optimize=not args.no_optimize,
            )
        except Exception as error:
            failures += 1
            print(f"{image_path.name}: failed: {error}", file=sys.stderr)
            continue
        print(
            f"{image_path.name}: {result.initial_score.total:.4f} -> {result.best_score.total:.4f} "
            f"({result.improvement:+.4f}) feasible={result.best_score.feasible}"
        )
        reasons = result.floorplan.metadata.get("reliability_reasons", [])
        if reasons:
            print(f"  warning: low-confidence geometry ({'; '.join(reasons)}) - verify the plan before use")
        print(f"  wrote {result.output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())