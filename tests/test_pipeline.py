import json

from PIL import Image

from src.perception.pipeline import PerceptionPipeline
from src.schemas import Detection


class FakeDetector:
    def detect(self, image):
        return [Detection("bed", 0.9, (1, 1, 5, 5)), Detection("chair", 0.4, (2, 2, 6, 6))]


def test_pipeline_runs_without_depth_or_structure():
    pipeline = PerceptionPipeline(FakeDetector())
    image = Image.new("RGB", (16, 12), (255, 255, 255))
    result = pipeline.run(image)
    assert result.image_size == (16, 12)
    assert len(result.detections) == 2
    assert result.depth is None
    assert result.structures is None
    assert result.class_counts() == {"bed": 1, "chair": 1}
    payload = result.to_dict()
    assert payload["num_detections"] == 2
    assert payload["image_size"] == {"width": 16, "height": 12}
    assert "detection_s" in payload["timings"]


def test_pipeline_from_config_disables_optional_stages():
    config = {
        "detector": {"weights": "yolov8s.pt", "classes": ["bed"]},
        "depth": {"enabled": False},
        "structure": {"enabled": False},
    }
    pipeline = PerceptionPipeline.from_config(config)
    assert pipeline.depth_estimator is None
    assert pipeline.segmenter is None
    assert pipeline.detector.classes == {"bed"}


def test_perception_result_from_cache(tmp_path):
    import json

    import numpy as np
    from PIL import Image

    from src.perception.pipeline import PerceptionResult

    cache = tmp_path / "cached"
    cache.mkdir()
    payload = {
        "image_size": {"width": 16, "height": 12},
        "detections": [
            {"class": "bed", "confidence": 0.9, "bbox_xyxy": [1, 1, 5, 5], "has_mask": False},
        ],
        "depth": {"metric": False, "source": "test", "height": 12, "width": 16},
        "structures": {"source": "test"},
        "timings": {"detection_s": 0.01},
    }
    (cache / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    np.save(cache / "depth.npy", np.ones((12, 16), dtype=np.float32))
    Image.fromarray(np.zeros((12, 16), dtype=np.uint8)).save(cache / "floor_mask.png")
    wall = np.zeros((12, 16), dtype=np.uint8)
    wall[:, :4] = 255
    Image.fromarray(wall).save(cache / "wall_mask.png")

    result = PerceptionResult.from_cache(cache)

    assert result.image_size == (16, 12)
    assert len(result.detections) == 1
    assert result.detections[0].cls_name == "bed"
    assert result.depth is not None and result.depth.depth.shape == (12, 16)
    assert result.structures is not None
    assert result.structures.floor.shape == (12, 16)
    assert result.structures.wall[:, :4].all()


def test_layout_pipeline_end_to_end(tmp_path):
    from PIL import Image

    from src.geometry.pipeline import GeometryPipeline
    from src.pipeline import LayoutPipeline
    from src.scoring.score import LayoutScorer
    from tests.test_geometry_pipeline import make_config, make_result
    from tests.synthetic_geometry import HEIGHT, WIDTH

    image_path = tmp_path / "room.png"
    Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255)).save(image_path)
    scoring_config = {"grid": {"resolution": 0.1}}
    optimizer_config = {
        "seed": 0,
        "rotation_step_deg": 90,
        "sampling": {"pose_attempts": 60, "wall_snap_probability": 0.8},
        "greedy": {"candidates_per_item": 80},
    }
    pipeline = LayoutPipeline(
        geometry=GeometryPipeline.from_config(make_config()),
        scorer=LayoutScorer.from_config(scoring_config),
        optimizer_config=optimizer_config,
        scoring_config=scoring_config,
        algorithm="greedy",
    )

    result = pipeline.run(image_path, output_dir=tmp_path / "out", perception=make_result())

    assert result.output_dir.exists()
    assert result.best_score.total >= result.initial_score.total
    assert (result.output_dir / "floorplan_initial.json").exists()
    assert (result.output_dir / "floorplan_optimized.json").exists()
    assert (result.output_dir / "before_after.png").exists()
    assert (result.output_dir / "run.json").exists()
    payload = result.to_dict()
    assert payload["algorithm"] == "greedy"
    assert payload["evaluations"] > 0
    assert isinstance(payload["geometry_reliable"], bool)
    assert isinstance(payload["reliability_reasons"], list)
    run_json = result.output_dir / "run.json"
    assert "geometry_reliable" in run_json.read_text(encoding="utf-8")


def test_layout_pipeline_saves_top_k_layouts(tmp_path):
    from PIL import Image

    from src.geometry.pipeline import GeometryPipeline
    from src.pipeline import LayoutPipeline
    from src.scoring.score import LayoutScorer
    from tests.test_geometry_pipeline import make_config, make_result
    from tests.synthetic_geometry import HEIGHT, WIDTH

    image_path = tmp_path / "room.png"
    Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255)).save(image_path)
    scoring_config = {"grid": {"resolution": 0.1}}
    optimizer_config = {
        "seed": 0,
        "rotation_step_deg": 90,
        "annealing": {"iterations": 60, "restarts": 1, "top_k": 2},
    }
    pipeline = LayoutPipeline(
        geometry=GeometryPipeline.from_config(make_config()),
        scorer=LayoutScorer.from_config(scoring_config),
        optimizer_config=optimizer_config,
        scoring_config=scoring_config,
        algorithm="annealing",
    )

    result = pipeline.run(image_path, output_dir=tmp_path / "out", perception=make_result())

    assert result.optimization is not None and result.optimization.top_plans
    layouts = json.loads((result.output_dir / "layouts_top.json").read_text(encoding="utf-8"))
    assert layouts["layouts"]
    for entry in layouts["layouts"]:
        assert (result.output_dir / f"floorplan_top_{entry['rank']:02d}.png").exists()
    assert result.artifacts["top_layouts"].endswith("layouts_top.json")
