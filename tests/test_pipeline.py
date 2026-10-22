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
