import matplotlib

matplotlib.use("Agg")

from PIL import Image

from src.geometry.pipeline import GeometryPipeline
from src.scoring.score import LayoutScorer
from src.viz.compare import (
    save_before_after,
    save_comparison_grid,
    save_optimizer_curves,
    save_score_bars,
    save_walkability,
)
from src.viz.overlay import can_overlay, overlay_plan, save_overlay
from tests.test_geometry_pipeline import make_config, make_result
from tests.synthetic_geometry import HEIGHT, WIDTH


def make_plan():
    return GeometryPipeline.from_config(make_config()).run(
        Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255)),
        make_result(),
    )


def test_save_before_after_and_grid(tmp_path):
    plan = make_plan()
    scorer = LayoutScorer()
    score = scorer.evaluate(plan)
    before = save_before_after(plan, score, plan, score, tmp_path / "before_after.png")
    grid = save_comparison_grid([("detected", plan, score)], tmp_path / "grid.png", columns=1)
    assert before.exists() and before.stat().st_size > 0
    assert grid.exists() and grid.stat().st_size > 0


def test_save_score_bars_and_walkability(tmp_path):
    plan = make_plan()
    scorer = LayoutScorer()
    bars = save_score_bars({"detected": scorer.evaluate(plan)}, tmp_path / "bars.png")
    walk = save_walkability(plan, tmp_path / "walk.png")
    assert bars.exists() and bars.stat().st_size > 0
    assert walk.exists() and walk.stat().st_size > 0


def test_save_optimizer_curves(tmp_path):
    curves = [
        {
            "room": "room_a",
            "algorithm": "greedy",
            "curve": {"evaluations": [0, 1, 2], "best_scores": [0.5, 0.6, 0.7]},
        },
        {
            "room": "room_a",
            "algorithm": "genetic",
            "curve": {"evaluations": [0, 1, 2], "best_scores": [0.5, 0.65, 0.8]},
        },
    ]
    output = save_optimizer_curves(curves, tmp_path / "curves.png")
    assert output.exists() and output.stat().st_size > 0


def test_overlay_roundtrip(tmp_path):
    plan = make_plan()
    assert can_overlay(plan)
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    overlaid = overlay_plan(image, plan)
    assert overlaid.size == (WIDTH, HEIGHT)
    assert overlaid.tobytes() != image.tobytes()
    output = save_overlay(image, plan, tmp_path / "overlay.png")
    assert output.exists() and output.stat().st_size > 0
