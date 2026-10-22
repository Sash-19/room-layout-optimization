import matplotlib
import pytest

matplotlib.use("Agg")

from PIL import Image

from src.geometry.pipeline import GeometryPipeline
from src.geometry.visualize import plot_floorplan, save_floorplan
from tests.test_geometry_pipeline import make_config, make_result
from tests.synthetic_geometry import HEIGHT, WIDTH


def test_plot_and_save_floorplan(tmp_path):
    plan = GeometryPipeline.from_config(make_config()).run(
        Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255)),
        make_result(),
    )
    figure, axes = plot_floorplan(plan)
    assert axes.get_aspect() == 1.0
    figure.canvas.draw()
    output = save_floorplan(plan, tmp_path / "floorplan.png")
    assert output.exists()
    assert output.stat().st_size > 0
