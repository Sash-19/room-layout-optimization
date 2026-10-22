import numpy as np
from PIL import Image

from src.perception.visualize import colorize_depth, draw_detections, draw_structure_overlay
from src.schemas import Detection, StructureMasks


def test_draw_detections_with_and_without_masks():
    image = Image.new("RGB", (32, 24), (255, 255, 255))
    mask = np.zeros((24, 32), dtype=bool)
    mask[2:8, 2:8] = True
    detections = [Detection("bed", 0.9, (2, 2, 8, 8), mask), Detection("door", 0.5, (20, 5, 30, 20))]
    drawn = draw_detections(image, detections)
    assert drawn.size == (32, 24)
    assert np.asarray(drawn).sum() != np.asarray(image).sum()


def test_colorize_depth_and_structure_overlay():
    depth = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    colored = colorize_depth(depth)
    assert colored.size == (4, 4)
    assert colored.mode == "RGB"

    floor = np.zeros((4, 4), dtype=bool)
    floor[:2] = True
    wall = np.zeros((4, 4), dtype=bool)
    wall[2:] = True
    overlay = draw_structure_overlay(Image.new("RGB", (4, 4), (0, 0, 0)), StructureMasks(floor, wall))
    assert overlay.size == (4, 4)
