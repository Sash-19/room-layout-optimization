import json

import pytest
from shapely.geometry import box

from src.floorplan.representation import Furniture, Room
from src.geometry.evaluation import evaluate_pair, format_markdown, summarize


def make_room(width, depth, furniture):
    return Room(box(0, 0, width, depth), [], furniture)


def test_geometry_evaluation_pair_metrics():
    ground_truth = make_room(
        4.0,
        3.0,
        [Furniture("bed", 1.6, 2.0, 1.0, 1.5), Furniture("desk", 1.2, 0.6, 3.0, 2.5)],
    )
    prediction = make_room(
        4.2,
        2.9,
        [Furniture("bed", 1.7, 2.0, 1.2, 1.5), Furniture("desk", 1.2, 0.6, 3.0, 2.5)],
    )
    result = evaluate_pair(prediction, ground_truth)
    assert result["room"]["width_abs_error_m"] == pytest.approx(0.2)
    assert result["room"]["width_pct_error"] == pytest.approx(5.0)
    assert result["room"]["depth_pct_error"] == pytest.approx(100 * 0.1 / 3.0)
    assert result["summary"]["matched_furniture"] == 2
    bed = next(item for item in result["furniture"] if item.get("class") == "bed" and "position_error_m" in item)
    assert bed["position_error_m"] == pytest.approx(0.2)
    assert bed["width_abs_error_m"] == pytest.approx(0.1)
    json.dumps(result)


def test_geometry_evaluation_false_positives_and_negatives():
    ground_truth = make_room(4.0, 3.0, [Furniture("bed", 1.6, 2.0, 1.0, 1.5)])
    prediction = make_room(
        4.0,
        3.0,
        [Furniture("chair", 0.5, 0.5, 3.0, 2.0)],
    )
    result = evaluate_pair(prediction, ground_truth)
    assert result["summary"]["matched_furniture"] == 0
    assert result["summary"]["false_positives"] == 1
    assert result["summary"]["false_negatives"] == 1


def test_geometry_evaluation_summary_and_markdown():
    ground_truth = make_room(4.0, 3.0, [Furniture("bed", 1.6, 2.0, 1.0, 1.5)])
    prediction = make_room(4.1, 3.0, [Furniture("bed", 1.6, 2.0, 1.1, 1.5)])
    result = evaluate_pair(prediction, ground_truth)
    summary = summarize([result])
    assert summary["rooms"] == 1
    assert summary["room_width_pct_error"]["mean"] == pytest.approx(2.5)
    assert summary["furniture"]["matched"] == 1
    assert summary["furniture"]["position_error_m"]["mean"] == pytest.approx(0.1)
    text = format_markdown(summary)
    assert "Geometry Evaluation" in text
    assert "Furniture position error" in text
    assert "1/1" in text
