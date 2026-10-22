from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..floorplan.representation import Furniture, Room


def _percent(error: float, reference: float) -> float | None:
    if reference <= 0:
        return None
    return 100.0 * error / reference


def _center(item: Furniture) -> tuple[float, float]:
    return (float(item.x), float(item.y))


def _iou(first: Furniture, second: Furniture) -> float:
    intersection = float(first.footprint.intersection(second.footprint).area)
    union = float(first.footprint.union(second.footprint).area)
    return intersection / union if union > 0 else 0.0


def _match_class(
    predictions: list[Furniture],
    ground_truth: list[Furniture],
    max_distance_m: float,
) -> tuple[list[tuple[Furniture, Furniture, float]], list[int], list[int]]:
    if not predictions or not ground_truth:
        return [], list(range(len(predictions))), list(range(len(ground_truth)))
    distances = np.array(
        [
            [math.hypot(pred.x - gt.x, pred.y - gt.y) for gt in ground_truth]
            for pred in predictions
        ],
        dtype=float,
    )
    rows, columns = linear_sum_assignment(distances)
    matches = []
    matched_predictions = set()
    matched_ground_truth = set()
    for row, column in zip(rows, columns):
        distance = float(distances[row, column])
        if distance <= max_distance_m:
            matches.append((predictions[row], ground_truth[column], distance))
            matched_predictions.add(int(row))
            matched_ground_truth.add(int(column))
    unmatched_predictions = [index for index in range(len(predictions)) if index not in matched_predictions]
    unmatched_ground_truth = [index for index in range(len(ground_truth)) if index not in matched_ground_truth]
    return matches, unmatched_predictions, unmatched_ground_truth


def evaluate_pair(
    prediction: Room,
    ground_truth: Room,
    match_distance_m: float = 1.5,
) -> dict[str, Any]:
    pred_width, pred_depth = prediction.width, prediction.depth
    gt_width, gt_depth = ground_truth.width, ground_truth.depth
    room_metrics = {
        "pred_width_m": round(pred_width, 4),
        "gt_width_m": round(gt_width, 4),
        "width_abs_error_m": round(abs(pred_width - gt_width), 4),
        "width_pct_error": _percent(abs(pred_width - gt_width), gt_width),
        "pred_depth_m": round(pred_depth, 4),
        "gt_depth_m": round(gt_depth, 4),
        "depth_abs_error_m": round(abs(pred_depth - gt_depth), 4),
        "depth_pct_error": _percent(abs(pred_depth - gt_depth), gt_depth),
        "pred_area_m2": round(prediction.area, 4),
        "gt_area_m2": round(ground_truth.area, 4),
        "area_abs_error_m2": round(abs(prediction.area - ground_truth.area), 4),
        "area_pct_error": _percent(abs(prediction.area - ground_truth.area), ground_truth.area),
    }

    classes = sorted({item.cls for item in prediction.furniture} | {item.cls for item in ground_truth.furniture})
    furniture_metrics: list[dict[str, Any]] = []
    matched_count = 0
    false_positives = 0
    false_negatives = 0
    for cls in classes:
        pred_items = [item for item in prediction.furniture if item.cls == cls]
        gt_items = [item for item in ground_truth.furniture if item.cls == cls]
        matches, unmatched_predictions, unmatched_ground_truth = _match_class(
            pred_items, gt_items, match_distance_m
        )
        matched_count += len(matches)
        false_positives += len(unmatched_predictions)
        false_negatives += len(unmatched_ground_truth)
        for pred_item, gt_item, distance in matches:
            width_error = abs(pred_item.w - gt_item.w)
            depth_error = abs(pred_item.d - gt_item.d)
            furniture_metrics.append(
                {
                    "class": cls,
                    "position_error_m": round(distance, 4),
                    "width_abs_error_m": round(width_error, 4),
                    "width_pct_error": _percent(width_error, gt_item.w),
                    "depth_abs_error_m": round(depth_error, 4),
                    "depth_pct_error": _percent(depth_error, gt_item.d),
                    "iou": round(_iou(pred_item, gt_item), 4),
                    "pred": pred_item.to_dict(),
                    "gt": gt_item.to_dict(),
                }
            )
        for index in unmatched_predictions:
            furniture_metrics.append({"class": cls, "status": "false_positive", "pred": pred_items[index].to_dict()})
        for index in unmatched_ground_truth:
            furniture_metrics.append({"class": cls, "status": "false_negative", "gt": gt_items[index].to_dict()})

    return {
        "room": room_metrics,
        "furniture": furniture_metrics,
        "summary": {
            "gt_furniture": len(ground_truth.furniture),
            "pred_furniture": len(prediction.furniture),
            "matched_furniture": matched_count,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "match_distance_m": match_distance_m,
        },
    }


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    if not records:
        return {"rooms": 0}

    def values(key: str, section: str = "room") -> list[float]:
        result = []
        for record in records:
            value = record.get(section, {}).get(key)
            if value is not None:
                result.append(float(value))
        return result

    def aggregate(key: str, section: str = "room") -> dict[str, float | None]:
        data = values(key, section)
        if not data:
            return {"mean": None, "median": None}
        return {"mean": round(float(np.mean(data)), 4), "median": round(float(np.median(data)), 4)}

    furniture = [item for record in records for item in record.get("furniture", [])]
    matched = [item for item in furniture if "position_error_m" in item]
    positions = [float(item["position_error_m"]) for item in matched]
    ious = [float(item["iou"]) for item in matched]
    dimension_errors = [
        value
        for item in matched
        for value in (item.get("width_pct_error"), item.get("depth_pct_error"))
        if value is not None
    ]
    gt_total = sum(int(record["summary"]["gt_furniture"]) for record in records)
    matched_total = sum(int(record["summary"]["matched_furniture"]) for record in records)
    false_positives = sum(int(record["summary"]["false_positives"]) for record in records)
    false_negatives = sum(int(record["summary"]["false_negatives"]) for record in records)

    return {
        "rooms": len(records),
        "room_width_pct_error": aggregate("width_pct_error"),
        "room_depth_pct_error": aggregate("depth_pct_error"),
        "room_area_pct_error": aggregate("area_pct_error"),
        "room_width_abs_error_m": aggregate("width_abs_error_m"),
        "room_depth_abs_error_m": aggregate("depth_abs_error_m"),
        "furniture": {
            "matched": matched_total,
            "gt_total": gt_total,
            "matched_fraction": round(matched_total / gt_total, 4) if gt_total else None,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "position_error_m": {
                "mean": round(float(np.mean(positions)), 4) if positions else None,
                "median": round(float(np.median(positions)), 4) if positions else None,
            },
            "dimension_pct_error": {
                "mean": round(float(np.mean(dimension_errors)), 4) if dimension_errors else None,
                "median": round(float(np.median(dimension_errors)), 4) if dimension_errors else None,
            },
            "iou": {
                "mean": round(float(np.mean(ious)), 4) if ious else None,
                "median": round(float(np.median(ious)), 4) if ious else None,
            },
        },
    }


def format_markdown(summary: dict[str, Any]) -> str:
    if not summary.get("rooms"):
        return "No matched predictions.\n"

    def value(entry: dict[str, Any] | None, key: str = "mean") -> str:
        if not entry or entry.get(key) is None:
            return "n/a"
        return f"{entry[key]:.3f}"

    furniture = summary["furniture"]
    lines = [
        "# Geometry Evaluation",
        "",
        f"Rooms evaluated: **{summary['rooms']}**",
        "",
        "| Metric | Mean | Median |",
        "|---|---:|---:|",
        f"| Room width error (%) | {value(summary['room_width_pct_error'])} | {value(summary['room_width_pct_error'], 'median')} |",
        f"| Room depth error (%) | {value(summary['room_depth_pct_error'])} | {value(summary['room_depth_pct_error'], 'median')} |",
        f"| Room area error (%) | {value(summary['room_area_pct_error'])} | {value(summary['room_area_pct_error'], 'median')} |",
        f"| Room width error (m) | {value(summary['room_width_abs_error_m'])} | {value(summary['room_width_abs_error_m'], 'median')} |",
        f"| Room depth error (m) | {value(summary['room_depth_abs_error_m'])} | {value(summary['room_depth_abs_error_m'], 'median')} |",
        f"| Furniture position error (m) | {value(furniture['position_error_m'])} | {value(furniture['position_error_m'], 'median')} |",
        f"| Furniture dimension error (%) | {value(furniture['dimension_pct_error'])} | {value(furniture['dimension_pct_error'], 'median')} |",
        f"| Furniture IoU | {value(furniture['iou'])} | {value(furniture['iou'], 'median')} |",
        "",
        f"Furniture matched: **{furniture['matched']}/{furniture['gt_total']}** "
        f"(false positives: {furniture['false_positives']}, false negatives: {furniture['false_negatives']})",
        "",
    ]
    return "\n".join(lines)
