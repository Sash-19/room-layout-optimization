from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..floorplan.representation import FloorPlan, Room
from .constraints import ConstraintViolation, check_constraints
from .grid import OccupancyGrid
from .terms import TERMS

DEFAULT_WEIGHTS: dict[str, float] = {
    "overlap": 0.06,
    "walk": 0.22,
    "path": 0.08,
    "wall": 0.14,
    "door": 0.18,
    "window": 0.04,
    "util": 0.10,
    "clear": 0.16,
    "pair": 0.02,
}


@dataclass(slots=True)
class ScoreResult:
    total: float
    terms: dict[str, float]
    details: dict[str, dict[str, Any]] = field(default_factory=dict)
    violations: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    feasible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(float(self.total), 4),
            "feasible": self.feasible,
            "terms": {key: round(float(value), 4) for key, value in self.terms.items()},
            "details": self.details,
            "violations": self.violations,
            "metrics": self.metrics,
        }


class LayoutScorer:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        weights = dict(DEFAULT_WEIGHTS)
        weights.update({key: float(value) for key, value in self.config.get("weights", {}).items()})
        self.weights = {key: value for key, value in weights.items() if key in TERMS}

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> LayoutScorer:
        return cls(config)

    def evaluate(self, layout: Room | FloorPlan, grid: OccupancyGrid | None = None) -> ScoreResult:
        room = layout.room if isinstance(layout, FloorPlan) else layout
        resolution = float(self.config.get("grid", {}).get("resolution", 0.05))
        grid = grid or OccupancyGrid.from_room(room, resolution=resolution)
        violations: list[ConstraintViolation] = check_constraints(room, self.config)

        terms: dict[str, float] = {}
        details: dict[str, dict[str, Any]] = {}
        for name, weight in self.weights.items():
            if weight == 0:
                continue
            result = TERMS[name](room, grid, self.config)
            terms[name] = float(max(0.0, min(1.0, result.score)))
            details[name] = result.details

        total = float(sum(self.weights[name] * terms[name] for name in terms))
        overlap_details = details.get("overlap", {})
        footprints = [item.footprint for item in room.furniture]
        furniture_area = sum(float(footprint.area) for footprint in footprints)
        furniture_inside_area = sum(
            float(footprint.intersection(room.polygon).area) for footprint in footprints
        )
        outside_furniture_area = max(0.0, furniture_area - furniture_inside_area)
        violation_area = sum(float(item.area) for item in violations)
        metrics = {
            "room_area_m2": round(room.area, 4),
            "free_area_m2": round(grid.free_area, 4),
            "usable_floor_fraction": round(grid.free_area / room.area, 4) if room.area > 0 else 0.0,
            "walking_area_m2": round(grid.walkable_area(float(self.config.get("terms", {}).get("walkway_width_m", 0.6))), 4),
            "overlap_area_m2": round(float(overlap_details.get("overlap_area", 0.0)), 4),
            "overlap_pairs": len(overlap_details.get("pairs", [])),
            "largest_free_component_fraction": round(float(terms.get("util", 0.0)), 4),
            "door_clearance": round(float(terms.get("door", 1.0)), 4),
            "furniture_area_m2": round(furniture_area, 4),
            "furniture_area_inside_m2": round(furniture_inside_area, 4),
            "outside_furniture_area_m2": round(outside_furniture_area, 4),
            "furniture_coverage_fraction": round(furniture_inside_area / room.area, 4) if room.area > 0 else 0.0,
            "violation_area_m2": round(violation_area, 4),
            "grid": grid.to_dict(),
        }
        return ScoreResult(
            total=total,
            terms=terms,
            details=details,
            violations=[violation.to_dict() for violation in violations],
            metrics=metrics,
            feasible=not violations,
        )


def score_room(room: Room, config: dict[str, Any] | None = None) -> ScoreResult:
    return LayoutScorer.from_config(config).evaluate(room)
