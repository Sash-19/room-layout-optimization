from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ..floorplan.representation import FloorPlan
from ..scoring.score import ScoreResult


@dataclass(slots=True)
class OptimizationResult:
    algorithm: str
    initial_plan: FloorPlan
    best_plan: FloorPlan
    initial_score: ScoreResult
    best_score: ScoreResult
    evaluations: int
    feasible_layouts: int
    history: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        return float(self.best_score.total - self.initial_score.total)

    @property
    def improved(self) -> bool:
        return self.improvement > 1e-9

    def to_dict(self, include_plan: bool = False) -> dict[str, Any]:
        payload = {
            "algorithm": self.algorithm,
            "initial_score": self.initial_score.to_dict(),
            "best_score": self.best_score.to_dict(),
            "improvement": round(self.improvement, 4),
            "improved": self.improved,
            "evaluations": self.evaluations,
            "feasible_layouts": self.feasible_layouts,
            "history": self.history,
            "config": self.config,
        }
        if include_plan:
            payload["best_plan"] = self.best_plan.to_dict()
        return payload


def clone_plan(plan: FloorPlan) -> FloorPlan:
    return deepcopy(plan)


def score_rank(score: ScoreResult) -> tuple[int, float, float]:
    violation_area = sum(float(item.get("area", 0.0)) for item in score.violations)
    return (1 if score.feasible else 0, float(score.total), -violation_area)
