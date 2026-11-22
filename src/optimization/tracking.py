from __future__ import annotations

from typing import Any

from ..floorplan.representation import FloorPlan
from ..scoring.score import LayoutScorer, ScoreResult
from .base import score_rank


class ScoreTracker:
    """LayoutScorer decorator that records every candidate evaluation.

    The optimizers only require an object with an ``evaluate`` method, so wrapping
    the scorer gives E4 score-vs-evaluations curves without changing any optimizer.
    """

    def __init__(self, scorer: LayoutScorer) -> None:
        self.scorer = scorer
        self.records: list[dict[str, Any]] = []

    def evaluate(self, layout: FloorPlan | Any) -> ScoreResult:
        score = self.scorer.evaluate(layout)
        self.records.append(
            {
                "evaluation": len(self.records),
                "score": round(float(score.total), 6),
                "feasible": bool(score.feasible),
                "violations": len(score.violations),
                "terms": {key: round(float(value), 6) for key, value in score.terms.items()},
            }
        )
        return score

    @property
    def evaluations(self) -> int:
        return len(self.records)

    def curve(self) -> dict[str, list[float]]:
        best = float("-inf")
        best_scores: list[float] = []
        best_feasible: list[float | None] = []
        feasible_best = float("-inf")
        for record in self.records:
            if record["score"] > best:
                best = float(record["score"])
            best_scores.append(round(best, 6))
            if record["feasible"] and record["score"] > feasible_best:
                feasible_best = float(record["score"])
            best_feasible.append(round(feasible_best, 6) if feasible_best > float("-inf") else None)
        return {
            "evaluations": list(range(len(self.records))),
            "scores": [record["score"] for record in self.records],
            "best_scores": best_scores,
            "best_feasible_scores": best_feasible,
            "feasible": [record["feasible"] for record in self.records],
        }

    def best_so_far(self) -> float:
        if not self.records:
            return 0.0
        return float(max(record["score"] for record in self.records))

    def rank(self) -> tuple[int, float, float]:
        if not self.records:
            return (0, 0.0, 0.0)
        return max(
            (int(record["feasible"]), float(record["score"]), -float(record["violations"]))
            for record in self.records
        )


def rank_records(records: list[dict[str, Any]]) -> tuple[int, float, float]:
    if not records:
        return (0, 0.0, 0.0)
    return max(
        (int(record["feasible"]), float(record["score"]), -float(record["violations"]))
        for record in records
    )


__all__ = ["ScoreTracker", "rank_records", "score_rank"]