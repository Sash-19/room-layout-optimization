from __future__ import annotations

from random import Random
from typing import Any

from ..floorplan.representation import FloorPlan
from ..scoring.score import LayoutScorer
from .base import OptimizationResult, clone_plan, score_rank
from .sampling import optimizer_settings, sample_layout, scoring_settings


def random_search(
    plan: FloorPlan,
    scorer: LayoutScorer | None = None,
    config: dict[str, Any] | None = None,
) -> OptimizationResult:
    settings = optimizer_settings(config)
    scorer = scorer or LayoutScorer.from_config(scoring_settings(config))
    rng = Random(int(settings.get("seed", 0)))
    iterations = int(settings.get("random", {}).get("iterations", 200))
    max_attempts = int(settings.get("random", {}).get("max_layout_attempts", 40))

    initial_score = scorer.evaluate(plan)
    best_plan = clone_plan(plan)
    best_score = initial_score
    evaluations = 1
    feasible_layouts = 1 if initial_score.feasible else 0
    history = [
        {
            "evaluation": 0,
            "score": round(initial_score.total, 4),
            "feasible": initial_score.feasible,
            "violations": len(initial_score.violations),
            "improved": False,
        }
    ]

    for iteration in range(1, iterations + 1):
        candidate = None
        for _ in range(max(1, max_attempts)):
            candidate = sample_layout(plan, rng, config)
            if candidate is not None:
                break
        if candidate is None:
            continue

        score = scorer.evaluate(candidate)
        evaluations += 1
        if score.feasible:
            feasible_layouts += 1
        improved = score_rank(score) > score_rank(best_score)
        if improved:
            best_plan = candidate
            best_score = score
            history.append(
                {
                    "evaluation": evaluations - 1,
                    "score": round(score.total, 4),
                    "feasible": score.feasible,
                    "violations": len(score.violations),
                    "improved": True,
                }
            )

    return OptimizationResult(
        algorithm="random_search",
        initial_plan=clone_plan(plan),
        best_plan=best_plan,
        initial_score=initial_score,
        best_score=best_score,
        evaluations=evaluations,
        feasible_layouts=feasible_layouts,
        history=history,
        config=settings,
    )
