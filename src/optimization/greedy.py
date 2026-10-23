from __future__ import annotations

from copy import deepcopy
from random import Random
from typing import Any

from ..floorplan.representation import FloorPlan, Furniture, Room
from ..scoring.score import LayoutScorer
from .base import OptimizationResult, clone_plan, score_rank
from .sampling import optimizer_settings, sample_pose, scoring_settings


def greedy_search(
    plan: FloorPlan,
    scorer: LayoutScorer | None = None,
    config: dict[str, Any] | None = None,
) -> OptimizationResult:
    settings = optimizer_settings(config)
    scorer = scorer or LayoutScorer.from_config(scoring_settings(config))
    rng = Random(int(settings.get("seed", 0)))
    candidates_per_item = int(settings.get("greedy", {}).get("candidates_per_item", 120))

    initial_score = scorer.evaluate(plan)
    room = plan.room
    evaluations = 1
    feasible_layouts = 1 if initial_score.feasible else 0
    history = [
        {
            "evaluation": 0,
            "stage": "initial",
            "score": round(initial_score.total, 4),
            "feasible": initial_score.feasible,
            "violations": len(initial_score.violations),
        }
    ]

    placed: dict[int, Furniture] = {
        index: deepcopy(item) for index, item in enumerate(room.furniture) if item.locked
    }
    movable = [index for index, item in enumerate(room.furniture) if not item.locked]
    movable.sort(key=lambda index: (not room.furniture[index].wall_affinity, -room.furniture[index].area))

    for index in movable:
        item = room.furniture[index]
        best_candidate: Furniture | None = None
        best_partial = None
        best_rank = None
        for _ in range(max(1, candidates_per_item)):
            candidate = sample_pose(room, item, list(placed.values()), rng, config)
            if candidate is None:
                continue
            partial = Room(room.polygon, list(room.openings), [*placed.values(), candidate])
            score = scorer.evaluate(partial)
            evaluations += 1
            if score.feasible:
                feasible_layouts += 1
            rank = score_rank(score)
            if best_rank is None or rank > best_rank:
                best_candidate = candidate
                best_partial = score
                best_rank = rank

        if best_candidate is None:
            best_candidate = deepcopy(item)
            partial = Room(room.polygon, list(room.openings), [*placed.values(), best_candidate])
            best_partial = scorer.evaluate(partial)
            evaluations += 1
        placed[index] = best_candidate
        history.append(
            {
                "evaluation": evaluations - 1,
                "stage": f"place:{item.cls}",
                "score": round(best_partial.total, 4),
                "feasible": best_partial.feasible,
                "violations": len(best_partial.violations),
            }
        )

    final_plan = FloorPlan(
        room=Room(room.polygon, list(room.openings), [placed[index] for index in range(len(room.furniture))]),
        scale=plan.scale,
        scale_mode=plan.scale_mode,
        camera_height=plan.camera_height,
        camera=deepcopy(plan.camera),
        metadata=deepcopy(plan.metadata),
    )
    final_score = scorer.evaluate(final_plan)
    evaluations += 1
    if final_score.feasible:
        feasible_layouts += 1
    history.append(
        {
            "evaluation": evaluations - 1,
            "stage": "final",
            "score": round(final_score.total, 4),
            "feasible": final_score.feasible,
            "violations": len(final_score.violations),
        }
    )

    if score_rank(final_score) >= score_rank(initial_score):
        best_plan, best_score = final_plan, final_score
    else:
        best_plan, best_score = clone_plan(plan), initial_score

    return OptimizationResult(
        algorithm="greedy",
        initial_plan=clone_plan(plan),
        best_plan=best_plan,
        initial_score=initial_score,
        best_score=best_score,
        evaluations=evaluations,
        feasible_layouts=feasible_layouts,
        history=history,
        config=settings,
    )
