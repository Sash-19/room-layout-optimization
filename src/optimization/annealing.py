from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from random import Random
from typing import Any

from ..floorplan.representation import FloorPlan, Furniture, Room
from ..scoring.constraints import check_constraints
from ..scoring.score import LayoutScorer, ScoreResult
from .base import OptimizationResult, clone_plan, score_rank
from .sampling import (
    candidate_is_valid,
    optimizer_settings,
    sample_layout,
    sample_pose,
    scoring_settings,
)


def _with_furniture(plan: FloorPlan, furniture: list[Furniture]) -> FloorPlan:
    return FloorPlan(
        room=Room(plan.room.polygon, list(plan.room.openings), furniture),
        scale=plan.scale,
        scale_mode=plan.scale_mode,
        camera_height=plan.camera_height,
        camera=plan.camera,
        metadata=plan.metadata,
    )


def _valid_layout(room: Room, furniture: list[Furniture], config: dict | None) -> bool:
    return not check_constraints(Room(room.polygon, list(room.openings), furniture), scoring_settings(config))


def _move_translate(plan: FloorPlan, index: int, rng: Random, config: dict) -> FloorPlan | None:
    settings = optimizer_settings(config)
    step = float(settings.get("annealing", {}).get("step_m", 0.4))
    attempts = int(settings.get("annealing", {}).get("move_attempts", 25))
    room = plan.room
    item = room.furniture[index]
    others = [other for other_index, other in enumerate(room.furniture) if other_index != index]
    for _ in range(max(1, attempts)):
        candidate = replace(item, x=item.x + rng.gauss(0.0, step), y=item.y + rng.gauss(0.0, step))
        if candidate_is_valid(room, others, candidate, config):
            furniture = list(room.furniture)
            furniture[index] = candidate
            return _with_furniture(plan, furniture)
    return None


def _move_rotate(plan: FloorPlan, index: int, rng: Random, config: dict) -> FloorPlan | None:
    settings = optimizer_settings(config)
    step = math.radians(float(settings.get("rotation_step_deg", 90.0)))
    attempts = int(settings.get("annealing", {}).get("move_attempts", 25))
    room = plan.room
    item = room.furniture[index]
    others = [other for other_index, other in enumerate(room.furniture) if other_index != index]
    for _ in range(max(1, attempts)):
        theta = item.theta + rng.choice((-step, step))
        candidate = replace(item, theta=theta)
        if candidate_is_valid(room, others, candidate, config):
            furniture = list(room.furniture)
            furniture[index] = candidate
            return _with_furniture(plan, furniture)
    return None


def _move_wall(plan: FloorPlan, index: int, rng: Random, config: dict) -> FloorPlan | None:
    room = plan.room
    item = room.furniture[index]
    others = [other for other_index, other in enumerate(room.furniture) if other_index != index]
    candidate = sample_pose(room, item, others, rng, config, wall_snap_probability=1.0)
    if candidate is None:
        return None
    furniture = list(room.furniture)
    furniture[index] = candidate
    return _with_furniture(plan, furniture)


def _move_swap(plan: FloorPlan, index: int, rng: Random, config: dict) -> FloorPlan | None:
    room = plan.room
    movable = [i for i, item in enumerate(room.furniture) if not item.locked]
    if len(movable) < 2:
        return None
    other_index = rng.choice([i for i in movable if i != index])
    first = room.furniture[index]
    second = room.furniture[other_index]
    furniture = list(room.furniture)
    furniture[index] = replace(first, x=second.x, y=second.y, theta=second.theta)
    furniture[other_index] = replace(second, x=first.x, y=first.y, theta=first.theta)
    return _with_furniture(plan, furniture) if _valid_layout(room, furniture, config) else None


def perturb(plan: FloorPlan, rng: Random, config: dict | None = None) -> FloorPlan | None:
    movable = [index for index, item in enumerate(plan.room.furniture) if not item.locked]
    if not movable:
        return None
    index = rng.choice(movable)
    item = plan.room.furniture[index]
    draw = rng.random()
    if item.wall_affinity and draw < 0.2:
        return _move_wall(plan, index, rng, config)
    if draw < 0.45:
        return _move_rotate(plan, index, rng, config)
    if draw < 0.55:
        return _move_swap(plan, index, rng, config)
    return _move_translate(plan, index, rng, config)


def _layout_key(plan: FloorPlan) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (round(item.x, 2), round(item.y, 2), round(item.theta, 2))
        for item in plan.room.furniture
    )


def simulated_annealing(
    plan: FloorPlan,
    scorer: LayoutScorer | None = None,
    config: dict[str, Any] | None = None,
) -> OptimizationResult:
    settings = optimizer_settings(config)
    annealing = settings.get("annealing", {})
    scorer = scorer or LayoutScorer.from_config(scoring_settings(config))
    rng = Random(int(settings.get("seed", 0)))
    iterations = int(annealing.get("iterations", 600))
    restarts = max(1, int(annealing.get("restarts", 2)))
    start_temp = float(annealing.get("start_temp", 0.05))
    end_temp = float(annealing.get("end_temp", 0.001))
    top_k = max(1, int(annealing.get("top_k", 3)))
    warm_start = str(annealing.get("warm_start", "")).lower()

    initial_score = scorer.evaluate(plan)
    best_plan = clone_plan(plan)
    best_score = initial_score
    evaluations = 1
    feasible_layouts = 1 if initial_score.feasible else 0
    history = [
        {
            "evaluation": 0,
            "restart": 0,
            "iteration": 0,
            "temperature": round(start_temp, 6),
            "score": round(initial_score.total, 4),
            "feasible": initial_score.feasible,
            "violations": len(initial_score.violations),
        }
    ]
    top_plans: list[FloorPlan] = []
    top_scores: list[ScoreResult] = []

    def remember(candidate: FloorPlan, score: ScoreResult) -> None:
        if not score.feasible:
            return
        key = _layout_key(candidate)
        for index, existing in enumerate(top_plans):
            if _layout_key(existing) == key:
                if score_rank(score) > score_rank(top_scores[index]):
                    top_plans[index] = clone_plan(candidate)
                    top_scores[index] = score
                return
        top_plans.append(clone_plan(candidate))
        top_scores.append(score)
        order = sorted(range(len(top_plans)), key=lambda i: score_rank(top_scores[i]), reverse=True)
        top_plans[:] = [top_plans[i] for i in order[:top_k]]
        top_scores[:] = [top_scores[i] for i in order[:top_k]]

    warm_plan: FloorPlan | None = None
    if warm_start in {"greedy", "random"}:
        if warm_start == "greedy":
            from .greedy import greedy_search

            warm = greedy_search(plan, scorer, config)
        else:
            from .random_search import random_search

            warm = random_search(plan, scorer, config)
        evaluations += max(0, warm.evaluations - 1)
        feasible_layouts += max(0, warm.feasible_layouts - (1 if warm.initial_score.feasible else 0))
        warm_plan = warm.best_plan
        if score_rank(warm.best_score) > score_rank(best_score):
            best_plan, best_score = clone_plan(warm.best_plan), warm.best_score
        remember(warm.best_plan, warm.best_score)
        history.append(
            {
                "evaluation": evaluations - 1,
                "restart": 0,
                "iteration": 0,
                "temperature": round(start_temp, 6),
                "score": round(warm.best_score.total, 4),
                "feasible": warm.best_score.feasible,
                "violations": len(warm.best_score.violations),
                "warm_start": warm_start,
            }
        )

    remember(plan, initial_score)
    for restart in range(restarts):
        if restart == 0:
            current_plan = clone_plan(warm_plan or plan)
        else:
            current_plan = sample_layout(plan, rng, config) or clone_plan(plan)
        current_score = scorer.evaluate(current_plan)
        evaluations += 1
        if current_score.feasible:
            feasible_layouts += 1
        if score_rank(current_score) > score_rank(best_score):
            best_plan, best_score = clone_plan(current_plan), current_score

        for iteration in range(1, iterations + 1):
            progress = iteration / max(1, iterations)
            temperature = start_temp * ((end_temp / start_temp) ** progress) if start_temp > 0 else 0.0
            candidate = perturb(current_plan, rng, config)
            if candidate is None:
                continue
            score = scorer.evaluate(candidate)
            evaluations += 1
            if score.feasible:
                feasible_layouts += 1

            current_rank = score_rank(current_score)
            candidate_rank = score_rank(score)
            if candidate_rank > current_rank:
                accept = True
            elif candidate_rank[0] == current_rank[0] and temperature > 0:
                accept = rng.random() < math.exp((score.total - current_score.total) / temperature)
            else:
                accept = False
            if accept:
                current_plan, current_score = candidate, score

            if score_rank(score) > score_rank(best_score):
                best_plan, best_score = clone_plan(candidate), score
                history.append(
                    {
                        "evaluation": evaluations - 1,
                        "restart": restart,
                        "iteration": iteration,
                        "temperature": round(temperature, 6),
                        "score": round(score.total, 4),
                        "feasible": score.feasible,
                        "violations": len(score.violations),
                    }
                )
            remember(candidate, score)

    if not top_plans:
        top_plans = [clone_plan(best_plan)]
        top_scores = [best_score]
    return OptimizationResult(
        algorithm="simulated_annealing",
        initial_plan=clone_plan(plan),
        best_plan=best_plan,
        initial_score=initial_score,
        best_score=best_score,
        evaluations=evaluations,
        feasible_layouts=feasible_layouts,
        history=history,
        config=settings,
        top_plans=top_plans,
        top_scores=top_scores,
    )
