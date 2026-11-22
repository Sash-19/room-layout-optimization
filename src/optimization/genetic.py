from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from random import Random
from typing import Any

from ..floorplan.representation import FloorPlan, Furniture, Room
from ..scoring.score import LayoutScorer, ScoreResult
from .base import OptimizationResult, clone_plan, score_rank
from .sampling import (
    candidate_is_valid,
    optimizer_settings,
    rotation_options,
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
        camera=deepcopy(plan.camera),
        metadata=deepcopy(plan.metadata),
    )


def _layout_key(plan: FloorPlan) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (round(item.x, 2), round(item.y, 2), round(item.theta, 2))
        for item in plan.room.furniture
    )


def _layout_distance(first: FloorPlan, second: FloorPlan) -> float:
    if len(first.room.furniture) != len(second.room.furniture):
        return float("inf")
    return sum(
        math.hypot(a.x - b.x, a.y - b.y)
        for a, b in zip(first.room.furniture, second.room.furniture)
    )


def _mutate_item(
    plan: FloorPlan,
    index: int,
    rng: Random,
    config: dict[str, Any],
) -> FloorPlan | None:
    settings = optimizer_settings(config)
    genetic = settings.get("genetic", {})
    step = float(genetic.get("mutation_step_m", 0.5))
    attempts = int(genetic.get("mutation_attempts", 20))
    room = plan.room
    item = room.furniture[index]
    others = [other for other_index, other in enumerate(room.furniture) if other_index != index]
    rotations = rotation_options(settings.get("rotation_step_deg", 90.0))

    for _ in range(max(1, attempts)):
        draw = rng.random()
        if item.wall_affinity and draw < 0.2:
            candidate = sample_pose(room, item, others, rng, config, wall_snap_probability=1.0)
        elif draw < 0.45:
            candidate = replace(item, theta=rng.choice(rotations))
        elif draw < 0.55:
            movable = [
                other_index
                for other_index, other in enumerate(room.furniture)
                if other_index != index and not other.locked
            ]
            if not movable:
                continue
            other_index = rng.choice(movable)
            other = room.furniture[other_index]
            furniture = list(room.furniture)
            furniture[index] = replace(item, x=other.x, y=other.y, theta=other.theta)
            furniture[other_index] = replace(other, x=item.x, y=item.y, theta=item.theta)
            changed = _with_furniture(plan, furniture)
            if all(
                candidate_is_valid(
                    room,
                    [other_item for other_pos, other_item in enumerate(furniture) if other_pos != pos],
                    furniture[pos],
                    config,
                )
                for pos in (index, other_index)
            ):
                return changed
            continue
        else:
            candidate = replace(
                item,
                x=item.x + rng.gauss(0.0, step),
                y=item.y + rng.gauss(0.0, step),
            )
        if candidate is not None and candidate_is_valid(room, others, candidate, config):
            furniture = list(room.furniture)
            furniture[index] = candidate
            return _with_furniture(plan, furniture)
    return None


def _mutate(plan: FloorPlan, rng: Random, config: dict[str, Any]) -> FloorPlan:
    genetic = optimizer_settings(config).get("genetic", {})
    probability = float(genetic.get("mutation_probability", 0.35))
    current = plan
    for index, item in enumerate(plan.room.furniture):
        if item.locked or rng.random() >= probability:
            continue
        mutated = _mutate_item(current, index, rng, config)
        if mutated is not None:
            current = mutated
    return current


def _crossover(
    first: FloorPlan,
    second: FloorPlan,
    rng: Random,
    config: dict[str, Any],
) -> FloorPlan:
    room = first.room
    placed: dict[int, Furniture] = {
        index: deepcopy(item) for index, item in enumerate(room.furniture) if item.locked
    }
    for index, item in enumerate(room.furniture):
        if item.locked:
            continue
        source = first if rng.random() < 0.5 else second
        candidate = deepcopy(source.room.furniture[index])
        if candidate_is_valid(room, list(placed.values()), candidate, config):
            placed[index] = candidate
            continue
        fallback = sample_pose(room, item, list(placed.values()), rng, config)
        placed[index] = fallback if fallback is not None else deepcopy(item)
    return _with_furniture(first, [placed[index] for index in range(len(room.furniture))])


def _tournament(
    population: list[FloorPlan],
    scores: list[ScoreResult],
    rng: Random,
    size: int,
) -> FloorPlan:
    indices = rng.sample(range(len(population)), k=min(max(1, size), len(population)))
    best = max(indices, key=lambda index: score_rank(scores[index]))
    return population[best]


def genetic_algorithm(
    plan: FloorPlan,
    scorer: LayoutScorer | None = None,
    config: dict[str, Any] | None = None,
) -> OptimizationResult:
    settings = optimizer_settings(config)
    genetic = settings.get("genetic", {})
    scorer = scorer or LayoutScorer.from_config(scoring_settings(config))
    rng = Random(int(settings.get("seed", 0)))
    population_size = max(2, int(genetic.get("population_size", 32)))
    generations = max(1, int(genetic.get("generations", 40)))
    tournament_size = int(genetic.get("tournament_size", 3))
    elitism = max(0, min(population_size, int(genetic.get("elitism", 2))))
    crossover_probability = float(genetic.get("crossover_probability", 0.75))
    top_k = max(1, int(genetic.get("top_k", 3)))
    min_distance = float(genetic.get("min_layout_distance_m", 0.5))
    max_evaluations = int(genetic.get("max_evaluations", 0))

    initial_score = scorer.evaluate(plan)
    evaluations = 1
    feasible_layouts = 1 if initial_score.feasible else 0
    best_plan = clone_plan(plan)
    best_score = initial_score
    history = [
        {
            "generation": 0,
            "evaluation": 0,
            "score": round(initial_score.total, 4),
            "feasible": initial_score.feasible,
            "violations": len(initial_score.violations),
        }
    ]

    population: list[FloorPlan] = [clone_plan(plan)]
    population_scores: list[ScoreResult] = [initial_score]
    while len(population) < population_size:
        candidate = sample_layout(plan, rng, config)
        if candidate is None:
            candidate = _mutate(clone_plan(plan), rng, config)
        population.append(candidate)
        score = scorer.evaluate(candidate)
        population_scores.append(score)
        evaluations += 1
        if score.feasible:
            feasible_layouts += 1
        if score_rank(score) > score_rank(best_score):
            best_plan, best_score = clone_plan(candidate), score
            history.append(
                {
                    "generation": 0,
                    "evaluation": evaluations - 1,
                    "score": round(score.total, 4),
                    "feasible": score.feasible,
                    "violations": len(score.violations),
                }
            )

    top_plans: list[FloorPlan] = []
    top_scores: list[ScoreResult] = []

    def remember(candidate: FloorPlan, score: ScoreResult) -> None:
        nonlocal top_plans, top_scores
        if not score.feasible:
            return
        key = _layout_key(candidate)
        for index, existing in enumerate(top_plans):
            if _layout_key(existing) == key:
                if score_rank(score) > score_rank(top_scores[index]):
                    top_plans[index] = clone_plan(candidate)
                    top_scores[index] = score
                return
        distances = [_layout_distance(candidate, existing) for existing in top_plans]
        if top_plans and distances and min(distances) < min_distance:
            nearest = min(range(len(distances)), key=distances.__getitem__)
            if score_rank(score) > score_rank(top_scores[nearest]):
                top_plans[nearest] = clone_plan(candidate)
                top_scores[nearest] = score
            return
        top_plans.append(clone_plan(candidate))
        top_scores.append(score)
        order = sorted(range(len(top_plans)), key=lambda index: score_rank(top_scores[index]), reverse=True)
        top_plans[:] = [top_plans[index] for index in order[:top_k]]
        top_scores[:] = [top_scores[index] for index in order[:top_k]]

    remember(plan, initial_score)
    for candidate, score in zip(population, population_scores):
        remember(candidate, score)

    for generation in range(1, generations + 1):
        if max_evaluations and evaluations >= max_evaluations:
            history.append(
                {
                    "generation": generation,
                    "evaluation": evaluations - 1,
                    "score": round(best_score.total, 4),
                    "feasible": best_score.feasible,
                    "violations": len(best_score.violations),
                    "stopped": "max_evaluations",
                }
            )
            break

        order = sorted(range(len(population)), key=lambda index: score_rank(population_scores[index]), reverse=True)
        next_population = [clone_plan(population[index]) for index in order[:elitism]]
        next_scores = [population_scores[index] for index in order[:elitism]]

        while len(next_population) < population_size:
            first = _tournament(population, population_scores, rng, tournament_size)
            second = _tournament(population, population_scores, rng, tournament_size)
            if rng.random() < crossover_probability:
                child = _crossover(first, second, rng, config)
            else:
                child = clone_plan(first)
            child = _mutate(child, rng, config)
            score = scorer.evaluate(child)
            evaluations += 1
            if score.feasible:
                feasible_layouts += 1
            next_population.append(child)
            next_scores.append(score)
            remember(child, score)
            if score_rank(score) > score_rank(best_score):
                best_plan, best_score = clone_plan(child), score
                history.append(
                    {
                        "generation": generation,
                        "evaluation": evaluations - 1,
                        "score": round(score.total, 4),
                        "feasible": score.feasible,
                        "violations": len(score.violations),
                    }
                )
            if max_evaluations and evaluations >= max_evaluations:
                break

        population, population_scores = next_population, next_scores

    if not top_plans:
        top_plans = [clone_plan(best_plan)]
        top_scores = [best_score]
    return OptimizationResult(
        algorithm="genetic",
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