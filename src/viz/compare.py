from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..geometry.visualize import plot_floorplan
from ..scoring.grid import OccupancyGrid
from ..scoring.score import ScoreResult


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def plot_walkability(plan, ax=None):
    """Draw the free-space distance transform used by the walk term."""
    plt = _pyplot()
    from matplotlib.patches import Polygon as MplPolygon

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    room = plan.room
    grid = OccupancyGrid.from_room(room, resolution=0.05)
    values = grid.distance.copy()
    values[~grid.inside] = float("nan")
    x0, y0, _, _ = room.bounds
    extent = (x0, x0 + grid.width * grid.resolution, y0, y0 + grid.height * grid.resolution)
    image = ax.imshow(values, origin="lower", extent=extent, cmap="viridis")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="distance to obstacle (m)")
    for item in room.furniture:
        coords = list(item.footprint.exterior.coords)
        ax.add_patch(
            MplPolygon(coords, closed=True, facecolor="none", edgecolor="#ffffff", linewidth=1.2)
        )
        ax.text(item.x, item.y, item.cls, color="#ffffff", ha="center", va="center", fontsize=7)
    coords = list(room.polygon.exterior.coords)
    ax.add_patch(MplPolygon(coords, closed=True, facecolor="none", edgecolor="#111111", linewidth=1.8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Walkability distance transform")
    return ax.figure, ax


def save_walkability(plan, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    figure, _ = plot_walkability(plan)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def plot_score_bars(scores: dict[str, ScoreResult], ax=None):
    plt = _pyplot()
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(scores)), 4.5))
    if not scores:
        ax.text(0.5, 0.5, "no scores", ha="center", va="center")
        return ax.figure, ax
    terms = sorted({term for score in scores.values() for term in score.terms})
    labels = list(scores)
    x = np.arange(len(terms))
    width = 0.8 / max(1, len(labels))
    for index, label in enumerate(labels):
        values = [scores[label].terms.get(term, 0.0) for term in terms]
        ax.bar(x + (index - (len(labels) - 1) / 2.0) * width, values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("normalized term score")
    ax.set_title("Score breakdown")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    return ax.figure, ax


def save_score_bars(scores: dict[str, ScoreResult], path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    figure, _ = plot_score_bars(scores)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def save_before_after(
    initial_plan,
    initial_score: ScoreResult,
    best_plan,
    best_score: ScoreResult,
    path: str | Path,
    title: str | None = None,
    dpi: int = 150,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_floorplan(
        initial_plan,
        title=(
            f"Detected layout\nscore {initial_score.total:.3f} "
            f"({'feasible' if initial_score.feasible else 'infeasible'})"
        ),
        ax=axes[0],
    )
    plot_floorplan(
        best_plan,
        title=(
            f"Optimized layout\nscore {best_score.total:.3f} "
            f"({'feasible' if best_score.feasible else 'infeasible'})"
        ),
        ax=axes[1],
    )
    delta = best_score.total - initial_score.total
    figure.suptitle(title or f"Before / after (delta {delta:+.3f})")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def save_comparison_grid(
    layouts: list[tuple[str, Any, ScoreResult]],
    path: str | Path,
    columns: int = 3,
    dpi: int = 150,
) -> Path:
    layouts = list(layouts)
    if not layouts:
        raise ValueError("layouts must not be empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    columns = max(1, min(columns, len(layouts)))
    rows = int(math.ceil(len(layouts) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (label, plan, score) in zip(axes.ravel(), layouts):
        ax.axis("on")
        plot_floorplan(
            plan,
            title=f"{label}\nscore {score.total:.3f} ({'feasible' if score.feasible else 'infeasible'})",
            ax=ax,
            show_camera=False,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def save_optimizer_curves(
    curves: list[dict[str, Any]],
    path: str | Path,
    max_rooms: int = 6,
    dpi: int = 150,
) -> Path:
    """Plot best-score-so-far against evaluation count for E4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    rooms = sorted({item["room"] for item in curves})[:max_rooms]
    if not rooms:
        raise ValueError("curves must not be empty")
    columns = min(3, len(rooms))
    rows = int(math.ceil(len(rooms) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.8 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, room in zip(axes.ravel(), rooms):
        ax.axis("on")
        for item in [curve for curve in curves if curve["room"] == room]:
            curve = item["curve"]
            ax.plot(curve["evaluations"], curve["best_scores"], label=item["algorithm"], linewidth=1.6)
        ax.set_title(room)
        ax.set_xlabel("scorer evaluations")
        ax.set_ylabel("best score so far")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path