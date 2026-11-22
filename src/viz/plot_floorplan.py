from __future__ import annotations

from pathlib import Path

from ..geometry.visualize import CLASS_COLORS, plot_floorplan, save_floorplan

__all__ = ["CLASS_COLORS", "plot_floorplan", "save_floorplan"]


def save_plan(floorplan, path: str | Path, dpi: int = 150) -> Path:
    return save_floorplan(floorplan, path, dpi=dpi)