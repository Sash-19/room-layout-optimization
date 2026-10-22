from __future__ import annotations

from pathlib import Path

from ..floorplan.representation import FloorPlan

CLASS_COLORS = {
    "bed": "#3478f6",
    "desk": "#f59222",
    "chair": "#2eb85c",
    "couch": "#9b59b6",
    "table": "#1abc9c",
    "cabinet": "#8b5e3c",
}


def plot_floorplan(
    floorplan: FloorPlan,
    title: str | None = None,
    ax=None,
    show_camera: bool = True,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Polygon as MplPolygon

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    room = floorplan.room
    room_min_x, room_min_y, room_max_x, room_max_y = room.bounds
    min_x, min_y, max_x, max_y = room.bounds
    coords = list(room.polygon.exterior.coords)
    ax.add_patch(MplPolygon(coords, closed=True, facecolor="#f2f2f2", edgecolor="#222222", linewidth=2.0))
    ax.plot([value[0] for value in coords], [value[1] for value in coords], color="#222222", linewidth=2.0)

    for item in room.furniture:
        footprint = list(item.footprint.exterior.coords)
        color = CLASS_COLORS.get(item.cls, "#7f8c8d")
        ax.add_patch(MplPolygon(footprint, closed=True, facecolor=color, edgecolor="#1b1b1b", alpha=0.65, linewidth=1.0))
        ax.text(item.x, item.y, f"{item.cls}\n{item.w:.2f}x{item.d:.2f}", ha="center", va="center", fontsize=7)

    width_x = max_x - min_x
    depth_y = max_y - min_y
    for item in room.furniture:
        item_min_x, item_min_y, item_max_x, item_max_y = item.footprint.bounds
        min_x = min(min_x, item_min_x)
        min_y = min(min_y, item_min_y)
        max_x = max(max_x, item_max_x)
        max_y = max(max_y, item_max_y)
    for opening in room.openings:
        color = "#e74c3c" if opening.kind == "door" else "#34c7de"
        if opening.wall_index == 0:
            start, end = (room_min_x + opening.offset, room_min_y), (room_min_x + opening.offset + opening.width, room_min_y)
        elif opening.wall_index == 2:
            start, end = (room_min_x + opening.offset, room_max_y), (room_min_x + opening.offset + opening.width, room_max_y)
        elif opening.wall_index == 1:
            start, end = (room_max_x, room_min_y + opening.offset), (room_max_x, room_min_y + opening.offset + opening.width)
        else:
            start, end = (room_min_x, room_min_y + opening.offset), (room_min_x, room_min_y + opening.offset + opening.width)
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=5, solid_capstyle="butt")
        if opening.kind == "door" and opening.swing_radius > 0:
            ax.add_patch(
                Arc(
                    end,
                    width=2 * opening.swing_radius,
                    height=2 * opening.swing_radius,
                    theta1=0,
                    theta2=90,
                    color=color,
                    linestyle="--",
                    linewidth=1,
                )
            )

    if show_camera:
        ax.plot([0.0], [0.0], marker="*", markersize=14, color="#111111")
        ax.text(0.0, 0.0, " camera", fontsize=8, va="bottom", clip_on=True)

    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(min_y - 0.5, max_y + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("x (m)" if floorplan.scale_mode in {"metric", "reference", "user"} else "x (units)")
    ax.set_ylabel("y (m)" if floorplan.scale_mode in {"metric", "reference", "user"} else "y (units)")
    if title is None:
        title = f"floor plan ({floorplan.scale_mode}, area {room.area:.1f})"
    ax.set_title(title)
    ax.text(
        0.01,
        0.01,
        f"{width_x:.2f} x {depth_y:.2f}  scale={floorplan.scale:.3g}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    return ax.figure, ax


def save_floorplan(floorplan: FloorPlan, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_floorplan(floorplan)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path
