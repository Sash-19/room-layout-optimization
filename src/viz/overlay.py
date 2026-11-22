from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..geometry.camera import CameraIntrinsics
from ..geometry.floor_plane import FloorPlane
from ..geometry.room_boundary import FloorFrame
from ..utils import load_image

CLASS_COLORS = {
    "bed": (52, 120, 246),
    "desk": (245, 146, 34),
    "chair": (46, 184, 92),
    "couch": (155, 89, 182),
    "table": (26, 188, 156),
    "cabinet": (139, 94, 60),
}
ROOM_COLOR = (230, 40, 40)


def _camera(plan) -> CameraIntrinsics | None:
    payload = plan.camera
    if not payload:
        return None
    try:
        return CameraIntrinsics(
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            source=str(payload.get("source", "floorplan")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _frame(plan) -> FloorFrame | None:
    metadata = plan.metadata or {}
    frame_payload = (metadata.get("room_estimate") or {}).get("frame")
    if frame_payload:
        try:
            return FloorFrame(
                normal=frame_payload["normal"],
                u=frame_payload["u"],
                v=frame_payload["v"],
                origin=frame_payload["origin"],
            )
        except (KeyError, TypeError, ValueError):
            return None
    plane_payload = metadata.get("plane")
    if plane_payload:
        try:
            plane = FloorPlane(
                normal=np.asarray(plane_payload["normal"], dtype=float),
                offset=float(plane_payload["offset"]),
            )
            return FloorFrame.from_plane(plane)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def can_overlay(plan) -> bool:
    return _camera(plan) is not None and _frame(plan) is not None


def _project_polygon(coords: list[tuple[float, float]], frame: FloorFrame, camera: CameraIntrinsics):
    points = frame.to_3d(np.asarray(coords, dtype=float))
    if np.any(points[:, 2] <= 1e-6):
        return None
    pixels = camera.project(points)
    if not np.all(np.isfinite(pixels)):
        return None
    return [(float(x), float(y)) for x, y in pixels]


def overlay_plan(
    image: str | Path | Image.Image,
    plan,
    draw_labels: bool = True,
    alpha: float = 1.0,
) -> Image.Image:
    if not isinstance(image, Image.Image):
        image = load_image(image)
    canvas = image.convert("RGB")
    camera = _camera(plan)
    frame = _frame(plan)
    if camera is None or frame is None:
        return canvas
    draw_image = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_image)
    line_width = max(2, canvas.size[0] // 320)

    room_coords = _project_polygon(list(plan.room.polygon.exterior.coords), frame, camera)
    if room_coords:
        draw.line(room_coords + [room_coords[0]], fill=(*ROOM_COLOR, 220), width=line_width)

    for item in plan.room.furniture:
        coords = _project_polygon(list(item.footprint.exterior.coords), frame, camera)
        if not coords:
            continue
        color = CLASS_COLORS.get(item.cls, (90, 90, 90))
        draw.polygon(coords, fill=(*color, 90), outline=(*color, 240), width=line_width)
        if draw_labels:
            center = _project_polygon([(item.x, item.y)], frame, camera)
            if center:
                x, y = center[0]
                text = f"{item.cls} ({item.w:.2f}x{item.d:.2f})"
                draw.text((x + 4, y - 8), text, fill=(*color, 255))
    return Image.alpha_composite(canvas.convert("RGBA"), draw_image).convert("RGB")


def save_overlay(
    image: str | Path | Image.Image,
    plan,
    path: str | Path,
    draw_labels: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay_plan(image, plan, draw_labels=draw_labels).save(path)
    return path