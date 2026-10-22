from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..schemas import Detection, StructureMasks

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "bed": (52, 120, 246),
    "desk": (245, 146, 34),
    "chair": (46, 184, 92),
    "couch": (155, 89, 182),
    "table": (26, 188, 156),
    "cabinet": (139, 94, 60),
    "door": (231, 76, 60),
    "window": (52, 199, 222),
}
DEFAULT_COLOR = (160, 160, 160)


def _blend_mask(canvas: np.ndarray, mask: np.ndarray, color: Iterable[int], alpha: float) -> None:
    if mask.shape != canvas.shape[:2]:
        return
    rgb = np.asarray(list(color), dtype=np.float32)
    canvas[mask] = canvas[mask] * (1.0 - alpha) + rgb * alpha


def draw_detections(
    image: Image.Image,
    detections: Iterable[Detection],
    show_masks: bool = True,
    mask_alpha: float = 0.35,
) -> Image.Image:
    detections = list(detections)
    base = image.convert("RGB")
    canvas = np.asarray(base, dtype=np.float32).copy()
    if show_masks:
        for detection in detections:
            if detection.mask is None:
                continue
            _blend_mask(canvas, detection.mask, CLASS_COLORS.get(detection.cls_name, DEFAULT_COLOR), mask_alpha)
    rendered = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    line_width = max(2, int(round(min(rendered.size) / 300)))

    for detection in detections:
        color = CLASS_COLORS.get(detection.cls_name, DEFAULT_COLOR)
        draw.rectangle(detection.bbox, outline=color, width=line_width)
        label = f"{detection.cls_name} {detection.conf:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = right - left, bottom - top
        x = max(0, int(detection.bbox[0]))
        y = max(0, int(detection.bbox[1]) - text_height - 4)
        draw.rectangle((x, y, x + text_width + 4, y + text_height + 4), fill=color)
        draw.text((x + 2, y + 2), label, fill=(255, 255, 255), font=font)
    return rendered


def draw_structure_overlay(
    image: Image.Image,
    structures: StructureMasks,
    alpha: float = 0.4,
) -> Image.Image:
    base = image.convert("RGB").resize((structures.shape[1], structures.shape[0]))
    canvas = np.asarray(base, dtype=np.float32).copy()
    _blend_mask(canvas, structures.floor, (64, 170, 255), alpha)
    _blend_mask(canvas, structures.wall, (235, 70, 70), alpha)
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def colorize_depth(
    depth: np.ndarray,
    invert: bool = False,
    colormap: str = "turbo",
) -> Image.Image:
    import matplotlib

    array = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return Image.fromarray(np.zeros((*array.shape, 3), dtype=np.uint8))
    values = array[finite]
    low, high = float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))
    if high <= low:
        low, high = float(values.min()), float(values.max())
    scaled = np.zeros_like(array, dtype=np.float32) if high <= low else (array - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    if invert:
        scaled = 1.0 - scaled
    scaled[~finite] = 0.0
    rgba = matplotlib.colormaps[colormap](scaled)
    return Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))


def save_mask(mask: np.ndarray, path: str) -> None:
    Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8), mode="L").save(path)
