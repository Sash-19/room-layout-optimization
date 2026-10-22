from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..schemas import DepthMap


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def resize_depth(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    array = np.asarray(depth, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"depth map must be 2D, got shape {array.shape}")
    width, height = int(size[0]), int(size[1])
    if array.shape == (height, width):
        return array
    image = Image.fromarray(np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0), mode="F")
    return np.asarray(image.resize((width, height), Image.BILINEAR), dtype=np.float32)


def postprocess_depth(
    depth: np.ndarray,
    target_size: tuple[int, int],
    metric: bool = False,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> np.ndarray:
    array = resize_depth(depth, target_size)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    if metric:
        array = np.where(finite, array, max_depth)
        return np.clip(array, min_depth, max_depth).astype(np.float32)
    values = array[finite]
    low, high = np.percentile(values, [2.0, 98.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    normalized[~finite] = 0.0
    return normalized.astype(np.float32)


class DepthEstimator:
    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str | int | None = None,
        metric: bool = False,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
    ) -> None:
        self.model_id = str(model_id)
        self.device = device
        self.metric = bool(metric)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self._processor: Any | None = None
        self._model: Any | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DepthEstimator:
        return cls(
            model_id=config.get("model_id", "depth-anything/Depth-Anything-V2-Small-hf"),
            device=config.get("device"),
            metric=config.get("metric", False),
            min_depth=config.get("min_depth_m", 0.1),
            max_depth=config.get("max_depth_m", 10.0),
        )

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for depth estimation; install it with `pip install transformers`"
                ) from exc
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        return self._processor, self._model

    def predict(self, image: Image.Image | str) -> DepthMap:
        if isinstance(image, str):
            with Image.open(image) as handle:
                image = handle.convert("RGB")
        processor, model = self._load()
        if self.device is not None and hasattr(model, "to"):
            model.to(self.device)
        inputs = processor(images=image, return_tensors="pt")
        if self.device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        import torch

        with torch.no_grad():
            outputs = model(**inputs)
        predicted = _to_numpy(outputs.predicted_depth)
        depth = postprocess_depth(
            predicted,
            target_size=image.size,
            metric=self.metric,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )
        return DepthMap(depth=depth, metric=self.metric, source=self.model_id)
