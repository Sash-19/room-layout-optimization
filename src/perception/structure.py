from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from PIL import Image

from ..schemas import StructureMasks


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def resize_labels(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"label map must be 2D, got shape {labels.shape}")
    width, height = int(size[0]), int(size[1])
    if labels.shape == (height, width):
        return labels.astype(np.int64, copy=False)
    image = Image.fromarray(labels.astype(np.int32), mode="I")
    resized = image.resize((width, height), Image.NEAREST)
    return np.asarray(resized, dtype=np.int64)


def labels_to_masks(
    labels: np.ndarray,
    target_size: tuple[int, int],
    floor_ids: Iterable[int] = (3,),
    wall_ids: Iterable[int] = (0,),
) -> StructureMasks:
    resized = resize_labels(labels, target_size)
    floor = np.isin(resized, list(floor_ids))
    wall = np.isin(resized, list(wall_ids))
    return StructureMasks(floor=floor, wall=wall)


def masks_from_logits(
    logits: Any,
    target_size: tuple[int, int],
    floor_ids: Iterable[int] = (3,),
    wall_ids: Iterable[int] = (0,),
) -> StructureMasks:
    array = _to_numpy(logits)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"logits must have shape (C, H, W), got {array.shape}")
    labels = np.argmax(array, axis=0)
    return labels_to_masks(labels, target_size, floor_ids=floor_ids, wall_ids=wall_ids)


class FloorWallSegmenter:
    def __init__(
        self,
        model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: str | int | None = None,
        floor_ids: Iterable[int] = (3,),
        wall_ids: Iterable[int] = (0,),
    ) -> None:
        self.model_id = str(model_id)
        self.device = device
        self.floor_ids = tuple(int(v) for v in floor_ids)
        self.wall_ids = tuple(int(v) for v in wall_ids)
        self._processor: Any | None = None
        self._model: Any | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FloorWallSegmenter:
        return cls(
            model_id=config.get("model_id", "nvidia/segformer-b0-finetuned-ade-512-512"),
            device=config.get("device"),
            floor_ids=config.get("floor_label_ids", (3,)),
            wall_ids=config.get("wall_label_ids", (0,)),
        )

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            try:
                from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for segmentation; install it with `pip install transformers`"
                ) from exc
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForSemanticSegmentation.from_pretrained(self.model_id)
        return self._processor, self._model

    def predict(self, image: Image.Image | str) -> StructureMasks:
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
        from torch.nn.functional import interpolate

        with torch.no_grad():
            outputs = model(**inputs)
        logits = interpolate(outputs.logits, size=(image.size[1], image.size[0]), mode="bilinear", align_corners=False)
        labels = logits.argmax(dim=1)[0].cpu().numpy()
        masks = labels_to_masks(labels, target_size=image.size, floor_ids=self.floor_ids, wall_ids=self.wall_ids)
        masks.source = self.model_id
        return masks
