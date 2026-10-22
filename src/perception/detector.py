from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from ..schemas import Detection
from .taxonomy import resolve_class_name

_IMAGE_TYPES = (str, Path, Image.Image, np.ndarray)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _names_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def _resize_masks(masks: np.ndarray, orig_shape: tuple[int, int]) -> np.ndarray:
    if masks.shape[1:] == orig_shape:
        return masks
    import torch
    from torch.nn.functional import interpolate

    tensor = interpolate(
        torch.from_numpy(np.ascontiguousarray(masks)).float().unsqueeze(1),
        size=orig_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return tensor.numpy()


def detections_from_result(
    result: Any,
    class_map: dict[str, str | None] | None = None,
    drop_unknown: bool = True,
) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
    conf = _to_numpy(boxes.conf).reshape(-1)
    classes = _to_numpy(boxes.cls).reshape(-1).astype(int)
    names = _names_dict(result.names)

    masks = None
    raw_masks = getattr(getattr(result, "masks", None), "data", None)
    if raw_masks is not None:
        mask_array = _to_numpy(raw_masks)
        if mask_array.size:
            orig_shape = tuple(getattr(result, "orig_shape", mask_array.shape[1:]))
            masks = _resize_masks(mask_array, orig_shape) > 0.5

    detections: list[Detection] = []
    for index, (box, score, class_id) in enumerate(zip(xyxy, conf, classes)):
        raw_name = names.get(int(class_id), str(class_id))
        name = resolve_class_name(raw_name, class_map)
        if name is None:
            if drop_unknown:
                continue
            name = raw_name
        mask = masks[index] if masks is not None else None
        detections.append(Detection(name, float(score), tuple(float(v) for v in box), mask))

    detections.sort(key=lambda item: item.conf, reverse=True)
    return detections


class Detector:
    def __init__(
        self,
        weights: str = "yolov8s.pt",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        device: str | int | None = None,
        classes: Iterable[str] | None = None,
        class_map: dict[str, str | None] | None = None,
        drop_unknown: bool = True,
        max_det: int = 300,
        half: bool = False,
        verbose: bool = False,
    ) -> None:
        self.weights = str(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device
        self.classes = set(classes) if classes else None
        self.class_map = dict(class_map) if class_map else None
        self.drop_unknown = bool(drop_unknown)
        self.max_det = int(max_det)
        self.half = bool(half)
        self.verbose = bool(verbose)
        self._model: Any | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Detector:
        return cls(
            weights=config.get("weights", "yolov8s.pt"),
            conf=config.get("conf", 0.25),
            iou=config.get("iou", 0.7),
            imgsz=config.get("imgsz", 640),
            device=config.get("device"),
            classes=config.get("classes") or None,
            class_map=config.get("class_map") or None,
            drop_unknown=config.get("drop_unknown", True),
            max_det=config.get("max_det", 300),
            half=config.get("half", False),
            verbose=config.get("verbose", False),
        )

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "ultralytics is required for detection; install it with `pip install ultralytics`"
                ) from exc
            self._model = YOLO(self.weights)
        return self._model

    def detect(self, image: str | Path | Image.Image | np.ndarray) -> list[Detection]:
        if not isinstance(image, _IMAGE_TYPES):
            raise TypeError(f"unsupported image type: {type(image)!r}")
        predict_args = {
            "source": image,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "max_det": self.max_det,
            "verbose": self.verbose,
            "stream": False,
        }
        if self.half:
            predict_args["half"] = True
        result = self.model.predict(**predict_args)[0]
        detections = detections_from_result(result, class_map=self.class_map, drop_unknown=self.drop_unknown)
        if self.classes is not None:
            detections = [detection for detection in detections if detection.cls_name in self.classes]
        return detections

    def train(self, data: str | Path, **overrides: Any) -> Any:
        data_path = Path(data)
        if not data_path.exists():
            raise FileNotFoundError(f"dataset yaml not found: {data_path}")
        return self.model.train(data=str(data_path), **overrides)

    def validate(self, data: str | Path, **overrides: Any) -> Any:
        data_path = Path(data)
        if not data_path.exists():
            raise FileNotFoundError(f"dataset yaml not found: {data_path}")
        return self.model.val(data=str(data_path), **overrides)
