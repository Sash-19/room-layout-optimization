from __future__ import annotations

import re

UNIFIED_CLASSES: tuple[str, ...] = (
    "bed",
    "desk",
    "chair",
    "couch",
    "table",
    "cabinet",
    "door",
    "window",
)

MOVABLE_CLASSES: frozenset[str] = frozenset({"bed", "desk", "chair", "couch", "table", "cabinet"})
FIXED_CLASSES: frozenset[str] = frozenset({"door", "window"})

CLASS_ALIASES: dict[str, str] = {
    "bed": "bed",
    "bunk bed": "bed",
    "double bed": "bed",
    "single bed": "bed",
    "desk": "desk",
    "writing desk": "desk",
    "office desk": "desk",
    "computer desk": "desk",
    "chair": "chair",
    "armchair": "chair",
    "dining chair": "chair",
    "office chair": "chair",
    "stool": "chair",
    "couch": "couch",
    "sofa": "couch",
    "settee": "couch",
    "loveseat": "couch",
    "table": "table",
    "dining table": "table",
    "coffee table": "table",
    "side table": "table",
    "kitchen table": "table",
    "cabinet": "cabinet",
    "cupboard": "cabinet",
    "wardrobe": "cabinet",
    "closet": "cabinet",
    "bookcase": "cabinet",
    "bookshelf": "cabinet",
    "book shelf": "cabinet",
    "shelf": "cabinet",
    "dresser": "cabinet",
    "nightstand": "cabinet",
    "chest of drawers": "cabinet",
    "tv stand": "cabinet",
    "door": "door",
    "doorway": "door",
    "glass door": "door",
    "sliding door": "door",
    "window": "window",
    "windows": "window",
    "windowpane": "window",
    "glass window": "window",
}

_ALIAS_LOOKUP = {re.sub(r"[\s_\-]+", " ", key.strip().lower()): value for key, value in CLASS_ALIASES.items()}


def normalize_name(name: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(name).strip().lower())


def canonical_class(name: str) -> str | None:
    return _ALIAS_LOOKUP.get(normalize_name(name))


def resolve_class_name(raw_name: str, class_map: dict[str, str | None] | None = None) -> str | None:
    normalized = normalize_name(raw_name)
    if class_map:
        for key, mapped in class_map.items():
            if normalize_name(key) != normalized:
                continue
            if mapped is None:
                return None
            return canonical_class(str(mapped))
    return canonical_class(raw_name)


def is_unified_class(name: str) -> bool:
    return name in UNIFIED_CLASSES


def class_index(name: str) -> int:
    return UNIFIED_CLASSES.index(name)
