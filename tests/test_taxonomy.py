from src.perception.taxonomy import (
    FIXED_CLASSES,
    MOVABLE_CLASSES,
    UNIFIED_CLASSES,
    canonical_class,
    normalize_name,
    resolve_class_name,
)


def test_taxonomy_sets():
    assert set(UNIFIED_CLASSES) == MOVABLE_CLASSES | FIXED_CLASSES
    assert MOVABLE_CLASSES.isdisjoint(FIXED_CLASSES)


def test_aliases_resolve_to_unified_classes():
    assert canonical_class("Dining Table") == "table"
    assert canonical_class("sofa") == "couch"
    assert canonical_class("Wardrobe") == "cabinet"
    assert canonical_class("windowpane") == "window"
    assert canonical_class("book_shelf") == "cabinet"
    assert canonical_class("toaster") is None


def test_normalize_name():
    assert normalize_name("  Dining-Table ") == "dining table"


def test_class_map_overrides_and_drops():
    class_map = {"Office Chair": "chair", "potted plant": None}
    assert resolve_class_name("office chair", class_map) == "chair"
    assert resolve_class_name("potted plant", class_map) is None
    assert resolve_class_name("Sofa") == "couch"
