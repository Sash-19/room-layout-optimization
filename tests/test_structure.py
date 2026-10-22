import numpy as np

from src.perception.structure import labels_to_masks, masks_from_logits, resize_labels


def test_labels_to_masks_resizes_and_splits_classes():
    labels = np.array([[3, 3], [0, 0]], dtype=np.int64)
    masks = labels_to_masks(labels, target_size=(4, 4), floor_ids=(3,), wall_ids=(0,))
    assert masks.floor.sum() == 8
    assert masks.wall.sum() == 8
    assert masks.floor[:2].all()
    assert masks.wall[2:].all()


def test_masks_from_logits_uses_argmax():
    logits = np.zeros((4, 2, 2), dtype=np.float32)
    logits[1] = 1.0
    logits[3, 0, 0] = 5.0
    logits[0, 1, 1] = 5.0
    masks = masks_from_logits(logits, target_size=(2, 2))
    assert masks.floor[0, 0]
    assert masks.wall[1, 1]
    assert masks.floor.sum() == 1
    assert masks.wall.sum() == 1


def test_resize_labels_is_nearest_neighbor():
    labels = np.array([[1, 2]], dtype=np.int64)
    resized = resize_labels(labels, (4, 2))
    np.testing.assert_array_equal(resized[0], [1, 1, 2, 2])
    np.testing.assert_array_equal(resized[1], [1, 1, 2, 2])
