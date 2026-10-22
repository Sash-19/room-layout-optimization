import numpy as np

from src.perception.depth import postprocess_depth, resize_depth


def test_resize_depth_matches_target_size():
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    resized = resize_depth(depth, (4, 6))
    assert resized.shape == (6, 4)
    assert np.all(np.isfinite(resized))


def test_metric_depth_is_clipped_and_filled():
    depth = np.array([[0.5, 2.0], [5.0, 20.0]], dtype=np.float32)
    result = postprocess_depth(depth, (2, 2), metric=True, min_depth=1.0, max_depth=10.0)
    np.testing.assert_allclose(result, np.array([[1.0, 2.0], [5.0, 10.0]], dtype=np.float32))


def test_relative_depth_is_normalized():
    depth = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    result = postprocess_depth(depth, (2, 2), metric=False)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    assert result.max() == 1.0
    assert result.min() == 0.0


def test_nonfinite_depth_is_handled():
    depth = np.array([[1.0, np.inf], [np.nan, 2.0]], dtype=np.float32)
    metric = postprocess_depth(depth, (2, 2), metric=True, min_depth=0.5, max_depth=5.0)
    assert np.all(np.isfinite(metric))
    relative = postprocess_depth(depth, (2, 2), metric=False)
    assert np.all(np.isfinite(relative))
