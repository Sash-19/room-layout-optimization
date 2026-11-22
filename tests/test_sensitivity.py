import pytest

from src.optimization.sensitivity import ablate_weight, normalize_weights, weight_variant, weight_variants


def test_normalize_weights_sums_to_one():
    weights = normalize_weights({"a": 2.0, "b": 1.0, "c": 1.0})
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] == pytest.approx(0.5)


def test_normalize_weights_rejects_all_zero():
    with pytest.raises(ValueError):
        normalize_weights({"a": 0.0, "b": 0.0})


def test_weight_variant_scales_and_renormalizes():
    base = {"walk": 0.2, "door": 0.2, "wall": 0.2, "util": 0.2, "clear": 0.2}
    variant = weight_variant(base, "walk", 4.0)
    assert sum(variant.values()) == pytest.approx(1.0)
    assert variant["walk"] > base["walk"]


def test_weight_variants_includes_baseline_and_perturbations():
    base = {"walk": 0.3, "door": 0.7}
    variants = weight_variants(base, factors=(0.5, 2.0))
    assert variants["baseline"] == pytest.approx(base, abs=1e-9)
    assert "walk_x0.5" in variants
    assert "walk_x2" in variants
    assert "door_x2" in variants


def test_ablate_weight_zeroes_term_and_renormalizes():
    base = {"walk": 0.2, "door": 0.8}
    ablated = ablate_weight(base, "walk")
    assert ablated["walk"] == pytest.approx(0.0)
    assert ablated["door"] == pytest.approx(1.0)


def test_ablate_weight_unknown_term():
    with pytest.raises(KeyError):
        ablate_weight({"walk": 1.0}, "nope")
