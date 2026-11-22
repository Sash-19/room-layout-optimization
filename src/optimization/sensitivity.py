from __future__ import annotations

from typing import Iterable


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(0.0, float(value)) for value in weights.values()))
    if total <= 0:
        raise ValueError("weights must contain at least one positive value")
    return {key: max(0.0, float(value)) / total for key, value in weights.items()}


def weight_variant(
    base: dict[str, float],
    key: str,
    factor: float,
) -> dict[str, float]:
    weights = dict(base)
    weights[key] = max(0.0, float(weights.get(key, 0.0)) * float(factor))
    return normalize_weights(weights)


def ablate_weight(base: dict[str, float], key: str) -> dict[str, float]:
    weights = dict(base)
    if key not in weights:
        raise KeyError(f"unknown weight {key!r}")
    weights[key] = 0.0
    return normalize_weights(weights)


def weight_variants(
    base: dict[str, float],
    keys: Iterable[str] | None = None,
    factors: tuple[float, ...] = (0.25, 4.0),
) -> dict[str, dict[str, float]]:
    keys = list(keys or base)
    variants = {"baseline": normalize_weights(base)}
    for key in keys:
        for factor in factors:
            variants[f"{key}_x{factor:g}"] = weight_variant(base, key, factor)
    return variants


__all__ = ["normalize_weights", "weight_variant", "weight_variants"]