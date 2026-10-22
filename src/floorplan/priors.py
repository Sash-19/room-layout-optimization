from __future__ import annotations

from dataclasses import dataclass

from .representation import Furniture, Opening


@dataclass(frozen=True, slots=True)
class FurniturePrior:
    cls: str
    width: float
    depth: float
    min_width: float
    max_width: float
    min_depth: float
    max_depth: float
    front_clearance: float = 0.0
    wall_affinity: bool = False
    height: float = 0.8


@dataclass(frozen=True, slots=True)
class OpeningPrior:
    kind: str
    width: float
    height: float


FURNITURE_PRIORS: dict[str, FurniturePrior] = {
    "bed": FurniturePrior("bed", 1.6, 2.0, 0.9, 2.4, 1.8, 2.4, 0.6, True, 0.6),
    "desk": FurniturePrior("desk", 1.2, 0.6, 0.8, 2.2, 0.45, 0.9, 0.75, False, 0.75),
    "chair": FurniturePrior("chair", 0.5, 0.5, 0.4, 0.8, 0.4, 0.8, 0.0, False, 0.9),
    "couch": FurniturePrior("couch", 1.8, 0.9, 1.2, 2.8, 0.7, 1.2, 0.5, True, 0.8),
    "table": FurniturePrior("table", 1.2, 0.8, 0.6, 2.6, 0.5, 1.2, 0.5, False, 0.75),
    "cabinet": FurniturePrior("cabinet", 1.0, 0.5, 0.5, 2.4, 0.3, 0.7, 0.5, True, 1.8),
}

OPENING_PRIORS: dict[str, OpeningPrior] = {
    "door": OpeningPrior("door", 0.8, 2.0),
    "window": OpeningPrior("window", 1.2, 1.2),
}


def get_prior(cls: str) -> FurniturePrior | None:
    return FURNITURE_PRIORS.get(cls)


def clamp_width(cls: str, width: float) -> float:
    prior = get_prior(cls)
    if prior is None:
        return max(0.1, float(width))
    return float(min(prior.max_width, max(prior.min_width, width)))


def clamp_depth(cls: str, depth: float) -> float:
    prior = get_prior(cls)
    if prior is None:
        return max(0.1, float(depth))
    return float(min(prior.max_depth, max(prior.min_depth, depth)))


def clamp_footprint(cls: str, width: float, depth: float) -> tuple[float, float]:
    return clamp_width(cls, width), clamp_depth(cls, depth)


def opening_prior(kind: str) -> OpeningPrior | None:
    return OPENING_PRIORS.get(kind)


def furniture_from_detection(
    cls: str,
    width: float,
    x: float,
    y: float,
    theta: float,
    confidence: float = 1.0,
) -> Furniture:
    prior = get_prior(cls)
    if prior is None:
        raise ValueError(f"no furniture prior for class {cls!r}")
    return Furniture(
        cls=cls,
        w=clamp_width(cls, width),
        d=prior.depth,
        x=x,
        y=y,
        theta=theta,
        front_clearance=prior.front_clearance,
        wall_affinity=prior.wall_affinity,
        confidence=confidence,
    )


def opening_from_dimensions(
    kind: str,
    wall_index: int,
    offset: float,
    width: float,
) -> Opening:
    prior = opening_prior(kind)
    if prior is None:
        raise ValueError(f"no opening prior for kind {kind!r}")
    if width <= 0:
        width = prior.width
    return Opening(
        kind=kind,
        wall_index=wall_index,
        offset=offset,
        width=width,
        swing_radius=width if kind == "door" else 0.0,
    )
