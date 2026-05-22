"""Synthetic benchmark runner for institutional weighting strategies.

The benchmark is deterministic and intended for engineering validation of weight
behavior. It is not evidence of clinical effectiveness.
"""

from dataclasses import dataclass
from typing import Dict, List
import math

from src.features.federated.pathology_fl.weighting.fair_weights_h import (
    FairWeightsHEngine,
)
from src.features.federated.pathology_fl.weighting.synthetic_federation import (
    SyntheticInstitution,
    equal_weights,
    prestige_weights,
    volume_weights,
)


@dataclass(frozen=True)
class WeightStrategyReport:
    strategy: str
    weights: Dict[str, float]
    normalized_entropy: float
    effective_institution_count: float
    rural_weight: float
    max_weight: float


def normalized_entropy(weights: Dict[str, float]) -> float:
    if len(weights) <= 1:
        return 0.0
    entropy = -sum(weight * math.log(weight + 1e-12) for weight in weights.values())
    return entropy / math.log(len(weights))


def effective_institution_count(weights: Dict[str, float]) -> float:
    return 1.0 / sum(weight * weight for weight in weights.values())


def summarize_weights(strategy: str, weights: Dict[str, float]) -> WeightStrategyReport:
    return WeightStrategyReport(
        strategy=strategy,
        weights=weights,
        normalized_entropy=normalized_entropy(weights),
        effective_institution_count=effective_institution_count(weights),
        rural_weight=weights.get("rural_hospital", 0.0),
        max_weight=max(weights.values()),
    )


def compare_weighting_strategies(
    institutions: List[SyntheticInstitution],
    fair_weights_engine: FairWeightsHEngine | None = None,
) -> List[WeightStrategyReport]:
    """Compare baseline and FAIR-WEIGHTS-H strategies on synthetic profiles."""

    if not institutions:
        raise ValueError("institutions cannot be empty")

    engine = fair_weights_engine or FairWeightsHEngine()
    fair_result = engine.compute([institution.to_weight_signals() for institution in institutions])

    return [
        summarize_weights("equal", equal_weights(institutions)),
        summarize_weights("volume", volume_weights(institutions)),
        summarize_weights("prestige", prestige_weights(institutions)),
        summarize_weights("fair_weights_h", fair_result.weights),
    ]
