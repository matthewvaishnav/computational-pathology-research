"""Experimental FAIR-WEIGHTS-H institutional weighting engine.

This module implements a conservative, auditable research scaffold for hybrid
institutional weighting in federated computational pathology. It is not a
clinically validated or regulatory-cleared weighting policy.
"""

from dataclasses import dataclass
from typing import Dict, List
import math


@dataclass(frozen=True)
class InstitutionWeightSignals:
    """Signals used to compute one institution's provisional training weight."""

    institution_id: str
    adjusted_quality: float
    process_quality: float
    useful_uniqueness: float
    fairness_score: float
    uncertainty_penalty: float
    contribution_score: float = 0.0
    volume_factor: float = 1.0
    integrity_ok: bool = True


@dataclass(frozen=True)
class FairWeightsHConfig:
    """Configuration for the experimental FAIR-WEIGHTS-H scoring model."""

    lambda_quality: float = 1.0
    lambda_diversity: float = 1.0
    lambda_fairness: float = 1.0
    lambda_contribution: float = 1.0
    lambda_volume: float = 0.25
    lambda_uncertainty: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 1.0
    conservative_mode: bool = False


@dataclass(frozen=True)
class WeightComputationResult:
    """Weight output plus lightweight audit diagnostics."""

    weights: Dict[str, float]
    normalized_entropy: float
    effective_institution_count: float
    diagnostics: Dict[str, float]


class FairWeightsHEngine:
    """Compute bounded, normalized institutional weights from auditable signals."""

    def __init__(self, config: FairWeightsHConfig | None = None):
        self.config = config or FairWeightsHConfig()
        self._validate_config()

    def compute(self, signals: List[InstitutionWeightSignals]) -> WeightComputationResult:
        if not signals:
            raise ValueError("Cannot compute FAIR-WEIGHTS-H weights for no institutions")

        ids = [s.institution_id for s in signals]
        if len(ids) != len(set(ids)):
            raise ValueError("Institution IDs must be unique")

        scores = {s.institution_id: self._score(s) for s in signals}
        weights = self._stable_softmax(scores)
        weights = self._apply_caps_and_normalize(weights)

        entropy = self._normalized_entropy(weights)
        effective_n = 1.0 / sum(w * w for w in weights.values())

        return WeightComputationResult(
            weights=weights,
            normalized_entropy=entropy,
            effective_institution_count=effective_n,
            diagnostics={
                "n_institutions": float(len(signals)),
                "max_weight": max(weights.values()),
                "min_weight": min(weights.values()),
                "integrity_exclusions": float(sum(not s.integrity_ok for s in signals)),
            },
        )

    def _score(self, s: InstitutionWeightSignals) -> float:
        self._validate_signal(s)
        if not s.integrity_ok:
            return -1e9

        cfg = self.config
        diversity_weight = 0.25 * cfg.lambda_diversity if cfg.conservative_mode else cfg.lambda_diversity
        fairness_weight = 0.5 * cfg.lambda_fairness if cfg.conservative_mode else cfg.lambda_fairness

        quality_term = cfg.lambda_quality * (s.adjusted_quality + s.process_quality)
        diversity_term = diversity_weight * s.useful_uniqueness
        fairness_term = fairness_weight * s.fairness_score
        contribution_term = cfg.lambda_contribution * s.contribution_score
        volume_term = cfg.lambda_volume * math.log1p(max(0.0, s.volume_factor))
        uncertainty_term = cfg.lambda_uncertainty * s.uncertainty_penalty

        return quality_term + diversity_term + fairness_term + contribution_term + volume_term - uncertainty_term

    def _stable_softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        max_score = max(scores.values())
        exps = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exps.values())
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("Unable to normalize FAIR-WEIGHTS-H scores")
        return {k: v / total for k, v in exps.items()}

    def _apply_caps_and_normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        capped = {k: min(cfg.max_weight, max(cfg.min_weight, w)) for k, w in weights.items()}
        total = sum(capped.values())
        if total <= 0.0:
            raise ValueError("Weight caps produced zero total weight")
        return {k: w / total for k, w in capped.items()}

    def _normalized_entropy(self, weights: Dict[str, float]) -> float:
        if len(weights) <= 1:
            return 0.0
        entropy = -sum(w * math.log(w + 1e-12) for w in weights.values())
        return entropy / math.log(len(weights))

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.min_weight < 0.0:
            raise ValueError("min_weight must be non-negative")
        if cfg.max_weight <= 0.0:
            raise ValueError("max_weight must be positive")
        if cfg.min_weight > cfg.max_weight:
            raise ValueError("min_weight cannot exceed max_weight")

    def _validate_signal(self, s: InstitutionWeightSignals) -> None:
        if not s.institution_id:
            raise ValueError("institution_id is required")
        numeric_fields = [
            s.adjusted_quality,
            s.process_quality,
            s.useful_uniqueness,
            s.fairness_score,
            s.uncertainty_penalty,
            s.contribution_score,
            s.volume_factor,
        ]
        if any(not math.isfinite(value) for value in numeric_fields):
            raise ValueError(f"Non-finite FAIR-WEIGHTS-H signal for {s.institution_id}")
