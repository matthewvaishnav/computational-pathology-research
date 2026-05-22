"""Experimental FAIR-WEIGHTS-H institutional weighting engine.

This module implements a conservative, auditable research scaffold for hybrid
institutional weighting in federated computational pathology. It is not a
clinically validated or regulatory-cleared weighting policy.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal
import math


ScoreTransform = Literal["linear", "log_linear"]
UpdateRule = Literal["softmax", "mirror_descent"]


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
    """Configuration for the experimental FAIR-WEIGHTS-H scoring model.

    The default keeps the original linear softmax behavior for compatibility.
    Use score_transform="log_linear" for a log-linear form aligned with the
    original multiplicative FAIR-WEIGHTS derivation. Use
    update_rule="mirror_descent" with previous weights for multiplicative
    weights / entropy-regularized mirror descent updates.
    """

    lambda_quality: float = 1.0
    lambda_diversity: float = 1.0
    lambda_fairness: float = 1.0
    lambda_contribution: float = 1.0
    lambda_volume: float = 0.25
    lambda_uncertainty: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 1.0
    conservative_mode: bool = False
    beta: float = 1.0
    eta: float = 1.0
    epsilon: float = 1e-8
    score_transform: ScoreTransform = "linear"
    update_rule: UpdateRule = "softmax"


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

    def compute(
        self,
        signals: List[InstitutionWeightSignals],
        previous_weights: Dict[str, float] | None = None,
    ) -> WeightComputationResult:
        if not signals:
            raise ValueError("Cannot compute FAIR-WEIGHTS-H weights for no institutions")

        ids = [s.institution_id for s in signals]
        if len(ids) != len(set(ids)):
            raise ValueError("Institution IDs must be unique")

        scores = {s.institution_id: self._score(s) for s in signals}
        if self.config.update_rule == "mirror_descent":
            weights = self._mirror_descent(scores, ids, previous_weights)
        else:
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
                "beta": self.config.beta,
                "eta": self.config.eta,
                "score_transform": 1.0 if self.config.score_transform == "log_linear" else 0.0,
                "update_rule": 1.0 if self.config.update_rule == "mirror_descent" else 0.0,
            },
        )

    def _score(self, s: InstitutionWeightSignals) -> float:
        self._validate_signal(s)
        if not s.integrity_ok:
            return -1e9

        if self.config.score_transform == "log_linear":
            return self._log_linear_score(s)
        return self._linear_score(s)

    def _linear_score(self, s: InstitutionWeightSignals) -> float:
        cfg = self.config
        diversity_weight = 0.25 * cfg.lambda_diversity if cfg.conservative_mode else cfg.lambda_diversity
        fairness_weight = 0.5 * cfg.lambda_fairness if cfg.conservative_mode else cfg.lambda_fairness

        quality_term = cfg.lambda_quality * (s.adjusted_quality + s.process_quality)
        diversity_term = diversity_weight * s.useful_uniqueness
        fairness_term = fairness_weight * s.fairness_score
        contribution_term = cfg.lambda_contribution * s.contribution_score
        volume_term = cfg.lambda_volume * math.log1p(max(0.0, s.volume_factor))
        uncertainty_term = cfg.lambda_uncertainty * s.uncertainty_penalty

        return cfg.beta * (
            quality_term
            + diversity_term
            + fairness_term
            + contribution_term
            + volume_term
            - uncertainty_term
        )

    def _log_linear_score(self, s: InstitutionWeightSignals) -> float:
        cfg = self.config
        diversity_weight = 0.25 * cfg.lambda_diversity if cfg.conservative_mode else cfg.lambda_diversity
        fairness_weight = 0.5 * cfg.lambda_fairness if cfg.conservative_mode else cfg.lambda_fairness

        eps = cfg.epsilon
        adjusted_quality_term = cfg.lambda_quality * math.log(max(eps, s.adjusted_quality))
        process_quality_term = cfg.lambda_quality * math.log(max(eps, s.process_quality))
        diversity_term = diversity_weight * math.log(max(eps, s.useful_uniqueness))
        fairness_term = fairness_weight * math.log(max(eps, s.fairness_score))
        contribution_term = cfg.lambda_contribution * s.contribution_score
        volume_term = cfg.lambda_volume * math.log1p(max(0.0, s.volume_factor))
        uncertainty_term = cfg.lambda_uncertainty * s.uncertainty_penalty

        return cfg.beta * (
            adjusted_quality_term
            + process_quality_term
            + diversity_term
            + fairness_term
            + contribution_term
            + volume_term
            - uncertainty_term
        )

    def _stable_softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        max_score = max(scores.values())
        exps = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exps.values())
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("Unable to normalize FAIR-WEIGHTS-H scores")
        return {k: v / total for k, v in exps.items()}

    def _mirror_descent(
        self,
        scores: Dict[str, float],
        ids: List[str],
        previous_weights: Dict[str, float] | None,
    ) -> Dict[str, float]:
        if previous_weights is None:
            previous_weights = {institution_id: 1.0 / len(ids) for institution_id in ids}
        self._validate_previous_weights(previous_weights, ids)

        logits = {
            institution_id: math.log(max(self.config.epsilon, previous_weights[institution_id]))
            + self.config.eta * scores[institution_id]
            for institution_id in ids
        }
        return self._stable_softmax(logits)

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
        if cfg.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if cfg.eta <= 0.0:
            raise ValueError("eta must be positive")
        if cfg.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if cfg.score_transform not in {"linear", "log_linear"}:
            raise ValueError("score_transform must be 'linear' or 'log_linear'")
        if cfg.update_rule not in {"softmax", "mirror_descent"}:
            raise ValueError("update_rule must be 'softmax' or 'mirror_descent'")

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

    def _validate_previous_weights(self, previous_weights: Dict[str, float], ids: List[str]) -> None:
        missing = sorted(set(ids) - set(previous_weights))
        extra = sorted(set(previous_weights) - set(ids))
        if missing or extra:
            raise ValueError(
                "previous_weights must match institution IDs; "
                f"missing={missing}, extra={extra}"
            )
        if any(weight < 0.0 or not math.isfinite(weight) for weight in previous_weights.values()):
            raise ValueError("previous_weights must be finite and non-negative")
        if sum(previous_weights.values()) <= 0.0:
            raise ValueError("previous_weights must have positive total mass")
