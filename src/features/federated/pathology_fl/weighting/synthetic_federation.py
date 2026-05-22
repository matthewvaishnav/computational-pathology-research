"""Synthetic federation utilities for FAIR-WEIGHTS-H experiments.

These helpers create deterministic institution profiles for testing weighting
behavior under controlled heterogeneity. They do not simulate clinical truth and
must not be used as evidence of clinical effectiveness.
"""

from dataclasses import dataclass
from typing import Dict, List

from src.features.federated.pathology_fl.weighting.fair_weights_h import (
    InstitutionWeightSignals,
)


@dataclass(frozen=True)
class SyntheticInstitution:
    institution_id: str
    institution_type: str
    dataset_size: int
    adjusted_quality: float
    process_quality: float
    useful_uniqueness: float
    fairness_score: float
    uncertainty_penalty: float
    contribution_score: float
    volume_factor: float

    def to_weight_signals(self) -> InstitutionWeightSignals:
        return InstitutionWeightSignals(
            institution_id=self.institution_id,
            adjusted_quality=self.adjusted_quality,
            process_quality=self.process_quality,
            useful_uniqueness=self.useful_uniqueness,
            fairness_score=self.fairness_score,
            uncertainty_penalty=self.uncertainty_penalty,
            contribution_score=self.contribution_score,
            volume_factor=self.volume_factor,
        )


def default_synthetic_federation() -> List[SyntheticInstitution]:
    """Return a small deterministic four-institution federation."""

    return [
        SyntheticInstitution(
            institution_id="cancer_center",
            institution_type="cancer_center",
            dataset_size=5000,
            adjusted_quality=0.92,
            process_quality=0.93,
            useful_uniqueness=0.25,
            fairness_score=0.10,
            uncertainty_penalty=0.04,
            contribution_score=0.30,
            volume_factor=3.0,
        ),
        SyntheticInstitution(
            institution_id="teaching_hospital",
            institution_type="teaching_hospital",
            dataset_size=3000,
            adjusted_quality=0.88,
            process_quality=0.86,
            useful_uniqueness=0.35,
            fairness_score=0.20,
            uncertainty_penalty=0.06,
            contribution_score=0.25,
            volume_factor=2.0,
        ),
        SyntheticInstitution(
            institution_id="community_hospital",
            institution_type="community_hospital",
            dataset_size=1400,
            adjusted_quality=0.82,
            process_quality=0.78,
            useful_uniqueness=0.55,
            fairness_score=0.45,
            uncertainty_penalty=0.12,
            contribution_score=0.18,
            volume_factor=1.0,
        ),
        SyntheticInstitution(
            institution_id="rural_hospital",
            institution_type="rural_hospital",
            dataset_size=600,
            adjusted_quality=0.76,
            process_quality=0.70,
            useful_uniqueness=0.88,
            fairness_score=0.90,
            uncertainty_penalty=0.20,
            contribution_score=0.12,
            volume_factor=0.6,
        ),
    ]


def equal_weights(institutions: List[SyntheticInstitution]) -> Dict[str, float]:
    if not institutions:
        raise ValueError("institutions cannot be empty")
    weight = 1.0 / len(institutions)
    return {institution.institution_id: weight for institution in institutions}


def volume_weights(institutions: List[SyntheticInstitution]) -> Dict[str, float]:
    if not institutions:
        raise ValueError("institutions cannot be empty")
    total = sum(institution.dataset_size for institution in institutions)
    if total <= 0:
        raise ValueError("total dataset size must be positive")
    return {
        institution.institution_id: institution.dataset_size / total
        for institution in institutions
    }


def prestige_weights(institutions: List[SyntheticInstitution]) -> Dict[str, float]:
    """Legacy prestige baseline retained only for comparison experiments."""

    multipliers = {
        "cancer_center": 2.0,
        "teaching_hospital": 1.5,
        "community_hospital": 1.0,
        "rural_hospital": 0.8,
    }
    raw = {
        institution.institution_id: multipliers.get(institution.institution_type, 1.0)
        for institution in institutions
    }
    total = sum(raw.values())
    return {institution_id: value / total for institution_id, value in raw.items()}
