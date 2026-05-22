"""Controlled synthetic perturbations for weighting experiments.

These functions modify synthetic institution profiles to test weight sensitivity.
They are engineering probes only and are not clinical simulations.
"""

from typing import Iterable, List

from src.features.federated.pathology_fl.weighting.synthetic_federation import (
    SyntheticInstitution,
)


def apply_uncertainty_spike(
    institutions: Iterable[SyntheticInstitution],
    institution_id: str,
    increase: float = 0.5,
) -> List[SyntheticInstitution]:
    """Increase uncertainty for one institution, e.g. from label noise or QC drift."""

    return [
        _replace(
            institution,
            uncertainty_penalty=institution.uncertainty_penalty + increase,
        )
        if institution.institution_id == institution_id
        else institution
        for institution in institutions
    ]


def apply_rare_population_enrichment(
    institutions: Iterable[SyntheticInstitution],
    institution_id: str,
    uniqueness_increase: float = 0.3,
    fairness_increase: float = 0.3,
) -> List[SyntheticInstitution]:
    """Increase useful uniqueness and fairness signals for one institution."""

    return [
        _replace(
            institution,
            useful_uniqueness=_clip01(institution.useful_uniqueness + uniqueness_increase),
            fairness_score=_clip01(institution.fairness_score + fairness_increase),
        )
        if institution.institution_id == institution_id
        else institution
        for institution in institutions
    ]


def apply_quality_degradation(
    institutions: Iterable[SyntheticInstitution],
    institution_id: str,
    decrease: float = 0.2,
) -> List[SyntheticInstitution]:
    """Decrease adjusted and process quality for one institution."""

    return [
        _replace(
            institution,
            adjusted_quality=_clip01(institution.adjusted_quality - decrease),
            process_quality=_clip01(institution.process_quality - decrease),
        )
        if institution.institution_id == institution_id
        else institution
        for institution in institutions
    ]


def apply_scanner_shift(
    institutions: Iterable[SyntheticInstitution],
    institution_id: str,
    uniqueness_increase: float = 0.2,
    uncertainty_increase: float = 0.2,
) -> List[SyntheticInstitution]:
    """Simulate a shift that makes a site more different but also less certain."""

    return [
        _replace(
            institution,
            useful_uniqueness=_clip01(institution.useful_uniqueness + uniqueness_increase),
            uncertainty_penalty=institution.uncertainty_penalty + uncertainty_increase,
        )
        if institution.institution_id == institution_id
        else institution
        for institution in institutions
    ]


def _replace(institution: SyntheticInstitution, **changes) -> SyntheticInstitution:
    data = institution.__dict__.copy()
    data.update(changes)
    return SyntheticInstitution(**data)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
