"""Regression tests for PathologyFL specialty-weighted aggregation."""

import pytest
import torch

from src.features.federated.pathology_fl.pathology_fl import (
    CancerType,
    HospitalMetadata,
    HospitalType,
    PathologyFederatedAggregator,
    SlideQuality,
)


def _quality() -> SlideQuality:
    return SlideQuality(
        image_sharpness=0.9,
        stain_consistency=0.9,
        label_confidence=0.9,
        artifact_level=0.1,
    )


def _metadata(cancer_type: CancerType) -> dict[str, HospitalMetadata]:
    return {
        "specialist": HospitalMetadata(
            hospital_id="specialist",
            hospital_type=HospitalType.TEACHING_HOSPITAL,
            annual_cases=8000,
            cancer_specialties=[cancer_type],
            diagnostic_accuracy=0.96,
            years_experience=15,
        ),
        "generalist": HospitalMetadata(
            hospital_id="generalist",
            hospital_type=HospitalType.COMMUNITY_HOSPITAL,
            annual_cases=500,
            cancer_specialties=[CancerType.GENERAL],
            diagnostic_accuracy=0.9,
            years_experience=5,
        ),
    }


def _updates() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "specialist": {"weight": torch.tensor([10.0])},
        "generalist": {"weight": torch.tensor([0.0])},
    }


@pytest.mark.parametrize("cancer_type", [CancerType.BREAST, CancerType.LUNG, CancerType.PROSTATE])
def test_pathology_fl_cancer_specialty_weights_drive_final_average(cancer_type):
    aggregator = PathologyFederatedAggregator(alpha=0.7, beta=0.2)
    metadata = _metadata(cancer_type)
    slide_quality = {client_id: _quality() for client_id in metadata}

    aggregated = aggregator.aggregate_updates(_updates(), metadata, slide_quality, cancer_type)

    expertise_weights = aggregator._cancer_specific_weights(cancer_type, metadata)
    quality_weights = {
        client_id: aggregator.calculate_quality_weight(quality)
        for client_id, quality in slide_quality.items()
    }
    combined_weights = {
        client_id: (
            aggregator.alpha * expertise_weights[client_id]
            + aggregator.beta * quality_weights[client_id]
            + (1 - aggregator.alpha - aggregator.beta)
        )
        for client_id in _updates()
    }
    expected = (
        _updates()["specialist"]["weight"] * combined_weights["specialist"]
        + _updates()["generalist"]["weight"] * combined_weights["generalist"]
    ) / sum(combined_weights.values())

    assert torch.allclose(aggregated["weight"], expected)
    assert aggregated["weight"].item() > 5.0


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        (-0.1, 0.2),
        (0.2, -0.1),
        (0.8, 0.3),
    ],
)
def test_pathology_fl_rejects_invalid_alpha_beta(alpha, beta):
    with pytest.raises(ValueError, match="alpha and beta"):
        PathologyFederatedAggregator(alpha=alpha, beta=beta)


def test_pathology_fl_rejects_missing_metadata_and_quality():
    aggregator = PathologyFederatedAggregator()
    updates = _updates()
    metadata = _metadata(CancerType.BREAST)
    slide_quality = {"specialist": _quality()}

    with pytest.raises(ValueError, match="missing slide quality for generalist"):
        aggregator.aggregate_updates(updates, metadata, slide_quality, CancerType.BREAST)

    with pytest.raises(ValueError, match="missing hospital metadata for generalist"):
        aggregator.aggregate_updates(
            updates,
            {"specialist": metadata["specialist"]},
            {client_id: _quality() for client_id in updates},
            CancerType.BREAST,
        )


def test_pathology_fl_rejects_invalid_metadata_quality_and_shapes():
    aggregator = PathologyFederatedAggregator()
    metadata = _metadata(CancerType.BREAST)
    slide_quality = {client_id: _quality() for client_id in metadata}

    bad_metadata = dict(metadata)
    bad_metadata["specialist"] = HospitalMetadata(
        hospital_id="specialist",
        hospital_type=HospitalType.TEACHING_HOSPITAL,
        annual_cases=0,
        cancer_specialties=[CancerType.BREAST],
        diagnostic_accuracy=0.96,
        years_experience=15,
    )
    with pytest.raises(ValueError, match="annual_cases"):
        aggregator.aggregate_updates(_updates(), bad_metadata, slide_quality, CancerType.BREAST)

    bad_quality = dict(slide_quality)
    bad_quality["specialist"] = SlideQuality(1.2, 0.9, 0.9, 0.1)
    with pytest.raises(ValueError, match="image_sharpness"):
        aggregator.aggregate_updates(_updates(), metadata, bad_quality, CancerType.BREAST)

    bad_updates = _updates()
    bad_updates["generalist"] = {"weight": torch.tensor([[0.0]])}
    with pytest.raises(ValueError, match="mismatched parameter shape"):
        aggregator.aggregate_updates(bad_updates, metadata, slide_quality, CancerType.BREAST)
