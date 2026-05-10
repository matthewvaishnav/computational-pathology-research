"""Production integration tests for the PathologyFL aggregator adapter."""

import torch

from src.federated.aggregator.factory import AggregatorFactory
from src.federated.aggregator.pathology_fl import PathologyFLAggregator
from src.federated.common.data_models import ClientUpdate


def _pathology_update(
    client_id: str,
    value: float,
    hospital_type: str,
    specialties: list[str],
    cancer_type: str = "breast",
) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        round_id=1,
        model_version=1,
        gradients={"weight": torch.tensor([value])},
        dataset_size=100,
        training_time_seconds=1.0,
        hospital_metadata={
            "hospital_id": client_id,
            "hospital_type": hospital_type,
            "annual_cases": 8000 if client_id == "specialist" else 500,
            "cancer_specialties": specialties,
            "diagnostic_accuracy": 0.96,
            "years_experience": 15,
        },
        slide_quality={
            "image_sharpness": 0.9,
            "stain_consistency": 0.9,
            "label_confidence": 0.9,
            "artifact_level": 0.1,
        },
        cancer_type=cancer_type,
    )


def test_factory_registers_pathology_fl():
    aggregator = AggregatorFactory.create_aggregator(
        "pathology_fl",
        {"alpha": 0.7, "beta": 0.2, "default_cancer_type": "breast"},
    )

    assert isinstance(aggregator, PathologyFLAggregator)
    assert "pathology_fl" in AggregatorFactory.get_available_algorithms()


def test_pathology_fl_adapter_aggregates_client_update_metadata():
    aggregator = AggregatorFactory.create_aggregator(
        "pathology_fl",
        {"alpha": 0.7, "beta": 0.2, "default_cancer_type": "breast"},
    )
    updates = [
        _pathology_update("specialist", 10.0, "teaching_hospital", ["breast"]),
        _pathology_update("generalist", 0.0, "community", ["general"]),
    ]

    aggregated = aggregator.aggregate(updates)

    assert "weight" in aggregated
    assert torch.isfinite(aggregated["weight"]).all()
    assert aggregated["weight"].item() > 5.0
