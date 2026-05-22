import pytest
import torch

from src.features.federated.pathology_fl.aggregator.weighted import ExplicitWeightedAggregator
from src.features.federated.pathology_fl.common.data_models import ClientUpdate


def _client(client_id, value):
    return ClientUpdate(
        client_id=client_id,
        gradients={"w": torch.tensor([float(value)])},
        num_samples=10,
        loss=0.1,
        metrics={},
    )


def test_weighted_average_matches_expected():
    agg = ExplicitWeightedAggregator({"a":0.75,"b":0.25})
    out = agg.aggregate([_client("a",2.0),_client("b",0.0)])
    assert out["w"].item() == pytest.approx(1.5)


def test_missing_weight_raises():
    agg = ExplicitWeightedAggregator({"a":1.0})
    with pytest.raises(ValueError, match="Missing explicit weights"):
        agg.aggregate([_client("a",1.0), _client("b",2.0)])


def test_negative_weight_invalid():
    with pytest.raises(ValueError):
        ExplicitWeightedAggregator({"a":-0.1})


def test_empty_updates_raise():
    agg = ExplicitWeightedAggregator({"a":1.0})
    with pytest.raises(ValueError):
        agg.aggregate([])
