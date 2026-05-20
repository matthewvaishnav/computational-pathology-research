"""Contract tests for secure aggregation payload handling."""

import pytest
import torch

from src.features.federated.pathology_fl.aggregator.secure import SecureAggregator
from src.features.federated.pathology_fl.common.data_models import ClientUpdate, EncryptedClientUpdate


def _secure_aggregator_with_protocol(protocol):
    aggregator = SecureAggregator.__new__(SecureAggregator)
    aggregator.expected_clients = None
    aggregator.min_clients_required = 2
    aggregator.protocol = protocol
    aggregator.algorithm_name = "SecureAggregation"
    return aggregator


def test_secure_aggregator_rejects_plaintext_client_update():
    aggregator = _secure_aggregator_with_protocol(protocol=None)
    update = ClientUpdate(
        client_id="hospital-a",
        round_id=1,
        model_version=0,
        gradients={"w": torch.ones(2)},
        dataset_size=2,
        training_time_seconds=1.0,
    )

    with pytest.raises(ValueError, match="EncryptedClientUpdate"):
        aggregator.aggregate([update])


def test_secure_aggregator_routes_encrypted_payloads_to_protocol():
    calls = {}

    class Protocol:
        def aggregate_encrypted_client_updates(self, client_data):
            calls["client_data"] = client_data
            return {"w": torch.tensor([1.0, 2.0])}

    aggregator = _secure_aggregator_with_protocol(Protocol())
    updates = [
        EncryptedClientUpdate(
            client_id="hospital-a",
            round_id=1,
            model_version=0,
            encrypted_gradients={"w": b"encrypted-a"},
            gradient_shapes={"w": (2,)},
            dataset_size=2,
            training_time_seconds=1.0,
        ),
        EncryptedClientUpdate(
            client_id="hospital-b",
            round_id=1,
            model_version=0,
            encrypted_gradients={"w": b"encrypted-b"},
            gradient_shapes={"w": (2,)},
            dataset_size=6,
            training_time_seconds=1.0,
        ),
    ]

    result = aggregator.aggregate(updates)

    assert torch.equal(result["w"], torch.tensor([1.0, 2.0]))
    assert calls["client_data"]["hospital-a"] == ({"w": b"encrypted-a"}, {"w": (2,)}, 0.25)
    assert calls["client_data"]["hospital-b"] == ({"w": b"encrypted-b"}, {"w": (2,)}, 0.75)
