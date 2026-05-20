"""Privacy/security regressions that support the PathologyFL production path."""

import torch

from src.features.federated.pathology_fl.privacy import dp_sgd
from src.features.federated.pathology_fl.privacy.secure_aggregation import SecureAggregationClient


def test_secure_client_packages_encrypted_update_metadata(monkeypatch):
    client = SecureAggregationClient(client_id="hospital-a")
    gradients = {
        "layer.weight": torch.ones(2, 3),
        "layer.bias": torch.zeros(3),
    }

    def fake_encrypt(payload):
        assert payload is gradients
        return {"layer.weight": b"encrypted-weight", "layer.bias": b"encrypted-bias"}

    monkeypatch.setattr(client, "encrypt_gradients", fake_encrypt)

    update = client.create_encrypted_update(
        gradients=gradients,
        round_id=7,
        model_version=3,
        dataset_size=42,
        training_time_seconds=1.5,
        privacy_epsilon=0.25,
        compression_method="quantize_8bit",
    )

    assert update.client_id == "hospital-a"
    assert update.encrypted_gradients["layer.weight"] == b"encrypted-weight"
    assert update.gradient_shapes == {"layer.weight": (2, 3), "layer.bias": (3,)}
    assert update.dataset_size == 42
    assert update.privacy_epsilon == 0.25
    assert update.compression_method == "quantize_8bit"


def test_dpsgd_engine_make_private_delegates_to_opacus(monkeypatch):
    calls = {}

    class FakePrivacyEngine:
        def __init__(self, secure_mode):
            calls["secure_mode"] = secure_mode

        def make_private(
            self,
            module,
            optimizer,
            data_loader,
            noise_multiplier,
            max_grad_norm,
        ):
            calls["make_private"] = {
                "module": module,
                "optimizer": optimizer,
                "data_loader": data_loader,
                "noise_multiplier": noise_multiplier,
                "max_grad_norm": max_grad_norm,
            }
            return "private-model", "private-optimizer", "private-loader"

    monkeypatch.setattr(dp_sgd, "OPACUS_AVAILABLE", True)
    monkeypatch.setattr(dp_sgd, "PrivacyEngine", FakePrivacyEngine)

    engine = dp_sgd.DPSGDEngine(
        max_grad_norm=0.5,
        noise_multiplier=1.2,
        sample_rate=0.1,
        secure_rng=True,
    )
    result = engine.make_private("model", "optimizer", "loader")

    assert result == ("private-model", "private-optimizer", "private-loader")
    assert calls["secure_mode"] is True
    assert calls["make_private"] == {
        "module": "model",
        "optimizer": "optimizer",
        "data_loader": "loader",
        "noise_multiplier": 1.2,
        "max_grad_norm": 0.5,
    }
