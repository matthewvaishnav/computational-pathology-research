"""Regression tests for DP-SGD integration boundaries."""

import pytest
import torch
import torch.nn as nn

from src.federated.client.trainer import LocalTrainer
from src.federated.privacy import dp_sgd


def test_dpsgd_engine_fails_closed_without_opacus(monkeypatch):
    monkeypatch.setattr(dp_sgd, "OPACUS_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="Opacus is required"):
        dp_sgd.DPSGDEngine()


def test_trainer_uses_privacy_engine_make_private():
    class FakePrivacyEngine:
        def __init__(self):
            self.make_private_called = False
            self.privatize_gradients_called = False

        def make_private(self, model, optimizer, data_loader):
            self.make_private_called = True
            return model, optimizer, data_loader

        def privatize_gradients(self, *args, **kwargs):
            self.privatize_gradients_called = True
            raise AssertionError("trainer should not manually privatize batch gradients")

        def get_privacy_spent(self):
            return 0.0, 1e-5

        def get_clipping_stats(self):
            return {}

    model = nn.Linear(4, 2)
    privacy_engine = FakePrivacyEngine()
    trainer = LocalTrainer(model=model, privacy_engine=privacy_engine)
    trainer.initialize_from_global(model.state_dict())
    trainer.set_data(torch.randn(8, 4), torch.randint(0, 2, (8,)))

    trainer.train_local_epochs(num_epochs=1, batch_size=4)

    assert privacy_engine.make_private_called
    assert not privacy_engine.privatize_gradients_called
