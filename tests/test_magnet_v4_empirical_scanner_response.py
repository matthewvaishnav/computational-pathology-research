"""Synthetic-only structural tests for the MAGNET v4 energy core."""
import numpy as np
import pytest
from experiments.paired_acquisition.magnet_v4_empirical_scanner_response import (
    EmpiricalScannerResponse, calibrate_training_gauge,
)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(20261103)
    biological = rng.normal(size=(40, 16))
    shifts = np.arange(5)[None, :, None] * np.linspace(-.1, .1, 16)[None, None, :]
    x = biological[:, None, :] + shifts
    samples = np.array([f"sample_{j // 2}" for j in range(40)])
    return EmpiricalScannerResponse(x[:30], samples[:30], k=4, gamma=.8), x


def test_unique_conditional_displacement_stationary(synthetic):
    model, x = synthetic
    source = x[30:35, 1, :]
    delta = model.transport(source, 1, 3) - source
    assert np.max(np.abs(model.energy_gradient(delta, source, 1, 3))) < 1e-10


def test_source_only_target_independence_and_identity(synthetic):
    model, x = synthetic
    a = x[30:, :, :].copy()
    pred = model.predict_all(a[:, 0], 0)
    assert np.array_equal(pred[:, 0], a[:, 0])
    a[:, 1:] += 123456
    assert np.array_equal(model.predict_all(a[:, 0], 0), pred)


def test_training_only_common_gauge(synthetic):
    model, _ = synthetic
    gauge = calibrate_training_gauge(model)
    assert gauge.shape == (5, 16)
    assert np.allclose(gauge.mean(axis=0), 0, atol=1e-12)


def test_invalid_samples_and_scanners_fail_closed(synthetic):
    model, x = synthetic
    with pytest.raises(ValueError):
        model.canonical(x[:3, 1, :], 1, np.array(["missing"]), True)
    with pytest.raises(ValueError):
        model.transport(x[:3, 1, :], 1, 99)
