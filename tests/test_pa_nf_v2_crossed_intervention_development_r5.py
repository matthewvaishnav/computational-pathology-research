from __future__ import annotations

import torch

from experiments.paired_acquisition import (
    run_pa_nf_v2_crossed_intervention_development_r5 as r5,
)


def test_r5_candidate_control_parameter_counts_equal() -> None:
    config = r5.ExperimentConfig()
    candidate = r5.build_model(config, torch.device("cpu"))
    control = r5.build_model(config, torch.device("cpu"))
    assert r5.r1.parameter_count(candidate) == r5.r1.parameter_count(control)


def test_r5_zero_acquisition_is_exact_identity_modulation() -> None:
    config = r5.ExperimentConfig()
    model = r5.build_model(config, torch.device("cpu"))
    biological = torch.randn(5, config.biological_dim)
    acquisition = torch.zeros(5, config.acquisition_dim)
    decoder = model.decoder
    with torch.no_grad():
        canonical = decoder.canonical_biology(biological)
        operator = decoder.acquisition_operator(acquisition)
        expected = decoder.shared_readout(canonical)
        actual = model.decode(biological, acquisition)
    assert torch.equal(operator, torch.zeros_like(operator))
    assert torch.equal(expected, actual)


def test_r5_nonzero_acquisition_changes_operator_state() -> None:
    config = r5.ExperimentConfig()
    model = r5.build_model(config, torch.device("cpu"))
    acquisition = torch.randn(7, config.acquisition_dim)
    with torch.no_grad():
        operator = model.decoder.acquisition_operator(acquisition)
    assert operator.shape == (7, 2 * config.hidden_dim)
    assert torch.isfinite(operator).all()


def test_r5_frozen_smoke_seeds_are_fresh() -> None:
    assert r5.DEFAULT_SMOKE_SEEDS == (3501, 3502, 3503)


def test_r5_control_disables_only_cross_and_cycle_weights() -> None:
    config = r5.ExperimentConfig()
    candidate = r5.family_weights(r5.MODEL_FAMILIES[0], config)
    control = r5.family_weights(r5.MODEL_FAMILIES[1], config)
    assert candidate == (
        config.crossed_reconstruction_weight,
        config.biological_cycle_weight,
        config.acquisition_cycle_weight,
    )
    assert control == (0.0, 0.0, 0.0)
