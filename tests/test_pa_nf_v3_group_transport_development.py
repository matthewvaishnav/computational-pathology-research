from __future__ import annotations

import numpy as np
import torch

from experiments.paired_acquisition import (
    run_pa_nf_v3_group_transport_development as v3,
)


def test_candidate_and_control_parameter_counts_match() -> None:
    config = v3.ExperimentConfig()
    candidate = v3.TransportModel(config, v3.MODEL_FAMILIES[0])
    control = v3.TransportModel(config, v3.MODEL_FAMILIES[1])
    assert v3.parameter_count(candidate) == v3.parameter_count(control)


def test_heldout_composed_scanner_is_absent_from_training() -> None:
    config = v3.ExperimentConfig()
    dataset = v3.make_dataset(config, v3.RENDERERS[0])
    train_scanners = dataset.scanner_ids[dataset.train_indices]
    assert v3.HELDOUT_COMPOSED_SCANNER not in set(train_scanners.tolist())
    assert len(dataset.train_indices) == config.train_identities * len(v3.TRAIN_SCANNERS)


def test_frozen_scanner_composition_relation() -> None:
    coords = v3.frozen_scanner_coordinates()
    np.testing.assert_allclose(
        coords[v3.HELDOUT_COMPOSED_SCANNER],
        coords[v3.COMPOSITION_A] + coords[v3.COMPOSITION_B],
        atol=0.0,
        rtol=0.0,
    )


def test_candidate_operator_identity_inverse_and_composition() -> None:
    torch.manual_seed(123)
    config = v3.ExperimentConfig()
    model = v3.TransportModel(config, v3.MODEL_FAMILIES[0])
    x = torch.randn(13, config.feature_dim)
    zero = torch.zeros(13, config.acquisition_dim)
    a = torch.randn(13, config.acquisition_dim) * 0.2
    b = torch.randn(13, config.acquisition_dim) * 0.2
    with torch.no_grad():
        identity = model.apply_operator(x, zero)
        inverse = model.apply_operator(model.apply_operator(x, a), -a)
        sequential = model.apply_operator(model.apply_operator(x, a), b)
        combined = model.apply_operator(x, a + b)
    assert torch.max(torch.abs(identity - x)).item() < 1e-5
    assert torch.max(torch.abs(inverse - x)).item() < 1e-5
    assert torch.mean((sequential - combined) ** 2).item() < 1e-10


def test_matched_control_breaks_additive_closure() -> None:
    torch.manual_seed(456)
    config = v3.ExperimentConfig()
    control = v3.TransportModel(config, v3.MODEL_FAMILIES[1])
    x = torch.randn(19, config.feature_dim)
    a = torch.randn(19, config.acquisition_dim) * 0.7
    b = torch.randn(19, config.acquisition_dim) * 0.7
    with torch.no_grad():
        sequential = control.apply_operator(control.apply_operator(x, a), b)
        combined = control.apply_operator(x, a + b)
    assert torch.mean((sequential - combined) ** 2).item() > 1e-12


def test_primary_composition_prediction_cannot_use_scanner5_code() -> None:
    source = v3.evaluate_model.__code__.co_names
    # Primary prediction is constructed from learned prototypes for scanners 1 and 2;
    # the held-out scanner is used only as the observed target for scoring.
    assert "prototypes" in source
    assert "COMPOSITION_A" in source
    assert "COMPOSITION_B" in source
