from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4


def test_v4_identity_splits_and_generator_claims() -> None:
    config = v4.ExperimentConfig()
    ds = v4.make_dataset(config, "linear_biology")
    assert not (set(ds.train_indices) & set(ds.calibration_indices))
    assert not (set(ds.train_indices) & set(ds.test_indices))
    assert not (set(ds.calibration_indices) & set(ds.test_indices))
    assert ds.true_metadata["shared_additive_scanner_coordinate_law"] is False
    assert ds.true_metadata["heldout_scanner_is_composition"] is False
    assert ds.observations.shape == (
        config.train_identities + config.calibration_identities + config.test_identities,
        len(v4.ALL_SCANNERS),
        config.feature_dim,
    )


def test_v4_candidate_control_exact_parameter_match_and_initialization() -> None:
    config = v4.ExperimentConfig()
    v4.set_deterministic_seed(4401)
    candidate = v4.ReferenceGaugeModel(config, "inverse_transport")
    v4.set_deterministic_seed(4401)
    control = v4.ReferenceGaugeModel(config, "no_inverse_transport_control")
    assert v4.parameter_count(candidate) == v4.parameter_count(control)
    for (cn, cv), (xn, xv) in zip(candidate.state_dict().items(), control.state_dict().items()):
        assert cn == xn
        assert torch.equal(cv, xv)


def test_v4_operator_roundtrip_and_reference_identity() -> None:
    config = v4.ExperimentConfig()
    v4.set_deterministic_seed(4402)
    model = v4.ReferenceGaugeModel(config, "inverse_transport")
    x = torch.randn(32, config.feature_dim)
    assert torch.equal(model.apply_operator(x, 0), x)
    assert torch.equal(model.invert_operator(x, 0), x)
    with torch.no_grad():
        for scanner in v4.ALL_SCANNERS:
            xr = model.invert_operator(model.apply_operator(x, scanner), scanner)
            assert float(F.mse_loss(xr, x)) < 1e-8


def test_v4_heldout_operator_excluded_from_shared_training_parameters() -> None:
    config = v4.ExperimentConfig()
    model = v4.ReferenceGaugeModel(config, "inverse_transport")
    shared = {id(p) for p in v4._shared_training_parameters(model)}
    heldout = {id(p) for p in model.operator_module(v4.HELDOUT_SCANNER).parameters()}
    assert not (shared & heldout)
    for scanner in v4.TRAIN_SCANNERS:
        if scanner == v4.REFERENCE_SCANNER:
            continue
        ids = {id(p) for p in model.operator_module(scanner).parameters()}
        assert ids.issubset(shared)


def test_v4_candidate_and_control_differ_only_in_source_inverse_behavior() -> None:
    config = v4.ExperimentConfig()
    v4.set_deterministic_seed(4403)
    candidate = v4.ReferenceGaugeModel(config, "inverse_transport")
    v4.set_deterministic_seed(4403)
    control = v4.ReferenceGaugeModel(config, "no_inverse_transport_control")
    x = torch.randn(16, config.feature_dim)
    # Reference scanner is identity, so both arms must agree exactly at scanner 0.
    assert torch.equal(candidate.biological_representation(x, 0), control.biological_representation(x, 0))
    # A nonreference scanner changes only candidate preprocessing; parameters remain identical.
    c = candidate.biological_representation(x, 1)
    k = control.biological_representation(x, 1)
    assert c.shape == k.shape == (16, config.biological_latent_dim)
    assert not np.allclose(c.detach().numpy(), k.detach().numpy())
