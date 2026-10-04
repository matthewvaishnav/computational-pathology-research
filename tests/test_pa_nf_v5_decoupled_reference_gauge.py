from __future__ import annotations

import numpy as np
import torch

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.paired_acquisition import run_pa_nf_v5_decoupled_reference_gauge as v5


def test_frozen_v5_seeds_and_parameter_budget():
    config = v5.frozen_config()
    assert config.dataset_seed == 16037
    assert config.bootstrap_seed == 20261004
    assert v5.FROZEN_MODEL_SEEDS == (4501, 4502, 4503, 4504, 4505)
    v4.set_deterministic_seed(4501)
    candidate = v4.ReferenceGaugeModel(config, "inverse_transport")
    v4.set_deterministic_seed(4501)
    control = v4.ReferenceGaugeModel(config, "no_inverse_transport_control")
    assert v4.parameter_count(candidate) == 10728
    assert v4.parameter_count(candidate) == v4.parameter_count(control)


def test_dataset_splits_disjoint_and_no_composition_law():
    ds = v4.make_dataset(v5.frozen_config(), "linear_biology")
    assert np.intersect1d(ds.train_indices, ds.calibration_indices).size == 0
    assert np.intersect1d(ds.train_indices, ds.test_indices).size == 0
    assert np.intersect1d(ds.calibration_indices, ds.test_indices).size == 0
    assert not ds.true_metadata["shared_additive_scanner_coordinate_law"]
    assert not ds.true_metadata["heldout_scanner_is_composition"]


def test_operator_inverse_roundtrip_exact_enough():
    config = v5.frozen_config()
    v4.set_deterministic_seed(4501)
    model = v4.ReferenceGaugeModel(config, "inverse_transport")
    g = torch.Generator(device="cpu")
    g.manual_seed(20261004)
    x = torch.randn(32, config.feature_dim, generator=g)
    for scanner in v4.ALL_SCANNERS:
        xr = model.invert_operator(model.apply_operator(x, scanner), scanner)
        assert torch.mean((xr - x) ** 2).item() < 1e-8


def test_heldout_operator_excluded_from_shared_optimizer_parameters():
    model = v4.ReferenceGaugeModel(v5.frozen_config(), "inverse_transport")
    heldout_ids = {id(p) for p in model.operator_module(v4.HELDOUT_SCANNER).parameters()}
    shared_ids = {id(p) for p in v4._shared_training_parameters(model)}
    assert not (heldout_ids & shared_ids)


def test_operator_only_transport_does_not_use_biological_encoder_or_decoder():
    config = v5.frozen_config()
    ds = v4.make_dataset(config, "linear_biology")
    v4.set_deterministic_seed(4501)
    model = v4.ReferenceGaugeModel(config, "inverse_transport")
    obs = ds.observations[ds.test_indices[:8]]
    before = v5._operator_only_transport_gain(model, obs, [(0, 1), (1, 2)], torch.device("cpu"))
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(torch.randn_like(p) * 10.0)
        for p in model.decoder.parameters():
            p.add_(torch.randn_like(p) * 10.0)
    after = v5._operator_only_transport_gain(model, obs, [(0, 1), (1, 2)], torch.device("cpu"))
    assert before == after
