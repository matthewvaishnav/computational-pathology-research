import torch

from experiments.paired_acquisition import (
    run_pa_nf_v2_crossed_intervention_development_r2 as experiment,
)


def test_candidate_and_control_are_parameter_matched():
    config = experiment.ExperimentConfig()
    candidate = experiment.r1.build_model(config, torch.device("cpu"))
    control = experiment.r1.build_model(config, torch.device("cpu"))
    assert experiment.r1.parameter_count(candidate) == experiment.r1.parameter_count(control)


def test_family_weights_only_disable_cross_and_cycle_in_control():
    config = experiment.ExperimentConfig()
    assert experiment.family_weights("pa_nf_v2_crossed_intervention_r2", config) == (
        config.crossed_reconstruction_weight,
        config.biological_cycle_weight,
        config.acquisition_cycle_weight,
    )
    assert experiment.family_weights("v2_r2_control_no_cross_cycle", config) == (
        0.0,
        0.0,
        0.0,
    )


def test_same_scanner_consistency_zero_for_identical_codes():
    acquisition = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    scanner_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss = experiment.same_scanner_acquisition_consistency(acquisition, scanner_ids, 2)
    assert float(loss) == 0.0


def test_same_scanner_consistency_penalizes_identity_specific_spread():
    acquisition = torch.tensor(
        [[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
        dtype=torch.float32,
    )
    scanner_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss = experiment.same_scanner_acquisition_consistency(acquisition, scanner_ids, 2)
    assert float(loss) > 0.0


def test_inference_forward_requires_no_scanner_id():
    config = experiment.ExperimentConfig(observation_dim=8, hidden_dim=16, biological_dim=4, acquisition_dim=2)
    model = experiment.r1.build_model(config, torch.device("cpu"))
    output = model(torch.randn(3, 8))
    assert set(output) == {"biological", "acquisition", "reconstruction"}
    assert output["biological"].shape == (3, 4)
    assert output["acquisition"].shape == (3, 2)
    assert output["reconstruction"].shape == (3, 8)
