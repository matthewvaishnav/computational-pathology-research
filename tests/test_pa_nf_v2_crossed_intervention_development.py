from __future__ import annotations

import numpy as np
import torch

from experiments.paired_acquisition import (
    run_pa_nf_v2_crossed_intervention_development as experiment,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as base,
)


def tiny_config() -> experiment.ExperimentConfig:
    return experiment.ExperimentConfig(
        identities=20,
        scanners=5,
        observation_dim=24,
        nonlinear_hidden_dim=32,
        biological_dim=12,
        acquisition_dim=6,
        hidden_dim=32,
        epochs=2,
        bootstrap_replicates=100,
        dataset_seed=5317,
    )


def make_dataset(config: experiment.ExperimentConfig) -> base.SyntheticDataset:
    return base.make_synthetic_dataset(experiment.to_base_config(config), "linear")


def test_inference_requires_images_only() -> None:
    config = tiny_config()
    model = experiment.build_model(config, torch.device("cpu"))
    inputs = torch.randn(7, config.observation_dim)
    output = model(inputs)
    assert output["biological"].shape == (7, config.biological_dim)
    assert output["acquisition"].shape == (7, config.acquisition_dim)
    assert output["reconstruction"].shape == (7, config.observation_dim)


def test_candidate_and_control_are_exactly_capacity_matched() -> None:
    config = tiny_config()
    candidate = experiment.build_model(config, torch.device("cpu"))
    control = experiment.build_model(config, torch.device("cpu"))
    assert experiment.parameter_count(candidate) == experiment.parameter_count(control)
    assert experiment.family_weights("pa_nf_v2_crossed_intervention", config) == (
        config.crossed_reconstruction_weight,
        config.cycle_weight,
    )
    assert experiment.family_weights("v2_control_no_cross_cycle", config) == (0.0, 0.0)


def test_crossed_pairs_are_train_only_same_identity_different_scanner() -> None:
    config = tiny_config()
    dataset = make_dataset(config)
    source, target = experiment.build_crossed_pairs(dataset)
    train_set = set(int(index) for index in dataset.train_indices.tolist())
    assert len(source) == config.identities * (config.scanners - 1) * (config.scanners - 2)
    assert len(source) == len(target)
    for left, right in zip(source.tolist(), target.tolist()):
        assert left in train_set
        assert right in train_set
        assert dataset.identity_ids[left] == dataset.identity_ids[right]
        assert dataset.scanner_ids[left] != dataset.scanner_ids[right]


def test_biological_consistency_pairs_are_train_only() -> None:
    config = tiny_config()
    dataset = make_dataset(config)
    left, right = experiment.build_biological_consistency_pairs(dataset)
    expected_per_identity = (config.scanners - 1) * (config.scanners - 2) // 2
    assert len(left) == config.identities * expected_per_identity
    train_set = set(int(index) for index in dataset.train_indices.tolist())
    for first, second in zip(left.tolist(), right.tolist()):
        assert first in train_set
        assert second in train_set
        assert dataset.identity_ids[first] == dataset.identity_ids[second]
        assert dataset.scanner_ids[first] != dataset.scanner_ids[second]


def test_one_epoch_training_is_finite_for_both_families() -> None:
    config = tiny_config()
    config = experiment.replace(config, epochs=1)
    dataset = make_dataset(config)
    for family in experiment.MODEL_FAMILIES:
        base.set_deterministic_seed(3101)
        model = experiment.build_model(config, torch.device("cpu"))
        training = experiment.train_model(
            family, model, dataset, config, torch.device("cpu")
        )
        assert training["optimizer_steps"] == 1
        assert np.isfinite(training["history"][-1]["total"])


def test_cycle_diagnostics_are_finite() -> None:
    config = tiny_config()
    dataset = make_dataset(config)
    model = experiment.build_model(config, torch.device("cpu"))
    diagnostics = experiment.cycle_diagnostics(model, dataset, torch.device("cpu"))
    assert all(np.isfinite(value) for value in diagnostics.values())
    assert diagnostics["cycle_total_mse"] >= 0.0
