from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "paired_acquisition"
    / "run_synthetic_crossed_factor_identifiability.py"
)
SPEC = importlib.util.spec_from_file_location("synthetic_identifiability", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


def small_config() -> experiment.ExperimentConfig:
    return experiment.ExperimentConfig(
        identities=20,
        scanners=5,
        biological_latent_dim=3,
        acquisition_latent_dim=2,
        observation_dim=12,
        nonlinear_hidden_dim=16,
        pa_nf_biological_dim=8,
        pa_nf_acquisition_dim=4,
        pa_nf_hidden_dim=16,
        epochs=2,
        bootstrap_replicates=50,
        dataset_seed=77,
    )


@pytest.mark.parametrize("renderer", experiment.RENDERERS)
def test_crossed_split_has_one_unseen_scanner_per_identity(renderer: str) -> None:
    config = small_config()
    dataset = experiment.make_synthetic_dataset(config, renderer)

    assert dataset.observations.shape == (
        config.identities * config.scanners,
        config.observation_dim,
    )
    assert len(dataset.train_indices) == config.identities * (config.scanners - 1)
    assert len(dataset.test_indices) == config.identities

    for identity in range(config.identities):
        identity_rows = np.flatnonzero(dataset.identity_ids == identity)
        train_rows = np.intersect1d(identity_rows, dataset.train_indices)
        test_rows = np.intersect1d(identity_rows, dataset.test_indices)
        assert len(train_rows) == config.scanners - 1
        assert len(test_rows) == 1
        assert (
            dataset.scanner_ids[test_rows[0]]
            == dataset.heldout_scanner_by_identity[identity]
        )

    counts = np.bincount(
        dataset.heldout_scanner_by_identity,
        minlength=config.scanners,
    )
    assert counts.max() - counts.min() <= 1


@pytest.mark.parametrize("renderer", experiment.RENDERERS)
def test_counterfactual_indices_are_valid(renderer: str) -> None:
    config = small_config()
    dataset = experiment.make_synthetic_dataset(config, renderer)
    train = set(dataset.train_indices.tolist())

    for identity in range(config.identities):
        target_index = int(dataset.test_indices[identity])
        source_index = int(dataset.source_index_by_identity[identity])
        donor_index = int(dataset.donor_index_by_identity[identity])
        target_scanner = int(dataset.scanner_ids[target_index])

        assert source_index in train
        assert donor_index in train
        assert dataset.identity_ids[source_index] == identity
        assert dataset.scanner_ids[donor_index] == target_scanner
        assert dataset.identity_ids[donor_index] != identity
        assert (
            dataset.donor_identity_by_identity[identity]
            == dataset.identity_ids[donor_index]
        )


def test_bootstrap_interval_is_deterministic() -> None:
    values = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    first = experiment.bootstrap_mean_interval(values, replicates=200, seed=123)
    second = experiment.bootstrap_mean_interval(values, replicates=200, seed=123)
    assert first == second
    assert first[0] <= np.mean(values) <= first[1]


def test_control_models_expose_separate_branches_and_decoder() -> None:
    config = small_config()
    inputs = torch.randn(7, config.observation_dim)

    joint = experiment.JointAutoencoder(
        config.observation_dim,
        config.pa_nf_biological_dim,
        config.pa_nf_acquisition_dim,
        config.pa_nf_hidden_dim,
    )
    joint_output = experiment.model_forward(joint, inputs)
    assert joint_output["biological"].shape == (
        7,
        config.pa_nf_biological_dim,
    )
    assert joint_output["acquisition"].shape == (
        7,
        config.pa_nf_acquisition_dim,
    )
    assert joint_output["reconstruction"].shape == inputs.shape

    oracle = experiment.OracleSupervisedFactorizer(
        config.observation_dim,
        config.biological_latent_dim,
        config.acquisition_latent_dim,
        config.nonlinear_hidden_dim,
    )
    oracle_output = experiment.model_forward(oracle, inputs)
    assert oracle_output["biological"].shape == (
        7,
        config.biological_latent_dim,
    )
    assert oracle_output["acquisition"].shape == (
        7,
        config.acquisition_latent_dim,
    )
    assert oracle_output["reconstruction"].shape == inputs.shape


def test_seed_parser_rejects_duplicates() -> None:
    assert experiment.parse_int_list("1,2,3") == (1, 2, 3)
    with pytest.raises(Exception):
        experiment.parse_int_list("1,1")
