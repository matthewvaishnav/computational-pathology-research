from __future__ import annotations

import numpy as np
import torch

from experiments.paired_acquisition import (
    run_crossed_target_scanner_prototype_factorization as experiment,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as base,
)


def _small_dataset() -> base.SyntheticDataset:
    config = base.ExperimentConfig(
        identities=10,
        scanners=5,
        biological_latent_dim=3,
        acquisition_latent_dim=2,
        observation_dim=8,
        nonlinear_hidden_dim=16,
        pa_nf_biological_dim=8,
        pa_nf_acquisition_dim=4,
        pa_nf_hidden_dim=16,
        epochs=1,
        bootstrap_replicates=20,
    )
    return base.make_synthetic_dataset(config, "linear")


def test_crossed_pairs_use_only_observed_same_identity_cells() -> None:
    dataset = _small_dataset()
    source, target = experiment.build_crossed_pairs(dataset)
    expected = 10 * 4 * 3
    assert len(source) == expected
    assert len(target) == expected
    assert set(source.tolist()).issubset(set(dataset.train_indices.tolist()))
    assert set(target.tolist()).issubset(set(dataset.train_indices.tolist()))
    assert np.all(dataset.identity_ids[source] == dataset.identity_ids[target])
    assert np.all(dataset.scanner_ids[source] != dataset.scanner_ids[target])


def test_scanner_prototype_is_independent_of_donor_observation() -> None:
    model = experiment.ScannerPrototypeFactorizer(
        input_dim=8,
        biological_dim=4,
        acquisition_dim=3,
        hidden_dim=12,
        scanners=5,
    )
    inputs = torch.randn(2, 8)
    scanners = torch.tensor([3, 3], dtype=torch.long)
    output = model(inputs, scanners)
    torch.testing.assert_close(
        output["acquisition"][0],
        output["acquisition"][1],
    )


def test_prototype_decoder_changes_only_through_requested_scanner() -> None:
    model = experiment.ScannerPrototypeFactorizer(
        input_dim=8,
        biological_dim=4,
        acquisition_dim=3,
        hidden_dim=12,
        scanners=5,
    )
    biological = torch.randn(1, 4).repeat(2, 1)
    same_scanner = torch.tensor([2, 2], dtype=torch.long)
    decoded = experiment.decode_prototype(model, biological, same_scanner)
    torch.testing.assert_close(decoded[0], decoded[1])


def _run(model: str, crossed: bool) -> dict:
    return {
        "model_family": model,
        "evaluation": {
            "gates": {
                "crossed_factorization_success": crossed,
            }
        },
    }


def test_path_gate_opens_for_valid_incremental_crossed_result() -> None:
    runs = [
        _run("oracle_supervised", True),
        _run("pa_nf", False),
        _run("prototype_reconstruction", False),
        _run("crossed_target_prototype", True),
    ]
    result = experiment.validate_controls(runs)
    assert result["path_forward_gate_open"] is True
    assert result["crossed_objective_incremental_value"] is True
    assert result["architecture_sufficient_without_crossed_loss"] is False


def test_path_gate_closes_when_proposed_model_fails() -> None:
    runs = [
        _run("oracle_supervised", True),
        _run("pa_nf", False),
        _run("prototype_reconstruction", False),
        _run("crossed_target_prototype", False),
    ]
    result = experiment.validate_controls(runs)
    assert result["path_forward_gate_open"] is False


def test_summary_csv_supports_heterogeneous_model_metrics(tmp_path) -> None:
    rows = [
        {
            "renderer": "linear",
            "shared_metric_mean": 1.0,
        },
        {
            "renderer": "nonlinear",
            "shared_metric_mean": 2.0,
            "counterfactual_delta_mean": 3.0,
        },
    ]

    fieldnames = experiment.summary_csv_fieldnames(rows)

    assert fieldnames == [
        "renderer",
        "shared_metric_mean",
        "counterfactual_delta_mean",
    ]

    output = tmp_path / "summary.csv"
    experiment.atomic_csv(output, fieldnames, rows)

    header = output.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(fieldnames)
