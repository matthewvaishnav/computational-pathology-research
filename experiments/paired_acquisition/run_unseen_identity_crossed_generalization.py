#!/usr/bin/env python3
"""Unseen-identity generalization for crossed-target scanner prototypes.

This post-confirmatory exploratory experiment follows the successful missing-cell
synthetic campaign. It removes identity overlap between training and evaluation:

* every training biological identity is observed under every scanner;
* every test biological identity is entirely absent from optimization;
* evaluation covers every ordered source-scanner -> target-scanner transfer;
* representation probes are fit only on training identities and evaluated only
  on unseen identities; and
* multiple independent dataset-generation seeds are required.

The four model families and factorization thresholds are unchanged from the
preceding crossed-target scanner-prototype experiment.
"""

from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.paired_acquisition import (  # noqa: E402
    run_crossed_target_scanner_prototype_factorization as parent,
)
from experiments.paired_acquisition import (  # noqa: E402
    run_synthetic_crossed_factor_identifiability as base,
)


SCHEMA_VERSION = "paired-acquisition-unseen-identity-generalization/v1"
DEFAULT_MODEL_SEEDS = tuple(range(2201, 2211))
DEFAULT_DATASET_SEEDS = (4301, 5301, 6301)
RENDERERS = parent.RENDERERS
MODEL_FAMILIES = parent.MODEL_FAMILIES


class ExperimentError(parent.ExperimentError):
    """Raised when the unseen-identity experiment cannot proceed safely."""


@dataclass(frozen=True)
class ExperimentConfig(parent.ExperimentConfig):
    """Parent architecture/training configuration plus unseen test identities."""

    test_identities: int = 128


def make_unseen_identity_dataset(
    config: ExperimentConfig,
    renderer: str,
) -> base.SyntheticDataset:
    """Create a complete grid split by biological identity, never by cell."""
    if renderer not in RENDERERS:
        raise ExperimentError("Unknown renderer: {}".format(renderer))
    if config.identities < config.scanners * 2:
        raise ExperimentError(
            "Training requires at least two identities per scanner."
        )
    if config.test_identities < 2:
        raise ExperimentError("At least two unseen test identities are required.")

    total_identities = config.identities + config.test_identities
    renderer_offset = 0 if renderer == "linear" else 100_000
    effective_seed = config.dataset_seed + renderer_offset
    rng = np.random.default_rng(effective_seed)

    biological_identity_latents = rng.normal(
        size=(total_identities, config.biological_latent_dim)
    )
    scanner_latents = rng.normal(
        size=(config.scanners, config.acquisition_latent_dim)
    )
    scanner_latents = scanner_latents / np.maximum(
        np.linalg.norm(scanner_latents, axis=1, keepdims=True),
        1e-12,
    )

    identity_ids = np.repeat(
        np.arange(total_identities, dtype=np.int64),
        config.scanners,
    )
    scanner_ids = np.tile(
        np.arange(config.scanners, dtype=np.int64),
        total_identities,
    )
    biological = biological_identity_latents[identity_ids]
    acquisition = scanner_latents[scanner_ids]

    if renderer == "linear":
        observations, renderer_metadata = base._render_linear(
            biological,
            acquisition,
            rng,
            config.observation_dim,
        )
    else:
        observations, renderer_metadata = base._render_nonlinear(
            biological,
            acquisition,
            rng,
            config.observation_dim,
            config.nonlinear_hidden_dim,
        )

    if config.noise_std > 0:
        observations = observations + rng.normal(
            scale=config.noise_std,
            size=observations.shape,
        )

    training_identity_mask = identity_ids < config.identities
    train_indices = np.flatnonzero(training_identity_mask)
    test_indices = np.flatnonzero(~training_identity_mask)

    expected_train = config.identities * config.scanners
    expected_test = config.test_identities * config.scanners
    if len(train_indices) != expected_train or len(test_indices) != expected_test:
        raise ExperimentError("Identity-level split produced unexpected cell counts.")

    train_mean = observations[train_indices].mean(axis=0, keepdims=True)
    train_std = observations[train_indices].std(axis=0, keepdims=True)
    train_std = np.where(train_std < 1e-6, 1.0, train_std)
    standardized = (observations - train_mean) / train_std

    # Parent pair builders iterate over the length of this vector. Giving it
    # exactly the number of training identities makes them construct crossed and
    # consistency pairs only from the training partition. Values are sentinel -1
    # because this experiment does not withhold scanner cells within identities.
    training_identity_sentinel = np.full(
        config.identities,
        -1,
        dtype=np.int64,
    )
    source_indices = (
        np.arange(config.identities, dtype=np.int64) * config.scanners
    )
    donor_identities = (
        np.arange(config.identities, dtype=np.int64) + 1
    ) % config.identities
    donor_indices = donor_identities * config.scanners

    renderer_metadata = {
        **dict(renderer_metadata),
        "dataset_seed": effective_seed,
        "noise_std": config.noise_std,
        "split": "entirely_disjoint_biological_identities",
        "training_identity_count": config.identities,
        "test_identity_count": config.test_identities,
        "training_identity_ids_sha256": base.sha256_bytes(
            np.arange(config.identities, dtype="<i8").tobytes()
        ),
        "test_identity_ids_sha256": base.sha256_bytes(
            np.arange(
                config.identities,
                total_identities,
                dtype="<i8",
            ).tobytes()
        ),
    }

    return base.SyntheticDataset(
        observations=standardized.astype(np.float32),
        biological_latents=biological.astype(np.float32),
        acquisition_latents=acquisition.astype(np.float32),
        identity_ids=identity_ids,
        scanner_ids=scanner_ids,
        train_indices=train_indices,
        test_indices=test_indices,
        heldout_scanner_by_identity=training_identity_sentinel,
        source_index_by_identity=source_indices,
        donor_index_by_identity=donor_indices,
        donor_identity_by_identity=donor_identities,
        train_mean=train_mean.astype(np.float64),
        train_std=train_std.astype(np.float64),
        renderer=renderer,
        renderer_metadata=renderer_metadata,
    )


def build_all_ordered_test_pairs(
    dataset: base.SyntheticDataset,
    scanners: int,
) -> Dict[str, np.ndarray]:
    """Enumerate every source != target scanner transfer for unseen identities."""
    test_identity_ids = np.unique(dataset.identity_ids[dataset.test_indices])
    if len(test_identity_ids) < 2:
        raise ExperimentError("At least two unseen identities are required.")

    sources: List[int] = []
    targets: List[int] = []
    donors: List[int] = []
    identity_positions: List[int] = []
    source_scanners: List[int] = []
    target_scanners: List[int] = []

    test_index_set = set(int(index) for index in dataset.test_indices.tolist())
    for identity_position, identity in enumerate(test_identity_ids.tolist()):
        donor_identity = int(
            test_identity_ids[(identity_position + 1) % len(test_identity_ids)]
        )
        if donor_identity == int(identity):
            raise ExperimentError("Counterfactual donor identity must differ.")

        for source_scanner, target_scanner in itertools.permutations(
            range(scanners),
            2,
        ):
            source = int(identity) * scanners + int(source_scanner)
            target = int(identity) * scanners + int(target_scanner)
            donor = donor_identity * scanners + int(target_scanner)

            if (
                source not in test_index_set
                or target not in test_index_set
                or donor not in test_index_set
            ):
                raise ExperimentError(
                    "Unseen-identity intervention referenced training data."
                )
            if dataset.identity_ids[source] != dataset.identity_ids[target]:
                raise ExperimentError("Source and target biology differ.")
            if dataset.scanner_ids[source] == dataset.scanner_ids[target]:
                raise ExperimentError("Ordered transfer did not change scanner.")
            if dataset.identity_ids[donor] == dataset.identity_ids[target]:
                raise ExperimentError("Donor biology must differ from target.")
            if dataset.scanner_ids[donor] != dataset.scanner_ids[target]:
                raise ExperimentError("Donor must provide the target scanner.")

            sources.append(source)
            targets.append(target)
            donors.append(donor)
            identity_positions.append(identity_position)
            source_scanners.append(source_scanner)
            target_scanners.append(target_scanner)

    expected = len(test_identity_ids) * scanners * (scanners - 1)
    if len(sources) != expected:
        raise ExperimentError("Ordered intervention count is incorrect.")

    return {
        "source": np.asarray(sources, dtype=np.int64),
        "target": np.asarray(targets, dtype=np.int64),
        "donor": np.asarray(donors, dtype=np.int64),
        "identity_position": np.asarray(identity_positions, dtype=np.int64),
        "source_scanner": np.asarray(source_scanners, dtype=np.int64),
        "target_scanner": np.asarray(target_scanners, dtype=np.int64),
        "test_identity_ids": test_identity_ids.astype(np.int64),
    }


def cross_scanner_identity_retrieval_top1(
    features: np.ndarray,
    identity_ids: np.ndarray,
    scanner_ids: np.ndarray,
) -> float:
    """Retrieve unseen identity using only gallery views from other scanners."""
    features = np.asarray(features, dtype=np.float64)
    identity_ids = np.asarray(identity_ids)
    scanner_ids = np.asarray(scanner_ids)
    if (
        features.ndim != 2
        or len(features) != len(identity_ids)
        or len(features) != len(scanner_ids)
        or len(features) == 0
    ):
        raise ExperimentError("Invalid unseen-identity retrieval inputs.")

    normalized = features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True),
        1e-12,
    )
    similarity = normalized @ normalized.T
    valid = scanner_ids[:, None] != scanner_ids[None, :]
    if not np.all(valid.any(axis=1)):
        raise ExperimentError("Every query needs an alternate-scanner gallery.")
    similarity = np.where(valid, similarity, -np.inf)
    nearest = np.argmax(similarity, axis=1)
    return float(np.mean(identity_ids[nearest] == identity_ids))


def acquisition_within_scanner_variance(
    acquisition: np.ndarray,
    scanner_ids: np.ndarray,
) -> float:
    """Measure donor-identity variation remaining within scanner codes."""
    values: List[float] = []
    for scanner in np.unique(scanner_ids):
        group = acquisition[scanner_ids == scanner]
        centered = group - group.mean(axis=0, keepdims=True)
        values.append(float(np.mean(centered**2)))
    return float(np.mean(values))


def _forward_all(
    model_family: str,
    model: nn.Module,
    observations: torch.Tensor,
    scanner_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model_family in parent.PROTOTYPE_FAMILIES:
        prototype_model = model
        biological = prototype_model.encode_biological(observations)  # type: ignore[attr-defined]
        acquisition = prototype_model.acquisition_from_scanner(scanner_ids)  # type: ignore[attr-defined]
        reconstruction = prototype_model.decode(  # type: ignore[attr-defined]
            biological,
            acquisition,
        )
        return biological, acquisition, reconstruction

    output = base.model_forward(model, observations)
    return (
        output["biological"],
        output["acquisition"],
        output["reconstruction"],
    )


def _decode(
    model_family: str,
    model: nn.Module,
    biological: torch.Tensor,
    acquisition: torch.Tensor,
) -> torch.Tensor:
    if model_family in parent.PROTOTYPE_FAMILIES:
        return model.decode(biological, acquisition)  # type: ignore[attr-defined]
    return base.decode_branches(model, biological, acquisition)


def make_gates(metrics: Mapping[str, float]) -> Dict[str, bool]:
    """Apply the unchanged v1 crossed-factorization thresholds."""
    gates = {
        "biology_retention_ci_positive": (
            metrics["biology_retention_delta_ci_025"] > 0
        ),
        "acquisition_transfer_ci_positive": (
            metrics["acquisition_transfer_delta_ci_025"] > 0
        ),
        "two_axis_majority_identities": (
            metrics["two_axis_identity_success_rate"] > 0.5
        ),
        "biological_factor_recovery": (
            metrics["biological_to_biological_r2"] >= 0.80
        ),
        "biological_acquisition_exclusion": (
            metrics["biological_to_acquisition_r2"] <= 0.10
        ),
        "acquisition_factor_recovery": (
            metrics["acquisition_to_acquisition_r2"] >= 0.80
        ),
        "acquisition_biological_exclusion": (
            metrics["acquisition_to_biological_r2"] <= 0.10
        ),
        "combined_factor_recovery": (
            metrics["combined_to_joint_factors_r2"] >= 0.80
        ),
    }
    gates["two_axis_counterfactual_success"] = bool(
        gates["biology_retention_ci_positive"]
        and gates["acquisition_transfer_ci_positive"]
        and gates["two_axis_majority_identities"]
    )
    gates["factor_allocation_success"] = bool(
        gates["biological_factor_recovery"]
        and gates["biological_acquisition_exclusion"]
        and gates["acquisition_factor_recovery"]
        and gates["acquisition_biological_exclusion"]
    )
    gates["crossed_factorization_success"] = bool(
        gates["two_axis_counterfactual_success"]
        and gates["factor_allocation_success"]
    )
    return gates


def evaluate_model(
    model_family: str,
    model: nn.Module,
    dataset: base.SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate all ordered scanner interventions on entirely unseen identities."""
    model.eval()
    observations = torch.as_tensor(
        dataset.observations,
        dtype=torch.float32,
        device=device,
    )
    scanner_ids = torch.as_tensor(
        dataset.scanner_ids,
        dtype=torch.long,
        device=device,
    )
    pairs = build_all_ordered_test_pairs(dataset, config.scanners)
    source = torch.as_tensor(pairs["source"], dtype=torch.long, device=device)
    target = torch.as_tensor(pairs["target"], dtype=torch.long, device=device)
    donor = torch.as_tensor(pairs["donor"], dtype=torch.long, device=device)

    with torch.no_grad():
        biological, acquisition, reconstruction = _forward_all(
            model_family,
            model,
            observations,
            scanner_ids,
        )
        zeros_b = torch.zeros_like(biological)
        zeros_a = torch.zeros_like(acquisition)
        content_only = _decode(
            model_family,
            model,
            biological,
            zeros_a,
        )
        acquisition_only = _decode(
            model_family,
            model,
            zeros_b,
            acquisition,
        )

        source_biological = biological.index_select(0, source)
        if model_family in parent.PROTOTYPE_FAMILIES:
            target_acquisition = model.acquisition_from_scanner(  # type: ignore[attr-defined]
                scanner_ids.index_select(0, target)
            )
        else:
            target_acquisition = acquisition.index_select(0, donor)

        swapped = _decode(
            model_family,
            model,
            source_biological,
            target_acquisition,
        )
        correct_target = observations.index_select(0, target)
        donor_target = observations.index_select(0, donor)
        source_target = observations.index_select(0, source)

        correct_mse = (
            (swapped - correct_target)
            .square()
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        donor_mse = (
            (swapped - donor_target)
            .square()
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        source_mse = (
            (swapped - source_target)
            .square()
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )

    biology_delta = donor_mse - correct_mse
    acquisition_delta = source_mse - correct_mse
    identity_position = pairs["identity_position"]
    test_identity_count = len(pairs["test_identity_ids"])

    biology_by_identity = np.asarray(
        [
            biology_delta[identity_position == position].mean()
            for position in range(test_identity_count)
        ],
        dtype=np.float64,
    )
    acquisition_by_identity = np.asarray(
        [
            acquisition_delta[identity_position == position].mean()
            for position in range(test_identity_count)
        ],
        dtype=np.float64,
    )
    pair_two_axis_success = (biology_delta > 0) & (acquisition_delta > 0)
    pair_success_by_identity = np.asarray(
        [
            pair_two_axis_success[identity_position == position].mean()
            for position in range(test_identity_count)
        ],
        dtype=np.float64,
    )

    biology_low, biology_high = base.bootstrap_mean_interval(
        biology_by_identity,
        config.bootstrap_replicates,
        seed + config.dataset_seed * 100 + 800_000,
    )
    acquisition_low, acquisition_high = base.bootstrap_mean_interval(
        acquisition_by_identity,
        config.bootstrap_replicates,
        seed + config.dataset_seed * 100 + 900_000,
    )

    ordered_scanner_pair_metrics: List[Dict[str, Any]] = []
    for source_scanner, target_scanner in itertools.permutations(
        range(config.scanners),
        2,
    ):
        mask = (
            (pairs["source_scanner"] == source_scanner)
            & (pairs["target_scanner"] == target_scanner)
        )
        ordered_scanner_pair_metrics.append(
            {
                "source_scanner": int(source_scanner),
                "target_scanner": int(target_scanner),
                "pair_count": int(mask.sum()),
                "biology_retention_delta": float(biology_delta[mask].mean()),
                "acquisition_transfer_delta": float(
                    acquisition_delta[mask].mean()
                ),
                "two_axis_pair_success_rate": float(
                    pair_two_axis_success[mask].mean()
                ),
            }
        )

    biological_np = biological.detach().cpu().numpy()
    acquisition_np = acquisition.detach().cpu().numpy()
    reconstruction_np = reconstruction.detach().cpu().numpy()
    content_only_np = content_only.detach().cpu().numpy()
    acquisition_only_np = acquisition_only.detach().cpu().numpy()

    train = dataset.train_indices
    test = dataset.test_indices
    combined_np = np.concatenate([biological_np, acquisition_np], axis=1)
    joint_truth = np.concatenate(
        [dataset.biological_latents, dataset.acquisition_latents],
        axis=1,
    )

    full_mse = float(
        np.mean(
            (reconstruction_np[test] - dataset.observations[test]) ** 2
        )
    )
    content_only_mse = float(
        np.mean((content_only_np[test] - dataset.observations[test]) ** 2)
    )
    acquisition_only_mse = float(
        np.mean(
            (acquisition_only_np[test] - dataset.observations[test]) ** 2
        )
    )

    metrics = {
        "test_reconstruction_mse": full_mse,
        "content_branch_ablation_penalty": acquisition_only_mse - full_mse,
        "acquisition_branch_ablation_penalty": content_only_mse - full_mse,
        "counterfactual_correct_target_mse": float(correct_mse.mean()),
        "counterfactual_donor_target_mse": float(donor_mse.mean()),
        "source_scanner_target_mse": float(source_mse.mean()),
        "biology_retention_delta": float(biology_by_identity.mean()),
        "biology_retention_delta_ci_025": float(biology_low),
        "biology_retention_delta_ci_975": float(biology_high),
        "biology_retention_identity_success_rate": float(
            np.mean(biology_by_identity > 0)
        ),
        "acquisition_transfer_delta": float(
            acquisition_by_identity.mean()
        ),
        "acquisition_transfer_delta_ci_025": float(acquisition_low),
        "acquisition_transfer_delta_ci_975": float(acquisition_high),
        "acquisition_transfer_identity_success_rate": float(
            np.mean(acquisition_by_identity > 0)
        ),
        "two_axis_identity_success_rate": float(
            np.mean(
                (biology_by_identity > 0)
                & (acquisition_by_identity > 0)
            )
        ),
        "all_ordered_pair_two_axis_success_rate": float(
            pair_two_axis_success.mean()
        ),
        "worst_identity_ordered_pair_success_rate": float(
            pair_success_by_identity.min()
        ),
        "worst_scanner_pair_biology_retention_delta": float(
            min(
                row["biology_retention_delta"]
                for row in ordered_scanner_pair_metrics
            )
        ),
        "worst_scanner_pair_acquisition_transfer_delta": float(
            min(
                row["acquisition_transfer_delta"]
                for row in ordered_scanner_pair_metrics
            )
        ),
        "worst_scanner_pair_two_axis_success_rate": float(
            min(
                row["two_axis_pair_success_rate"]
                for row in ordered_scanner_pair_metrics
            )
        ),
        "biological_to_biological_r2": base.ridge_r2(
            biological_np[train],
            dataset.biological_latents[train],
            biological_np[test],
            dataset.biological_latents[test],
        ),
        "biological_to_acquisition_r2": base.ridge_r2(
            biological_np[train],
            dataset.acquisition_latents[train],
            biological_np[test],
            dataset.acquisition_latents[test],
        ),
        "acquisition_to_acquisition_r2": base.ridge_r2(
            acquisition_np[train],
            dataset.acquisition_latents[train],
            acquisition_np[test],
            dataset.acquisition_latents[test],
        ),
        "acquisition_to_biological_r2": base.ridge_r2(
            acquisition_np[train],
            dataset.biological_latents[train],
            acquisition_np[test],
            dataset.biological_latents[test],
        ),
        "combined_to_joint_factors_r2": base.ridge_r2(
            combined_np[train],
            joint_truth[train],
            combined_np[test],
            joint_truth[test],
        ),
        "biological_scanner_balanced_accuracy": (
            base.scanner_balanced_accuracy(
                biological_np[train],
                dataset.scanner_ids[train],
                biological_np[test],
                dataset.scanner_ids[test],
            )
        ),
        "acquisition_scanner_balanced_accuracy": (
            base.scanner_balanced_accuracy(
                acquisition_np[train],
                dataset.scanner_ids[train],
                acquisition_np[test],
                dataset.scanner_ids[test],
            )
        ),
        "unseen_biological_identity_retrieval_top1": (
            cross_scanner_identity_retrieval_top1(
                biological_np[test],
                dataset.identity_ids[test],
                dataset.scanner_ids[test],
            )
        ),
        "acquisition_donor_invariance_mse": (
            acquisition_within_scanner_variance(
                acquisition_np[test],
                dataset.scanner_ids[test],
            )
        ),
    }
    gates = make_gates(metrics)

    return {
        "model_family": model_family,
        "metrics": metrics,
        "gates": gates,
        "ordered_pair_count": int(len(biology_delta)),
        "test_identity_count": int(test_identity_count),
        "biology_retention_delta_by_identity": (
            biology_by_identity.tolist()
        ),
        "acquisition_transfer_delta_by_identity": (
            acquisition_by_identity.tolist()
        ),
        "ordered_pair_success_rate_by_identity": (
            pair_success_by_identity.tolist()
        ),
        "ordered_scanner_pair_metrics": ordered_scanner_pair_metrics,
    }


def summarize_runs(
    runs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Summarize model seeds separately for every dataset/renderer condition."""
    grouped: Dict[Tuple[int, str, str], List[Mapping[str, Any]]] = {}
    for run in runs:
        key = (
            int(run["dataset_seed"]),
            str(run["renderer"]),
            str(run["model_family"]),
        )
        grouped.setdefault(key, []).append(run)

    metric_names = sorted(
        {
            metric
            for run in runs
            for metric in run["evaluation"]["metrics"].keys()
        }
    )
    summaries: List[Dict[str, Any]] = []
    for (dataset_seed, renderer, model_family), group in sorted(
        grouped.items()
    ):
        metrics: Dict[str, Any] = {}
        for name in metric_names:
            values = [
                float(run["evaluation"]["metrics"][name])
                for run in group
                if name in run["evaluation"]["metrics"]
            ]
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            metrics[name] = {
                "mean": float(array.mean()),
                "std": (
                    float(array.std(ddof=1))
                    if len(array) > 1
                    else 0.0
                ),
                "min": float(array.min()),
                "max": float(array.max()),
            }

        summaries.append(
            {
                "dataset_seed": int(dataset_seed),
                "renderer": renderer,
                "model_family": model_family,
                "model_seed_count": len(group),
                "metrics": metrics,
                "all_seed_two_axis_counterfactual_success": all(
                    bool(
                        run["evaluation"]["gates"].get(
                            "two_axis_counterfactual_success",
                            False,
                        )
                    )
                    for run in group
                ),
                "all_seed_factor_allocation_success": all(
                    bool(
                        run["evaluation"]["gates"].get(
                            "factor_allocation_success",
                            False,
                        )
                    )
                    for run in group
                ),
                "all_seed_crossed_factorization_success": all(
                    bool(
                        run["evaluation"]["gates"].get(
                            "crossed_factorization_success",
                            False,
                        )
                    )
                    for run in group
                ),
            }
        )
    return summaries


def flattened_summary_rows(
    summaries: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        row: Dict[str, Any] = {
            "dataset_seed": summary["dataset_seed"],
            "renderer": summary["renderer"],
            "model_family": summary["model_family"],
            "model_seed_count": summary["model_seed_count"],
            "all_seed_two_axis_counterfactual_success": summary[
                "all_seed_two_axis_counterfactual_success"
            ],
            "all_seed_factor_allocation_success": summary[
                "all_seed_factor_allocation_success"
            ],
            "all_seed_crossed_factorization_success": summary[
                "all_seed_crossed_factorization_success"
            ],
        }
        for metric, values in summary["metrics"].items():
            for statistic in ("mean", "std", "min", "max"):
                row["{}_{}".format(metric, statistic)] = values[statistic]
        rows.append(row)
    return rows


def validate_controls(
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Require the same controls to behave correctly in every condition."""
    condition_results: List[Dict[str, Any]] = []
    conditions = sorted(
        {
            (int(run["dataset_seed"]), str(run["renderer"]))
            for run in runs
        }
    )

    for dataset_seed, renderer in conditions:
        condition = [
            run
            for run in runs
            if int(run["dataset_seed"]) == dataset_seed
            and str(run["renderer"]) == renderer
        ]
        family_all_success: Dict[str, bool] = {}
        family_all_rejected: Dict[str, bool] = {}
        for family in MODEL_FAMILIES:
            family_runs = [
                run
                for run in condition
                if run["model_family"] == family
            ]
            if not family_runs:
                raise ExperimentError(
                    "Missing {} runs for dataset {} renderer {}.".format(
                        family,
                        dataset_seed,
                        renderer,
                    )
                )
            statuses = [
                bool(
                    run["evaluation"]["gates"].get(
                        "crossed_factorization_success",
                        False,
                    )
                )
                for run in family_runs
            ]
            family_all_success[family] = all(statuses)
            family_all_rejected[family] = all(not status for status in statuses)

        condition_results.append(
            {
                "dataset_seed": dataset_seed,
                "renderer": renderer,
                "oracle_all_model_seeds_success": family_all_success[
                    "oracle_supervised"
                ],
                "pa_nf_all_model_seeds_rejected": family_all_rejected[
                    "pa_nf"
                ],
                "prototype_reconstruction_all_model_seeds_success": (
                    family_all_success["prototype_reconstruction"]
                ),
                "crossed_target_prototype_all_model_seeds_success": (
                    family_all_success["crossed_target_prototype"]
                ),
                "condition_gate_open": bool(
                    family_all_success["oracle_supervised"]
                    and family_all_rejected["pa_nf"]
                    and family_all_success["crossed_target_prototype"]
                ),
            }
        )

    oracle_passed = all(
        result["oracle_all_model_seeds_success"]
        for result in condition_results
    )
    pa_nf_rejected = all(
        result["pa_nf_all_model_seeds_rejected"]
        for result in condition_results
    )
    proposed_passed = all(
        result["crossed_target_prototype_all_model_seeds_success"]
        for result in condition_results
    )
    reconstruction_only_passed = all(
        result["prototype_reconstruction_all_model_seeds_success"]
        for result in condition_results
    )
    path_open = bool(
        oracle_passed
        and pa_nf_rejected
        and proposed_passed
        and all(
            result["condition_gate_open"]
            for result in condition_results
        )
    )

    return {
        "condition_results": condition_results,
        "oracle_all_condition_all_seed_success": oracle_passed,
        "pa_nf_all_condition_all_seed_rejected": pa_nf_rejected,
        "crossed_target_prototype_all_condition_all_seed_success": (
            proposed_passed
        ),
        "prototype_reconstruction_all_condition_all_seed_success": (
            reconstruction_only_passed
        ),
        "crossed_objective_incremental_value": bool(
            proposed_passed and not reconstruction_only_passed
        ),
        "architecture_sufficient_without_crossed_loss": bool(
            proposed_passed and reconstruction_only_passed
        ),
        "unseen_identity_generalization_gate_open": path_open,
        "path_forward_gate_open": path_open,
    }


def run_experiment(
    config: ExperimentConfig,
    dataset_seeds: Sequence[int],
    model_seeds: Sequence[int],
    renderers: Sequence[str],
    model_families: Sequence[str],
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if output_root.exists():
        raise ExperimentError(
            "Output root already exists; overwrite is prohibited: {}".format(
                output_root
            )
        )
    output_root.mkdir(parents=True, exist_ok=False)

    datasets: Dict[Tuple[int, str], base.SyntheticDataset] = {}
    dataset_manifest: Dict[str, Any] = {}
    for dataset_seed in dataset_seeds:
        seeded_config = replace(config, dataset_seed=int(dataset_seed))
        seed_manifest: Dict[str, Any] = {}
        for renderer in renderers:
            dataset = make_unseen_identity_dataset(
                seeded_config,
                renderer,
            )
            datasets[(int(dataset_seed), renderer)] = dataset
            train_identity_ids = np.unique(
                dataset.identity_ids[dataset.train_indices]
            )
            test_identity_ids = np.unique(
                dataset.identity_ids[dataset.test_indices]
            )
            if np.intersect1d(
                train_identity_ids,
                test_identity_ids,
            ).size:
                raise ExperimentError("Training and test identities overlap.")

            seed_manifest[renderer] = {
                "renderer_metadata": dict(dataset.renderer_metadata),
                "observation_shape": list(dataset.observations.shape),
                "train_cell_count": int(len(dataset.train_indices)),
                "test_cell_count": int(len(dataset.test_indices)),
                "training_identity_count": int(len(train_identity_ids)),
                "test_identity_count": int(len(test_identity_ids)),
                "identity_overlap_count": 0,
                "all_scanners_per_training_identity": True,
                "all_scanners_per_test_identity": True,
                "ordered_test_transfer_count": int(
                    len(test_identity_ids)
                    * config.scanners
                    * (config.scanners - 1)
                ),
                "train_observation_sha256": base.sha256_bytes(
                    dataset.observations[dataset.train_indices]
                    .astype("<f4")
                    .tobytes()
                ),
                "test_observation_sha256": base.sha256_bytes(
                    dataset.observations[dataset.test_indices]
                    .astype("<f4")
                    .tobytes()
                ),
            }
        dataset_manifest[str(int(dataset_seed))] = seed_manifest

    base.atomic_json(
        output_root / "unseen_identity_dataset_manifest.json",
        dataset_manifest,
    )

    runs: List[Dict[str, Any]] = []
    for dataset_seed in dataset_seeds:
        seeded_config = replace(config, dataset_seed=int(dataset_seed))
        for renderer in renderers:
            dataset = datasets[(int(dataset_seed), renderer)]
            for model_family in model_families:
                for model_seed in model_seeds:
                    print(
                        "[dataset_seed={}] [{}] model={} seed={}".format(
                            dataset_seed,
                            renderer,
                            model_family,
                            model_seed,
                        ),
                        flush=True,
                    )
                    base.set_deterministic_seed(int(model_seed))
                    model = parent.build_model(
                        model_family,
                        seeded_config,
                        device,
                    )
                    training = parent.train_model(
                        model_family,
                        model,
                        dataset,
                        seeded_config,
                        device,
                    )
                    evaluation = evaluate_model(
                        model_family,
                        model,
                        dataset,
                        seeded_config,
                        device,
                        int(model_seed),
                    )
                    run = {
                        "dataset_seed": int(dataset_seed),
                        "renderer": renderer,
                        "model_family": model_family,
                        "model_seed": int(model_seed),
                        "training": training,
                        "evaluation": evaluation,
                        "parameter_count": int(
                            sum(
                                parameter.numel()
                                for parameter in model.parameters()
                            )
                        ),
                    }
                    runs.append(run)
                    run_dir = (
                        output_root
                        / "dataset_seed_{}".format(dataset_seed)
                        / renderer
                        / model_family
                        / "seed_{}".format(model_seed)
                    )
                    base.atomic_json(run_dir / "run_result.json", run)

    summaries = summarize_runs(runs)
    control_validation = validate_controls(runs)
    status = (
        "complete_unseen_identity_generalization_path_open"
        if control_validation[
            "unseen_identity_generalization_gate_open"
        ]
        else "complete_unseen_identity_generalization_path_closed"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "claim_scope": {
            "post_confirmatory_exploratory": True,
            "does_not_modify_frozen_private_campaign": True,
            "does_not_overwrite_prior_synthetic_campaigns": True,
            "not_pathology_domain_evidence": True,
            "primary_question": (
                "Does crossed-target scanner-prototype factorization "
                "generalize intervention-consistent semantics to entirely "
                "unseen biological identities?"
            ),
        },
        "generalization_design": {
            "identity_split": (
                "all scanner views of every biological identity belong "
                "exclusively to either optimization or evaluation"
            ),
            "test_interventions": (
                "all ordered source-scanner to target-scanner transfers "
                "for every unseen biological identity"
            ),
            "probe_protocol": (
                "linear probes fit on training identities and evaluated "
                "on disjoint unseen identities"
            ),
            "thresholds": "unchanged from crossed-target prototype v1",
        },
        "config": asdict(config),
        "dataset_seeds": [int(seed) for seed in dataset_seeds],
        "model_seeds": [int(seed) for seed in model_seeds],
        "renderers": list(renderers),
        "model_families": list(model_families),
        "device": str(device),
        "dataset_manifest": dataset_manifest,
        "runs": runs,
        "summaries": summaries,
        "control_validation": control_validation,
    }
    result["result_sha256"] = base.sha256_bytes(
        base.canonical_json_bytes(result)
    )
    base.atomic_json(
        output_root / "unseen_identity_generalization_result.json",
        result,
    )

    csv_rows = flattened_summary_rows(summaries)
    if csv_rows:
        parent.atomic_csv(
            output_root / "unseen_identity_generalization_summary.csv",
            parent.summary_csv_fieldnames(csv_rows),
            csv_rows,
        )
    return result


def parse_int_list(value: str) -> Tuple[int, ...]:
    return base.parse_int_list(value)


def parse_choice_list(
    value: str,
    allowed: Sequence[str],
) -> Tuple[str, ...]:
    return base.parse_choice_list(value, allowed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-identities", type=int)
    parser.add_argument("--test-identities", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--model-seeds", type=parse_int_list)
    parser.add_argument("--dataset-seeds", type=parse_int_list)
    parser.add_argument(
        "--renderers",
        type=lambda value: parse_choice_list(value, RENDERERS),
        default=RENDERERS,
    )
    parser.add_argument(
        "--models",
        type=lambda value: parse_choice_list(value, MODEL_FAMILIES),
        default=MODEL_FAMILIES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        config = ExperimentConfig(
            identities=args.train_identities or 40,
            test_identities=args.test_identities or 20,
            epochs=args.epochs or 100,
            bootstrap_replicates=args.bootstrap_replicates or 500,
        )
        dataset_seeds = args.dataset_seeds or (4301, 5301)
        model_seeds = args.model_seeds or (2201, 2202)
    else:
        config = ExperimentConfig(
            identities=args.train_identities or 256,
            test_identities=args.test_identities or 128,
            epochs=args.epochs or 250,
            bootstrap_replicates=args.bootstrap_replicates or 5000,
        )
        dataset_seeds = args.dataset_seeds or DEFAULT_DATASET_SEEDS
        model_seeds = args.model_seeds or DEFAULT_MODEL_SEEDS

    if (
        config.epochs <= 0
        or config.bootstrap_replicates <= 0
        or config.identities <= 0
        or config.test_identities <= 0
    ):
        raise ExperimentError(
            "Epochs, bootstrap replicates, and identity counts must be positive."
        )
    if len(set(dataset_seeds)) != len(dataset_seeds):
        raise ExperimentError("Dataset seeds must be unique.")
    if len(set(model_seeds)) != len(model_seeds):
        raise ExperimentError("Model seeds must be unique.")
    if set(args.models) != set(MODEL_FAMILIES):
        raise ExperimentError(
            "All four model families are required for control validation."
        )

    result = run_experiment(
        config=config,
        dataset_seeds=dataset_seeds,
        model_seeds=model_seeds,
        renderers=args.renderers,
        model_families=args.models,
        output_root=args.output_root.resolve(),
        device=base.resolve_device(args.device),
    )
    print(
        base.json.dumps(
            {
                "status": result["status"],
                "output_root": str(args.output_root.resolve()),
                "result_sha256": result["result_sha256"],
                "run_count": len(result["runs"]),
                "dataset_seed_count": len(result["dataset_seeds"]),
                "unseen_identity_generalization_gate_open": result[
                    "control_validation"
                ]["unseen_identity_generalization_gate_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (
        ExperimentError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        raise SystemExit(
            "UNSEEN-IDENTITY GENERALIZATION EXPERIMENT FAILED: {}".format(
                exc
            )
        ) from exc
