#!/usr/bin/env python3
"""Development-only PA-NF v2 crossed-intervention factorization.

This runner deliberately uses synthetic known-factor data only. It does not read
or optimize against the frozen 2026-09-29 SCORPION shortcut/conflict outcomes.

The primary causal comparison uses identical-capacity models:

* pa_nf_v2_crossed_intervention: crossed reconstruction + re-encoding cycle.
* v2_control_no_cross_cycle: exact same architecture, but those two weights are 0.

Scanner identity is never required at inference. Learned scanner prototypes are
training-only anchors for the image-inferred acquisition representation.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.paired_acquisition import (  # noqa: E402
    run_synthetic_crossed_factor_identifiability as base,
)
from experiments.paired_acquisition import (  # noqa: E402
    run_synthetic_crossed_factor_identifiability_v2 as eval_v2,
)


SCHEMA_VERSION = "pa-nf-v2-crossed-intervention-development/v1"
MODEL_FAMILIES = (
    "pa_nf_v2_crossed_intervention",
    "v2_control_no_cross_cycle",
)
DEFAULT_FULL_SEEDS = tuple(range(3101, 3111))
DEFAULT_SMOKE_SEEDS = (3101, 3102, 3103)


class ExperimentError(base.ExperimentError):
    """Raised when the development experiment cannot proceed safely."""


@dataclass(frozen=True)
class ExperimentConfig:
    identities: int = 256
    scanners: int = 5
    biological_latent_dim: int = 8
    acquisition_latent_dim: int = 4
    observation_dim: int = 64
    nonlinear_hidden_dim: int = 128
    biological_dim: int = 64
    acquisition_dim: int = 16
    hidden_dim: int = 256
    noise_std: float = 0.01
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 250
    bootstrap_replicates: int = 5000
    dataset_seed: int = 5317
    self_reconstruction_weight: float = 1.0
    crossed_reconstruction_weight: float = 1.0
    cycle_weight: float = 1.0
    biological_consistency_weight: float = 1.0
    acquisition_prototype_weight: float = 0.25
    biological_variance_weight: float = 0.05
    prototype_center_weight: float = 0.01
    prototype_separation_weight: float = 0.01


def to_base_config(config: ExperimentConfig) -> base.ExperimentConfig:
    return base.ExperimentConfig(
        identities=config.identities,
        scanners=config.scanners,
        biological_latent_dim=config.biological_latent_dim,
        acquisition_latent_dim=config.acquisition_latent_dim,
        observation_dim=config.observation_dim,
        nonlinear_hidden_dim=config.nonlinear_hidden_dim,
        pa_nf_biological_dim=config.biological_dim,
        pa_nf_acquisition_dim=config.acquisition_dim,
        pa_nf_hidden_dim=config.hidden_dim,
        noise_std=config.noise_std,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        bootstrap_replicates=config.bootstrap_replicates,
        dataset_seed=config.dataset_seed,
    )


class CrossedInterventionFactorizer(nn.Module):
    """Image-inferred biological/acquisition codes with a compositional decoder."""

    def __init__(
        self,
        input_dim: int,
        biological_dim: int,
        acquisition_dim: int,
        hidden_dim: int,
        scanners: int,
    ) -> None:
        super().__init__()
        self.biological_dim = int(biological_dim)
        self.acquisition_dim = int(acquisition_dim)
        self.scanners = int(scanners)

        self.biological_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, biological_dim),
        )
        self.acquisition_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, acquisition_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(biological_dim + acquisition_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.scanner_prototypes = nn.Embedding(scanners, acquisition_dim)
        nn.init.normal_(self.scanner_prototypes.weight, mean=0.0, std=0.1)

    def encode_biological(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.biological_encoder(inputs)

    def encode_acquisition(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.acquisition_encoder(inputs)

    def decode(self, biological: torch.Tensor, acquisition: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([biological, acquisition], dim=1))

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        biological = self.encode_biological(inputs)
        acquisition = self.encode_acquisition(inputs)
        reconstruction = self.decode(biological, acquisition)
        return {
            "biological": biological,
            "acquisition": acquisition,
            "reconstruction": reconstruction,
        }


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def build_model(config: ExperimentConfig, device: torch.device) -> CrossedInterventionFactorizer:
    return CrossedInterventionFactorizer(
        input_dim=config.observation_dim,
        biological_dim=config.biological_dim,
        acquisition_dim=config.acquisition_dim,
        hidden_dim=config.hidden_dim,
        scanners=config.scanners,
    ).to(device)


def build_crossed_pairs(dataset: base.SyntheticDataset) -> Tuple[np.ndarray, np.ndarray]:
    """All ordered observed pairs with same identity and different scanners."""
    train_set = set(int(index) for index in dataset.train_indices.tolist())
    source: List[int] = []
    target: List[int] = []
    identities = int(dataset.heldout_scanner_by_identity.shape[0])
    for identity in range(identities):
        indices = [
            int(index)
            for index in dataset.train_indices[
                dataset.identity_ids[dataset.train_indices] == identity
            ].tolist()
        ]
        indices.sort(key=lambda index: int(dataset.scanner_ids[index]))
        if len(indices) < 2:
            raise ExperimentError("Every identity requires at least two observed scanner views.")
        for left in indices:
            for right in indices:
                if left == right:
                    continue
                if left not in train_set or right not in train_set:
                    raise ExperimentError("Crossed pair leaked held-out data.")
                if dataset.identity_ids[left] != dataset.identity_ids[right]:
                    raise ExperimentError("Crossed pair changed biological identity.")
                if dataset.scanner_ids[left] == dataset.scanner_ids[right]:
                    raise ExperimentError("Crossed pair did not change scanner.")
                source.append(left)
                target.append(right)
    return np.asarray(source, dtype=np.int64), np.asarray(target, dtype=np.int64)


def build_biological_consistency_pairs(
    dataset: base.SyntheticDataset,
) -> Tuple[np.ndarray, np.ndarray]:
    left: List[int] = []
    right: List[int] = []
    identities = int(dataset.heldout_scanner_by_identity.shape[0])
    for identity in range(identities):
        indices = [
            int(index)
            for index in dataset.train_indices[
                dataset.identity_ids[dataset.train_indices] == identity
            ].tolist()
        ]
        indices.sort(key=lambda index: int(dataset.scanner_ids[index]))
        for offset, first in enumerate(indices):
            for second in indices[offset + 1 :]:
                left.append(first)
                right.append(second)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def biological_variance_floor(biological: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(biological.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - std).mean()


def prototype_regularization(prototypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    center_penalty = prototypes.mean(dim=0).square().mean()
    centered = prototypes - prototypes.mean(dim=0, keepdim=True)
    normalized = F.normalize(centered, dim=1)
    similarity = normalized @ normalized.T
    mask = ~torch.eye(
        prototypes.shape[0], dtype=torch.bool, device=prototypes.device
    )
    separation_penalty = similarity[mask].square().mean()
    return center_penalty, separation_penalty


def family_weights(model_family: str, config: ExperimentConfig) -> Tuple[float, float]:
    if model_family == "pa_nf_v2_crossed_intervention":
        return config.crossed_reconstruction_weight, config.cycle_weight
    if model_family == "v2_control_no_cross_cycle":
        return 0.0, 0.0
    raise ExperimentError("Unknown model family: {}".format(model_family))


def train_model(
    model_family: str,
    model: CrossedInterventionFactorizer,
    dataset: base.SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    scanner_ids = torch.as_tensor(dataset.scanner_ids, dtype=torch.long, device=device)
    train = torch.as_tensor(dataset.train_indices, dtype=torch.long, device=device)
    source_np, target_np = build_crossed_pairs(dataset)
    consistency_left_np, consistency_right_np = build_biological_consistency_pairs(dataset)
    source = torch.as_tensor(source_np, dtype=torch.long, device=device)
    target = torch.as_tensor(target_np, dtype=torch.long, device=device)
    consistency_left = torch.as_tensor(consistency_left_np, dtype=torch.long, device=device)
    consistency_right = torch.as_tensor(consistency_right_np, dtype=torch.long, device=device)
    crossed_weight, cycle_weight = family_weights(model_family, config)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        all_biological = model.encode_biological(observations)
        all_acquisition = model.encode_acquisition(observations)

        train_biological = all_biological.index_select(0, train)
        train_acquisition = all_acquisition.index_select(0, train)
        train_inputs = observations.index_select(0, train)
        self_prediction = model.decode(train_biological, train_acquisition)
        self_reconstruction = F.mse_loss(self_prediction, train_inputs)

        crossed_prediction = model.decode(
            all_biological.index_select(0, source),
            all_acquisition.index_select(0, target),
        )
        crossed_reconstruction = F.mse_loss(
            crossed_prediction, observations.index_select(0, target)
        )

        cycle_biological = model.encode_biological(crossed_prediction)
        cycle_acquisition = model.encode_acquisition(crossed_prediction)
        cycle_biological_loss = F.mse_loss(
            cycle_biological,
            all_biological.index_select(0, source).detach(),
        )
        cycle_acquisition_loss = F.mse_loss(
            cycle_acquisition,
            all_acquisition.index_select(0, target).detach(),
        )
        cycle_loss = cycle_biological_loss + cycle_acquisition_loss

        biological_consistency = F.mse_loss(
            all_biological.index_select(0, consistency_left),
            all_biological.index_select(0, consistency_right),
        )
        prototype_targets = model.scanner_prototypes(
            scanner_ids.index_select(0, train)
        )
        acquisition_prototype = F.mse_loss(train_acquisition, prototype_targets)
        variance_floor = biological_variance_floor(train_biological)
        prototype_center, prototype_separation = prototype_regularization(
            model.scanner_prototypes.weight
        )

        loss = (
            config.self_reconstruction_weight * self_reconstruction
            + crossed_weight * crossed_reconstruction
            + cycle_weight * cycle_loss
            + config.biological_consistency_weight * biological_consistency
            + config.acquisition_prototype_weight * acquisition_prototype
            + config.biological_variance_weight * variance_floor
            + config.prototype_center_weight * prototype_center
            + config.prototype_separation_weight * prototype_separation
        )
        if not torch.isfinite(loss):
            raise ExperimentError(
                "Non-finite loss for {} at epoch {}".format(model_family, epoch + 1)
            )
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ExperimentError(
                    "Non-finite gradient for {} at epoch {}".format(
                        model_family, epoch + 1
                    )
                )
        optimizer.step()

        if epoch in {0, config.epochs - 1} or (epoch + 1) % max(1, config.epochs // 10) == 0:
            history.append(
                {
                    "epoch": epoch + 1,
                    "total": float(loss.detach().cpu()),
                    "self_reconstruction": float(self_reconstruction.detach().cpu()),
                    "crossed_reconstruction": float(crossed_reconstruction.detach().cpu()),
                    "cycle": float(cycle_loss.detach().cpu()),
                    "biological_consistency": float(biological_consistency.detach().cpu()),
                    "acquisition_prototype": float(acquisition_prototype.detach().cpu()),
                    "biological_variance_floor": float(variance_floor.detach().cpu()),
                    "crossed_weight": float(crossed_weight),
                    "cycle_weight": float(cycle_weight),
                }
            )

    return {
        "epochs": int(config.epochs),
        "optimizer_steps": int(config.epochs),
        "crossed_pair_count": int(len(source_np)),
        "biological_consistency_pair_count": int(len(consistency_left_np)),
        "history": history,
    }


def cycle_diagnostics(
    model: CrossedInterventionFactorizer,
    dataset: base.SyntheticDataset,
    device: torch.device,
) -> Dict[str, float]:
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    source = torch.as_tensor(dataset.source_index_by_identity, dtype=torch.long, device=device)
    donor = torch.as_tensor(dataset.donor_index_by_identity, dtype=torch.long, device=device)
    target = torch.as_tensor(dataset.test_indices, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        biological = model.encode_biological(observations)
        acquisition = model.encode_acquisition(observations)
        crossed = model.decode(
            biological.index_select(0, source),
            acquisition.index_select(0, donor),
        )
        re_biological = model.encode_biological(crossed)
        re_acquisition = model.encode_acquisition(crossed)
        bio_cycle = F.mse_loss(
            re_biological, biological.index_select(0, source)
        )
        acq_cycle = F.mse_loss(
            re_acquisition, acquisition.index_select(0, donor)
        )
        crossed_target_mse = F.mse_loss(crossed, observations.index_select(0, target))
    return {
        "cycle_biological_mse": float(bio_cycle.cpu()),
        "cycle_acquisition_mse": float(acq_cycle.cpu()),
        "cycle_total_mse": float((bio_cycle + acq_cycle).cpu()),
        "crossed_target_mse": float(crossed_target_mse.cpu()),
    }


def evaluate_model(
    model_family: str,
    model: CrossedInterventionFactorizer,
    dataset: base.SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    result = eval_v2.evaluate_model_v2(
        model_family,
        model,
        dataset,
        to_base_config(config),
        device,
        seed,
    )
    result["metrics"].update(cycle_diagnostics(model, dataset, device))
    result["metrics"]["parameter_count"] = parameter_count(model)
    return result


def summarize_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"by_renderer": {}, "promotion_gate": {}}
    required_metrics = (
        "biology_retention_delta",
        "acquisition_transfer_delta",
        "cycle_total_mse",
        "biological_scanner_balanced_accuracy",
        "acquisition_scanner_balanced_accuracy",
        "biological_identity_retrieval_top1",
    )
    paired_cycle_differences: List[float] = []
    paired_transfer_differences: List[float] = []
    paired_biology_differences: List[float] = []
    equal_parameters = True
    candidate_all_success = True

    for renderer in base.RENDERERS:
        renderer_runs = [run for run in runs if run["renderer"] == renderer]
        family_summary: Dict[str, Any] = {}
        by_family_seed = {
            (run["model_family"], int(run["seed"])): run for run in renderer_runs
        }
        for family in MODEL_FAMILIES:
            family_runs = [run for run in renderer_runs if run["model_family"] == family]
            metric_summary: Dict[str, Any] = {}
            for metric in required_metrics:
                values = np.asarray(
                    [float(run["evaluation"]["metrics"][metric]) for run in family_runs],
                    dtype=np.float64,
                )
                metric_summary[metric] = {
                    "mean": float(values.mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            family_summary[family] = {
                "seed_count": len(family_runs),
                "metrics": metric_summary,
                "all_seed_crossed_factorization_success": all(
                    bool(run["evaluation"]["gates"]["crossed_factorization_success"])
                    for run in family_runs
                ),
            }
            if family == "pa_nf_v2_crossed_intervention":
                candidate_all_success = candidate_all_success and family_summary[family][
                    "all_seed_crossed_factorization_success"
                ]

        seeds = sorted({int(run["seed"]) for run in renderer_runs})
        for seed in seeds:
            candidate = by_family_seed[("pa_nf_v2_crossed_intervention", seed)]
            control = by_family_seed[("v2_control_no_cross_cycle", seed)]
            candidate_metrics = candidate["evaluation"]["metrics"]
            control_metrics = control["evaluation"]["metrics"]
            equal_parameters = equal_parameters and (
                int(candidate_metrics["parameter_count"])
                == int(control_metrics["parameter_count"])
            )
            paired_cycle_differences.append(
                float(control_metrics["cycle_total_mse"])
                - float(candidate_metrics["cycle_total_mse"])
            )
            paired_transfer_differences.append(
                float(candidate_metrics["acquisition_transfer_delta"])
                - float(control_metrics["acquisition_transfer_delta"])
            )
            paired_biology_differences.append(
                float(candidate_metrics["biology_retention_delta"])
                - float(control_metrics["biology_retention_delta"])
            )
        summary["by_renderer"][renderer] = family_summary

    summary["promotion_gate"] = {
        "parameter_counts_equal": bool(equal_parameters),
        "candidate_all_seed_crossed_factorization_success": bool(candidate_all_success),
        "candidate_mean_acquisition_transfer_delta_greater_than_control": bool(
            np.mean(paired_transfer_differences) > 0
        ),
        "candidate_mean_biology_retention_delta_greater_than_control": bool(
            np.mean(paired_biology_differences) > 0
        ),
        "candidate_mean_cycle_error_lower_than_control": bool(
            np.mean(paired_cycle_differences) > 0
        ),
        "paired_mean_candidate_minus_control_acquisition_transfer_delta": float(
            np.mean(paired_transfer_differences)
        ),
        "paired_mean_candidate_minus_control_biology_retention_delta": float(
            np.mean(paired_biology_differences)
        ),
        "paired_mean_control_minus_candidate_cycle_total_mse": float(
            np.mean(paired_cycle_differences)
        ),
    }
    gate_keys = [
        "parameter_counts_equal",
        "candidate_all_seed_crossed_factorization_success",
        "candidate_mean_acquisition_transfer_delta_greater_than_control",
        "candidate_mean_biology_retention_delta_greater_than_control",
        "candidate_mean_cycle_error_lower_than_control",
    ]
    summary["promotion_gate"]["development_promotion_pass"] = all(
        bool(summary["promotion_gate"][key]) for key in gate_keys
    )
    return summary


def run_experiment(
    config: ExperimentConfig,
    model_seeds: Sequence[int],
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if output_root.exists():
        raise ExperimentError("Output root already exists; overwrite prohibited: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=False)

    datasets = {
        renderer: base.make_synthetic_dataset(to_base_config(config), renderer)
        for renderer in base.RENDERERS
    }
    manifest = {
        renderer: {
            "observation_shape": list(dataset.observations.shape),
            "train_count": int(len(dataset.train_indices)),
            "test_count": int(len(dataset.test_indices)),
            "renderer_metadata": dict(dataset.renderer_metadata),
        }
        for renderer, dataset in datasets.items()
    }
    base.atomic_json(output_root / "dataset_manifest.json", manifest)

    runs: List[Dict[str, Any]] = []
    for renderer, dataset in datasets.items():
        for seed in model_seeds:
            for model_family in MODEL_FAMILIES:
                print(
                    "[{}] model={} seed={}".format(renderer, model_family, seed),
                    flush=True,
                )
                base.set_deterministic_seed(int(seed))
                model = build_model(config, device)
                training = train_model(
                    model_family, model, dataset, config, device
                )
                evaluation = evaluate_model(
                    model_family, model, dataset, config, device, int(seed)
                )
                runs.append(
                    {
                        "renderer": renderer,
                        "model_family": model_family,
                        "seed": int(seed),
                        "parameter_count": parameter_count(model),
                        "training": training,
                        "evaluation": evaluation,
                    }
                )

    summary = summarize_runs(runs)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_only_not_confirmation",
        "config": asdict(config),
        "model_seeds": [int(seed) for seed in model_seeds],
        "model_families": list(MODEL_FAMILIES),
        "development_isolation": {
            "frozen_2026_09_29_scorpion_outcomes_used_for_tuning": False,
            "confirmation_claim_allowed": False,
        },
        "runs": runs,
        "summary": summary,
        "claim_boundary": (
            "Development diagnostics may select whether PA-NF v2 is worth freezing. "
            "They do not establish real-pathology or unseen-scanner superiority."
        ),
    }
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    base.atomic_json(output_root / "pa_nf_v2_development_result.json", result)
    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))
    print(
        "DEVELOPMENT PROMOTION PASS: {}".format(
            summary["promotion_gate"]["development_promotion_pass"]
        )
    )
    print("Artifacts: {}".format(output_root.resolve()))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/pa_nf_v2_crossed_intervention_development_smoke"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = ExperimentConfig()
    seeds: Sequence[int] = DEFAULT_FULL_SEEDS
    output_root = args.output_root
    if args.mode == "smoke":
        config = replace(
            config,
            identities=64,
            epochs=80,
            bootstrap_replicates=1000,
        )
        seeds = DEFAULT_SMOKE_SEEDS
    run_experiment(config, seeds, output_root, device)


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit("PA-NF V2 DEVELOPMENT FAILED: {}".format(exc)) from exc
