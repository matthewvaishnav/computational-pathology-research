#!/usr/bin/env python3
"""PA-NF v2 development revision 5: canonical biology + bounded acquisition operator.

R5 is an architectural redesign after R4. R4 showed that donor-invariant
acquisition pooling substantially reduced, but did not eliminate, the persistent
candidate-minus-control biology-retention deficit. The remaining direct structural
entanglement path was the unconstrained MLP over concat(z_b, z_a).

R5 replaces that decoder with an operator factorization:

    h_b = C(z_b)
    (gamma, beta) = A(z_a)
    h = (1 + tanh(gamma)) * h_b + tanh(beta)
    x_hat = R(h)

Thus biology supplies canonical content and acquisition can only act through a
bounded feature-wise affine operator before a shared readout. With z_a = 0, the
acquisition operator is exactly the identity by construction. Crossed interventions
retain R4's leave-query-out same-scanner acquisition pooling.

Candidate and control have exactly identical architecture and parameter count; the
control disables only crossed reconstruction and biological/acquisition cycle losses.
This is development-only and does not read frozen SCORPION outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.paired_acquisition import (
    run_pa_nf_v2_crossed_intervention_development as r1,
)
from experiments.paired_acquisition import (
    run_pa_nf_v2_crossed_intervention_development_r2 as r2,
)
from experiments.paired_acquisition import (
    run_pa_nf_v2_crossed_intervention_development_r4 as r4,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as base,
)

SCHEMA_VERSION = "pa-nf-v2-crossed-intervention-development/v5"
MODEL_FAMILIES = (
    "pa_nf_v2_r5_operator_factorization",
    "v2_r5_operator_control_no_cross_cycle",
)
DEFAULT_SMOKE_SEEDS = (3501, 3502, 3503)
ExperimentConfig = r4.ExperimentConfig


class BoundedAcquisitionOperatorDecoder(nn.Module):
    """Decode biology canonically, then let acquisition act only as a bounded operator."""

    def __init__(
        self,
        biological_dim: int,
        acquisition_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        operator_hidden = max(32, hidden_dim // 2)
        self.biological_dim = int(biological_dim)
        self.acquisition_dim = int(acquisition_dim)
        self.hidden_dim = int(hidden_dim)

        self.canonical_biology = nn.Sequential(
            nn.Linear(biological_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        # Bias-free operator network guarantees A(0)=0, hence zero acquisition
        # is exactly the identity operator before the shared readout.
        self.acquisition_operator = nn.Sequential(
            nn.Linear(acquisition_dim, operator_hidden, bias=False),
            nn.GELU(),
            nn.Linear(operator_hidden, 2 * hidden_dim, bias=False),
        )
        self.shared_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, joined: torch.Tensor) -> torch.Tensor:
        if joined.ndim != 2:
            raise r1.ExperimentError("Operator decoder input must be a matrix")
        expected = self.biological_dim + self.acquisition_dim
        if joined.shape[1] != expected:
            raise r1.ExperimentError(
                "Operator decoder expected {} latent columns, got {}".format(
                    expected, joined.shape[1]
                )
            )
        biological = joined[:, : self.biological_dim]
        acquisition = joined[:, self.biological_dim :]
        canonical = self.canonical_biology(biological)
        operator = self.acquisition_operator(acquisition)
        gamma, beta = operator.chunk(2, dim=1)
        modulated = (1.0 + torch.tanh(gamma)) * canonical + torch.tanh(beta)
        return self.shared_readout(modulated)


class OperatorFactorizer(nn.Module):
    """Image-inferred factors with acquisition restricted to an operator on biology."""

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
        self.decoder = BoundedAcquisitionOperatorDecoder(
            biological_dim=biological_dim,
            acquisition_dim=acquisition_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
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


def build_model(config: ExperimentConfig, device: torch.device) -> OperatorFactorizer:
    return OperatorFactorizer(
        input_dim=config.observation_dim,
        biological_dim=config.biological_dim,
        acquisition_dim=config.acquisition_dim,
        hidden_dim=config.hidden_dim,
        scanners=config.scanners,
    ).to(device)


def family_weights(
    model_family: str, config: ExperimentConfig
) -> Tuple[float, float, float]:
    if model_family == MODEL_FAMILIES[0]:
        return (
            config.crossed_reconstruction_weight,
            config.biological_cycle_weight,
            config.acquisition_cycle_weight,
        )
    if model_family == MODEL_FAMILIES[1]:
        return 0.0, 0.0, 0.0
    raise r1.ExperimentError("Unknown model family: {}".format(model_family))


def train_model(
    model_family: str,
    model: OperatorFactorizer,
    dataset: base.SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    scanner_ids = torch.as_tensor(dataset.scanner_ids, dtype=torch.long, device=device)
    train = torch.as_tensor(dataset.train_indices, dtype=torch.long, device=device)
    source_np, target_np = r1.build_crossed_pairs(dataset)
    consistency_left_np, consistency_right_np = r1.build_biological_consistency_pairs(dataset)
    source = torch.as_tensor(source_np, dtype=torch.long, device=device)
    target = torch.as_tensor(target_np, dtype=torch.long, device=device)
    consistency_left = torch.as_tensor(consistency_left_np, dtype=torch.long, device=device)
    consistency_right = torch.as_tensor(consistency_right_np, dtype=torch.long, device=device)
    crossed_weight, bio_cycle_weight, acq_cycle_weight = family_weights(model_family, config)

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
        train_scanners = scanner_ids.index_select(0, train)
        train_inputs = observations.index_select(0, train)

        self_prediction = model.decode(train_biological, train_acquisition)
        self_reconstruction = F.mse_loss(self_prediction, train_inputs)

        pooled_target_acquisition = r4.pooled_acquisition_for_queries(
            all_acquisition,
            scanner_ids,
            train,
            target,
            config.scanners,
        )
        crossed_prediction = model.decode(
            all_biological.index_select(0, source), pooled_target_acquisition
        )
        crossed_reconstruction = F.mse_loss(
            crossed_prediction, observations.index_select(0, target)
        )

        cycle_biological = model.encode_biological(crossed_prediction)
        cycle_acquisition = model.encode_acquisition(crossed_prediction)
        biological_cycle_loss = F.mse_loss(
            cycle_biological,
            all_biological.index_select(0, source).detach(),
        )
        acquisition_cycle_loss = F.mse_loss(
            cycle_acquisition, pooled_target_acquisition.detach()
        )

        biological_consistency = F.mse_loss(
            all_biological.index_select(0, consistency_left),
            all_biological.index_select(0, consistency_right),
        )
        acquisition_same_scanner = r2.same_scanner_acquisition_consistency(
            train_acquisition, train_scanners, config.scanners
        )
        prototype_targets = model.scanner_prototypes(train_scanners)
        acquisition_prototype = F.mse_loss(train_acquisition, prototype_targets)
        variance_floor = r1.biological_variance_floor(train_biological)
        prototype_center, prototype_separation = r1.prototype_regularization(
            model.scanner_prototypes.weight
        )

        loss = (
            config.self_reconstruction_weight * self_reconstruction
            + crossed_weight * crossed_reconstruction
            + bio_cycle_weight * biological_cycle_loss
            + acq_cycle_weight * acquisition_cycle_loss
            + config.biological_consistency_weight * biological_consistency
            + config.acquisition_same_scanner_consistency_weight * acquisition_same_scanner
            + config.acquisition_prototype_weight * acquisition_prototype
            + config.biological_variance_weight * variance_floor
            + config.prototype_center_weight * prototype_center
            + config.prototype_separation_weight * prototype_separation
        )
        if not torch.isfinite(loss):
            raise r1.ExperimentError(
                "Non-finite R5 loss for {} at epoch {}".format(model_family, epoch + 1)
            )
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise r1.ExperimentError(
                    "Non-finite R5 gradient for {} at epoch {}".format(
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
                    "pooled_crossed_reconstruction": float(crossed_reconstruction.detach().cpu()),
                    "biological_cycle": float(biological_cycle_loss.detach().cpu()),
                    "acquisition_cycle": float(acquisition_cycle_loss.detach().cpu()),
                    "biological_consistency": float(biological_consistency.detach().cpu()),
                    "acquisition_same_scanner_consistency": float(
                        acquisition_same_scanner.detach().cpu()
                    ),
                    "crossed_weight": float(crossed_weight),
                    "biological_cycle_weight": float(bio_cycle_weight),
                    "acquisition_cycle_weight": float(acq_cycle_weight),
                }
            )

    return {
        "epochs": int(config.epochs),
        "optimizer_steps": int(config.epochs),
        "crossed_pair_count": int(len(source_np)),
        "biological_consistency_pair_count": int(len(consistency_left_np)),
        "crossed_acquisition_source": "leave-query-out_same-scanner_training_pool",
        "decoder_factorization": "canonical_biology_plus_bounded_acquisition_operator",
        "history": history,
    }


def pooled_cycle_diagnostics(
    model: OperatorFactorizer,
    dataset: base.SyntheticDataset,
    device: torch.device,
) -> Dict[str, float]:
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    scanner_ids = torch.as_tensor(dataset.scanner_ids, dtype=torch.long, device=device)
    train = torch.as_tensor(dataset.train_indices, dtype=torch.long, device=device)
    source = torch.as_tensor(dataset.source_index_by_identity, dtype=torch.long, device=device)
    target = torch.as_tensor(dataset.test_indices, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        biological = model.encode_biological(observations)
        acquisition = model.encode_acquisition(observations)
        biological_target = biological.index_select(0, source)
        pooled_target = r4.pooled_acquisition_for_queries(
            acquisition, scanner_ids, train, target, int(model.scanners)
        )
        crossed = model.decode(biological_target, pooled_target)
        re_biological = model.encode_biological(crossed)
        re_acquisition = model.encode_acquisition(crossed)
        bio_cycle = F.mse_loss(re_biological, biological_target)
        acq_cycle = F.mse_loss(re_acquisition, pooled_target)
        bio_scale = biological_target.var(dim=0, unbiased=False).mean().clamp_min(1e-6)
        acq_scale = pooled_target.var(dim=0, unbiased=False).mean().clamp_min(1e-6)
        normalized_bio = bio_cycle / bio_scale
        normalized_acq = acq_cycle / acq_scale
        crossed_target_mse = F.mse_loss(crossed, observations.index_select(0, target))
    return {
        "cycle_biological_mse": float(bio_cycle.cpu()),
        "cycle_acquisition_mse": float(acq_cycle.cpu()),
        "cycle_total_mse": float((bio_cycle + acq_cycle).cpu()),
        "cycle_biological_variance_normalized": float(normalized_bio.cpu()),
        "cycle_acquisition_variance_normalized": float(normalized_acq.cpu()),
        "cycle_variance_normalized_total": float((normalized_bio + normalized_acq).cpu()),
        "crossed_target_mse": float(crossed_target_mse.cpu()),
    }


def evaluate_model(
    model_family: str,
    model: OperatorFactorizer,
    dataset: base.SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    # R4's evaluator already recomputes the crossed metrics with donor-invariant
    # pooling. It is architecture-agnostic as long as the model exposes the same
    # encode/decode interface and a decoder callable on concatenated latents.
    result = r4.evaluate_model(model_family, model, dataset, config, device, seed)
    result["metrics"].update(pooled_cycle_diagnostics(model, dataset, device))
    result["metrics"]["parameter_count"] = r1.parameter_count(model)
    result["r5_architecture"] = {
        "decoder": "canonical_biology_plus_bounded_acquisition_operator",
        "zero_acquisition_is_identity_operator": True,
        "individual_donor_code_used": False,
    }
    return result


def summarize_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    paired_cycle: List[float] = []
    paired_transfer: List[float] = []
    paired_biology: List[float] = []
    equal_parameters = True
    candidate_all_success = True
    by_renderer: Dict[str, Any] = {}

    for renderer in base.RENDERERS:
        renderer_runs = [run for run in runs if run["renderer"] == renderer]
        by_key = {
            (run["model_family"], int(run["seed"])): run for run in renderer_runs
        }
        seeds = sorted({int(run["seed"]) for run in renderer_runs})
        cb: List[float] = []
        xb: List[float] = []
        ct: List[float] = []
        xt: List[float] = []
        for seed in seeds:
            candidate = by_key[(MODEL_FAMILIES[0], seed)]
            control = by_key[(MODEL_FAMILIES[1], seed)]
            cm = candidate["evaluation"]["metrics"]
            xm = control["evaluation"]["metrics"]
            equal_parameters = equal_parameters and (
                int(cm["parameter_count"]) == int(xm["parameter_count"])
            )
            candidate_all_success = candidate_all_success and bool(
                candidate["evaluation"]["gates"]["crossed_factorization_success"]
            )
            paired_cycle.append(
                float(xm["cycle_variance_normalized_total"])
                - float(cm["cycle_variance_normalized_total"])
            )
            paired_transfer.append(
                float(cm["acquisition_transfer_delta"])
                - float(xm["acquisition_transfer_delta"])
            )
            paired_biology.append(
                float(cm["biology_retention_delta"])
                - float(xm["biology_retention_delta"])
            )
            cb.append(float(cm["biology_retention_delta"]))
            xb.append(float(xm["biology_retention_delta"]))
            ct.append(float(cm["acquisition_transfer_delta"]))
            xt.append(float(xm["acquisition_transfer_delta"]))
        by_renderer[renderer] = {
            "seed_count": len(seeds),
            "candidate_mean_biology_retention_delta": float(np.mean(cb)),
            "control_mean_biology_retention_delta": float(np.mean(xb)),
            "candidate_mean_acquisition_transfer_delta": float(np.mean(ct)),
            "control_mean_acquisition_transfer_delta": float(np.mean(xt)),
        }

    gate = {
        "parameter_counts_equal": bool(equal_parameters),
        "candidate_all_seed_crossed_factorization_success": bool(candidate_all_success),
        "candidate_mean_acquisition_transfer_delta_greater_than_control": bool(
            np.mean(paired_transfer) > 0
        ),
        "candidate_mean_biology_retention_delta_greater_than_control": bool(
            np.mean(paired_biology) > 0
        ),
        "candidate_mean_variance_normalized_cycle_error_lower_than_control": bool(
            np.mean(paired_cycle) > 0
        ),
        "paired_mean_candidate_minus_control_acquisition_transfer_delta": float(
            np.mean(paired_transfer)
        ),
        "paired_mean_candidate_minus_control_biology_retention_delta": float(
            np.mean(paired_biology)
        ),
        "paired_mean_control_minus_candidate_variance_normalized_cycle_error": float(
            np.mean(paired_cycle)
        ),
    }
    required = [
        "parameter_counts_equal",
        "candidate_all_seed_crossed_factorization_success",
        "candidate_mean_acquisition_transfer_delta_greater_than_control",
        "candidate_mean_biology_retention_delta_greater_than_control",
        "candidate_mean_variance_normalized_cycle_error_lower_than_control",
    ]
    gate["development_promotion_pass"] = all(bool(gate[key]) for key in required)
    return {"by_renderer": by_renderer, "promotion_gate": gate}


def run_experiment(
    config: ExperimentConfig,
    model_seeds: Sequence[int],
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if output_root.exists():
        raise r1.ExperimentError(
            "Output root already exists; overwrite prohibited: {}".format(output_root)
        )
    output_root.mkdir(parents=True, exist_ok=False)
    datasets = {
        renderer: base.make_synthetic_dataset(r1.to_base_config(config), renderer)
        for renderer in base.RENDERERS
    }
    base.atomic_json(
        output_root / "dataset_manifest.json",
        {
            renderer: {
                "observation_shape": list(dataset.observations.shape),
                "train_count": int(len(dataset.train_indices)),
                "test_count": int(len(dataset.test_indices)),
                "renderer_metadata": dict(dataset.renderer_metadata),
            }
            for renderer, dataset in datasets.items()
        },
    )

    runs: List[Dict[str, Any]] = []
    for renderer, dataset in datasets.items():
        for seed in model_seeds:
            for model_family in MODEL_FAMILIES:
                print("[{}] model={} seed={}".format(renderer, model_family, seed), flush=True)
                base.set_deterministic_seed(int(seed))
                model = build_model(config, device)
                training = train_model(model_family, model, dataset, config, device)
                evaluation = evaluate_model(
                    model_family, model, dataset, config, device, int(seed)
                )
                runs.append(
                    {
                        "renderer": renderer,
                        "model_family": model_family,
                        "seed": int(seed),
                        "parameter_count": r1.parameter_count(model),
                        "training": training,
                        "evaluation": evaluation,
                    }
                )

    summary = summarize_runs(runs)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_only_not_confirmation",
        "development_iteration": 5,
        "config": asdict(config),
        "model_seeds": [int(seed) for seed in model_seeds],
        "model_families": list(MODEL_FAMILIES),
        "architecture": {
            "name": "canonical_biology_plus_bounded_acquisition_operator",
            "zero_acquisition_is_identity_operator": True,
            "crossed_acquisition_source": "leave-query-out_same-scanner_training_pool",
            "individual_donor_acquisition_code_used": False,
        },
        "development_isolation": {
            "frozen_2026_09_29_scorpion_outcomes_used_for_tuning": False,
            "r1_r2_r3_r4_synthetic_development_used": True,
            "confirmation_claim_allowed": False,
        },
        "runs": runs,
        "summary": summary,
        "claim_boundary": (
            "R5 may determine whether an operator-structured decoder resolves the synthetic "
            "development tradeoff. It is not real-pathology confirmation."
        ),
    }
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    base.atomic_json(output_root / "pa_nf_v2_development_r5_result.json", result)
    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))
    print(
        "DEVELOPMENT R5 PROMOTION PASS: {}".format(
            summary["promotion_gate"]["development_promotion_pass"]
        )
    )
    print("Artifacts: {}".format(output_root.resolve()))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/pa_nf_v2_crossed_intervention_development_r5_smoke"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        ExperimentConfig(),
        identities=64,
        epochs=80,
        bootstrap_replicates=1000,
    )
    run_experiment(config, DEFAULT_SMOKE_SEEDS, args.output_root, torch.device(args.device))


if __name__ == "__main__":
    try:
        main()
    except (r1.ExperimentError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit("PA-NF V2 DEVELOPMENT R5 FAILED: {}".format(exc)) from exc
