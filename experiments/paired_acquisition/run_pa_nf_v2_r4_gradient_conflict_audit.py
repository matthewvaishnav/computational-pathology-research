#!/usr/bin/env python3
"""Post-hoc R4 gradient-conflict audit performed before inspecting R5 outcomes.

This audit asks whether the R4 candidate-specific crossed/cycle objective
systematically points the biological encoder away from a differentiable surrogate
for recovery of the known synthetic biological latent.

It reproduces R4 candidate training on the already-exposed R4 smoke seeds and
records gradient cosines at frozen checkpoints. It does not alter R4, does not
read any R5 outcomes, and cannot promote either model.
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

SCHEMA_VERSION = "pa-nf-v2-r4-gradient-conflict-audit/v1"
MODEL_FAMILY = "pa_nf_v2_crossed_intervention_r4_pooled_acquisition"
DEFAULT_SEEDS = (3401, 3402, 3403)
CHECKPOINTS = (1, 10, 20, 40, 80)
RIDGE_ALPHA = 1e-3


class AuditError(r1.ExperimentError):
    """Raised when the diagnostic cannot be interpreted safely."""


def _identity_split_indices(
    dataset: base.SyntheticDataset,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    train = np.asarray(dataset.train_indices, dtype=np.int64)
    train_identities = np.asarray(dataset.identity_ids[train], dtype=np.int64)
    calibration = train[(train_identities % 2) == 0]
    diagnostic = train[(train_identities % 2) == 1]
    if calibration.size == 0 or diagnostic.size == 0:
        raise AuditError("Calibration/diagnostic identity split is empty")
    if set(dataset.identity_ids[calibration].tolist()) & set(
        dataset.identity_ids[diagnostic].tolist()
    ):
        raise AuditError("Calibration and diagnostic identity sets overlap")
    return (
        torch.as_tensor(calibration, dtype=torch.long, device=device),
        torch.as_tensor(diagnostic, dtype=torch.long, device=device),
    )


def _fit_detached_affine_ridge_probe(
    biological: torch.Tensor,
    biological_truth: torch.Tensor,
    calibration_indices: torch.Tensor,
    alpha: float = RIDGE_ALPHA,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit z_b -> ground-truth biology on detached calibration identities."""
    with torch.no_grad():
        z = biological.index_select(0, calibration_indices).detach()
        y = biological_truth.index_select(0, calibration_indices).detach()
        mean = z.mean(dim=0, keepdim=True)
        std = z.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        standardized = (z - mean) / std
        ones = torch.ones(
            standardized.shape[0], 1, dtype=standardized.dtype, device=standardized.device
        )
        design = torch.cat([standardized, ones], dim=1)
        gram = design.T @ design
        penalty = torch.eye(
            gram.shape[0], dtype=gram.dtype, device=gram.device
        ) * float(alpha)
        penalty[-1, -1] = 0.0
        rhs = design.T @ y
        weights = torch.linalg.solve(gram + penalty, rhs)
    return mean.detach(), std.detach(), weights.detach()


def _retention_surrogate(
    biological: torch.Tensor,
    biological_truth: torch.Tensor,
    diagnostic_indices: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    z = biological.index_select(0, diagnostic_indices)
    y = biological_truth.index_select(0, diagnostic_indices)
    standardized = (z - mean) / std
    ones = torch.ones(
        standardized.shape[0], 1, dtype=standardized.dtype, device=standardized.device
    )
    design = torch.cat([standardized, ones], dim=1)
    prediction = design @ weights
    return F.mse_loss(prediction, y)


def _flatten_gradients(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    pieces: List[torch.Tensor] = []
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None:
            pieces.append(torch.zeros_like(parameter).reshape(-1))
        else:
            pieces.append(gradient.reshape(-1))
    if not pieces:
        raise AuditError("No biological-encoder parameters found")
    return torch.cat(pieces)


def _gradient_cosine(
    left_loss: torch.Tensor,
    right_loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> Dict[str, float]:
    left_grad = torch.autograd.grad(
        left_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    right_grad = torch.autograd.grad(
        right_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    left = _flatten_gradients(left_grad, parameters)
    right = _flatten_gradients(right_grad, parameters)
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.detach().cpu()) < 1e-12 or float(right_norm.detach().cpu()) < 1e-12:
        raise AuditError("Gradient norm too small for cosine interpretation")
    cosine = torch.dot(left, right) / (left_norm * right_norm)
    return {
        "cosine": float(cosine.detach().cpu()),
        "left_norm": float(left_norm.detach().cpu()),
        "right_norm": float(right_norm.detach().cpu()),
    }


def train_and_audit(
    dataset: base.SyntheticDataset,
    config: r4.ExperimentConfig,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    base.set_deterministic_seed(int(seed))
    model = r1.build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    observations = torch.as_tensor(
        dataset.observations, dtype=torch.float32, device=device
    )
    scanner_ids = torch.as_tensor(dataset.scanner_ids, dtype=torch.long, device=device)
    biological_truth = torch.as_tensor(
        dataset.biological_latents, dtype=torch.float32, device=device
    )
    train = torch.as_tensor(dataset.train_indices, dtype=torch.long, device=device)
    calibration, diagnostic = _identity_split_indices(dataset, device)

    source_np, target_np = r1.build_crossed_pairs(dataset)
    consistency_left_np, consistency_right_np = r1.build_biological_consistency_pairs(dataset)
    source = torch.as_tensor(source_np, dtype=torch.long, device=device)
    target = torch.as_tensor(target_np, dtype=torch.long, device=device)
    consistency_left = torch.as_tensor(
        consistency_left_np, dtype=torch.long, device=device
    )
    consistency_right = torch.as_tensor(
        consistency_right_np, dtype=torch.long, device=device
    )
    crossed_weight, bio_cycle_weight, acq_cycle_weight = r4.family_weights(
        MODEL_FAMILY, config
    )
    biological_parameters = tuple(model.biological_encoder.parameters())

    measurements: List[Dict[str, Any]] = []
    for epoch in range(1, int(config.epochs) + 1):
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

        candidate_specific = (
            crossed_weight * crossed_reconstruction
            + bio_cycle_weight * biological_cycle_loss
            + acq_cycle_weight * acquisition_cycle_loss
        )
        full_loss = (
            config.self_reconstruction_weight * self_reconstruction
            + candidate_specific
            + config.biological_consistency_weight * biological_consistency
            + config.acquisition_same_scanner_consistency_weight * acquisition_same_scanner
            + config.acquisition_prototype_weight * acquisition_prototype
            + config.biological_variance_weight * variance_floor
            + config.prototype_center_weight * prototype_center
            + config.prototype_separation_weight * prototype_separation
        )

        if epoch in CHECKPOINTS:
            mean, std, weights = _fit_detached_affine_ridge_probe(
                all_biological,
                biological_truth,
                calibration,
            )
            retention = _retention_surrogate(
                all_biological,
                biological_truth,
                diagnostic,
                mean,
                std,
                weights,
            )
            cross_vs_retention = _gradient_cosine(
                crossed_reconstruction, retention, biological_parameters
            )
            extra_vs_retention = _gradient_cosine(
                candidate_specific, retention, biological_parameters
            )
            measurements.append(
                {
                    "epoch": int(epoch),
                    "retention_surrogate_mse": float(retention.detach().cpu()),
                    "crossed_reconstruction_mse": float(
                        crossed_reconstruction.detach().cpu()
                    ),
                    "candidate_specific_loss": float(candidate_specific.detach().cpu()),
                    "crossed_vs_retention": cross_vs_retention,
                    "candidate_specific_vs_retention": extra_vs_retention,
                }
            )

        if not torch.isfinite(full_loss):
            raise AuditError("Non-finite R4 audit training loss at epoch {}".format(epoch))
        full_loss.backward()
        optimizer.step()

    evaluation = r4.evaluate_model(
        MODEL_FAMILY, model, dataset, config, device, int(seed)
    )
    return {
        "seed": int(seed),
        "measurements": measurements,
        "final_r4_metrics": {
            "biology_retention_delta": float(
                evaluation["metrics"]["biology_retention_delta"]
            ),
            "acquisition_transfer_delta": float(
                evaluation["metrics"]["acquisition_transfer_delta"]
            ),
            "cycle_variance_normalized_total": float(
                evaluation["metrics"]["cycle_variance_normalized_total"]
            ),
        },
    }


def summarize(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_renderer: Dict[str, Any] = {}
    all_extra: List[float] = []
    all_cross: List[float] = []
    renderer_rules: List[bool] = []

    for renderer in base.RENDERERS:
        renderer_runs = [run for run in runs if run["renderer"] == renderer]
        extra = [
            float(measurement["candidate_specific_vs_retention"]["cosine"])
            for run in renderer_runs
            for measurement in run["measurements"]
        ]
        cross = [
            float(measurement["crossed_vs_retention"]["cosine"])
            for run in renderer_runs
            for measurement in run["measurements"]
        ]
        if not extra or not cross:
            raise AuditError("Missing gradient measurements for {}".format(renderer))
        extra_mean = float(np.mean(extra))
        extra_negative_fraction = float(np.mean(np.asarray(extra) < 0.0))
        cross_mean = float(np.mean(cross))
        cross_negative_fraction = float(np.mean(np.asarray(cross) < 0.0))
        rule = bool(extra_mean < 0.0 and extra_negative_fraction >= 0.80)
        renderer_rules.append(rule)
        all_extra.extend(extra)
        all_cross.extend(cross)
        by_renderer[renderer] = {
            "measurement_count": len(extra),
            "candidate_specific_vs_retention_cosine_mean": extra_mean,
            "candidate_specific_vs_retention_negative_fraction": extra_negative_fraction,
            "crossed_vs_retention_cosine_mean": cross_mean,
            "crossed_vs_retention_negative_fraction": cross_negative_fraction,
            "systematic_negative_candidate_specific_gradient": rule,
        }

    supported = bool(all(renderer_rules))
    return {
        "by_renderer": by_renderer,
        "overall_candidate_specific_vs_retention_cosine_mean": float(
            np.mean(all_extra)
        ),
        "overall_candidate_specific_vs_retention_negative_fraction": float(
            np.mean(np.asarray(all_extra) < 0.0)
        ),
        "overall_crossed_vs_retention_cosine_mean": float(np.mean(all_cross)),
        "overall_crossed_vs_retention_negative_fraction": float(
            np.mean(np.asarray(all_cross) < 0.0)
        ),
        "gradient_conflict_supported_by_predeclared_rule": supported,
        "interpretation": (
            "Systematic negative candidate-specific gradients support optimization conflict as a contributor. "
            "Failure of this rule does not prove structural identifiability, but it weakens the claim that the persistent R4 retention deficit is merely negative gradient transfer."
        ),
    }


def run_audit(
    output_root: Path,
    device: torch.device,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> Dict[str, Any]:
    if output_root.exists():
        raise AuditError("Output root already exists; overwrite prohibited: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=False)

    config = replace(
        r4.ExperimentConfig(),
        identities=64,
        epochs=80,
        bootstrap_replicates=1000,
    )
    datasets = {
        renderer: base.make_synthetic_dataset(r1.to_base_config(config), renderer)
        for renderer in base.RENDERERS
    }

    runs: List[Dict[str, Any]] = []
    for renderer, dataset in datasets.items():
        for seed in seeds:
            print("[{}] R4 gradient audit seed={}".format(renderer, seed), flush=True)
            run = train_and_audit(dataset, config, int(seed), device)
            run["renderer"] = renderer
            runs.append(run)

    summary = summarize(runs)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "post_hoc_mechanism_diagnostic_not_confirmation",
        "source_model": MODEL_FAMILY,
        "config": asdict(config),
        "checkpoints": list(CHECKPOINTS),
        "ridge_alpha": RIDGE_ALPHA,
        "model_seeds": [int(seed) for seed in seeds],
        "r5_outcomes_inspected": False,
        "frozen_scorpion_outcomes_used": False,
        "runs": runs,
        "summary": summary,
    }
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    base.atomic_json(output_root / "pa_nf_v2_r4_gradient_conflict_audit.json", result)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(
        "R4 GRADIENT CONFLICT SUPPORTED: {}".format(
            summary["gradient_conflict_supported_by_predeclared_rule"]
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
        default=Path("results/pa_nf_v2_r4_gradient_conflict_audit_20260930"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_audit(args.output_root, torch.device(args.device))


if __name__ == "__main__":
    try:
        main()
    except (AuditError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit("PA-NF V2 R4 GRADIENT AUDIT FAILED: {}".format(exc)) from exc
