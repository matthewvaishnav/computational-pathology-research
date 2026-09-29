#!/usr/bin/env python3
"""PA-NF v2 crossed-intervention development revision 2.

Revision 2 is development-only. It responds only to the synthetic smoke result from
revision 1 and still does not read or optimize against the frozen 2026-09-29
SCORPION shortcut/conflict outcomes.

Changes from revision 1:
- add efficient same-scanner acquisition consistency across different identities;
- weight biological and acquisition cycle preservation separately;
- evaluate cycle fidelity normalized by latent variance so collapsed/small codes
  are not rewarded by a deceptively small raw cycle MSE.

Candidate and primary control remain exactly parameter matched. They share all
architecture and regularizers. The control disables only crossed reconstruction
and both cycle-preservation losses.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from experiments.paired_acquisition import (
    run_pa_nf_v2_crossed_intervention_development as r1,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as base,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability_v2 as eval_v2,
)


SCHEMA_VERSION = "pa-nf-v2-crossed-intervention-development/v2"
MODEL_FAMILIES = (
    "pa_nf_v2_crossed_intervention_r2",
    "v2_r2_control_no_cross_cycle",
)
DEFAULT_FULL_SEEDS = tuple(range(3201, 3211))
DEFAULT_SMOKE_SEEDS = (3201, 3202, 3203)


@dataclass(frozen=True)
class ExperimentConfig(r1.ExperimentConfig):
    biological_cycle_weight: float = 2.0
    acquisition_cycle_weight: float = 1.0
    acquisition_same_scanner_consistency_weight: float = 0.5


def family_weights(
    model_family: str, config: ExperimentConfig
) -> Tuple[float, float, float]:
    if model_family == "pa_nf_v2_crossed_intervention_r2":
        return (
            config.crossed_reconstruction_weight,
            config.biological_cycle_weight,
            config.acquisition_cycle_weight,
        )
    if model_family == "v2_r2_control_no_cross_cycle":
        return 0.0, 0.0, 0.0
    raise r1.ExperimentError("Unknown model family: {}".format(model_family))


def same_scanner_acquisition_consistency(
    acquisition: torch.Tensor,
    scanner_ids: torch.Tensor,
    scanners: int,
) -> torch.Tensor:
    """Average within-scanner acquisition variance over represented scanners."""
    losses: List[torch.Tensor] = []
    for scanner in range(int(scanners)):
        mask = scanner_ids == scanner
        values = acquisition[mask]
        if values.shape[0] < 2:
            continue
        center = values.mean(dim=0, keepdim=True)
        losses.append((values - center).square().mean())
    if not losses:
        raise r1.ExperimentError("No scanner has enough samples for acquisition consistency")
    return torch.stack(losses).mean()


def train_model(
    model_family: str,
    model: r1.CrossedInterventionFactorizer,
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

        crossed_prediction = model.decode(
            all_biological.index_select(0, source),
            all_acquisition.index_select(0, target),
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
            cycle_acquisition,
            all_acquisition.index_select(0, target).detach(),
        )

        biological_consistency = F.mse_loss(
            all_biological.index_select(0, consistency_left),
            all_biological.index_select(0, consistency_right),
        )
        acquisition_same_scanner = same_scanner_acquisition_consistency(
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
                "Non-finite loss for {} at epoch {}".format(model_family, epoch + 1)
            )
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise r1.ExperimentError(
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
                    "biological_cycle": float(biological_cycle_loss.detach().cpu()),
                    "acquisition_cycle": float(acquisition_cycle_loss.detach().cpu()),
                    "biological_consistency": float(biological_consistency.detach().cpu()),
                    "acquisition_same_scanner_consistency": float(
                        acquisition_same_scanner.detach().cpu()
                    ),
                    "acquisition_prototype": float(acquisition_prototype.detach().cpu()),
                    "biological_variance_floor": float(variance_floor.detach().cpu()),
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
        "history": history,
    }


def cycle_diagnostics(
    model: r1.CrossedInterventionFactorizer,
    dataset: base.SyntheticDataset,
    device: torch.device,
) -> Dict[str, float]:
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    source = torch.as_tensor(dataset.source_index_by_identity, dtype=torch.long, device=device)
    donor = torch.as_tensor(dataset.donor_index_by_identity, dtype=torch.long, device=device)
    target = torch.as_tensor(dataset.test_indices, dtype=torch.long, device=device)
    train = torch.as_tensor(dataset.train_indices, dtype=torch.long, device=device)
    scanner_ids = torch.as_tensor(dataset.scanner_ids, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        biological = model.encode_biological(observations)
        acquisition = model.encode_acquisition(observations)
        biological_target = biological.index_select(0, source)
        acquisition_target = acquisition.index_select(0, donor)
        crossed = model.decode(biological_target, acquisition_target)
        re_biological = model.encode_biological(crossed)
        re_acquisition = model.encode_acquisition(crossed)

        bio_cycle = F.mse_loss(re_biological, biological_target)
        acq_cycle = F.mse_loss(re_acquisition, acquisition_target)
        bio_scale = biological_target.var(dim=0, unbiased=False).mean().clamp_min(1e-6)
        acq_scale = acquisition_target.var(dim=0, unbiased=False).mean().clamp_min(1e-6)
        normalized_bio = bio_cycle / bio_scale
        normalized_acq = acq_cycle / acq_scale
        normalized_total = normalized_bio + normalized_acq
        crossed_target_mse = F.mse_loss(crossed, observations.index_select(0, target))
        train_acquisition = acquisition.index_select(0, train)
        train_scanners = scanner_ids.index_select(0, train)
        within_scanner = same_scanner_acquisition_consistency(
            train_acquisition, train_scanners, int(model.scanners)
        )

    return {
        "cycle_biological_mse": float(bio_cycle.cpu()),
        "cycle_acquisition_mse": float(acq_cycle.cpu()),
        "cycle_total_mse": float((bio_cycle + acq_cycle).cpu()),
        "cycle_biological_variance_normalized": float(normalized_bio.cpu()),
        "cycle_acquisition_variance_normalized": float(normalized_acq.cpu()),
        "cycle_variance_normalized_total": float(normalized_total.cpu()),
        "crossed_target_mse": float(crossed_target_mse.cpu()),
        "acquisition_within_scanner_variance": float(within_scanner.cpu()),
    }


def evaluate_model(
    model_family: str,
    model: r1.CrossedInterventionFactorizer,
    dataset: base.SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    result = eval_v2.evaluate_model_v2(
        model_family,
        model,
        dataset,
        r1.to_base_config(config),
        device,
        seed,
    )
    result["metrics"].update(cycle_diagnostics(model, dataset, device))
    result["metrics"]["parameter_count"] = r1.parameter_count(model)
    return result


def summarize_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"by_renderer": {}, "promotion_gate": {}}
    required_metrics = (
        "biology_retention_delta",
        "acquisition_transfer_delta",
        "cycle_variance_normalized_total",
        "cycle_total_mse",
        "acquisition_within_scanner_variance",
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
            if family == "pa_nf_v2_crossed_intervention_r2":
                candidate_all_success = candidate_all_success and family_summary[family][
                    "all_seed_crossed_factorization_success"
                ]

        seeds = sorted({int(run["seed"]) for run in renderer_runs})
        for seed in seeds:
            candidate = by_family_seed[("pa_nf_v2_crossed_intervention_r2", seed)]
            control = by_family_seed[("v2_r2_control_no_cross_cycle", seed)]
            candidate_metrics = candidate["evaluation"]["metrics"]
            control_metrics = control["evaluation"]["metrics"]
            equal_parameters = equal_parameters and (
                int(candidate_metrics["parameter_count"])
                == int(control_metrics["parameter_count"])
            )
            paired_cycle_differences.append(
                float(control_metrics["cycle_variance_normalized_total"])
                - float(candidate_metrics["cycle_variance_normalized_total"])
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
        "candidate_mean_variance_normalized_cycle_error_lower_than_control": bool(
            np.mean(paired_cycle_differences) > 0
        ),
        "paired_mean_candidate_minus_control_acquisition_transfer_delta": float(
            np.mean(paired_transfer_differences)
        ),
        "paired_mean_candidate_minus_control_biology_retention_delta": float(
            np.mean(paired_biology_differences)
        ),
        "paired_mean_control_minus_candidate_variance_normalized_cycle_error": float(
            np.mean(paired_cycle_differences)
        ),
    }
    gate_keys = [
        "parameter_counts_equal",
        "candidate_all_seed_crossed_factorization_success",
        "candidate_mean_acquisition_transfer_delta_greater_than_control",
        "candidate_mean_biology_retention_delta_greater_than_control",
        "candidate_mean_variance_normalized_cycle_error_lower_than_control",
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
        raise r1.ExperimentError(
            "Output root already exists; overwrite prohibited: {}".format(output_root)
        )
    output_root.mkdir(parents=True, exist_ok=False)

    datasets = {
        renderer: base.make_synthetic_dataset(r1.to_base_config(config), renderer)
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
                model = r1.build_model(config, device)
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
        "development_iteration": 2,
        "config": asdict(config),
        "model_seeds": [int(seed) for seed in model_seeds],
        "model_families": list(MODEL_FAMILIES),
        "development_isolation": {
            "frozen_2026_09_29_scorpion_outcomes_used_for_tuning": False,
            "revision_1_synthetic_smoke_used_for_development": True,
            "confirmation_claim_allowed": False,
        },
        "runs": runs,
        "summary": summary,
        "claim_boundary": (
            "Revision-2 diagnostics may decide whether PA-NF v2 should advance in "
            "development. They do not establish real-pathology or unseen-scanner superiority."
        ),
    }
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    base.atomic_json(output_root / "pa_nf_v2_development_r2_result.json", result)
    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))
    print(
        "DEVELOPMENT R2 PROMOTION PASS: {}".format(
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
        default=Path("results/pa_nf_v2_crossed_intervention_development_r2_smoke"),
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
    except (r1.ExperimentError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit("PA-NF V2 DEVELOPMENT R2 FAILED: {}".format(exc)) from exc
