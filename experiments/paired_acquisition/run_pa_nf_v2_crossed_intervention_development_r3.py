#!/usr/bin/env python3
"""PA-NF v2 development revision 3: biological-retention contrastive grid.

R3 responds only to the R1/R2 synthetic development outcomes. It does not read
or optimize against the frozen 2026-09-29 SCORPION shortcut/conflict outcomes.

The remaining R2 failure was biological retention relative to an exact-capacity
control. R3 adds a cross-scanner supervised-contrastive objective to the
biological code using only paired-region identity already available from the
paired-acquisition design. Candidate and control receive the same contrastive
objective; the control still differs only by disabling crossed reconstruction
and biological/acquisition cycle losses.

A frozen four-point weight grid is evaluated on fresh model seeds. The selected
configuration, if any, is the lowest contrastive weight satisfying every R2
promotion gate. No R3 result is confirmatory.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, replace
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
    run_synthetic_crossed_factor_identifiability as base,
)


SCHEMA_VERSION = "pa-nf-v2-crossed-intervention-development/v3"
MODEL_FAMILIES = (
    "pa_nf_v2_crossed_intervention_r3",
    "v2_r3_control_no_cross_cycle",
)
CONTRASTIVE_WEIGHT_GRID = (0.1, 0.25, 0.5, 1.0)
DEFAULT_SMOKE_SEEDS = (3301, 3302, 3303)


@dataclass(frozen=True)
class ExperimentConfig(r2.ExperimentConfig):
    biological_contrastive_weight: float = 0.1
    biological_contrastive_temperature: float = 0.1


def family_weights(
    model_family: str, config: ExperimentConfig
) -> Tuple[float, float, float]:
    if model_family == "pa_nf_v2_crossed_intervention_r3":
        return (
            config.crossed_reconstruction_weight,
            config.biological_cycle_weight,
            config.acquisition_cycle_weight,
        )
    if model_family == "v2_r3_control_no_cross_cycle":
        return 0.0, 0.0, 0.0
    raise r1.ExperimentError("Unknown model family: {}".format(model_family))


def biological_contrastive_loss(
    biological: torch.Tensor,
    identity_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Full-batch supervised contrastive loss over paired biological identities."""
    if biological.ndim != 2 or identity_ids.ndim != 1:
        raise r1.ExperimentError("Contrastive inputs have invalid rank")
    if biological.shape[0] != identity_ids.shape[0] or biological.shape[0] < 2:
        raise r1.ExperimentError("Contrastive inputs have incompatible shapes")
    if not float(temperature) > 0:
        raise r1.ExperimentError("Contrastive temperature must be positive")

    normalized = F.normalize(biological, dim=1)
    logits = (normalized @ normalized.T) / float(temperature)
    n = logits.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=logits.device)
    positive = identity_ids[:, None].eq(identity_ids[None, :]) & ~eye
    valid = positive.sum(dim=1) > 0
    if not bool(valid.all()):
        raise r1.ExperimentError("Every contrastive anchor requires a positive pair")

    logits = logits.masked_fill(eye, float("-inf"))
    log_denominator = torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = logits - log_denominator
    positive_log_prob = torch.where(positive, log_prob, torch.zeros_like(log_prob))
    mean_positive = positive_log_prob.sum(dim=1) / positive.sum(dim=1)
    return -mean_positive.mean()


def train_model(
    model_family: str,
    model: r1.CrossedInterventionFactorizer,
    dataset: base.SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    scanner_ids = torch.as_tensor(dataset.scanner_ids, dtype=torch.long, device=device)
    identity_ids = torch.as_tensor(dataset.identity_ids, dtype=torch.long, device=device)
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
        train_identities = identity_ids.index_select(0, train)
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
        biological_contrastive = biological_contrastive_loss(
            train_biological,
            train_identities,
            config.biological_contrastive_temperature,
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
            + config.biological_contrastive_weight * biological_contrastive
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
                    "biological_contrastive": float(biological_contrastive.detach().cpu()),
                    "acquisition_same_scanner_consistency": float(
                        acquisition_same_scanner.detach().cpu()
                    ),
                    "biological_contrastive_weight": float(
                        config.biological_contrastive_weight
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
        "history": history,
    }


def summarize_weight_runs(
    runs: Sequence[Mapping[str, Any]],
    weight: float,
) -> Dict[str, Any]:
    weight_runs = [
        run
        for run in runs
        if abs(float(run["biological_contrastive_weight"]) - float(weight)) < 1e-12
    ]
    if not weight_runs:
        raise r1.ExperimentError("Missing runs for contrastive weight {}".format(weight))

    paired_cycle: List[float] = []
    paired_transfer: List[float] = []
    paired_biology: List[float] = []
    equal_parameters = True
    candidate_all_success = True
    by_renderer: Dict[str, Any] = {}

    for renderer in base.RENDERERS:
        renderer_runs = [run for run in weight_runs if run["renderer"] == renderer]
        by_key = {
            (run["model_family"], int(run["seed"])): run for run in renderer_runs
        }
        seeds = sorted({int(run["seed"]) for run in renderer_runs})
        renderer_candidate_biology: List[float] = []
        renderer_control_biology: List[float] = []
        renderer_candidate_transfer: List[float] = []
        renderer_control_transfer: List[float] = []
        for seed in seeds:
            candidate = by_key[("pa_nf_v2_crossed_intervention_r3", seed)]
            control = by_key[("v2_r3_control_no_cross_cycle", seed)]
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
            renderer_candidate_biology.append(float(cm["biology_retention_delta"]))
            renderer_control_biology.append(float(xm["biology_retention_delta"]))
            renderer_candidate_transfer.append(float(cm["acquisition_transfer_delta"]))
            renderer_control_transfer.append(float(xm["acquisition_transfer_delta"]))
        by_renderer[renderer] = {
            "seed_count": len(seeds),
            "candidate_mean_biology_retention_delta": float(
                np.mean(renderer_candidate_biology)
            ),
            "control_mean_biology_retention_delta": float(
                np.mean(renderer_control_biology)
            ),
            "candidate_mean_acquisition_transfer_delta": float(
                np.mean(renderer_candidate_transfer)
            ),
            "control_mean_acquisition_transfer_delta": float(
                np.mean(renderer_control_transfer)
            ),
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
    return {"weight": float(weight), "by_renderer": by_renderer, "promotion_gate": gate}


def run_experiment(
    base_config: ExperimentConfig,
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
        renderer: base.make_synthetic_dataset(r1.to_base_config(base_config), renderer)
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
    for weight in CONTRASTIVE_WEIGHT_GRID:
        config = replace(base_config, biological_contrastive_weight=float(weight))
        for renderer, dataset in datasets.items():
            for seed in model_seeds:
                for model_family in MODEL_FAMILIES:
                    print(
                        "[{}] weight={} model={} seed={}".format(
                            renderer, weight, model_family, seed
                        ),
                        flush=True,
                    )
                    base.set_deterministic_seed(int(seed))
                    model = r1.build_model(config, device)
                    training = train_model(
                        model_family, model, dataset, config, device
                    )
                    evaluation = r2.evaluate_model(
                        model_family, model, dataset, config, device, int(seed)
                    )
                    runs.append(
                        {
                            "renderer": renderer,
                            "biological_contrastive_weight": float(weight),
                            "model_family": model_family,
                            "seed": int(seed),
                            "parameter_count": r1.parameter_count(model),
                            "training": training,
                            "evaluation": evaluation,
                        }
                    )

    weight_summaries = [
        summarize_weight_runs(runs, weight) for weight in CONTRASTIVE_WEIGHT_GRID
    ]
    passing = [
        summary
        for summary in weight_summaries
        if bool(summary["promotion_gate"]["development_promotion_pass"])
    ]
    selected_weight = None if not passing else float(min(item["weight"] for item in passing))
    summary = {
        "selection_rule": "lowest contrastive weight passing every unchanged R2 promotion gate",
        "selected_biological_contrastive_weight": selected_weight,
        "development_r3_grid_pass": selected_weight is not None,
        "weight_summaries": weight_summaries,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_only_not_confirmation",
        "development_iteration": 3,
        "base_config": asdict(base_config),
        "contrastive_weight_grid": list(CONTRASTIVE_WEIGHT_GRID),
        "model_seeds": [int(seed) for seed in model_seeds],
        "model_families": list(MODEL_FAMILIES),
        "development_isolation": {
            "frozen_2026_09_29_scorpion_outcomes_used_for_tuning": False,
            "r1_r2_synthetic_development_outcomes_informed_r3": True,
            "confirmation_claim_allowed": False,
        },
        "runs": runs,
        "summary": summary,
        "claim_boundary": (
            "R3 selects at most a development candidate. Any selected configuration "
            "must be frozen and tested on fresh development-holdout seeds before "
            "prospective real-data confirmation."
        ),
    }
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    base.atomic_json(output_root / "pa_nf_v2_development_r3_result.json", result)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DEVELOPMENT R3 GRID PASS: {}".format(summary["development_r3_grid_pass"]))
    print("SELECTED CONTRASTIVE WEIGHT: {}".format(selected_weight))
    print("Artifacts: {}".format(output_root.resolve()))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/pa_nf_v2_crossed_intervention_development_r3_smoke"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = ExperimentConfig(
        identities=64,
        epochs=80,
        bootstrap_replicates=1000,
    )
    run_experiment(config, DEFAULT_SMOKE_SEEDS, args.output_root, device)


if __name__ == "__main__":
    try:
        main()
    except (r1.ExperimentError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit("PA-NF V2 DEVELOPMENT R3 FAILED: {}".format(exc)) from exc
