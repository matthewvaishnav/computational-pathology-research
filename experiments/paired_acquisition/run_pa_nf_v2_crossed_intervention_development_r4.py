#!/usr/bin/env python3
"""PA-NF v2 development revision 4: donor-invariant acquisition pooling.

R4 is development-only. R3 showed that adding biological contrastive pressure did
not change the persistent candidate-minus-control biology-retention deficit. R4
therefore changes the crossed intervention mechanism rather than tuning another
loss weight.

For crossed reconstruction, the target acquisition state is represented by the
mean image-inferred acquisition embedding from *other* training identities on the
requested scanner. The individual target/donor acquisition vector is never passed
to the crossed decoder. This closes a direct route for donor biological identity
to hitchhike through z_a while preserving image-only encoders at inference.
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
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability_v2 as eval_v2,
)


SCHEMA_VERSION = "pa-nf-v2-crossed-intervention-development/v4"
MODEL_FAMILIES = (
    "pa_nf_v2_crossed_intervention_r4_pooled_acquisition",
    "v2_r4_control_no_cross_cycle",
)
DEFAULT_SMOKE_SEEDS = (3401, 3402, 3403)


@dataclass(frozen=True)
class ExperimentConfig(r2.ExperimentConfig):
    pass


def family_weights(
    model_family: str, config: ExperimentConfig
) -> Tuple[float, float, float]:
    if model_family == "pa_nf_v2_crossed_intervention_r4_pooled_acquisition":
        return (
            config.crossed_reconstruction_weight,
            config.biological_cycle_weight,
            config.acquisition_cycle_weight,
        )
    if model_family == "v2_r4_control_no_cross_cycle":
        return 0.0, 0.0, 0.0
    raise r1.ExperimentError("Unknown model family: {}".format(model_family))


def pooled_acquisition_for_queries(
    acquisition: torch.Tensor,
    scanner_ids: torch.Tensor,
    train_indices: torch.Tensor,
    query_indices: torch.Tensor,
    scanners: int,
) -> torch.Tensor:
    """Return scanner means while excluding the queried sample when it is training data.

    The synthetic design has at most one sample for an identity-scanner cell. Thus,
    excluding a training query removes that identity's direct contribution to the
    requested scanner pool. Held-out queries are not in training and contribute
    nothing by construction.
    """
    if acquisition.ndim != 2:
        raise r1.ExperimentError("Acquisition representation must be a matrix")
    if scanner_ids.ndim != 1 or query_indices.ndim != 1 or train_indices.ndim != 1:
        raise r1.ExperimentError("Pooling indices must be vectors")

    train_scanners = scanner_ids.index_select(0, train_indices)
    train_acquisition = acquisition.index_select(0, train_indices)
    sums: List[torch.Tensor] = []
    counts: List[int] = []
    for scanner in range(int(scanners)):
        values = train_acquisition[train_scanners == scanner]
        if values.shape[0] < 2:
            raise r1.ExperimentError(
                "Scanner {} needs at least two training identities for pooling".format(scanner)
            )
        sums.append(values.sum(dim=0))
        counts.append(int(values.shape[0]))
    scanner_sums = torch.stack(sums, dim=0)
    scanner_counts = torch.as_tensor(
        counts, dtype=acquisition.dtype, device=acquisition.device
    )

    query_scanners = scanner_ids.index_select(0, query_indices)
    pooled_sum = scanner_sums.index_select(0, query_scanners)
    pooled_count = scanner_counts.index_select(0, query_scanners)

    train_mask = torch.zeros(
        acquisition.shape[0], dtype=torch.bool, device=acquisition.device
    )
    train_mask[train_indices] = True
    query_is_train = train_mask.index_select(0, query_indices)
    subtraction = acquisition.index_select(0, query_indices) * query_is_train.to(
        acquisition.dtype
    ).unsqueeze(1)
    denominator = pooled_count - query_is_train.to(acquisition.dtype)
    if bool((denominator < 1).any()):
        raise r1.ExperimentError("Donor-invariant pool became empty")
    return (pooled_sum - subtraction) / denominator.unsqueeze(1)


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

        pooled_target_acquisition = pooled_acquisition_for_queries(
            all_acquisition,
            scanner_ids,
            train,
            target,
            config.scanners,
        )
        crossed_prediction = model.decode(
            all_biological.index_select(0, source),
            pooled_target_acquisition,
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
            pooled_target_acquisition.detach(),
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
                    "pooled_crossed_reconstruction": float(
                        crossed_reconstruction.detach().cpu()
                    ),
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
        "history": history,
    }


def pooled_cycle_diagnostics(
    model: r1.CrossedInterventionFactorizer,
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
        pooled_target = pooled_acquisition_for_queries(
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
    model: r1.CrossedInterventionFactorizer,
    dataset: base.SyntheticDataset,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    # Reuse the existing factor-allocation/probe diagnostics, then replace every
    # crossed-intervention metric/gate with the R4 donor-invariant operator.
    result = eval_v2.evaluate_model_v2(
        model_family,
        model,
        dataset,
        r1.to_base_config(config),
        device,
        seed,
    )

    observations = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    scanner_ids = torch.as_tensor(dataset.scanner_ids, dtype=torch.long, device=device)
    train = torch.as_tensor(dataset.train_indices, dtype=torch.long, device=device)
    source = torch.as_tensor(dataset.source_index_by_identity, dtype=torch.long, device=device)
    donor = torch.as_tensor(dataset.donor_index_by_identity, dtype=torch.long, device=device)
    target = torch.as_tensor(dataset.test_indices, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        biological = model.encode_biological(observations)
        acquisition = model.encode_acquisition(observations)
        pooled_target = pooled_acquisition_for_queries(
            acquisition, scanner_ids, train, target, config.scanners
        )
        swapped = model.decode(biological.index_select(0, source), pooled_target)
        correct_target = observations.index_select(0, target)
        donor_target = observations.index_select(0, donor)
        source_target = observations.index_select(0, source)
        correct_mse = (swapped - correct_target).square().mean(dim=1).cpu().numpy()
        donor_mse = (swapped - donor_target).square().mean(dim=1).cpu().numpy()
        source_mse = (swapped - source_target).square().mean(dim=1).cpu().numpy()

    biology_delta = donor_mse - correct_mse
    bio_low, bio_high = base.bootstrap_mean_interval(
        biology_delta, config.bootstrap_replicates, seed + 800_000
    )
    acquisition_delta = source_mse - correct_mse
    acq_low, acq_high = base.bootstrap_mean_interval(
        acquisition_delta, config.bootstrap_replicates, seed + 900_000
    )

    metrics = result["metrics"]
    metrics.update(
        {
            "counterfactual_correct_target_mse": float(correct_mse.mean()),
            "counterfactual_donor_target_mse": float(donor_mse.mean()),
            "counterfactual_delta": float(biology_delta.mean()),
            "counterfactual_delta_ci_025": float(bio_low),
            "counterfactual_delta_ci_975": float(bio_high),
            "counterfactual_identity_success_rate": float(np.mean(biology_delta > 0)),
            "biology_retention_delta": float(biology_delta.mean()),
            "biology_retention_delta_ci_025": float(bio_low),
            "biology_retention_delta_ci_975": float(bio_high),
            "biology_retention_identity_success_rate": float(np.mean(biology_delta > 0)),
            "source_scanner_target_mse": float(source_mse.mean()),
            "acquisition_transfer_delta": float(acquisition_delta.mean()),
            "acquisition_transfer_delta_ci_025": float(acq_low),
            "acquisition_transfer_delta_ci_975": float(acq_high),
            "acquisition_transfer_identity_success_rate": float(
                np.mean(acquisition_delta > 0)
            ),
            "two_axis_identity_success_rate": float(
                np.mean((biology_delta > 0) & (acquisition_delta > 0))
            ),
            "crossed_acquisition_uses_individual_donor_code": 0.0,
        }
    )
    metrics.update(pooled_cycle_diagnostics(model, dataset, device))
    metrics["parameter_count"] = r1.parameter_count(model)

    gates = result["gates"]
    gates["counterfactual_point_positive"] = bool(metrics["counterfactual_delta"] > 0)
    gates["counterfactual_ci_positive"] = bool(metrics["counterfactual_delta_ci_025"] > 0)
    gates["counterfactual_majority_identities"] = bool(
        metrics["counterfactual_identity_success_rate"] > 0.5
    )
    gates["biology_retention_ci_positive"] = bool(metrics["biology_retention_delta_ci_025"] > 0)
    gates["acquisition_transfer_point_positive"] = bool(metrics["acquisition_transfer_delta"] > 0)
    gates["acquisition_transfer_ci_positive"] = bool(
        metrics["acquisition_transfer_delta_ci_025"] > 0
    )
    gates["acquisition_transfer_majority_identities"] = bool(
        metrics["acquisition_transfer_identity_success_rate"] > 0.5
    )
    gates["two_axis_counterfactual_success"] = bool(
        gates["biology_retention_ci_positive"]
        and gates["acquisition_transfer_ci_positive"]
        and metrics["two_axis_identity_success_rate"] > 0.5
    )
    gates["crossed_factorization_success"] = bool(
        gates["two_axis_counterfactual_success"]
        and gates["factor_allocation_success"]
    )

    result["counterfactual_delta_by_identity"] = biology_delta.tolist()
    result["acquisition_transfer_delta_by_identity"] = acquisition_delta.tolist()
    result["r4_intervention"] = {
        "operator": "donor_invariant_same_scanner_pool",
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
        "development_iteration": 4,
        "config": asdict(config),
        "model_seeds": [int(seed) for seed in model_seeds],
        "model_families": list(MODEL_FAMILIES),
        "development_isolation": {
            "frozen_2026_09_29_scorpion_outcomes_used_for_tuning": False,
            "r1_r2_r3_synthetic_development_used": True,
            "confirmation_claim_allowed": False,
        },
        "intervention_operator": {
            "name": "donor_invariant_same_scanner_pool",
            "individual_donor_acquisition_code_used": False,
            "training_pool_excludes_query_sample": True,
        },
        "runs": runs,
        "summary": summary,
        "claim_boundary": (
            "R4 may determine whether donor-invariant acquisition pooling resolves the "
            "synthetic development failure. It is not real-pathology confirmation."
        ),
    }
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    base.atomic_json(output_root / "pa_nf_v2_development_r4_result.json", result)
    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))
    print(
        "DEVELOPMENT R4 PROMOTION PASS: {}".format(
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
        default=Path("results/pa_nf_v2_crossed_intervention_development_r4_smoke"),
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
        raise SystemExit("PA-NF V2 DEVELOPMENT R4 FAILED: {}".format(exc)) from exc
