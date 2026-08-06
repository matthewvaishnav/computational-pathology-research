#!/usr/bin/env python3
"""Versioned no-training fixed-estimand adjudication using recovered neural artifacts.

This is Phase C of the real-bottleneck artifact recovery. After the exact replay
recovered and archived the 50 canine B32/B64 per-cell projected representations,
this runner performs the previously blocked no-training, fixed-estimand
adjudication using only those recovered arrays.

It imports and reuses the original adjudication implementation wherever possible:

- accepts an explicit ``--recovery-manifest``;
- requires recovery status ``complete_exact_real_bottleneck_representation_recovery``;
- verifies all 50 projected-feature and checkpoint hashes;
- initializes zero optimizers, executes zero backward passes, trains zero models;
- loads the recovered arrays only;
- reproduces the frozen seven-category neural metrics before fixed-estimand scoring;
- uses the exact corrected five-category implementation;
- retains the seven-category endpoint as exploratory;
- uses identical candidate pools across methods;
- recomputes neural and simple-baseline scanner/category/retrieval metrics;
- preserves the existing 0.02 dominance margins and performs the previously
  specified Pareto and fold-aware adjudication;
- keeps Layer-2 and pixel-space work prohibited.

The previous ``fixed_estimand_adjudication_not_ready`` result is not overwritten
or reinterpreted; the v2 result is a new forward-valid artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.paired_acquisition import (
    run_fixed_estimand_real_feature_space_adjudication as adj,
)
from experiments.paired_acquisition import (
    run_real_paired_scanner_bottleneck_allocation_validation as real_validation,
)


SCHEMA_VERSION = "fixed-estimand-real-feature-space-adjudication/v2"
RECOVERY_STATUS_REQUIRED = "complete_exact_real_bottleneck_representation_recovery"
FROZEN_REPRODUCTION_TOLERANCE = 1e-6
MATERIAL_MARGIN = 0.02
CROSS_FOLD_REQUIRED = 4

NEURAL_METHODS = adj.NEURAL_METHODS
DETERMINISTIC_METHODS = adj.DETERMINISTIC_METHODS
FOLDS = adj.FOLDS
FAMILIES = adj.FAMILIES

CLAIM_SCOPE = {
    **adj.CLAIM_SCOPE,
    "schema_version": SCHEMA_VERSION,
    "uses_recovered_neural_artifacts": True,
    "recovery_status_required": RECOVERY_STATUS_REQUIRED,
    "does_not_modify_prior_adjudication_result": True,
    "layer2_prohibited": True,
    "pixel_space_prohibited": True,
}


class AdjudicationV2Error(RuntimeError):
    """A structural or execution failure distinct from a poor scientific result."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    adj.atomic_json(path, value)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=adj.heterogeneous_fieldnames(rows))
            writer.writeheader()
            writer.writerows(dict(row) for row in rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_hash(value: Mapping[str, Any]) -> str:
    return adj.canonical_hash(value)


def load_recovery_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AdjudicationV2Error(f"Recovery manifest missing: {path}")
    recovery = json.loads(path.read_text(encoding="utf-8"))
    if recovery.get("status") != RECOVERY_STATUS_REQUIRED:
        raise AdjudicationV2Error(
            f"Recovery status must be {RECOVERY_STATUS_REQUIRED}, found {recovery.get('status')}"
        )
    cells = recovery.get("cells")
    if not isinstance(cells, list) or len(cells) != 50:
        raise AdjudicationV2Error("Recovery manifest must contain exactly 50 cells.")
    for cell in cells:
        if not cell.get("accepted"):
            raise AdjudicationV2Error("Recovery manifest contains an unaccepted cell.")
    return recovery


def verify_recovery_hashes(
    recovery: Mapping[str, Any],
    recovery_output_root: Path,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for cell in recovery["cells"]:
        projected_path = Path(cell["projected_features_path"])
        checkpoint_path = Path(cell["checkpoint_path"])
        if not projected_path.is_file():
            raise AdjudicationV2Error(f"Recovered projected features missing: {projected_path}")
        if not checkpoint_path.is_file():
            raise AdjudicationV2Error(f"Recovered checkpoint missing: {checkpoint_path}")
        observed_projected = sha256_file(projected_path)
        observed_checkpoint = sha256_file(checkpoint_path)
        if observed_projected != cell["projected_features_sha256"]:
            raise AdjudicationV2Error(
                f"Recovered projected-feature hash mismatch: {projected_path}"
            )
        if observed_checkpoint != cell["checkpoint_sha256"]:
            raise AdjudicationV2Error(f"Recovered checkpoint hash mismatch: {checkpoint_path}")
        records[
            (cell["fold"], cell["seed"], cell["family"])
        ] = {
            "projected_features_path": str(projected_path.resolve()),
            "projected_features_sha256": observed_projected,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": observed_checkpoint,
        }
    del recovery_output_root
    return {"verified_cells": len(records), "all_hashes_verified": True}


def load_cell_arrays(
    recovery: Mapping[str, Any],
    fold: int,
    seed: int,
    family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for cell in recovery["cells"]:
        if (
            cell["fold"] == fold
            and cell["seed"] == seed
            and cell["family"] == family
        ):
            with np.load(cell["projected_features_path"], allow_pickle=False) as archive:
                biological = np.asarray(archive["biological_features"], dtype=np.float32)
                acquisition = np.asarray(archive["acquisition_features"], dtype=np.float32)
                combined = np.asarray(archive["combined_features"], dtype=np.float32)
            return biological, acquisition, combined
    raise AdjudicationV2Error(
        f"No recovered cell for fold={fold} seed={seed} family={family}"
    )


def reproduction_metrics(
    biological: np.ndarray,
    loaded: adj.LoadedFold,
    fold: int,
) -> dict[str, Any]:
    frame = loaded.frame
    train, _, test = real_validation.split_indices(frame)
    labels = frame["category_name"].astype(str).to_numpy()
    category = real_validation.probe(
        biological, labels, train, test, nonlinear=False, seeds=real_validation.PROBE_SEEDS
    )
    scanner_labels = frame["scanner_id"].astype(str).to_numpy()
    scanner = real_validation.probe(
        biological,
        scanner_labels,
        train,
        test,
        nonlinear=False,
        seeds=real_validation.PROBE_SEEDS,
    )
    retrieval = real_validation.retrieval_metrics(biological, frame, test)
    return {
        "fold": fold,
        "category_balanced_accuracy_seven": category["balanced_accuracy_median"],
        "scanner_balanced_accuracy": scanner["balanced_accuracy_median"],
        "overall_retrieval": retrieval["overall_top1"],
        "worst_pair_retrieval": retrieval["worst_ordered_scanner_pair_top1"],
    }


def verify_frozen_reproduction(
    metrics: Mapping[str, Any],
    frozen_run: Mapping[str, Any],
) -> dict[str, Any]:
    expected = [
        (
            "category_balanced_accuracy_seven",
            "layer1.biological_category_accessibility.linear.balanced_accuracy_median",
            frozen_run["layer1"]["biological_category_accessibility"]["linear"]["balanced_accuracy_median"],
        ),
        (
            "scanner_balanced_accuracy",
            "layer1.biological_scanner_probe.linear.balanced_accuracy_median",
            frozen_run["layer1"]["biological_scanner_probe"]["linear"]["balanced_accuracy_median"],
        ),
        (
            "overall_retrieval",
            "layer1.paired_region_preservation.overall_top1",
            frozen_run["layer1"]["paired_region_preservation"]["overall_top1"],
        ),
        (
            "worst_pair_retrieval",
            "layer1.paired_region_preservation.worst_ordered_scanner_pair_top1",
            frozen_run["layer1"]["paired_region_preservation"]["worst_ordered_scanner_pair_top1"],
        ),
    ]
    mismatches = []
    for metric, path, frozen_value in expected:
        replay_value = metrics.get(metric)
        if replay_value is None or abs(float(replay_value) - float(frozen_value)) > FROZEN_REPRODUCTION_TOLERANCE:
            mismatches.append(
                {"metric": metric, "path": path, "frozen": frozen_value, "replay": replay_value}
            )
    return {
        "frozen_metrics_reproduced": not mismatches,
        "mismatches": mismatches,
        "tolerance": FROZEN_REPRODUCTION_TOLERANCE,
    }


def evaluate_neural_cell(
    biological: np.ndarray,
    acquisition: np.ndarray,
    loaded: adj.LoadedFold,
    fold: int,
    frozen_run: Mapping[str, Any],
) -> dict[str, Any]:
    reproduction = verify_frozen_reproduction(
        reproduction_metrics(biological, loaded, fold), frozen_run
    )
    corrected_category = adj.fixed_category_evaluation(biological, loaded.frame, fold)
    scanner = adj.scanner_evaluation(biological, loaded.frame, fold)
    retrieval = adj.retrieval_evaluation(biological, loaded.frame, fold)
    acquisition_scanner = adj.scanner_evaluation(acquisition, loaded.frame, fold)
    combined = np.concatenate([biological, acquisition], axis=1).astype(np.float32)
    combined_retrieval = adj.retrieval_evaluation(combined, loaded.frame, fold)
    return {
        "fold": fold,
        "frozen_reproduction": reproduction,
        "corrected_category": corrected_category,
        "scanner": scanner,
        "retrieval": retrieval,
        "acquisition_scanner": acquisition_scanner,
        "combined_retrieval": combined_retrieval,
    }


def average_neural_within_fold(
    evaluations: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> Mapping[int, Mapping[str, dict[str, Any]]]:
    averaged: dict[int, dict[str, dict[str, Any]]] = {}
    for fold in FOLDS:
        averaged[fold] = {}
        for family in FAMILIES:
            per_seed_category = []
            per_seed_scanner = []
            per_seed_worst = []
            per_seed_overall = []
            seed_distribution = {}
            for seed in adj.MODEL_SEEDS:
                evaluation = evaluations[fold][seed][family]
                category_ba = evaluation["corrected_category"]["balanced_accuracy"]
                scanner_ba = evaluation["scanner"]["linear_balanced_accuracy"]
                worst = evaluation["retrieval"]["worst_ordered_scanner_pair_top1"]
                overall = evaluation["retrieval"]["overall_top1"]
                per_seed_category.append(category_ba)
                per_seed_scanner.append(scanner_ba)
                per_seed_worst.append(worst)
                per_seed_overall.append(overall)
                seed_distribution[seed] = {
                    "category_balanced_accuracy": category_ba,
                    "scanner_balanced_accuracy": scanner_ba,
                    "worst_pair_retrieval": worst,
                    "overall_retrieval": overall,
                }
            averaged[fold][family] = {
                "category_balanced_accuracy": float(np.mean(per_seed_category)),
                "scanner_balanced_accuracy": float(np.mean(per_seed_scanner)),
                "worst_pair_retrieval": float(np.mean(per_seed_worst)),
                "overall_retrieval": float(np.mean(per_seed_overall)),
                "seed_distribution": seed_distribution,
                "seed_averaged_before_inference": True,
            }
    return averaged


# ---------------------------------------------------------------------------
# Adjudication
# ---------------------------------------------------------------------------


def neural_increment_analysis(
    table: Mapping[int, Mapping[str, Mapping[str, float | None]]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for family in NEURAL_METHODS:
        gain_folds = 0
        for fold in FOLDS:
            family_cat = table[fold][family]["category_balanced_accuracy"]
            baseline_cats = [
                table[fold][baseline]["category_balanced_accuracy"]
                for baseline in DETERMINISTIC_METHODS
            ]
            if family_cat is not None and all(
                value is not None and family_cat - value >= MATERIAL_MARGIN
                for value in baseline_cats
            ):
                gain_folds += 1
        family_scanner = float(
            np.mean([table[fold][family]["scanner_balanced_accuracy"] for fold in FOLDS])
        )
        family_worst = float(
            np.mean([table[fold][family]["worst_pair_retrieval"] for fold in FOLDS])
        )
        scanner_violation = False
        retrieval_violation = False
        for baseline in DETERMINISTIC_METHODS:
            baseline_scanner = float(
                np.mean([table[fold][baseline]["scanner_balanced_accuracy"] for fold in FOLDS])
            )
            baseline_worst = float(
                np.mean([table[fold][baseline]["worst_pair_retrieval"] for fold in FOLDS])
            )
            if family_scanner - baseline_scanner > MATERIAL_MARGIN:
                scanner_violation = True
            if family_worst - baseline_worst < -MATERIAL_MARGIN:
                retrieval_violation = True
        supported = bool(
            gain_folds >= CROSS_FOLD_REQUIRED
            and not scanner_violation
            and not retrieval_violation
        )
        results[family] = {
            "category_gain_folds": gain_folds,
            "category_gain_supported": gain_folds >= CROSS_FOLD_REQUIRED,
            "scanner_margin_violation": scanner_violation,
            "worst_pair_retrieval_margin_violation": retrieval_violation,
            "supported": supported,
        }
    any_supported = any(item["supported"] for item in results.values())
    return {"families": results, "neural_feature_space_increment_supported": any_supported}


def simple_baseline_dominance_analysis(
    canine_table: Mapping[int, Mapping[str, Mapping[str, float | None]]],
    scorpion_table: Mapping[int, Mapping[str, Mapping[str, float | None]]],
) -> dict[str, Any]:
    axes = [
        "scanner_balanced_accuracy",
        "category_balanced_accuracy",
        "worst_pair_retrieval",
        "overall_retrieval",
    ]
    scorpion_axes = ["scanner_balanced_accuracy", "worst_pair_retrieval"]
    results: dict[str, Any] = {}
    for simple in ("centroid_qr_scanner_subspace_projection", "paired_linear_scanner_transform"):
        simple_folds = {fold: canine_table[fold][simple] for fold in FOLDS}
        b32_folds = {fold: canine_table[fold]["real_b32_reference"] for fold in FOLDS}
        b64_folds = {fold: canine_table[fold]["real_b64_parameter_matched"] for fold in FOLDS}
        dom_b32 = adj.cross_fold_material_dominance(simple_folds, b32_folds, axes)
        dom_b64 = adj.cross_fold_material_dominance(simple_folds, b64_folds, axes)
        scorpion_front_count = 0
        for fold in FOLDS:
            methods = [
                {"method": method, **scorpion_table[fold][method]}
                for method in DETERMINISTIC_METHODS
                if method in scorpion_table[fold]
            ]
            front = adj.pareto_front(
                methods,
                ["scanner_balanced_accuracy", "worst_pair_retrieval"],
                lower_is_better=["scanner_balanced_accuracy"],
            )
            if simple in front:
                scorpion_front_count += 1
        scorpion_non_inferior = scorpion_front_count >= CROSS_FOLD_REQUIRED
        supported = bool(
            dom_b32["cross_fold_material_dominance"]
            and dom_b64["cross_fold_material_dominance"]
            and scorpion_non_inferior
        )
        results[simple] = {
            "dominates_b32": dom_b32["cross_fold_material_dominance"],
            "dominates_b64": dom_b64["cross_fold_material_dominance"],
            "scorpion_front_fold_count": scorpion_front_count,
            "scorpion_non_inferior": scorpion_non_inferior,
            "supported": supported,
        }
    any_supported = any(item["supported"] for item in results.values())
    return {
        "baselines": results,
        "simple_baseline_pareto_dominance_supported": any_supported,
    }


def synthetic_transport_decision(
    contrasts: Mapping[str, Any],
    neural_increment: Mapping[str, Any],
) -> dict[str, Any]:
    del neural_increment
    b64_b32 = contrasts.get("b64_minus_b32")
    if not b64_b32 or not b64_b32.get("available"):
        return {
            "synthetic_accessibility_effect_transported": False,
            "reason": "B64-vs-B32 corrected category contrast unavailable",
        }
    category = b64_b32["axes"]["category_balanced_accuracy"]
    scanner = b64_b32["axes"]["scanner_balanced_accuracy"]
    worst = b64_b32["axes"]["worst_pair_retrieval"]
    gain_folds = category["positive_fold_count"]
    category_gain = bool(
        category["mean"] >= MATERIAL_MARGIN and gain_folds >= CROSS_FOLD_REQUIRED
    )
    scanner_ok = bool(scanner["mean"] <= MATERIAL_MARGIN)
    retrieval_ok = bool(worst["mean"] >= -MATERIAL_MARGIN)
    transported = bool(category_gain and scanner_ok and retrieval_ok)
    return {
        "synthetic_bottleneck_width_effect": "frozen synthetic factorial reported a capacity gain with scanner tradeoff",
        "real_retrieval_effect": contrasts["b64_minus_b32"]["axes"]["worst_pair_retrieval"],
        "real_scanner_recoverability_effect": scanner,
        "corrected_real_category_effect": category,
        "synthetic_accessibility_effect_transported": transported,
        "category_gain_in_four_folds": category_gain,
        "scanner_margin_ok": scanner_ok,
        "retrieval_margin_ok": retrieval_ok,
        "reason": "transport requires B64 corrected five-category gain in at least four folds without scanner or retrieval margin violations; retrieval gain alone is never counted as biological accessibility.",
    }


def v2_claim_adjudication(
    contrasts: Mapping[str, Any],
    neural_increment: Mapping[str, Any],
    transport: Mapping[str, Any],
    frozen_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    del frozen_metrics
    b64_b32 = contrasts.get("b64_minus_b32", {})
    b64_b32_available = bool(b64_b32.get("available"))
    category_gain = bool(
        b64_b32_available
        and b64_b32["axes"]["category_balanced_accuracy"]["mean"] >= MATERIAL_MARGIN
        and b64_b32["axes"]["category_balanced_accuracy"]["positive_fold_count"] >= CROSS_FOLD_REQUIRED
    )
    retrieval_gain = bool(
        b64_b32_available
        and (
            b64_b32["axes"]["worst_pair_retrieval"]["mean"] >= MATERIAL_MARGIN
            or b64_b32["axes"]["overall_retrieval"]["mean"] >= MATERIAL_MARGIN
        )
    )
    scanner_increase = bool(
        b64_b32_available
        and b64_b32["axes"]["scanner_balanced_accuracy"]["mean"] > MATERIAL_MARGIN
    )
    increment = neural_increment["neural_feature_space_increment_supported"]
    transported = transport["synthetic_accessibility_effect_transported"]

    claims = [
        {
            "id": 1,
            "claim": "Wider biological bottlenecks improve canine category accessibility.",
            "verdict": "supported" if category_gain else "unsupported",
            "rationale": "corrected five-category B64-vs-B32 category contrast",
        },
        {
            "id": 2,
            "claim": "Wider biological bottlenecks improve region retrieval.",
            "verdict": "supported" if retrieval_gain else "unsupported",
            "rationale": "corrected worst-pair/overall retrieval contrast",
        },
        {
            "id": 3,
            "claim": "Wider biological bottlenecks increase scanner recoverability.",
            "verdict": "supported" if scanner_increase else "unsupported",
            "rationale": "corrected scanner balanced-accuracy contrast",
        },
        {
            "id": 4,
            "claim": "Paired supervision improves region preservation.",
            "verdict": "supported",
            "rationale": "frozen true-pair vs broken-pair controls demonstrated paired supervision",
        },
        {
            "id": 5,
            "claim": "Neural factorization outperforms centroid/QR.",
            "verdict": "supported" if increment else "unsupported",
            "rationale": "neural feature-space increment against every simple scanner-removal baseline",
        },
        {
            "id": 6,
            "claim": "Neural factorization outperforms paired linear transforms.",
            "verdict": "supported" if increment else "unsupported",
            "rationale": "neural feature-space increment against paired-linear baseline",
        },
        {
            "id": 7,
            "claim": "The acquisition branch retains scanner information.",
            "verdict": "supported",
            "rationale": "frozen acquisition-branch scanner probes show high scanner recoverability",
        },
        {
            "id": 8,
            "claim": "The acquisition branch enables validated scanner swapping.",
            "verdict": "prohibited by evidence scope",
            "rationale": "verified Layer-2 swap metadata absent; swap utility remains unverified",
        },
        {
            "id": 9,
            "claim": "Synthetic accessibility effects transport to real pathology features.",
            "verdict": "supported" if transported else "unsupported",
            "rationale": "corrected five-category transport decision",
        },
        {
            "id": 10,
            "claim": "Feature-space evidence establishes pixel-space scanner translation.",
            "verdict": "prohibited by evidence scope",
            "rationale": "fixed-feature evidence does not establish pixel behavior; pixel-space evaluation prohibited",
        },
        {
            "id": 11,
            "claim": "Feature-space category preservation establishes clinical validity.",
            "verdict": "prohibited by evidence scope",
            "rationale": "canine tissue categories are descriptive labels, not clinical endpoints",
        },
    ]
    return claims


def v2_dataset_conclusions(
    neural_increment: Mapping[str, Any],
    simple_dominance: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if neural_increment["neural_feature_space_increment_supported"]:
        canine = "neural_feature_space_increment_supported"
    elif simple_dominance["simple_baseline_pareto_dominance_supported"]:
        canine = "simple_baseline_pareto_dominance_supported"
    else:
        canine = "no_neural_feature_space_increment_supported"
    return {
        "canine_scc": {"conclusion": canine},
        "scorpion": {"conclusion": "feature_only_no_biological_claim"},
    }


def v2_top_level_status(canine_conclusion: str) -> str:
    mapping = {
        "simple_baseline_pareto_dominance_supported": "complete_simple_baseline_pareto_dominance_supported",
        "neural_feature_space_increment_supported": "complete_neural_feature_space_increment_supported",
        "mixed_fixed_estimand_feature_space_evidence": "complete_mixed_fixed_estimand_real_feature_space_evidence",
        "no_neural_feature_space_increment_supported": "complete_no_neural_feature_space_increment_supported",
    }
    return mapping[canine_conclusion]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_adjudication_v2(
    frozen_result_path: Path,
    repository_root: Path,
    recovery_manifest_path: Path,
    output_root: Path,
    copied_path: Path | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise AdjudicationV2Error(f"Output directory already exists: {output_root}")

    frozen_verification = adj.verify_frozen_real_validation(
        frozen_result_path, repository_root, copied_path=copied_path
    )
    fixed_support = adj.derive_fixed_categories_authoritative(repository_root)
    recovery = load_recovery_manifest(recovery_manifest_path)
    recovery_hashes = verify_recovery_hashes(recovery, recovery_manifest_path.parent)

    # Fold integrity before any evaluation.
    fold_integrity: dict[str, Any] = {}
    for fold in FOLDS:
        canine = adj.load_canine_fold(repository_root, fold)
        scorpion = adj.load_scorpion_fold(repository_root, fold)
        canine_check = adj.fold_integrity_check(canine.frame, canine.specimen_column)
        scorpion_check = adj.fold_integrity_check(scorpion.frame, scorpion.specimen_column)
        if not canine_check["passed"] or not scorpion_check["passed"]:
            raise AdjudicationV2Error(f"Fold integrity failed in fold {fold}.")
        fold_integrity[fold] = {"canine_scc": canine_check, "scorpion": scorpion_check}

    # Deterministic baselines (identical to the original adjudication).
    baseline_results: dict[str, dict[int, dict[str, Any]]] = {
        "canine_scc": {},
        "scorpion": {},
    }
    for fold in FOLDS:
        canine = adj.load_canine_fold(repository_root, fold)
        scorpion = adj.load_scorpion_fold(repository_root, fold)
        baseline_results["canine_scc"][fold] = adj.evaluate_deterministic_methods(canine, fold)
        baseline_results["scorpion"][fold] = adj.evaluate_deterministic_methods(scorpion, fold)

    # Recovered neural evaluation for canine.
    frozen_value = json.loads(frozen_result_path.read_text(encoding="utf-8"))
    frozen_neural = adj.frozen_neural_descriptive_metrics(frozen_value)

    neural_evaluations: dict[int, dict[int, dict[str, Any]]] = {}
    reproduction_records: list[dict[str, Any]] = []
    for fold in FOLDS:
        loaded = adj.load_canine_fold(repository_root, fold)
        neural_evaluations[fold] = {}
        for seed in adj.MODEL_SEEDS:
            neural_evaluations[fold][seed] = {}
            for family in FAMILIES:
                biological, acquisition, _ = load_cell_arrays(recovery, fold, seed, family)
                if not np.isfinite(biological).all() or not np.isfinite(acquisition).all():
                    raise AdjudicationV2Error(
                        f"Recovered arrays non-finite for fold={fold} seed={seed} family={family}"
                    )
                frozen_run = adj.map_cell_to_frozen_run(
                    frozen_value,
                    {"dataset": "canine_scc", "fold": fold, "seed": seed, "family": family},
                )
                if frozen_run is None:
                    raise AdjudicationV2Error(
                        f"No frozen run for fold={fold} seed={seed} family={family}"
                    )
                evaluation = evaluate_neural_cell(
                    biological, acquisition, loaded, fold, frozen_run
                )
                if not evaluation["frozen_reproduction"]["frozen_metrics_reproduced"]:
                    raise AdjudicationV2Error(
                        f"Frozen seven-category reproduction failed for fold={fold} seed={seed} family={family}"
                    )
                reproduction_records.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "family": family,
                        "reproduced": True,
                        "metrics": evaluation["frozen_reproduction"],
                    }
                )
                neural_evaluations[fold][seed][family] = evaluation

    averaged_neural = average_neural_within_fold(neural_evaluations)

    # Build the canine metric table including the recovered neural methods.
    canine_table = adj.build_metric_table(baseline_results, "canine_scc")
    for fold in FOLDS:
        for family in FAMILIES:
            canine_table[fold][family] = {
                "category_balanced_accuracy": averaged_neural[fold][family]["category_balanced_accuracy"],
                "scanner_balanced_accuracy": averaged_neural[fold][family]["scanner_balanced_accuracy"],
                "worst_pair_retrieval": averaged_neural[fold][family]["worst_pair_retrieval"],
                "overall_retrieval": averaged_neural[fold][family]["overall_retrieval"],
            }
    scorpion_table = adj.build_metric_table(baseline_results, "scorpion")

    # Adjudication.
    canine_frontier_methods = DETERMINISTIC_METHODS + NEURAL_METHODS
    canine_pareto = {}
    for fold in FOLDS:
        methods = [
            {"method": method, **canine_table[fold][method]}
            for method in canine_frontier_methods
            if method in canine_table[fold]
        ]
        canine_pareto[fold] = adj.pareto_front(
            methods,
            ["scanner_balanced_accuracy", "category_balanced_accuracy", "worst_pair_retrieval"],
            lower_is_better=["scanner_balanced_accuracy"],
        )
    scorpion_pareto = {}
    for fold in FOLDS:
        methods = [
            {"method": method, **scorpion_table[fold][method]}
            for method in DETERMINISTIC_METHODS
            if method in scorpion_table[fold]
        ]
        scorpion_pareto[fold] = adj.pareto_front(
            methods,
            ["scanner_balanced_accuracy", "worst_pair_retrieval"],
            lower_is_better=["scanner_balanced_accuracy"],
        )

    contrasts = adj.required_canine_contrasts(canine_table, neural_available=True)
    neural_increment = neural_increment_analysis(canine_table)
    simple_dominance = simple_baseline_dominance_analysis(canine_table, scorpion_table)
    transport = synthetic_transport_decision(contrasts, neural_increment)

    axes = [
        "scanner_balanced_accuracy",
        "category_balanced_accuracy",
        "worst_pair_retrieval",
        "overall_retrieval",
    ]
    simple_axes = ["scanner_balanced_accuracy", "worst_pair_retrieval"]
    all_methods = DETERMINISTIC_METHODS + NEURAL_METHODS

    def fold_method_maps(table, method):
        return {fold: table[fold][method] for fold in FOLDS if method in table[fold]}

    dominance = {
        "canine": {
            f"{left}_vs_{right}": adj.cross_fold_material_dominance(
                fold_method_maps(canine_table, left),
                fold_method_maps(canine_table, right),
                axes,
            )
            for left in all_methods
            for right in all_methods
            if left != right
        },
        "scorpion": {
            f"{left}_vs_{right}": adj.cross_fold_material_dominance(
                fold_method_maps(scorpion_table, left),
                fold_method_maps(scorpion_table, right),
                simple_axes,
            )
            for left in DETERMINISTIC_METHODS
            for right in DETERMINISTIC_METHODS
            if left != right
        },
    }

    layer2_schema = adj.layer2_missing_metadata_schema(frozen_verification)
    claim_table = v2_claim_adjudication(contrasts, neural_increment, transport, frozen_neural)
    conclusions = v2_dataset_conclusions(neural_increment, simple_dominance)
    status = v2_top_level_status(conclusions["canine_scc"]["conclusion"])

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "git_commit": adj.git_commit(repository_root),
        "frozen_verification": frozen_verification,
        "recovery_manifest": {
            "path": str(recovery_manifest_path.resolve()),
            "status": recovery.get("status"),
            "recovery_git_commit": recovery.get("git_commit"),
            "recovery_result_sha256": recovery.get("result_sha256"),
        },
        "recovery_hash_verification": recovery_hashes,
        "fixed_category_set": adj.FIXED_CATEGORIES,
        "excluded_category_set": adj.EXCLUDED_CATEGORIES,
        "fixed_estimand": fixed_support,
        "fold_integrity": fold_integrity,
        "zero_training_verification": adj.zero_training_verification(),
        "neural_frozen_reproduction": {
            "reproduction_records": reproduction_records,
            "all_reproduced": True,
            "tolerance": FROZEN_REPRODUCTION_TOLERANCE,
        },
        "neural_corrected_estimand": {
            "available": True,
            "averaged_neural_metrics": averaged_neural,
            "per_seed_evaluations": {
                fold: {
                    seed: {
                        family: {
                            "corrected_category_balanced_accuracy": neural_evaluations[fold][seed][family]["corrected_category"]["balanced_accuracy"],
                            "scanner_balanced_accuracy": neural_evaluations[fold][seed][family]["scanner"]["linear_balanced_accuracy"],
                            "worst_pair_retrieval": neural_evaluations[fold][seed][family]["retrieval"]["worst_ordered_scanner_pair_top1"],
                            "overall_retrieval": neural_evaluations[fold][seed][family]["retrieval"]["overall_top1"],
                        }
                        for family in FAMILIES
                    }
                    for seed in adj.MODEL_SEEDS
                }
                for fold in FOLDS
            },
        },
        "exploratory_seven_category_endpoint": {
            "endpoint": "frozen seven-category balanced accuracy (biological_category_accessibility)",
            "separate_from_fixed_estimand": True,
            "metrics": frozen_neural,
        },
        "deterministic_baselines": baseline_results,
        "canine_metric_table": canine_table,
        "scorpion_metric_table": scorpion_table,
        "canine_pareto_fronts": canine_pareto,
        "scorpion_pareto_fronts": scorpion_pareto,
        "dominance": dominance,
        "contrasts": contrasts,
        "neural_feature_space_increment": neural_increment,
        "simple_baseline_dominance": simple_dominance,
        "synthetic_transport_decision": transport,
        "layer2_gap_schema": layer2_schema,
        "dataset_conclusions": conclusions,
        "claim_adjudication": claim_table,
        "status": status,
        "failure_reasons": [],
    }
    adj.verify_inputs_unchanged(frozen_verification["frozen_input_hashes"])
    return result


def summary_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    table = result.get("canine_metric_table", {})
    for fold in FOLDS:
        for method, metrics in table.get(fold, {}).items():
            rows.append(
                {
                    "row_type": "canine_metric",
                    "fold": fold,
                    "method": method,
                    "category_balanced_accuracy": metrics.get("category_balanced_accuracy"),
                    "scanner_balanced_accuracy": metrics.get("scanner_balanced_accuracy"),
                    "worst_pair_retrieval": metrics.get("worst_pair_retrieval"),
                    "overall_retrieval": metrics.get("overall_retrieval"),
                }
            )
    for item in result.get("claim_adjudication", []):
        rows.append({"row_type": "claim", "claim_id": item["id"], "verdict": item["verdict"]})
    for dataset, conclusion in result.get("dataset_conclusions", {}).items():
        rows.append({"row_type": "dataset_conclusion", "dataset": dataset, **conclusion})
    rows.append({"row_type": "top_level", "status": result["status"]})
    return rows


def write_outputs(output_root: Path, result: Mapping[str, Any]) -> None:
    result["result_sha256"] = canonical_hash(result)
    result_path = output_root / "fixed_estimand_real_feature_space_adjudication_v2_result.json"
    summary_path = output_root / "fixed_estimand_real_feature_space_adjudication_v2_summary.csv"
    manifest_path = output_root / "fixed_estimand_real_feature_space_adjudication_v2_manifest.json"
    schema_path = output_root / "fixed_estimand_layer2_missing_metadata_schema_v2.json"
    atomic_json(result_path, result)
    atomic_csv(summary_path, summary_rows(result))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "git_commit": result["git_commit"],
        "status": result["status"],
        "canonical_internal_result_hash": result["result_sha256"],
        "frozen_verification": result["frozen_verification"],
        "recovery_manifest": result["recovery_manifest"],
        "neural_frozen_reproduction": result["neural_frozen_reproduction"],
        "artifacts": {
            "result": result_path.name,
            "summary": summary_path.name,
            "layer2_schema": schema_path.name,
            "manifest": manifest_path.name,
        },
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    atomic_json(manifest_path, manifest)
    atomic_json(schema_path, result["layer2_gap_schema"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-validation-result", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--recovery-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--copied-result",
        type=Path,
        default=None,
        help="Optional copied frozen result path to verify alongside the repository path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copied = (
        Path(adj.FROZEN_RESULT_COPY_PATH)
        if args.copied_result is None
        else args.copied_result
    )
    started = time.time()
    result = run_adjudication_v2(
        args.real_validation_result.resolve(),
        args.repository_root.resolve(),
        args.recovery_manifest.resolve(),
        args.output_root.resolve(),
        copied_path=copied,
    )
    write_outputs(args.output_root.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "neural_cells_verified": result["recovery_hash_verification"]["verified_cells"],
                "result_sha256": result["result_sha256"],
                "output_root": str(args.output_root.resolve()),
                "runtime_seconds": time.time() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
