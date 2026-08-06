#!/usr/bin/env python3
"""Calibrate only nonlinear capacity of the frozen v1 residual probe."""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from experiments.paired_acquisition import (
    run_crossed_target_scanner_prototype_factorization as csv_support,
)
from experiments.paired_acquisition import (
    run_representation_geometry_instrument_calibration as v1,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as base,
)


SCHEMA_VERSION = "paired-acquisition-residual-probe-nonlinear-capacity-calibration/v2"
V1_FILE_SHA256 = "72b00083d789141c9abde67986a36765c8c9127ec49981e4ab0edeecb8f2d634"
V1_INTERNAL_SHA256 = "a571e854df12f4c68053576abbf5b90b28ade51fa7c7cab45e9a4dfe84529225"
PRIMARY_SHA256 = v1.PRIMARY_SHA256
GEOMETRY_SHA256 = v1.FAILED_FILE_SHA256
TRAIN_IDENTITIES = 512
TEST_IDENTITIES = 256
SCANNERS = 5
INPUT_DIM = 8
OUTPUT_DIM = 8
GENERATION_SEEDS = {
    "ridge_preservation": 9101,
    "teacher_in_hypothesis_class": 9102,
    "analytic_interaction": 9103,
    "permuted_target": 9104,
}
PROBE_SEEDS = (7203, 7204)
TEACHER_PARAMETER_SEED = 9151
TEACHER_HIDDEN_WIDTH = 8
TEACHER_ALPHA = 0.75
INTERACTION_BETA = 0.50
TARGET_NOISE_STD = 0.002
NORMALIZATION_SAMPLE_SIZE = 65_536
INTERACTION_PAIRS = tuple((index, (index + 1) % INPUT_DIM) for index in range(OUTPUT_DIM))
TIGHT_TOLERANCE = 1e-7


class ExperimentError(v1.ExperimentError):
    """Raised when residual-capacity calibration cannot proceed safely."""


@dataclass(frozen=True)
class CapacityDataset:
    features: np.ndarray
    targets: np.ndarray
    identity_ids: np.ndarray
    scanner_ids: np.ndarray
    split: Any
    control: str
    generation_seed: int
    metadata: Mapping[str, Any]


class FrozenResidualTeacher(nn.Module):
    """A deterministic two-hidden-layer GELU teacher that is never trained."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_width: int = TEACHER_HIDDEN_WIDTH,
        output_dim: int = OUTPUT_DIM,
        seed: int = TEACHER_PARAMETER_SEED,
    ) -> None:
        super().__init__()
        base.set_deterministic_seed(seed)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, output_dim),
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return base.sha256_bytes(np.ascontiguousarray(values).tobytes())


def teacher_parameter_sha256(teacher: FrozenResidualTeacher) -> str:
    pieces = []
    for name, value in sorted(teacher.state_dict().items()):
        pieces.append(name.encode("utf-8"))
        pieces.append(value.detach().cpu().numpy().astype("<f4").tobytes())
    return base.sha256_bytes(b"".join(pieces))


def ensure_new_output_root(output_root: Path) -> None:
    if output_root.exists():
        raise ExperimentError(
            "Output root already exists; overwrite is prohibited: {}".format(
                output_root
            )
        )
    output_root.mkdir(parents=True, exist_ok=False)


def verify_inherited_v1(v1_path: Path) -> Dict[str, Any]:
    if not v1_path.is_file():
        raise ExperimentError("Frozen v1 calibration result is missing.")
    v1_file_hash = _sha256_file(v1_path)
    if v1_file_hash != V1_FILE_SHA256:
        raise ExperimentError("Frozen v1 calibration file SHA-256 does not match.")
    payload = base.json.loads(v1_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != v1.SCHEMA_VERSION:
        raise ExperimentError("Frozen v1 calibration schema does not match.")
    if payload.get("status") != "regression_probe_calibration_failed":
        raise ExperimentError("Frozen v1 calibration status was reinterpreted.")
    if payload.get("result_sha256") != V1_INTERNAL_SHA256:
        raise ExperimentError("Frozen v1 internal SHA-256 does not match.")
    frozen = payload.get("frozen_inputs", {})
    primary_path = Path(frozen.get("primary_reference_path", ""))
    geometry_path = Path(frozen.get("failed_diagnostic_path", ""))
    if _sha256_file(primary_path) != PRIMARY_SHA256:
        raise ExperimentError("Frozen primary input hash does not match.")
    if _sha256_file(geometry_path) != GEOMETRY_SHA256:
        raise ExperimentError("Frozen geometry input hash does not match.")
    scanner_passed = bool(payload["scanner_classifier_controls"]["passed"])
    decoder_passed = bool(payload["true_factor_decoder_controls"]["passed"])
    oracle = payload["oracle_representation_calibration"]
    oracle_preserved = bool(
        oracle["flags"]["all_ridge_positive_oracles_preserved"]
    )
    oracle_runs = list(oracle["runs"])
    zero_crossed_target_fits = bool(
        len(oracle_runs) == 8
        and all(run.get("model_family") == "oracle_supervised" for run in oracle_runs)
        and payload["claim_scope"].get("does_not_test_crossed_target_representation")
    )
    inherited_verified = bool(
        scanner_passed
        and decoder_passed
        and oracle_preserved
        and zero_crossed_target_fits
    )
    return {
        "payload": payload,
        "source_path": str(v1_path.resolve()),
        "source_file_sha256": v1_file_hash,
        "source_internal_sha256": payload["result_sha256"],
        "primary_reference_path": str(primary_path.resolve()),
        "primary_reference_sha256": PRIMARY_SHA256,
        "failed_geometry_path": str(geometry_path.resolve()),
        "failed_geometry_sha256": GEOMETRY_SHA256,
        "v1_status": payload["status"],
        "scanner_calibration_passed": scanner_passed,
        "true_factor_decoder_calibration_passed": decoder_passed,
        "oracle_ridge_positive_representations_preserved": oracle_preserved,
        "oracle_run_count": len(oracle_runs),
        "crossed_target_fit_count": 0,
        "zero_crossed_target_fits_verified": zero_crossed_target_fits,
        "inherited_evidence_verified": inherited_verified,
    }


def _full_rank_linear_map(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.normal(size=(INPUT_DIM, OUTPUT_DIM)))
    right, _ = np.linalg.qr(rng.normal(size=(OUTPUT_DIM, OUTPUT_DIM)))
    singular = np.linspace(0.8, 1.2, OUTPUT_DIM)
    return (left @ np.diag(singular) @ right.T).astype(np.float64)


def _teacher_outputs(teacher: FrozenResidualTeacher, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        values = teacher(torch.as_tensor(features, dtype=torch.float32)).numpy()
    return values.astype(np.float64)


def _identity_features(seed: int, train_count: int, test_count: int) -> np.ndarray:
    training_rng = np.random.default_rng(seed + 100_000)
    test_rng = np.random.default_rng(seed + 200_000)
    training = training_rng.normal(size=(train_count, INPUT_DIM))
    test = test_rng.normal(size=(test_count, INPUT_DIM))
    return np.concatenate([training, test], axis=0).astype(np.float64)


def _repeat_by_scanner(values: np.ndarray, scanners: int) -> np.ndarray:
    return np.repeat(values, scanners, axis=0)


def analytic_interaction_values(features: np.ndarray) -> np.ndarray:
    """Return the frozen pairwise products used by analytic control C."""
    return np.stack(
        [features[:, left] * features[:, right] for left, right in INTERACTION_PAIRS],
        axis=1,
    )


def make_capacity_dataset(
    control: str,
    generation_seed: int,
    train_identities: int = TRAIN_IDENTITIES,
    test_identities: int = TEST_IDENTITIES,
    scanners: int = SCANNERS,
) -> CapacityDataset:
    if control not in {
        "ridge_preservation",
        "teacher_in_hypothesis_class",
        "analytic_interaction",
        "permuted_target",
    }:
        raise ExperimentError("Unknown v2 capacity control: {}".format(control))
    identity_features = _identity_features(
        generation_seed, train_identities, test_identities
    )
    identity_count = train_identities + test_identities
    identity_ids = np.repeat(np.arange(identity_count, dtype=np.int64), scanners)
    scanner_ids = np.tile(np.arange(scanners, dtype=np.int64), identity_count)
    metadata: Dict[str, Any] = {}
    if control == "ridge_preservation":
        identity_targets = identity_features.copy()
    elif control == "teacher_in_hypothesis_class":
        teacher = FrozenResidualTeacher()
        linear_map = _full_rank_linear_map(generation_seed + 1)
        normalization_rng = np.random.default_rng(generation_seed + 300_000)
        normalization_features = normalization_rng.normal(
            size=(NORMALIZATION_SAMPLE_SIZE, INPUT_DIM)
        )
        reference_linear = normalization_features @ linear_map
        reference_residual = _teacher_outputs(teacher, normalization_features)
        linear_mean = reference_linear.mean(axis=0)
        linear_std = np.maximum(reference_linear.std(axis=0), 1e-8)
        residual_mean = reference_residual.mean(axis=0)
        residual_std = np.maximum(reference_residual.std(axis=0), 1e-8)
        linear_component = (identity_features @ linear_map - linear_mean) / linear_std
        residual_component = (
            _teacher_outputs(teacher, identity_features) - residual_mean
        ) / residual_std
        identity_targets = linear_component + TEACHER_ALPHA * residual_component
        noise_rng = np.random.default_rng(generation_seed + 400_000)
        identity_targets += noise_rng.normal(
            scale=TARGET_NOISE_STD, size=identity_targets.shape
        )
        metadata = {
            "teacher_parameter_seed": TEACHER_PARAMETER_SEED,
            "teacher_hidden_width": TEACHER_HIDDEN_WIDTH,
            "teacher_parameter_sha256": teacher_parameter_sha256(teacher),
            "teacher_parameters_frozen": all(
                not parameter.requires_grad for parameter in teacher.parameters()
            ),
            "alpha": TEACHER_ALPHA,
            "linear_map_sha256": _sha256_array(linear_map.astype("<f8")),
            "normalization_sample_size": NORMALIZATION_SAMPLE_SIZE,
            "linear_component_mean_sha256": _sha256_array(
                linear_mean.astype("<f8")
            ),
            "linear_component_std_sha256": _sha256_array(
                linear_std.astype("<f8")
            ),
            "residual_component_mean_sha256": _sha256_array(
                residual_mean.astype("<f8")
            ),
            "residual_component_std_sha256": _sha256_array(
                residual_std.astype("<f8")
            ),
            "target_noise_std": TARGET_NOISE_STD,
        }
    elif control == "analytic_interaction":
        linear_map = _full_rank_linear_map(generation_seed + 1)
        linear_component = identity_features @ linear_map
        interactions = analytic_interaction_values(identity_features)
        identity_targets = linear_component + INTERACTION_BETA * interactions
        noise_rng = np.random.default_rng(generation_seed + 400_000)
        identity_targets += noise_rng.normal(
            scale=TARGET_NOISE_STD, size=identity_targets.shape
        )
        metadata = {
            "interaction_pairs": [list(pair) for pair in INTERACTION_PAIRS],
            "beta": INTERACTION_BETA,
            "linear_map_sha256": _sha256_array(linear_map.astype("<f8")),
            "target_noise_std": TARGET_NOISE_STD,
        }
    else:
        identity_targets = identity_features.copy()
        rng = np.random.default_rng(generation_seed + 500_000)
        permutation = rng.permutation(identity_count)
        while np.any(permutation == np.arange(identity_count)):
            permutation = rng.permutation(identity_count)
        identity_targets = identity_targets[permutation]
        metadata = {
            "identity_permutation": permutation.tolist(),
            "identity_permutation_sha256": _sha256_array(
                permutation.astype("<i8")
            ),
        }
    features = _repeat_by_scanner(identity_features, scanners).astype(np.float32)
    targets = _repeat_by_scanner(identity_targets, scanners).astype(np.float32)
    split = v1.make_identity_split(
        identity_ids,
        scanner_ids,
        train_identities,
        split_seed=generation_seed + 50_000,
    )
    return CapacityDataset(
        features=features,
        targets=targets,
        identity_ids=identity_ids,
        scanner_ids=scanner_ids,
        split=split,
        control=control,
        generation_seed=generation_seed,
        metadata=metadata,
    )


def dataset_manifest(dataset: CapacityDataset) -> Dict[str, Any]:
    return {
        "control": dataset.control,
        "generation_seed": dataset.generation_seed,
        "feature_sha256": _sha256_array(dataset.features.astype("<f4")),
        "target_sha256": _sha256_array(dataset.targets.astype("<f4")),
        "identity_sha256": _sha256_array(dataset.identity_ids.astype("<i8")),
        "scanner_sha256": _sha256_array(dataset.scanner_ids.astype("<i8")),
        "identity_split": v1.geometry.split_manifest(dataset.split),
        "metadata": dict(dataset.metadata),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_flat.std() < 1e-12 or right_flat.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def run_probe(
    dataset: CapacityDataset,
    probe_seed: int,
    config: v1.ResidualConfig,
) -> Dict[str, Any]:
    result, fit = v1.residual_probe_result(
        dataset.features,
        dataset.targets,
        dataset.split,
        probe_seed,
        config,
    )
    train_metrics = v1.evaluate_residual_fit(
        fit,
        dataset.features,
        dataset.targets,
        dataset.split.probe_training_indices,
    )
    test = dataset.split.unseen_test_indices
    ridge_prediction = fit.ridge_predict(dataset.features[test])
    residual_prediction = fit.predict(dataset.features[test])
    residual_values = residual_prediction - ridge_prediction
    residual_targets = dataset.targets[test] - ridge_prediction
    selected = result["selected_epoch"]
    selected_history = next(
        row for row in result["history"] if int(row["epoch"]) == selected
    )
    result["optimization_diagnostics"] = {
        "epoch_zero_validation_loss": result["history"][0]["validation_loss"],
        "best_validation_loss": selected_history["validation_loss"],
        "validation_loss_improvement_from_epoch_zero": (
            result["history"][0]["validation_loss"]
            - selected_history["validation_loss"]
        ),
        "final_training_loss": result["history"][-1]["train_loss"],
        "train_r2": train_metrics["residual_r2"],
        "validation_r2": result["validation"]["residual_r2"],
        "unseen_test_r2": result["unseen_test"]["residual_r2"],
        "residual_prediction_variance": float(np.var(residual_values)),
        "residual_target_variance": float(np.var(residual_targets)),
        "residual_target_correlation": _correlation(
            residual_values, residual_targets
        ),
        "gradient_norms": {
            "recorded": False,
            "reason": "The frozen v1 probe API does not expose per-epoch gradients.",
        },
    }
    return result


def control_flags(
    ridge: Mapping[str, Any],
    teacher_repeats: Sequence[Mapping[str, Any]],
    interaction_repeats: Sequence[Mapping[str, Any]],
    negative: Mapping[str, Any],
) -> Dict[str, Any]:
    ridge_test = ridge["unseen_test"]
    ridge_flags = {
        "ridge_test_r2_at_least_0_95": ridge_test["ridge_r2"] >= 0.95,
        "residual_no_worse_than_ridge": (
            ridge_test["residual_r2"] >= ridge_test["ridge_r2"] - TIGHT_TOLERANCE
        ),
        "epoch_zero_exactly_equals_ridge": (
            ridge["epoch_zero_max_abs_difference"] <= TIGHT_TOLERANCE
        ),
    }
    teacher_seed_flags = []
    for repeat in teacher_repeats:
        teacher_seed_flags.append(
            {
                "seed": repeat["seed"],
                "checkpoint_after_epoch_zero": repeat["selected_epoch"] > 0,
                "validation_improvement_positive": (
                    repeat["validation"]["residual_minus_ridge_mse"] < 0.0
                ),
                "test_r2_improvement_at_least_0_05": (
                    repeat["unseen_test"]["residual_minus_ridge_r2"] >= 0.05
                ),
                "test_residual_r2_at_least_0_90": (
                    repeat["unseen_test"]["residual_r2"] >= 0.90
                ),
            }
        )
    interaction_seed_flags = []
    for repeat in interaction_repeats:
        interaction_seed_flags.append(
            {
                "seed": repeat["seed"],
                "validation_improvement_positive": (
                    repeat["validation"]["residual_minus_ridge_mse"] < 0.0
                ),
                "test_r2_improvement_at_least_0_03": (
                    repeat["unseen_test"]["residual_minus_ridge_r2"] >= 0.03
                ),
                "test_residual_r2_at_least_0_85": (
                    repeat["unseen_test"]["residual_r2"] >= 0.85
                ),
            }
        )
    negative_flags = {
        "ridge_r2_below_0_80": negative["unseen_test"]["ridge_r2"] < 0.80,
        "residual_r2_below_0_80": (
            negative["unseen_test"]["residual_r2"] < 0.80
        ),
    }
    return {
        "ridge_preservation": ridge_flags,
        "ridge_preservation_passed": all(ridge_flags.values()),
        "teacher_in_class_by_seed": teacher_seed_flags,
        "teacher_in_class_passed": all(
            all(value for key, value in row.items() if key != "seed")
            for row in teacher_seed_flags
        ),
        "analytic_interaction_by_seed": interaction_seed_flags,
        "analytic_interaction_passed": all(
            all(value for key, value in row.items() if key != "seed")
            for row in interaction_seed_flags
        ),
        "negative_control": negative_flags,
        "negative_control_passed": all(negative_flags.values()),
    }


def aggregate_status(
    flags: Mapping[str, Any],
    inherited_verified: bool,
    hashes_unchanged: bool,
) -> Dict[str, Any]:
    failures = []
    if not flags["ridge_preservation_passed"]:
        failures.append("Ridge preservation control failed")
    if not flags["teacher_in_class_passed"]:
        failures.append("teacher-in-hypothesis-class nonlinear control failed")
    if not flags["analytic_interaction_passed"]:
        failures.append("analytic interaction nonlinear control failed")
    if not flags["negative_control_passed"]:
        failures.append("permuted-target negative control failed")
    if not inherited_verified:
        failures.append("inherited v1 evidence verification failed")
    if not hashes_unchanged:
        failures.append("a frozen artifact hash changed")
    if not hashes_unchanged:
        status = "residual_capacity_calibration_failed"
    elif not inherited_verified:
        status = "inherited_evidence_verification_failed"
    elif not flags["ridge_preservation_passed"]:
        status = "ridge_preservation_failed"
    elif not flags["teacher_in_class_passed"]:
        status = "teacher_residual_control_failed"
    elif not flags["analytic_interaction_passed"]:
        status = "analytic_interaction_control_failed"
    elif not flags["negative_control_passed"]:
        status = "negative_control_failed"
    else:
        status = "complete_residual_nonlinear_capacity_calibrated"
    composite = (
        "complete_instrument_families_calibrated_v2"
        if status == "complete_residual_nonlinear_capacity_calibrated"
        and inherited_verified
        and hashes_unchanged
        else "instrument_family_adjudication_not_complete"
    )
    return {
        "status": status,
        "failure_reasons": failures,
        "instrument_family_adjudication": composite,
        "post_hoc_composite_across_immutable_v1_and_v2": True,
        "does_not_replace_or_edit_source_results": True,
        "primary_unseen_identity_gate_remains_closed": True,
        "v1_calibration_status_remains_failed": True,
    }


def summary_rows(controls: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    rows.append(
        {
            "control": "ridge_preservation",
            "probe_seed": controls["ridge_preservation"]["seed"],
            "selected_epoch": controls["ridge_preservation"]["selected_epoch"],
            "ridge_test_r2": controls["ridge_preservation"]["unseen_test"][
                "ridge_r2"
            ],
            "residual_test_r2": controls["ridge_preservation"]["unseen_test"][
                "residual_r2"
            ],
        }
    )
    for name in ("teacher_in_hypothesis_class", "analytic_interaction"):
        for repeat in controls[name]["probe_repeats"]:
            rows.append(
                {
                    "control": name,
                    "probe_seed": repeat["seed"],
                    "selected_epoch": repeat["selected_epoch"],
                    "ridge_validation_r2": repeat["validation"]["ridge_r2"],
                    "residual_validation_r2": repeat["validation"]["residual_r2"],
                    "ridge_test_r2": repeat["unseen_test"]["ridge_r2"],
                    "residual_test_r2": repeat["unseen_test"]["residual_r2"],
                    "test_r2_improvement": repeat["unseen_test"][
                        "residual_minus_ridge_r2"
                    ],
                }
            )
    rows.append(
        {
            "control": "permuted_target",
            "probe_seed": controls["permuted_target"]["seed"],
            "selected_epoch": controls["permuted_target"]["selected_epoch"],
            "ridge_test_r2": controls["permuted_target"]["unseen_test"][
                "ridge_r2"
            ],
            "residual_test_r2": controls["permuted_target"]["unseen_test"][
                "residual_r2"
            ],
        }
    )
    return rows


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def run_calibration(
    v1_calibration_result: Path,
    output_root: Path,
    probe_config: v1.ResidualConfig | None = None,
) -> Dict[str, Any]:
    probe_config = probe_config or v1.ResidualConfig()
    inherited = verify_inherited_v1(v1_calibration_result)
    ensure_new_output_root(output_root)
    frozen_paths = (
        Path(inherited["primary_reference_path"]),
        Path(inherited["failed_geometry_path"]),
        v1_calibration_result,
    )
    before_hashes = [_sha256_file(path) for path in frozen_paths]
    datasets = {
        name: make_capacity_dataset(name, seed)
        for name, seed in GENERATION_SEEDS.items()
    }
    print("[v2 calibration] Ridge preservation", flush=True)
    ridge = run_probe(datasets["ridge_preservation"], 7201, probe_config)
    print("[v2 calibration] teacher-in-hypothesis-class residual", flush=True)
    teacher_repeats = [
        run_probe(
            datasets["teacher_in_hypothesis_class"], seed, probe_config
        )
        for seed in PROBE_SEEDS
    ]
    print("[v2 calibration] analytic interaction residual", flush=True)
    interaction_repeats = [
        run_probe(datasets["analytic_interaction"], seed, probe_config)
        for seed in PROBE_SEEDS
    ]
    print("[v2 calibration] permuted-target negative control", flush=True)
    negative = run_probe(datasets["permuted_target"], 7205, probe_config)
    controls = {
        "ridge_preservation": ridge,
        "teacher_in_hypothesis_class": {
            "probe_repeats": teacher_repeats,
        },
        "analytic_interaction": {
            "probe_repeats": interaction_repeats,
        },
        "permuted_target": negative,
    }
    flags = control_flags(ridge, teacher_repeats, interaction_repeats, negative)
    after_hashes = [_sha256_file(path) for path in frozen_paths]
    hashes_unchanged = before_hashes == after_hashes
    aggregate = aggregate_status(
        flags,
        inherited["inherited_evidence_verified"],
        hashes_unchanged,
    )
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": aggregate["status"],
        "claim_scope": {
            "residual_regression_instrument_capacity_only": True,
            "no_factorizer_initialized_or_trained": True,
            "crossed_target_fit_count": 0,
            "oracle_fit_count": 0,
            "does_not_validate_crossed_target_representation": True,
            "does_not_reinterpret_v1_calibration": True,
            "synthetic_evidence_only": True,
            "not_pathology_domain_validation": True,
        },
        "git_commit": _git_commit(),
        "frozen_inputs": {
            "v1_calibration_result_path": str(v1_calibration_result.resolve()),
            "v1_calibration_file_sha256": inherited["source_file_sha256"],
            "v1_calibration_internal_sha256": inherited["source_internal_sha256"],
            "primary_reference_path": inherited["primary_reference_path"],
            "primary_reference_sha256": inherited["primary_reference_sha256"],
            "failed_geometry_path": inherited["failed_geometry_path"],
            "failed_geometry_sha256": inherited["failed_geometry_sha256"],
            "all_hashes_unchanged_after_execution": hashes_unchanged,
        },
        "inherited_v1_evidence": {
            key: value for key, value in inherited.items() if key != "payload"
        },
        "unresolved_v1_nonlinear_control": {
            "transformation": "x = tanh(1.25 b) Q + epsilon",
            "classification": "unresolved_nonlinear_inversion_challenge",
            "standalone_capacity_positive_control": False,
            "v1_status_changed": False,
            "v1_ridge_r2_approximate": 0.869,
            "v1_residual_r2_approximate": 0.868,
        },
        "probe_implementation": {
            "imported_from": (
                "run_representation_geometry_instrument_calibration.fit_residual_regressor"
            ),
            "implementation_reused_unchanged": True,
            "configuration": asdict(probe_config),
        },
        "fixed_generation_configuration": {
            "training_identities": TRAIN_IDENTITIES,
            "unseen_test_identities": TEST_IDENTITIES,
            "scanners": SCANNERS,
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "generation_seeds": GENERATION_SEEDS,
            "probe_seeds": list(PROBE_SEEDS),
            "teacher_parameter_seed": TEACHER_PARAMETER_SEED,
            "teacher_hidden_width": TEACHER_HIDDEN_WIDTH,
            "teacher_alpha": TEACHER_ALPHA,
            "interaction_beta": INTERACTION_BETA,
            "interaction_pairs": [list(pair) for pair in INTERACTION_PAIRS],
            "target_noise_std": TARGET_NOISE_STD,
            "normalization_sample_size": NORMALIZATION_SAMPLE_SIZE,
        },
        "dataset_manifests": {
            name: dataset_manifest(dataset) for name, dataset in datasets.items()
        },
        "controls": controls,
        "control_flags": flags,
        "aggregate_interpretation": aggregate,
        "instrument_family_adjudication": aggregate[
            "instrument_family_adjudication"
        ],
    }
    result["result_sha256"] = base.sha256_bytes(base.canonical_json_bytes(result))
    rows = summary_rows(controls)
    result_path = output_root / "residual_probe_nonlinear_capacity_calibration_v2_result.json"
    summary_path = output_root / "residual_probe_nonlinear_capacity_calibration_v2_summary.csv"
    manifest_path = output_root / "residual_probe_nonlinear_capacity_calibration_v2_manifest.json"
    base.atomic_json(result_path, result)
    csv_support.atomic_csv(
        summary_path,
        csv_support.summary_csv_fieldnames(rows),
        rows,
    )
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": result["claim_scope"],
        "git_commit": result["git_commit"],
        "frozen_inputs": result["frozen_inputs"],
        "inherited_v1_evidence": result["inherited_v1_evidence"],
        "probe_implementation": result["probe_implementation"],
        "fixed_generation_configuration": result[
            "fixed_generation_configuration"
        ],
        "dataset_manifests": result["dataset_manifests"],
        "status": result["status"],
        "instrument_family_adjudication": result[
            "instrument_family_adjudication"
        ],
        "artifacts": {
            "result": result_path.name,
            "summary": summary_path.name,
            "manifest": manifest_path.name,
        },
        "canonical_internal_result_hash": result["result_sha256"],
    }
    manifest["manifest_sha256"] = base.sha256_bytes(
        base.canonical_json_bytes(manifest)
    )
    base.atomic_json(manifest_path, manifest)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-calibration-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_calibration(
        args.v1_calibration_result.resolve(), args.output_root.resolve()
    )
    print(
        base.json.dumps(
            {
                "status": result["status"],
                "instrument_family_adjudication": result[
                    "instrument_family_adjudication"
                ],
                "result_sha256": result["result_sha256"],
                "output_root": str(args.output_root.resolve()),
                "factorizer_fit_count": 0,
                "crossed_target_fit_count": 0,
                "oracle_fit_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(
            "RESIDUAL-PROBE NONLINEAR-CAPACITY CALIBRATION V2 FAILED: {}".format(
                exc
            )
        ) from exc
