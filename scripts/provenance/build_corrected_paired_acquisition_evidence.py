#!/usr/bin/env python3
"""Build the tracked corrected paired-acquisition evidence release."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.provenance.validate_corrected_paired_acquisition_evidence import (
    SCHEMA_VERSION,
    sha256_bytes,
    sha256_file,
    validate_evidence,
)

PACKAGE_ROOT = REPO_ROOT / "evidence/paired_acquisition/corrected-20260726"
SOURCE_COMMIT = "ad03723b7967b7d08f2d43f3814bf36d69187587"
EQUIVALENT_EXECUTION_COMMIT = "1c6e3197216fd29f4f0fa3db4daf4ae124f2f6f6"
CANINE_SOURCE = REPO_ROOT / (
    "results/paired_acquisition_factorization_" "biological_label_preservation_fixed_estimand"
)
SCORPION_SOURCE = REPO_ROOT / "results/paired_acquisition_factorization_scorpion_fold_aware_v2"


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def path_record(
    path: Path,
    *,
    base: Path,
    role: str,
    include_row_count: bool = False,
) -> dict[str, Any]:
    path = path.resolve()
    record: dict[str, Any] = {
        "path": path.relative_to(base.resolve()).as_posix(),
        "role": role,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if include_row_count:
        record["row_count"] = csv_row_count(path)
    return record


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        stderr=subprocess.STDOUT,
    ).strip()


def source_script_record(path: str, role: str) -> dict[str, Any]:
    content = subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"],
        cwd=REPO_ROOT,
        stderr=subprocess.STDOUT,
    )
    return {
        "path": path,
        "role": role,
        "sha256": sha256_bytes(content),
        "size_bytes": len(content),
    }


def copy_promoted(source: Path, relative_destination: str) -> Path:
    destination = PACKAGE_ROOT / relative_destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_text(encoding="utf-8")
    canonical = content.replace("\r\n", "\n").replace("\r", "\n")
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical)
    return destination


def external_record(path: str, role: str) -> dict[str, Any]:
    source = REPO_ROOT / path
    return path_record(
        source,
        base=REPO_ROOT,
        role=role,
        include_row_count=source.suffix == ".csv",
    )


def canine_inputs() -> list[dict[str, Any]]:
    records = [
        external_record(
            "results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz",
            "base_feature_archive",
        )
    ]
    for fold in range(5):
        records.append(
            external_record(
                "data/external_multiscanner_caninescc/patch_manifests/splits/"
                f"fold_{fold}_patch_manifest.csv",
                f"fold_{fold}_split_manifest",
            )
        )
        for condition in ("true_pairs", "shuffled_sample_pairs"):
            for seed in range(911, 916):
                records.append(
                    external_record(
                        "results/paired_acquisition_factorization_pair_integrity_"
                        f"caninescc/fold_{fold}/runs/{condition}_seed_{seed}/"
                        "projected_features.npz",
                        f"fold_{fold}_{condition}_seed_{seed}_projected_features",
                    )
                )
    return records


def key_canine_metrics(summary_path: Path) -> dict[str, dict[str, float]]:
    selected = {
        "linear_projection_k4",
        "original_frozen_features",
        "true_pair_acquisition",
        "true_pair_biological",
    }
    result: dict[str, dict[str, float]] = {}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            representation = row["representation"]
            if representation in selected:
                result[representation] = {
                    "category_probe_balanced_accuracy": float(
                        row["category_probe_balanced_accuracy_mean"]
                    ),
                    "purity_fit_pool_k5": float(row["purity_fit_pool_k5_mean"]),
                    "scanner_probe_balanced_accuracy": float(
                        row["scanner_probe_balanced_accuracy_mean"]
                    ),
                }
    if set(result) != selected:
        raise RuntimeError("missing expected canine key metrics")
    return result


def key_scorpion_metrics(summary_path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["metric"]] = {
                "cluster_bootstrap_ci_025": float(row["cluster_bootstrap_ci_025"]),
                "cluster_bootstrap_ci_975": float(row["cluster_bootstrap_ci_975"]),
                "mean_difference": float(row["mean_difference"]),
            }
    return result


def environment_record() -> dict[str, Any]:
    package_versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy"):
        package_versions[package] = importlib.metadata.version(package)
    return {
        "accelerator": "not used by these analysis scripts",
        "captured_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "environment_scope": (
            "Artifact-validation and publication environment. The completed "
            "analyses did not emit a separate immutable pre-run environment record."
        ),
        "git": git_bytes("--version").decode(),
        "packages": package_versions,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "schema_version": "corrected-paired-acquisition-environment/v1",
    }


def main() -> None:
    if not (PACKAGE_ROOT / "README.md").is_file():
        raise RuntimeError("evidence package README must exist before building")
    source_tree = git_bytes("rev-parse", f"{SOURCE_COMMIT}^{{tree}}").decode()
    execution_tree = git_bytes("rev-parse", f"{EQUIVALENT_EXECUTION_COMMIT}^{{tree}}").decode()
    if source_tree != execution_tree:
        raise RuntimeError("source and execution commits are not tree-equivalent")

    canine_promoted_paths = {
        "experiment_design": copy_promoted(
            CANINE_SOURCE / "experiment_design.json",
            "canine/experiment_design.json",
        ),
        "five_fold_descriptive_summary": copy_promoted(
            CANINE_SOURCE / "five_fold_descriptive_summary.csv",
            "canine/five_fold_descriptive_summary.csv",
        ),
        "fixed_category_support": copy_promoted(
            CANINE_SOURCE / "fixed_category_support.csv",
            "canine/fixed_category_support.csv",
        ),
        "fold_seed_averaged_metrics": copy_promoted(
            CANINE_SOURCE / "fold_seed_averaged_metrics.csv",
            "canine/fold_seed_averaged_metrics.csv",
        ),
    }
    scorpion_promoted_paths = {
        "analysis_design": copy_promoted(
            SCORPION_SOURCE / "analysis_design.json",
            "scorpion/analysis_design.json",
        ),
        "fold_aware_contrasts": copy_promoted(
            SCORPION_SOURCE / "fold_aware_contrasts.csv",
            "scorpion/fold_aware_contrasts.csv",
        ),
    }
    claim_snapshot = copy_promoted(
        REPO_ROOT / "CLAIM_BOUNDARY.md",
        "claim_boundary_snapshot.md",
    )
    environment_path = PACKAGE_ROOT / "environment.json"
    write_json(environment_path, environment_record())

    canine_outputs = [
        external_record(
            (
                "results/paired_acquisition_factorization_"
                "biological_label_preservation_fixed_estimand/"
                f"{name}"
            ),
            role,
        )
        for role, name in (
            ("experiment_design", "experiment_design.json"),
            ("five_fold_descriptive_summary", "five_fold_descriptive_summary.csv"),
            ("fixed_category_support", "fixed_category_support.csv"),
            ("fold_seed_averaged_metrics", "fold_seed_averaged_metrics.csv"),
            ("raw_metrics", "raw_metrics.csv"),
        )
    ]
    scorpion_outputs = [
        external_record(
            "results/paired_acquisition_factorization_scorpion_fold_aware_v2/" f"{name}",
            role,
        )
        for role, name in (
            ("analysis_design", "analysis_design.json"),
            ("fold_aware_contrasts", "fold_aware_contrasts.csv"),
            (
                "slide_seed_averaged_contrasts",
                "slide_seed_averaged_contrasts.csv",
            ),
        )
    ]
    canine_promoted = [
        path_record(
            path,
            base=PACKAGE_ROOT,
            role=role,
            include_row_count=path.suffix == ".csv",
        )
        for role, path in canine_promoted_paths.items()
    ]
    scorpion_promoted = [
        path_record(
            path,
            base=PACKAGE_ROOT,
            role=role,
            include_row_count=path.suffix == ".csv",
        )
        for role, path in scorpion_promoted_paths.items()
    ]
    claim_hash = sha256_file(REPO_ROOT / "CLAIM_BOUNDARY.md")

    manifest = {
        "claim_boundary": {
            "authoritative_repository_path": "CLAIM_BOUNDARY.md",
            "central_claim": (
                "Under corrected biological-sample-blocked and fold-aware "
                "evaluation, Paired-Acquisition Neural Factorization substantially "
                "reduces linearly recoverable scanner identity in its tissue-oriented "
                "representation while preserving descriptive tissue-category "
                "structure and same-region retrieval. The acquisition branch retains "
                "strong scanner information. These results support partial structured "
                "separation under the tested conditions, not pure biological factors, "
                "complete independence, or clinical utility."
            ),
            "publication_sha256": claim_hash,
            "snapshot_sha256": sha256_file(claim_snapshot),
        },
        "evidence_families": [
            {
                "claim_scope": [
                    "lower linearly recoverable scanner identity in the tissue-oriented branch",
                    "descriptive category structure close to frozen features",
                    "linear centroid/QR removes scanner information more aggressively",
                    "explicit acquisition branch retains strong scanner information",
                ],
                "command": [
                    "python",
                    "experiments/paired_acquisition/"
                    "run_biological_label_preservation_fixed_estimand.py",
                ],
                "configuration": {
                    "category_neighbour_pool": "fit_only",
                    "excluded_categories": ["Bone", "Cartilage"],
                    "fit_only_probe_standardization": True,
                    "fixed_categories": [
                        "Dermis",
                        "Epidermis",
                        "Inflamm/Necrosis",
                        "SCC",
                        "Subcutis",
                    ],
                    "folds": [0, 1, 2, 3, 4],
                    "minimum_fit_samples_per_category_per_fold": 2,
                    "minimum_test_samples_per_category_per_fold": 2,
                    "same_region_neighbours_excluded": True,
                    "same_sample_neighbours_excluded": True,
                    "seeds": [911, 912, 913, 914, 915],
                },
                "evidence_status": "current_corrected",
                "external_inputs": canine_inputs(),
                "external_outputs": canine_outputs,
                "family_id": "canine_fixed_estimand_v1",
                "fold_design": (
                    "Five biological-sample-blocked folds; one fixed category "
                    "estimand supported by at least two fit and two test biological "
                    "samples per category in every fold; optimization seeds averaged "
                    "within fold before five-fold descriptive summaries."
                ),
                "historical_boundary": (
                    "Supersedes but does not modify or delete historical contaminated "
                    "canine category metrics, which remain withdrawn."
                ),
                "key_metrics": key_canine_metrics(
                    CANINE_SOURCE / "five_fold_descriptive_summary.csv"
                ),
                "promoted_artifacts": canine_promoted,
            },
            {
                "claim_scope": [
                    "substantial reduction in linearly recoverable scanner identity",
                    "improved cross-view cosine geometry",
                    "same-region retrieval effectively unchanged",
                    "retrieval improvement is not supported",
                ],
                "command": [
                    "python",
                    "scripts/scorpion/analyze_pathoalign_crossfold_v2.py",
                    "--raw-slide-metrics",
                    "results/scorpion/pathoalign_dinov2_crossfold_analysis/"
                    "raw_slide_metrics.csv",
                    "--out-dir",
                    "results/paired_acquisition_factorization_scorpion_fold_aware_v2",
                    "--bootstrap-draws",
                    "100000",
                ],
                "configuration": {
                    "bootstrap": "resample folds, then slides within sampled folds",
                    "bootstrap_draws": 100000,
                    "difference": "pathoalign_dep20_minus_paired_reference",
                    "folds": 5,
                    "seed_averaging": "within fold/slide/method before contrast",
                    "sign_flip_p_values": "not reported",
                    "slides": 48,
                },
                "evidence_status": "current_corrected",
                "external_inputs": [
                    external_record(
                        "results/scorpion/pathoalign_dinov2_crossfold_analysis/"
                        "raw_slide_metrics.csv",
                        "raw_slide_metrics",
                    )
                ],
                "external_outputs": scorpion_outputs,
                "family_id": "scorpion_fold_aware_v2",
                "fold_design": (
                    "Seed average within fold/slide/method, followed by paired "
                    "pathoalign_dep20 minus paired_reference contrasts and a "
                    "two-stage bootstrap that resamples folds then slides."
                ),
                "historical_boundary": (
                    "Supersedes but does not modify or delete historical "
                    "slide-independent sign-flip p-values, which remain withdrawn."
                ),
                "key_metrics": key_scorpion_metrics(SCORPION_SOURCE / "fold_aware_contrasts.csv"),
                "promoted_artifacts": scorpion_promoted,
            },
        ],
        "historical_evidence": {
            "files_modified_or_deleted": False,
            "status": "withdrawn_and_preserved",
            "withdrawn_claims": [
                "biological purity or pure biological factors",
                "causal identification beyond the paired design",
                "information-theoretic independence",
                "complete disentanglement or complete scanner invariance",
                "diagnostic, clinical, patient-benefit, or deployment claims",
                "historical canine category metrics",
                "historical slide-independent sign-flip p-values",
                "historical TransnnMIL fusion and topology claims",
                "unified-scoreboard rankings",
                "claims that cosine proves biological preservation or destruction",
            ],
        },
        "publication": {
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "raw_outputs_committed": False,
            "repository": (
                "https://github.com/matthewvaishnav/" "computational-pathology-research"
            ),
            "tracked_content": (
                "validated summaries, design records, support table, environment, "
                "hash bindings, claim-boundary snapshot, and release metadata"
            ),
        },
        "release_artifacts": [
            path_record(
                claim_snapshot,
                base=PACKAGE_ROOT,
                role="claim_boundary_snapshot",
            ),
            path_record(
                environment_path,
                base=PACKAGE_ROOT,
                role="environment",
            ),
            path_record(
                PACKAGE_ROOT / "README.md",
                base=PACKAGE_ROOT,
                role="readme",
            ),
        ],
        "schema_version": SCHEMA_VERSION,
        "source_code": {
            "commit": SOURCE_COMMIT,
            "equivalent_execution_commit": EQUIVALENT_EXECUTION_COMMIT,
            "repository": (
                "https://github.com/matthewvaishnav/" "computational-pathology-research"
            ),
            "scripts": [
                source_script_record(
                    "experiments/paired_acquisition/"
                    "run_biological_label_preservation_fixed_estimand.py",
                    "canine_fixed_estimand_runner",
                ),
                source_script_record(
                    "experiments/paired_acquisition/"
                    "run_biological_label_preservation_audit_v2.py",
                    "canine_audit_v2",
                ),
                source_script_record(
                    "experiments/paired_acquisition/" "run_biological_label_preservation_audit.py",
                    "canine_legacy_loaders",
                ),
                source_script_record(
                    "scripts/scorpion/analyze_pathoalign_crossfold_v2.py",
                    "scorpion_fold_aware_analyzer",
                ),
            ],
            "tree": source_tree,
            "tree_equivalence_note": (
                "The corrected artifacts were generated from the exact tree "
                "committed as 1c6e319; squash merge ad03723 has the same tree. "
                "The direct-execution bootstrap changed no scientific calculation."
            ),
        },
        "status": "completed",
    }
    manifest_path = PACKAGE_ROOT / "release_manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            validate_evidence(
                manifest_path,
                repo_root=REPO_ROOT,
                require_external_inputs=True,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
