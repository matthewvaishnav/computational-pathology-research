#!/usr/bin/env python3
"""Build a deterministic, read-only provenance manifest for paired archives.

The builder hashes raw archive bytes, reads only the scalar ``metadata_json``
NPZ member, validates reconstructed family-level Git evidence, and never loads feature
payloads or runs training. Default execution writes only the two CSV outputs
and the validation report beside this script. ``--check`` and ``--format
json`` are read-only.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_FILE = PACKAGE_DIR / "provenance_manifest.csv"
CONFLICT_FILE = PACKAGE_DIR / "provenance_conflicts.csv"
REPORT_FILE = PACKAGE_DIR / "provenance_validation_report.md"

EXPECTED_ARCHIVE_COUNT = 426
EXPECTED_SOURCE_CONFLICTS = 200
EXPECTED_MODEL_CONFLICTS = 150
EXPECTED_BACKBONE_PATH_CONFLICTS = 0
EXPECTED_DATASET_PATH_CONFLICTS = 0
EXPECTED_CONFLICT_OVERLAP = 0
EXPECTED_OPTIONAL_BACKBONE_ABSENCES = 226
EXPECTED_FALLBACK_CONFLICTS = 0
EXPECTED_GROUP_EVIDENCE_ARCHIVES = 425
EXPECTED_CONFLICT_COMPATIBLE_PER_RUN_RECORDS = 275
EXPECTED_CONFLICT_AGGREGATE_ONLY_RECORDS = 75
EXPECTED_RESOLUTION_COUNTS = {
    "confirmed": 50,
    "corrected": 0,
    "legacy-optional": 26,
    "unresolved": 350,
}

ALLOWED_RESOLUTION_STATUSES = {
    "confirmed",
    "corrected",
    "unresolved",
    "legacy-optional",
}
ALLOWED_CONFIDENCE_VALUES = {"high", "medium", "low", "not_applicable"}
ALLOWED_CONFIDENCE_BY_STATUS = {
    "confirmed": {"high", "medium"},
    "corrected": {"high", "medium"},
    "unresolved": {"medium", "low"},
    "legacy-optional": {"not_applicable"},
}
CORRECTION_EVIDENCE_TYPES = {
    "content_hash_verified_manifest",
    "generator_config",
    "generator_config_and_run_log",
    "run_log_with_output_hash",
    "source_commit_and_code_path",
    "deterministic_internal_proof",
}
PROPOSED_CANONICAL_EVIDENCE_TYPE = "family_level_reconstructed_lineage_evidence"
UNRESOLVED_ADJUDICATION_EVIDENCE_NEEDED = (
    "verified run manifest linking archive path, run ID, producing invocation, "
    "fold, seed, condition, variant, evaluation split, model, backbone, and "
    "historical output hash"
)
ARCHIVE_BINDING_ROW_FIELDS = (
    "archive_id",
    "canonical_path",
    "content_sha256",
    "archive_family",
    "dataset",
    "fold",
    "seed",
    "condition",
    "variant",
    "evaluation_split",
    "canonical_source",
    "canonical_model",
    "canonical_backbone",
)
ARCHIVE_BINDING_EVIDENCE_FIELDS = (
    "commit",
    "generator_path",
    "config_identifier",
    "log_record_identifier",
    "run_identifier",
)
CONFLICT_ORDER = {
    "source_label_conflict": 0,
    "model_backbone_label_conflict": 1,
    "backbone_path_conflict": 2,
    "dataset_path_conflict": 3,
    "duplicate_content": 4,
}
CONFLICT_FLAG_TO_CLASS = {
    "source_label_conflict": "source_label_conflict",
    "model_backbone_label_conflict": "model_backbone_label_conflict",
    "backbone_path_conflict": "backbone_path_conflict",
    "dataset_path_conflict": "dataset_path_conflict",
}

MANIFEST_COLUMNS = (
    "archive_id",
    "canonical_path",
    "relative_path",
    "content_sha256",
    "file_size_bytes",
    "archive_family",
    "dataset",
    "fold",
    "seed",
    "condition",
    "variant",
    "evaluation_split",
    "observed_source",
    "observed_model",
    "observed_backbone",
    "observed_metadata_json_sha256",
    "observed_metadata_keys",
    "metadata_json_present",
    "metadata_json_record_count",
    "expected_source_from_path",
    "expected_model_family",
    "expected_backbone_from_path",
    "expected_dataset_from_path",
    "expected_condition_from_path",
    "expected_variant_from_path",
    "source_label_conflict",
    "model_backbone_label_conflict",
    "backbone_path_conflict",
    "dataset_path_conflict",
    "duplicate_path_conflict",
    "metadata_missing",
    "metadata_malformed",
    "conflict_class",
    "conflict_evidence_basis",
    "canonical_source",
    "canonical_model",
    "canonical_backbone",
    "canonical_resolution_status",
    "resolution_confidence",
    "resolution_evidence_type",
    "resolution_evidence_reference",
    "resolution_notes",
    "evidence_needed_for_adjudication",
    "generator_config_needed",
    "run_log_needed",
    "source_commit_needed",
    "archive_hash_comparison_needed",
    "human_review_needed",
)

CONFLICT_COLUMNS = (
    "archive_id",
    "canonical_path",
    "conflict_class",
    "observed_value",
    "expected_value",
    "evidence_basis",
    "current_resolution_status",
    "resolution_confidence",
    "evidence_available",
    "evidence_missing",
    "recommended_next_action",
    "scientific_impact_statement",
)


class ManifestValidationError(ValueError):
    """Deterministic fail-closed validation error."""


class ArchiveMetadataError(ValueError):
    """Expected malformed NPZ or metadata content."""


@dataclass(frozen=True)
class EvidenceSpec:
    commit: str
    generator_path: str
    log_path: str
    design_path: str = ""
    generator_tokens: tuple[str, ...] = ()
    design_tokens: tuple[str, ...] = ()
    log_tokens: tuple[str, ...] = ()
    record_scope: str = "compatible_per_run"

    def references(self) -> str:
        paths = [self.generator_path, self.log_path]
        if self.design_path:
            paths.append(self.design_path)
        paths.append("experiments/scorpion/run_pathoalign_projection.py")
        return "|".join(f"git:{self.commit}:{path}" for path in paths)


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    dataset: str
    pattern: str
    expected_count: int
    expected_source: str
    expected_source_token: str
    expected_model_family: str
    expected_backbone: str
    expected_condition: str = ""
    expected_variant: str = ""
    require_explicit_backbone: bool = False
    projected: bool = True
    evidence: EvidenceSpec | None = None


@dataclass(frozen=True)
class HashSnapshot:
    sha256: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class ArchiveObservation:
    family: FamilySpec
    path: Path
    canonical_path: str
    relative_path: str
    content_sha256: str
    file_size_bytes: int
    metadata_text: str
    metadata: dict[str, Any]
    metadata_keys: str
    metadata_sha256: str
    fold: int | None
    seed: int | None
    condition: str
    variant: str
    evaluation_split: str


@dataclass(frozen=True)
class BuildResult:
    manifest_rows: list[dict[str, Any]]
    conflict_rows: list[dict[str, Any]]
    family_counts: dict[str, int]
    dataset_counts: dict[str, int]
    duplicate_content_groups: list[list[str]]
    summary: dict[str, Any]
    outputs: dict[str, bytes]


CANINE_PAIR_EVIDENCE = EvidenceSpec(
    commit="43520b86210d0bfed8a2869d514639af6ce8e15a",
    generator_path="experiments/canine/run_pair_integrity_falsification_caninescc.py",
    log_path=(
        "results/paired_acquisition_factorization_pair_integrity_caninescc/run_log.txt"
    ),
    generator_tokens=("external_multiscanner_caninescc", "canine_pair_integrity_falsification"),
    log_tokens=(
        "canine pair-integrity run start",
        "Training fold=4 seed=915 condition=true_pairs",
    ),
)
SCORPION_PAIR_EVIDENCE = EvidenceSpec(
    commit="f435bfa28c438588df5bee53bb3e5843e1d3b0d8",
    generator_path="experiments/scorpion/run_pair_integrity_falsification.py",
    log_path="results/paired_acquisition_factorization_pair_integrity_scorpion/run_log.txt",
    generator_tokens=("SCORPION", "DINOv2-Base"),
    log_tokens=(
        "pair-integrity run start",
        "Training fold=4 seed=705 condition=true_pairs",
    ),
)
SCORPION_PHIKON_PAIR_EVIDENCE = EvidenceSpec(
    commit="14726b13e7c0f23f9fe494399bab9fd902fecd7a",
    generator_path=(
        "experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py"
    ),
    log_path=(
        "results/paired_acquisition_factorization_pair_integrity_scorpion_phikon/"
        "run_log.txt"
    ),
    generator_tokens=("phikon", "resnet50", "backbone"),
    log_tokens=(
        "cross-backbone pair-integrity run start",
        "Training backbone=phikon fold=4 seed=705 condition=true_pairs",
    ),
)
SCORPION_RESNET_PAIR_EVIDENCE = EvidenceSpec(
    commit="14726b13e7c0f23f9fe494399bab9fd902fecd7a",
    generator_path=(
        "experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py"
    ),
    log_path=(
        "results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50/"
        "run_log.txt"
    ),
    generator_tokens=("phikon", "resnet50", "backbone"),
    log_tokens=(
        "cross-backbone pair-integrity run start",
        "Training backbone=resnet50 fold=4 seed=705 condition=true_pairs",
    ),
)
CANINE_BOUNDARY_EVIDENCE = EvidenceSpec(
    commit="e4819c42e49f9c4a1e7a652fc8bf8651a2f6b628",
    generator_path="experiments/paired_acquisition/run_pair_structure_boundary_test.py",
    design_path=(
        "results/paired_acquisition_factorization_pair_structure_boundary_test/"
        "experiment_design.json"
    ),
    log_path=(
        "results/paired_acquisition_factorization_pair_structure_boundary_test/run_log.txt"
    ),
    generator_tokens=("canineSCC_DINOv2", "paired_acquisition_factorization_pair_integrity_caninescc"),
    design_tokens=("canineSCC_DINOv2",),
    log_tokens=(
        "Pair-structure boundary test started",
        "[canineSCC_DINOv2]   Trained 75 new runs",
    ),
    record_scope="aggregate_only",
)
CANINE_BOTTLENECK_EVIDENCE = EvidenceSpec(
    commit="a89bfb32977dc723ef895f150ab4ae720a345ac5",
    generator_path=(
        "experiments/paired_acquisition/"
        "run_acquisition_bottleneck_separation_frontier.py"
    ),
    design_path=(
        "results/paired_acquisition_factorization_acquisition_bottleneck_"
        "separation_frontier/experiment_design.json"
    ),
    log_path=(
        "results/paired_acquisition_factorization_acquisition_bottleneck_"
        "separation_frontier/run_log.txt"
    ),
    generator_tokens=("external_multiscanner_caninescc", "canine"),
    design_tokens=("external_multiscanner_caninescc",),
    log_tokens=(
        "PHASE B FULL",
        "ACQUISITION BOTTLENECK SEPARATION-FRONTIER SWEEP COMPLETE",
    ),
)
SCORPION_FRONTIER_EVIDENCE = EvidenceSpec(
    commit="0e2af24730a0a298fbf0363dfbab7682dc65a1af",
    generator_path=(
        "experiments/paired_acquisition/"
        "run_frontier_selected_crossbackbone_validation.py"
    ),
    design_path=(
        "results/paired_acquisition_factorization_frontier_selected_"
        "crossbackbone_validation/experiment_design.json"
    ),
    log_path=(
        "results/paired_acquisition_factorization_frontier_selected_"
        "crossbackbone_validation/run_log.txt"
    ),
    generator_tokens=("dinov2", "phikon", "resnet50", "SCORPION"),
    design_tokens=("dinov2", "phikon", "resnet50"),
    log_tokens=(
        "BACKBONE dinov2",
        "BACKBONE phikon",
        "BACKBONE resnet50",
        "Validation checks passed.",
    ),
)


FAMILIES = (
    FamilySpec(
        "canine_original_dinov2",
        "canine_scc",
        "results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz",
        1,
        "external_multiscanner_caninescc",
        "canine",
        "dinov2_feature_extractor",
        "dinov2",
        projected=False,
    ),
    FamilySpec(
        "canine_true_pair_biological",
        "canine_scc",
        (
            "results/paired_acquisition_factorization_pair_integrity_caninescc/"
            "fold_*/runs/true_pairs_seed_*/projected_features.npz"
        ),
        25,
        "external_multiscanner_caninescc",
        "canine",
        "pathoalign",
        "dinov2",
        expected_condition="true_pairs",
        evidence=CANINE_PAIR_EVIDENCE,
    ),
    FamilySpec(
        "canine_acq_dim8_biological",
        "canine_scc",
        (
            "results/paired_acquisition_factorization_acquisition_bottleneck_"
            "separation_frontier/trained_runs/full/fold_*/runs/"
            "acq_dim8_default_seed_*/projected_features.npz"
        ),
        25,
        "external_multiscanner_caninescc",
        "canine",
        "pathoalign",
        "dinov2",
        expected_variant="acq_dim8_default",
        evidence=CANINE_BOTTLENECK_EVIDENCE,
    ),
    FamilySpec(
        "canine_acq_dim16_biological",
        "canine_scc",
        (
            "results/paired_acquisition_factorization_acquisition_bottleneck_"
            "separation_frontier/trained_runs/full/fold_*/runs/"
            "acq_dim16_stronger_xcov_seed_*/projected_features.npz"
        ),
        25,
        "external_multiscanner_caninescc",
        "canine",
        "pathoalign",
        "dinov2",
        expected_variant="acq_dim16_stronger_xcov",
        evidence=CANINE_BOTTLENECK_EVIDENCE,
    ),
    FamilySpec(
        "canine_shuffled_region_control",
        "canine_scc",
        (
            "results/paired_acquisition_factorization_pair_integrity_caninescc/"
            "fold_*/runs/shuffled_region_pairs_seed_*/projected_features.npz"
        ),
        25,
        "external_multiscanner_caninescc",
        "canine",
        "pathoalign",
        "dinov2",
        expected_condition="shuffled_region_pairs",
        evidence=CANINE_PAIR_EVIDENCE,
    ),
    FamilySpec(
        "canine_shuffled_sample_control",
        "canine_scc",
        (
            "results/paired_acquisition_factorization_pair_integrity_caninescc/"
            "fold_*/runs/shuffled_sample_pairs_seed_*/projected_features.npz"
        ),
        25,
        "external_multiscanner_caninescc",
        "canine",
        "pathoalign",
        "dinov2",
        expected_condition="shuffled_sample_pairs",
        evidence=CANINE_PAIR_EVIDENCE,
    ),
    FamilySpec(
        "canine_same_category_different_sample_control",
        "canine_scc",
        (
            "results/paired_acquisition_factorization_pair_structure_boundary_test/"
            "canineSCC_DINOv2/fold_*/runs/same_category_different_sample_pairs_"
            "seed_*/projected_features.npz"
        ),
        25,
        "external_multiscanner_caninescc",
        "canine",
        "pathoalign",
        "dinov2",
        expected_condition="same_category_different_sample_pairs",
        evidence=CANINE_BOUNDARY_EVIDENCE,
    ),
    FamilySpec(
        "canine_scanner_balanced_random_control",
        "canine_scc",
        (
            "results/paired_acquisition_factorization_pair_structure_boundary_test/"
            "canineSCC_DINOv2/fold_*/runs/scanner_balanced_random_pairs_seed_*/"
            "projected_features.npz"
        ),
        25,
        "external_multiscanner_caninescc",
        "canine",
        "pathoalign",
        "dinov2",
        expected_condition="scanner_balanced_random_pairs",
        evidence=CANINE_BOUNDARY_EVIDENCE,
    ),
    FamilySpec(
        "canine_fully_random_control",
        "canine_scc",
        (
            "results/paired_acquisition_factorization_pair_structure_boundary_test/"
            "canineSCC_DINOv2/fold_*/runs/fully_random_pairs_seed_*/"
            "projected_features.npz"
        ),
        25,
        "external_multiscanner_caninescc",
        "canine",
        "pathoalign",
        "dinov2",
        expected_condition="fully_random_pairs",
        evidence=CANINE_BOUNDARY_EVIDENCE,
    ),
    FamilySpec(
        "scorpion_true_pair_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_pair_integrity_scorpion/"
            "fold_*/runs/true_pairs_seed_*/projected_features.npz"
        ),
        25,
        "SCORPION",
        "scorpion",
        "pathoalign",
        "dinov2",
        expected_condition="true_pairs",
        evidence=SCORPION_PAIR_EVIDENCE,
    ),
    FamilySpec(
        "scorpion_phikon_true_pair_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_pair_integrity_scorpion_phikon/"
            "fold_*/runs/true_pairs_seed_*/projected_features.npz"
        ),
        25,
        "SCORPION",
        "scorpion",
        "pathoalign",
        "phikon",
        expected_condition="true_pairs",
        require_explicit_backbone=True,
        evidence=SCORPION_PHIKON_PAIR_EVIDENCE,
    ),
    FamilySpec(
        "scorpion_resnet50_true_pair_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50/"
            "fold_*/runs/true_pairs_seed_*/projected_features.npz"
        ),
        25,
        "SCORPION",
        "scorpion",
        "pathoalign",
        "resnet50",
        expected_condition="true_pairs",
        require_explicit_backbone=True,
        evidence=SCORPION_RESNET_PAIR_EVIDENCE,
    ),
    FamilySpec(
        "scorpion_dinov2_acq_dim8_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_frontier_selected_"
            "crossbackbone_validation/trained_runs/dinov2/fold_*/runs/"
            "acq_dim8_default_seed_*/projected_features.npz"
        ),
        25,
        "SCORPION",
        "scorpion",
        "pathoalign",
        "dinov2",
        expected_condition="true_pairs",
        expected_variant="acq_dim8_default",
        require_explicit_backbone=True,
        evidence=SCORPION_FRONTIER_EVIDENCE,
    ),
    FamilySpec(
        "scorpion_dinov2_acq_dim16_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_frontier_selected_"
            "crossbackbone_validation/trained_runs/dinov2/fold_*/runs/"
            "acq_dim16_stronger_xcov_seed_*/projected_features.npz"
        ),
        25,
        "SCORPION",
        "scorpion",
        "pathoalign",
        "dinov2",
        expected_condition="true_pairs",
        expected_variant="acq_dim16_stronger_xcov",
        require_explicit_backbone=True,
        evidence=SCORPION_FRONTIER_EVIDENCE,
    ),
    FamilySpec(
        "scorpion_phikon_acq_dim8_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_frontier_selected_"
            "crossbackbone_validation/trained_runs/phikon/fold_*/runs/"
            "acq_dim8_default_seed_*/projected_features.npz"
        ),
        25,
        "SCORPION",
        "scorpion",
        "pathoalign",
        "phikon",
        expected_condition="true_pairs",
        expected_variant="acq_dim8_default",
        require_explicit_backbone=True,
        evidence=SCORPION_FRONTIER_EVIDENCE,
    ),
    FamilySpec(
        "scorpion_phikon_acq_dim16_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_frontier_selected_"
            "crossbackbone_validation/trained_runs/phikon/fold_*/runs/"
            "acq_dim16_stronger_xcov_seed_*/projected_features.npz"
        ),
        25,
        "SCORPION",
        "scorpion",
        "pathoalign",
        "phikon",
        expected_condition="true_pairs",
        expected_variant="acq_dim16_stronger_xcov",
        require_explicit_backbone=True,
        evidence=SCORPION_FRONTIER_EVIDENCE,
    ),
    FamilySpec(
        "scorpion_resnet50_acq_dim8_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_frontier_selected_"
            "crossbackbone_validation/trained_runs/resnet50/fold_*/runs/"
            "acq_dim8_default_seed_*/projected_features.npz"
        ),
        25,
        "SCORPION",
        "scorpion",
        "pathoalign",
        "resnet50",
        expected_condition="true_pairs",
        expected_variant="acq_dim8_default",
        require_explicit_backbone=True,
        evidence=SCORPION_FRONTIER_EVIDENCE,
    ),
    FamilySpec(
        "scorpion_resnet50_acq_dim16_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_frontier_selected_"
            "crossbackbone_validation/trained_runs/resnet50/fold_*/runs/"
            "acq_dim16_stronger_xcov_seed_*/projected_features.npz"
        ),
        25,
        "SCORPION",
        "scorpion",
        "pathoalign",
        "resnet50",
        expected_condition="true_pairs",
        expected_variant="acq_dim16_stronger_xcov",
        require_explicit_backbone=True,
        evidence=SCORPION_FRONTIER_EVIDENCE,
    ),
    FamilySpec(
        "oldstyle_keep_k4_row_level",
        "canine_scc",
        (
            "results/paired_acquisition_factorization_oldstyle_residual_branch_"
            "separation_audit/**/projected_features.npz"
        ),
        0,
        "external_multiscanner_caninescc",
        "canine",
        "oldstyle_keep_k4",
        "",
        projected=False,
    ),
)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def string_value(value: Any, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ManifestValidationError(f"metadata_field_not_string:{field}")
    return value


def git_show(repo_root: Path, commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ManifestValidationError(f"evidence_blob_missing:{commit}:{path}")
    try:
        return completed.stdout.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ManifestValidationError(
            f"evidence_blob_not_utf8:{commit}:{path}"
        ) from exc


def verify_evidence_specs(
    repo_root: Path,
    families: Sequence[FamilySpec] | None = None,
) -> set[str]:
    verified: set[str] = set()
    cache: dict[tuple[str, str], str] = {}
    selected_families = FAMILIES if families is None else families

    def content(commit: str, path: str) -> str:
        key = (commit, path)
        if key not in cache:
            cache[key] = git_show(repo_root, commit, path)
        return cache[key]

    for family in selected_families:
        evidence = family.evidence
        if evidence is None:
            continue
        if evidence.record_scope not in {"compatible_per_run", "aggregate_only"}:
            raise ManifestValidationError(
                f"invalid_evidence_record_scope:{family.family_id}:"
                f"{evidence.record_scope}"
            )
        generator = content(evidence.commit, evidence.generator_path)
        log = content(evidence.commit, evidence.log_path)
        shared = content(
            evidence.commit, "experiments/scorpion/run_pathoalign_projection.py"
        )
        if not generator.strip() or not log.strip():
            raise ManifestValidationError(
                f"empty_generator_or_log_evidence:{family.family_id}"
            )
        if evidence.design_path:
            design = content(evidence.commit, evidence.design_path)
            if not design.strip():
                raise ManifestValidationError(
                    f"empty_design_evidence:{family.family_id}"
                )
            for token in evidence.design_tokens:
                if token not in design:
                    raise ManifestValidationError(
                        f"design_evidence_token_missing:{family.family_id}:{token}"
                    )
        for token in evidence.generator_tokens:
            if token not in generator:
                raise ManifestValidationError(
                    f"generator_evidence_token_missing:{family.family_id}:{token}"
                )
        for token in evidence.log_tokens:
            if token not in log:
                raise ManifestValidationError(
                    f"run_log_evidence_token_missing:{family.family_id}:{token}"
                )
        hardcoded_tokens = (
            '"model": f"dinov2_{method}"',
            '"source": "SCORPION DINOv2 projection experiment"',
        )
        for token in hardcoded_tokens:
            if token not in shared:
                raise ManifestValidationError(
                    f"shared_generator_evidence_token_missing:{family.family_id}"
                )
        verified.add(family.family_id)
    return verified


def read_npy_header(handle: BinaryIO) -> dict[str, Any]:
    if handle.read(6) != b"\x93NUMPY":
        raise ArchiveMetadataError("invalid_npy_magic")
    version_raw = handle.read(2)
    if len(version_raw) != 2:
        raise ArchiveMetadataError("truncated_npy_version")
    major, _minor = version_raw
    if major == 1:
        size_raw = handle.read(2)
        if len(size_raw) != 2:
            raise ArchiveMetadataError("truncated_npy_header_length")
        header_size = struct.unpack("<H", size_raw)[0]
    elif major in {2, 3}:
        size_raw = handle.read(4)
        if len(size_raw) != 4:
            raise ArchiveMetadataError("truncated_npy_header_length")
        header_size = struct.unpack("<I", size_raw)[0]
    else:
        raise ArchiveMetadataError("unsupported_npy_version")
    if header_size > 1_000_000:
        raise ArchiveMetadataError("oversized_npy_header")
    header_raw = handle.read(header_size)
    if len(header_raw) != header_size:
        raise ArchiveMetadataError("truncated_npy_header")
    encoding = "utf-8" if major == 3 else "latin1"
    try:
        header = ast.literal_eval(header_raw.decode(encoding).strip())
    except (SyntaxError, TypeError, UnicodeError, ValueError) as exc:
        raise ArchiveMetadataError("invalid_npy_header") from exc
    if not isinstance(header, dict):
        raise ArchiveMetadataError("invalid_npy_header")
    shape = header.get("shape")
    descriptor = header.get("descr")
    fortran_order = header.get("fortran_order")
    if not isinstance(shape, tuple) or any(
        type(item) is not int or item < 0 for item in shape
    ):
        raise ArchiveMetadataError("invalid_npy_shape")
    if not isinstance(descriptor, str) or not descriptor:
        raise ArchiveMetadataError("invalid_npy_descriptor")
    if type(fortran_order) is not bool:
        raise ArchiveMetadataError("invalid_npy_fortran_order")
    return header


def open_npz_member(archive: zipfile.ZipFile, member: str) -> BinaryIO:
    try:
        return archive.open(member)
    except NotImplementedError as exc:
        raise ArchiveMetadataError("unsupported_zip_compression") from exc
    except RecursionError:
        raise
    except RuntimeError as exc:
        message = str(exc).lower()
        if "password" in message or "encrypted" in message:
            raise ArchiveMetadataError("encrypted_npz_member") from exc
        raise


def read_npz_text(path: Path, key: str, max_items: int = 20_000) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        members: dict[str, str] = {}
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            basename = Path(name).name[:-4]
            if basename in members:
                raise ArchiveMetadataError(
                    f"duplicate_npz_member_basename:{basename}"
                )
            members[basename] = name
        member = members.get(key)
        if not member:
            raise ArchiveMetadataError(f"npz_member_missing:{key}")
        if archive.getinfo(member).file_size > 20_000_000:
            raise ArchiveMetadataError("text_member_file_size_limit_exceeded")
        with open_npz_member(archive, member) as handle:
            header = read_npy_header(handle)
            shape = tuple(int(value) for value in header["shape"])
            count = math.prod(shape) if shape else 1
            if len(shape) > 1 or count > max_items:
                raise ArchiveMetadataError("text_array_shape_or_size_unsupported")
            descriptor = str(header["descr"])
            unicode_match = re.fullmatch(r"([<>=|])U(\d+)", descriptor)
            bytes_match = re.fullmatch(r"[|<>=]S(\d+)", descriptor)
            if unicode_match:
                width = int(unicode_match.group(2)) * 4
                encoding = "utf-32-be" if unicode_match.group(1) == ">" else "utf-32-le"
            elif bytes_match:
                width = int(bytes_match.group(1))
                encoding = "utf-8"
            else:
                raise ArchiveMetadataError(f"unsupported_text_dtype:{descriptor}")
            if width > 65_536 or width * count > 16_000_000:
                raise ArchiveMetadataError("text_array_byte_limit_exceeded")
            raw = handle.read(width * count)
            if len(raw) != width * count:
                raise ArchiveMetadataError("truncated_npy_data")
            values: list[str] = []
            for index in range(count):
                item = raw[index * width : (index + 1) * width]
                try:
                    value = item.decode(encoding, errors="strict").rstrip("\x00")
                except UnicodeError as exc:
                    raise ArchiveMetadataError(f"text_decode_error:{key}") from exc
                values.append(value)
            return values


def parse_metadata_json(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        records = read_npz_text(path, "metadata_json", max_items=10)
    except (ArchiveMetadataError, OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ManifestValidationError(f"metadata_json_unreadable:{path.name}:{exc}") from exc
    if len(records) != 1:
        raise ManifestValidationError(
            f"metadata_json_record_count_invalid:{path.name}:{len(records)}"
        )
    text = records[0]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"metadata_json_malformed:{path.name}") from exc
    if not isinstance(parsed, dict):
        raise ManifestValidationError(f"metadata_json_not_mapping:{path.name}")
    if any(not isinstance(key, str) for key in parsed):
        raise ManifestValidationError(f"metadata_json_key_not_string:{path.name}")
    return text, parsed


def hash_once(path: Path) -> HashSnapshot:
    before = path.stat()
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or byte_count != before.st_size
    ):
        raise ManifestValidationError(f"archive_mutated_during_hash:{path.name}")
    return HashSnapshot(digest.hexdigest(), before.st_size, before.st_mtime_ns)


def hash_file_twice(
    path: Path, between_passes: Callable[[], None] | None = None
) -> HashSnapshot:
    first = hash_once(path)
    if between_passes is not None:
        between_passes()
    second = hash_once(path)
    if first != second:
        raise ManifestValidationError(f"archive_mutated_between_hash_passes:{path.name}")
    return first


def normalize_repo_path(path: Path, repo_root: Path) -> tuple[str, str]:
    repo_resolved = repo_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        canonical = resolved.relative_to(repo_resolved).as_posix()
        relative = path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ManifestValidationError(f"archive_outside_repository:{path}") from exc
    for value in (canonical, relative):
        if not value or value.startswith("/") or ".." in Path(value).parts:
            raise ManifestValidationError(f"invalid_repository_relative_path:{value}")
        if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("\\\\"):
            raise ManifestValidationError(f"machine_specific_absolute_path:{value}")
    return canonical, relative


def discover_archives(repo_root: Path) -> list[tuple[FamilySpec, Path, str, str]]:
    found: list[tuple[FamilySpec, Path, str, str]] = []
    logical_owners: dict[str, list[str]] = defaultdict(list)
    resolved_owners: dict[str, list[str]] = defaultdict(list)
    for family in FAMILIES:
        paths = sorted(repo_root.glob(family.pattern), key=lambda item: item.as_posix())
        if len(paths) != family.expected_count:
            raise ManifestValidationError(
                "family_archive_count_mismatch:"
                f"{family.family_id}:expected={family.expected_count}:observed={len(paths)}"
            )
        for path in paths:
            if not path.is_file():
                raise ManifestValidationError(f"archive_file_missing:{path}")
            canonical, relative = normalize_repo_path(path, repo_root)
            logical_key = os.path.normcase(canonical)
            resolved_key = os.path.normcase(str(path.resolve(strict=True)))
            logical_owners[logical_key].append(family.family_id)
            resolved_owners[resolved_key].append(family.family_id)
            found.append((family, path, canonical, relative))
    duplicate_keys = sorted(
        key for owners in (logical_owners, resolved_owners) for key, value in owners.items()
        if len(value) > 1
    )
    if duplicate_keys:
        raise ManifestValidationError(
            "duplicate_canonical_paths:" + ";".join(sorted(set(duplicate_keys)))
        )
    if len(found) != EXPECTED_ARCHIVE_COUNT:
        raise ManifestValidationError(
            f"total_archive_count_mismatch:expected={EXPECTED_ARCHIVE_COUNT}:observed={len(found)}"
        )
    return sorted(found, key=lambda item: item[2])


def require_exact_int(metadata: dict[str, Any], key: str, path: str) -> int:
    value = metadata.get(key)
    if type(value) is not int:
        raise ManifestValidationError(f"metadata_integer_missing_or_wrong:{path}:{key}")
    return value


def validate_metadata(
    family: FamilySpec,
    canonical_path: str,
    metadata: dict[str, Any],
) -> tuple[int | None, int | None, str, str, str]:
    model = string_value(metadata.get("model"), "model")
    if not model:
        raise ManifestValidationError(f"metadata_model_missing:{canonical_path}")
    source = string_value(metadata.get("source"), "source")
    backbone = string_value(metadata.get("backbone"), "backbone")
    condition = string_value(metadata.get("condition"), "condition")
    variant = string_value(metadata.get("variant"), "variant")
    evaluation_split = string_value(
        metadata.get("evaluation_split"), "evaluation_split"
    )
    dataset_value = string_value(metadata.get("dataset"), "dataset")
    if family.projected:
        if not source:
            raise ManifestValidationError(f"metadata_source_missing:{canonical_path}")
        fold = require_exact_int(metadata, "fold", canonical_path)
        seed = require_exact_int(metadata, "seed", canonical_path)
        if not isinstance(metadata.get("config"), dict):
            raise ManifestValidationError(f"metadata_config_not_mapping:{canonical_path}")
        if metadata.get("contains_test_rows") is not True:
            raise ManifestValidationError(
                f"metadata_contains_test_rows_not_true:{canonical_path}"
            )
        if evaluation_split != "test":
            raise ManifestValidationError(
                f"metadata_evaluation_split_not_test:{canonical_path}"
            )
    else:
        fold_match = re.search(r"(?:^|/)fold_(\d+)", canonical_path)
        fold = int(fold_match.group(1)) if fold_match else None
        seed = None
    fold_match = re.search(r"(?:^|/)fold_(\d+)(?:/|_)", canonical_path)
    seed_match = re.search(r"_seed_(\d+)(?:/|$)", canonical_path)
    path_fold = int(fold_match.group(1)) if fold_match else None
    path_seed = int(seed_match.group(1)) if seed_match else None
    if family.projected and (fold != path_fold or seed != path_seed):
        raise ManifestValidationError(f"metadata_fold_seed_path_mismatch:{canonical_path}")
    run_match = re.search(r"/runs/(.+)_seed_\d+/projected_features\.npz$", canonical_path)
    run_label = run_match.group(1) if run_match else ""
    expected_run_label = family.expected_variant or family.expected_condition
    if family.projected and run_label != expected_run_label:
        raise ManifestValidationError(f"path_run_label_mismatch:{canonical_path}")
    if family.expected_condition and condition != family.expected_condition:
        raise ManifestValidationError(f"metadata_condition_path_mismatch:{canonical_path}")
    if family.expected_variant and variant != family.expected_variant:
        raise ManifestValidationError(f"metadata_variant_path_mismatch:{canonical_path}")
    if family.require_explicit_backbone and not backbone:
        raise ManifestValidationError(f"required_backbone_missing:{canonical_path}")
    if dataset_value and not dataset_matches(family.dataset, dataset_value):
        # This remains a row-level conflict, not a malformed metadata failure.
        pass
    return fold, seed, condition, variant, evaluation_split


def dataset_matches(expected: str, observed: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", observed.lower())
    if expected == "canine_scc":
        return "canine" in normalized
    return "scorpion" in normalized


def inspect_archives(
    repo_root: Path,
    discovered: Sequence[tuple[FamilySpec, Path, str, str]],
) -> list[ArchiveObservation]:
    observations: list[ArchiveObservation] = []
    for family, path, canonical, relative in discovered:
        first = hash_once(path)
        metadata_text, metadata = parse_metadata_json(path)
        second = hash_once(path)
        if first != second:
            raise ManifestValidationError(
                f"archive_mutated_between_hash_passes:{canonical}"
            )
        fold, seed, condition, variant, evaluation_split = validate_metadata(
            family, canonical, metadata
        )
        observations.append(
            ArchiveObservation(
                family=family,
                path=path,
                canonical_path=canonical,
                relative_path=relative,
                content_sha256=first.sha256,
                file_size_bytes=first.size,
                metadata_text=metadata_text,
                metadata=metadata,
                metadata_keys=json.dumps(
                    sorted(metadata), ensure_ascii=True, separators=(",", ":")
                ),
                metadata_sha256=hashlib.sha256(
                    metadata_text.encode("utf-8")
                ).hexdigest(),
                fold=fold,
                seed=seed,
                condition=condition,
                variant=variant,
                evaluation_split=evaluation_split,
            )
        )
    return observations


def duplicate_content_groups(
    observations: Sequence[ArchiveObservation],
) -> list[list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for observation in observations:
        by_hash[observation.content_sha256].append(observation.canonical_path)
    return sorted(
        [sorted(paths) for paths in by_hash.values() if len(paths) > 1],
        key=lambda paths: paths[0],
    )


def archive_id(canonical_path: str, content_sha256: str) -> str:
    payload = canonical_path.encode("utf-8") + b"\0" + content_sha256.encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def classify_conflicts(
    observation: ArchiveObservation,
    duplicate_paths: set[str],
) -> tuple[list[str], str, int]:
    family = observation.family
    metadata = observation.metadata
    source = string_value(metadata.get("source"), "source")
    model = string_value(metadata.get("model"), "model")
    backbone = string_value(metadata.get("backbone"), "backbone")
    dataset_value = string_value(metadata.get("dataset"), "dataset")
    classes: list[str] = []
    basis: list[str] = []
    if family.projected and family.expected_source_token.lower() not in source.lower():
        classes.append("source_label_conflict")
        basis.append("metadata_json.source versus family/path-expected dataset token")
    backbone_matches = bool(
        backbone and family.expected_backbone.lower() in backbone.lower()
    )
    if backbone and not backbone_matches:
        classes.append("backbone_path_conflict")
        basis.append("metadata_json.backbone versus family/path-expected backbone")
    if backbone and backbone_matches and backbone.lower() not in model.lower():
        classes.append("model_backbone_label_conflict")
        basis.append(
            "metadata_json.model versus explicit path-consistent metadata_json.backbone"
        )
    if dataset_value and not dataset_matches(family.dataset, dataset_value):
        classes.append("dataset_path_conflict")
        basis.append("metadata_json.dataset versus family/path-expected dataset")
    if observation.canonical_path in duplicate_paths:
        classes.append("duplicate_content")
        basis.append("raw SHA-256 shared by distinct canonical paths")
    fallback_conflict = int(
        not backbone
        and bool(family.expected_backbone)
        and family.expected_backbone.lower() not in model.lower()
    )
    classes.sort(key=lambda item: CONFLICT_ORDER[item])
    return classes, "; ".join(basis), fallback_conflict


def canonical_values(
    observation: ArchiveObservation, classes: Sequence[str], status: str
) -> tuple[str, str, str]:
    metadata = observation.metadata
    family = observation.family
    observed_source = string_value(metadata.get("source"), "source")
    observed_model = string_value(metadata.get("model"), "model")
    observed_backbone = string_value(metadata.get("backbone"), "backbone")
    canonical_source = observed_source
    canonical_model = observed_model
    canonical_backbone = observed_backbone
    if "source_label_conflict" in classes and status in {"corrected", "unresolved"}:
        canonical_source = family.expected_source
    elif "source_label_conflict" in classes:
        canonical_source = ""
    if "model_backbone_label_conflict" in classes and status in {
        "corrected",
        "unresolved",
    }:
        canonical_model = f"{family.expected_backbone}_pathoalign"
        canonical_backbone = family.expected_backbone
    elif "model_backbone_label_conflict" in classes:
        canonical_model = ""
    if "backbone_path_conflict" in classes:
        canonical_backbone = (
            family.expected_backbone
            if status in {"corrected", "unresolved"}
            else ""
        )
    return canonical_source, canonical_model, canonical_backbone


def resolution_for(
    observation: ArchiveObservation,
    classes: Sequence[str],
    evidence_verified: bool,
) -> tuple[str, str, str, str, str]:
    family = observation.family
    observed_backbone = string_value(observation.metadata.get("backbone"), "backbone")
    lineage_classes = set(classes) - {"duplicate_content"}
    if lineage_classes:
        reference = family.evidence.references() if family.evidence else ""
        return (
            "unresolved",
            "medium" if evidence_verified else "low",
            PROPOSED_CANONICAL_EVIDENCE_TYPE,
            reference,
            (
                "Each populated conflicting canonical field is a medium-confidence "
                "proposed canonical value supported by path/family lineage and reachable "
                "family-level generator/result history; it is not adjudicated because "
                "archive-specific run and historical output-hash binding are absent."
            ),
        )
    if not observed_backbone and family.expected_backbone:
        reference = (
            family.evidence.references()
            if family.evidence
            else (
                "results/external_multiscanner_caninescc/features/"
                "fold_0_dinov2_base.summary.json"
            )
        )
        return (
            "legacy-optional",
            "not_applicable",
            "optional_field_absent_without_contradiction",
            reference,
            "Optional explicit backbone metadata is absent without a contradictory label.",
        )
    return (
        "confirmed",
        "medium",
        "observed_metadata_agreement",
        family.evidence.references() if family.evidence else "",
        "Observed metadata agrees with the explicit family/path lineage evidence.",
    )


def make_manifest_rows(
    observations: Sequence[ArchiveObservation],
    verified_families: set[str],
    duplicate_groups: Sequence[Sequence[str]],
) -> tuple[list[dict[str, Any]], int]:
    duplicate_paths = {path for group in duplicate_groups for path in group}
    rows: list[dict[str, Any]] = []
    fallback_conflicts = 0
    for observation in observations:
        family = observation.family
        metadata = observation.metadata
        classes, basis, fallback = classify_conflicts(observation, duplicate_paths)
        fallback_conflicts += fallback
        evidence_verified = family.family_id in verified_families
        status, confidence, evidence_type, evidence_reference, notes = resolution_for(
            observation, classes, evidence_verified
        )
        canonical_source, canonical_model, canonical_backbone = canonical_values(
            observation, classes, status
        )
        unresolved = status == "unresolved"
        source = string_value(metadata.get("source"), "source")
        model = string_value(metadata.get("model"), "model")
        backbone = string_value(metadata.get("backbone"), "backbone")
        row: dict[str, Any] = {
            "archive_id": archive_id(
                observation.canonical_path, observation.content_sha256
            ),
            "canonical_path": observation.canonical_path,
            "relative_path": observation.relative_path,
            "content_sha256": observation.content_sha256,
            "file_size_bytes": observation.file_size_bytes,
            "archive_family": family.family_id,
            "dataset": family.dataset,
            "fold": "" if observation.fold is None else observation.fold,
            "seed": "" if observation.seed is None else observation.seed,
            "condition": observation.condition or family.expected_condition,
            "variant": observation.variant or family.expected_variant,
            "evaluation_split": observation.evaluation_split,
            "observed_source": source,
            "observed_model": model,
            "observed_backbone": backbone,
            "_observed_dataset": string_value(metadata.get("dataset"), "dataset"),
            "_expected_source_token": family.expected_source_token,
            "_family_projected": family.projected,
            "_requires_explicit_backbone": family.require_explicit_backbone,
            "observed_metadata_json_sha256": observation.metadata_sha256,
            "observed_metadata_keys": observation.metadata_keys,
            "metadata_json_present": "true",
            "metadata_json_record_count": 1,
            "expected_source_from_path": family.expected_source,
            "expected_model_family": family.expected_model_family,
            "expected_backbone_from_path": family.expected_backbone,
            "expected_dataset_from_path": family.dataset,
            "expected_condition_from_path": family.expected_condition,
            "expected_variant_from_path": family.expected_variant,
            "source_label_conflict": bool_text("source_label_conflict" in classes),
            "model_backbone_label_conflict": bool_text(
                "model_backbone_label_conflict" in classes
            ),
            "backbone_path_conflict": bool_text("backbone_path_conflict" in classes),
            "dataset_path_conflict": bool_text("dataset_path_conflict" in classes),
            "duplicate_path_conflict": "false",
            "metadata_missing": "false",
            "metadata_malformed": "false",
            "conflict_class": ";".join(classes) if classes else "none",
            "conflict_evidence_basis": basis,
            "canonical_source": canonical_source,
            "canonical_model": canonical_model,
            "canonical_backbone": canonical_backbone,
            "canonical_resolution_status": status,
            "resolution_confidence": confidence,
            "resolution_evidence_type": evidence_type,
            "resolution_evidence_reference": evidence_reference,
            "resolution_notes": notes,
            "evidence_needed_for_adjudication": (
                UNRESOLVED_ADJUDICATION_EVIDENCE_NEEDED
                if unresolved
                else ""
            ),
            "generator_config_needed": bool_text(unresolved),
            "run_log_needed": bool_text(unresolved),
            "source_commit_needed": bool_text(unresolved),
            "archive_hash_comparison_needed": "true",
            "human_review_needed": bool_text(unresolved),
            "_evidence_verified": evidence_verified,
            "_evidence_record_scope": (
                family.evidence.record_scope if family.evidence else "none"
            ),
            "_archive_specific_evidence_verified": False,
            "_referenced_objects_available": evidence_verified,
            "_referenced_commit_available": evidence_verified,
            "_referenced_config_log_available": evidence_verified,
            "_historical_output_binding_verified": False,
            "_deterministic_internal_proof_verified": False,
            "_evidence_candidate_count": 0,
            "_archive_specific_evidence": {},
        }
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["canonical_path"])), fallback_conflicts


def issue_values(row: dict[str, Any], conflict_class: str) -> tuple[str, str]:
    if conflict_class == "source_label_conflict":
        return str(row["observed_source"]), str(row["expected_source_from_path"])
    if conflict_class == "model_backbone_label_conflict":
        return str(row["observed_model"]), str(row["canonical_model"])
    if conflict_class == "backbone_path_conflict":
        return str(row["observed_backbone"]), str(row["expected_backbone_from_path"])
    if conflict_class == "dataset_path_conflict":
        return str(row.get("_observed_dataset", "")), str(
            row["expected_dataset_from_path"]
        )
    return str(row["content_sha256"]), "distinct paths may legitimately share content"


def scientific_impact_statement(conflict_class: str) -> str:
    if conflict_class == "source_label_conflict":
        return (
            "Metadata-lineage inconsistency only; the proposed source value is not an "
            "adjudicated correction and does not imply that training or metrics are invalid."
        )
    if conflict_class == "model_backbone_label_conflict":
        return (
            "Observed model label conflicts with path-consistent backbone metadata; the "
            "proposed model value is not proof of the historical producing backbone."
        )
    if conflict_class == "duplicate_content":
        return (
            "Content equivalence under distinct paths may reflect copied or deterministic "
            "outputs and does not establish a scientific defect."
        )
    return (
        "Metadata-lineage inconsistency only; no scientific-invalidity conclusion is made."
    )


def make_conflict_rows(manifest_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        classes = [
            item
            for item in str(manifest_row["conflict_class"]).split(";")
            if item and item != "none"
        ]
        basis_items = str(manifest_row["conflict_evidence_basis"]).split("; ")
        for index, conflict_class in enumerate(classes):
            observed, expected = issue_values(manifest_row, conflict_class)
            rows.append(
                {
                    "archive_id": manifest_row["archive_id"],
                    "canonical_path": manifest_row["canonical_path"],
                    "conflict_class": conflict_class,
                    "observed_value": observed,
                    "expected_value": expected,
                    "evidence_basis": (
                        basis_items[index]
                        if index < len(basis_items)
                        else manifest_row["conflict_evidence_basis"]
                    ),
                    "current_resolution_status": manifest_row[
                        "canonical_resolution_status"
                    ],
                    "resolution_confidence": manifest_row["resolution_confidence"],
                    "evidence_available": (
                        f"{manifest_row['resolution_evidence_type']}|"
                        f"{manifest_row['resolution_evidence_reference']}"
                    ),
                    "evidence_missing": (
                        "archive_specific_run_id|exact_output_path_binding|"
                        "exact_fold_seed_condition_variant_record|producing_invocation|"
                        "verified_run_manifest|historical_archive_output_hash_binding"
                    ),
                    "recommended_next_action": (
                        "Locate or create a verified historical run manifest that uniquely "
                        "links this archive path and invocation identifiers to an output "
                        "hash; retain the expected value as a proposal until then."
                    ),
                    "scientific_impact_statement": scientific_impact_statement(
                        conflict_class
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            str(row["canonical_path"]),
            CONFLICT_ORDER[str(row["conflict_class"])],
        ),
    )


def manifest_conflict_classes(row: dict[str, Any]) -> set[str]:
    raw = str(row.get("conflict_class", "none"))
    if raw == "none":
        return set()
    parts = raw.split(";")
    if not raw or any(not part or part == "none" for part in parts):
        raise ManifestValidationError(f"invalid_conflict_class_encoding:{raw}")
    unknown = sorted(set(parts) - set(CONFLICT_ORDER))
    if unknown:
        raise ManifestValidationError(f"invalid_conflict_class:{unknown[0]}")
    if len(parts) != len(set(parts)):
        raise ManifestValidationError(f"duplicate_manifest_conflict_class:{raw}")
    return set(parts)


def manifest_conflict_flag_classes(row: dict[str, Any]) -> set[str]:
    classes: set[str] = set()
    for flag, conflict_class in CONFLICT_FLAG_TO_CLASS.items():
        value = str(row.get(flag, "false"))
        if value not in {"true", "false"}:
            raise ManifestValidationError(f"invalid_conflict_flag_value:{flag}:{value}")
        if value == "true":
            classes.add(conflict_class)
    duplicate_path_value = str(row.get("duplicate_path_conflict", "false"))
    if duplicate_path_value not in {"true", "false"}:
        raise ManifestValidationError(
            "invalid_conflict_flag_value:duplicate_path_conflict:"
            + duplicate_path_value
        )
    if duplicate_path_value == "true":
        raise ManifestValidationError("duplicate_path_conflict_flag_true")
    return classes


def conflict_key(
    row: dict[str, Any], conflict_class: str
) -> tuple[str, str, str]:
    return (
        str(row.get("archive_id", "")),
        str(row.get("canonical_path", "")),
        conflict_class,
    )


def validate_identity_rows(rows: Sequence[dict[str, Any]]) -> None:
    ids = [str(row["archive_id"]) for row in rows]
    paths = [str(row["canonical_path"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ManifestValidationError("duplicate_archive_id")
    normalized_paths = [os.path.normcase(path) for path in paths]
    if len(normalized_paths) != len(set(normalized_paths)):
        raise ManifestValidationError("duplicate_canonical_path")


def derive_lineage_classes_from_row(row: dict[str, Any]) -> set[str]:
    classes: set[str] = set()
    source = str(row.get("observed_source", ""))
    model = str(row.get("observed_model", ""))
    backbone = str(row.get("observed_backbone", ""))
    expected_backbone = str(row.get("expected_backbone_from_path", ""))
    if row.get("_family_projected") is True:
        source_token = str(row.get("_expected_source_token", "")).lower()
        if not source_token or source_token not in source.lower():
            classes.add("source_label_conflict")
    backbone_matches = bool(
        backbone
        and expected_backbone
        and expected_backbone.lower() in backbone.lower()
    )
    if backbone and expected_backbone and not backbone_matches:
        classes.add("backbone_path_conflict")
    if backbone and backbone_matches and backbone.lower() not in model.lower():
        classes.add("model_backbone_label_conflict")
    observed_dataset = str(row.get("_observed_dataset", ""))
    expected_dataset = str(
        row.get("expected_dataset_from_path", row.get("dataset", ""))
    )
    if observed_dataset and expected_dataset and not dataset_matches(
        expected_dataset, observed_dataset
    ):
        classes.add("dataset_path_conflict")
    expected_condition = str(row.get("expected_condition_from_path", ""))
    if expected_condition and str(row.get("condition", "")) != expected_condition:
        raise ManifestValidationError("manifest_condition_expectation_mismatch")
    expected_variant = str(row.get("expected_variant_from_path", ""))
    if expected_variant and str(row.get("variant", "")) != expected_variant:
        raise ManifestValidationError("manifest_variant_expectation_mismatch")
    return classes


def archive_binding_reference(binding: dict[str, Any]) -> str:
    ordered_fields = (
        *ARCHIVE_BINDING_ROW_FIELDS,
        *ARCHIVE_BINDING_EVIDENCE_FIELDS,
        "historical_output_sha256",
        "verified_manifest_reference",
        "internal_proof_identifier",
    )
    parts = ["archive_binding_v1"]
    for field in ordered_fields:
        value = str(binding.get(field, ""))
        if "|" in value or "\n" in value or "\r" in value:
            raise ManifestValidationError(
                f"archive_specific_evidence_value_not_serializable:{field}"
            )
        parts.append(f"{field}={value}")
    return "|".join(parts)


def validate_correction_git_path(field: str, value: str) -> None:
    if (
        not value
        or re.match(r"^[A-Za-z]:[\\/]", value)
        or value.startswith(("/", "\\\\"))
        or "\\" in value
        or ":" in value
        or "\0" in value
        or ".." in Path(value).parts
    ):
        raise ManifestValidationError(f"corrected_invalid_git_evidence_path:{field}")


def correction_git_blob(
    repo_root: Path, commit: str, path: str, evidence_class: str
) -> bytes:
    completed = subprocess.run(
        ["git", "cat-file", "blob", f"{commit}:{path}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ManifestValidationError(
            f"corrected_referenced_{evidence_class}_unavailable:{path}"
        )
    return completed.stdout


def correction_evidence_records(blob: bytes, evidence_class: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(blob.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(
            f"corrected_{evidence_class}_record_malformed"
        ) from exc
    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        records = payload
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ManifestValidationError(
            f"corrected_{evidence_class}_records_not_mappings"
        )
    return records


def unique_correction_record(
    records: Sequence[dict[str, Any]],
    binding: dict[str, Any],
    evidence_class: str,
) -> dict[str, Any]:
    run_identifier = str(binding["run_identifier"])
    candidates = [
        record
        for record in records
        if str(record.get("run_identifier", "")) == run_identifier
    ]
    if not candidates:
        raise ManifestValidationError(
            f"corrected_archive_specific_record_missing:{evidence_class}"
        )
    if len(candidates) != 1:
        raise ManifestValidationError(
            f"corrected_ambiguous_evidence_candidates:{len(candidates)}:"
            f"{evidence_class}"
        )
    record = candidates[0]
    for field in ARCHIVE_BINDING_ROW_FIELDS:
        if str(record.get(field, "")) != str(binding[field]):
            raise ManifestValidationError(
                f"corrected_{evidence_class}_binding_mismatch:{field}"
            )
    for field in (
        "generator_path",
        "config_identifier",
        "log_record_identifier",
        "run_identifier",
    ):
        if str(record.get(field, "")) != str(binding[field]):
            raise ManifestValidationError(
                f"corrected_{evidence_class}_binding_mismatch:{field}"
            )
    return record


def validate_archive_specific_correction_evidence(
    row: dict[str, Any], repo_root: Path | None
) -> None:
    binding = row.get("_archive_specific_evidence")
    if not isinstance(binding, dict):
        raise ManifestValidationError("corrected_archive_specific_binding_not_mapping")
    for field in ARCHIVE_BINDING_ROW_FIELDS:
        row_value = str(row.get(field, ""))
        if not row_value:
            raise ManifestValidationError(
                f"corrected_missing_archive_identity_binding:{field}"
            )
        if str(binding.get(field, "")) != row_value:
            raise ManifestValidationError(
                f"corrected_archive_binding_mismatch:{field}"
            )
    if not str(row.get("condition", "")) and not str(row.get("variant", "")):
        raise ManifestValidationError("corrected_missing_condition_or_variant_binding")
    archive_id_value = str(row.get("archive_id", ""))
    content_sha = str(row.get("content_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", archive_id_value):
        raise ManifestValidationError("corrected_invalid_archive_id_binding")
    if not re.fullmatch(r"[0-9a-f]{64}", content_sha):
        raise ManifestValidationError("corrected_invalid_content_sha256_binding")
    if archive_id_value != archive_id(
        str(row.get("canonical_path", "")), content_sha
    ):
        raise ManifestValidationError("corrected_archive_id_derivation_mismatch")
    for field in ARCHIVE_BINDING_EVIDENCE_FIELDS:
        value = str(binding.get(field, ""))
        if not value:
            raise ManifestValidationError(
                f"corrected_missing_archive_specific_evidence:{field}"
            )
        if value.lower() in {"generic", "unknown", "unspecified", "not_available"}:
            raise ManifestValidationError(
                f"corrected_generic_archive_specific_evidence:{field}"
            )
    commit = str(binding["commit"])
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ManifestValidationError("corrected_invalid_evidence_commit")
    for field in ("generator_path", "config_identifier", "log_record_identifier"):
        validate_correction_git_path(field, str(binding[field]))
    verified_manifest_reference = str(binding.get("verified_manifest_reference", ""))
    if verified_manifest_reference:
        validate_correction_git_path(
            "verified_manifest_reference", verified_manifest_reference
        )
    expected_reference = archive_binding_reference(binding)
    if str(row.get("resolution_evidence_reference", "")) != expected_reference:
        raise ManifestValidationError(
            "corrected_evidence_reference_not_archive_specific"
        )
    evidence_type = str(row.get("resolution_evidence_type", ""))
    historical_output_sha256 = str(binding.get("historical_output_sha256", ""))
    internal_proof_identifier = str(binding.get("internal_proof_identifier", ""))
    if evidence_type == "deterministic_internal_proof":
        if not internal_proof_identifier or not verified_manifest_reference:
            raise ManifestValidationError(
                "corrected_without_verified_deterministic_internal_proof"
            )
    elif not historical_output_sha256 or not verified_manifest_reference:
        if evidence_type == "content_hash_verified_manifest":
            raise ManifestValidationError(
                "current_state_hash_not_historical_output_proof"
            )
        raise ManifestValidationError(
            "corrected_without_historical_or_internal_proof"
        )
    elif historical_output_sha256 != content_sha:
        raise ManifestValidationError("historical_output_hash_binding_mismatch")
    if repo_root is None:
        raise ManifestValidationError("corrected_evidence_repository_context_missing")
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ManifestValidationError(
            f"corrected_referenced_commit_unavailable:{commit}"
        )
    correction_git_blob(
        repo_root, commit, str(binding["generator_path"]), "generator"
    )
    config_records = correction_evidence_records(
        correction_git_blob(
            repo_root, commit, str(binding["config_identifier"]), "config"
        ),
        "config",
    )
    log_records = correction_evidence_records(
        correction_git_blob(
            repo_root, commit, str(binding["log_record_identifier"]), "log"
        ),
        "log",
    )
    manifest_records = correction_evidence_records(
        correction_git_blob(
            repo_root,
            commit,
            verified_manifest_reference,
            "verified_manifest",
        ),
        "verified_manifest",
    )
    unique_correction_record(config_records, binding, "config")
    unique_correction_record(log_records, binding, "log")
    manifest_record = unique_correction_record(
        manifest_records, binding, "verified_manifest"
    )
    if evidence_type == "deterministic_internal_proof":
        if str(manifest_record.get("internal_proof_identifier", "")) != (
            internal_proof_identifier
        ):
            raise ManifestValidationError(
                "deterministic_internal_proof_id_mismatch"
            )
    elif str(manifest_record.get("historical_output_sha256", "")) != (
        historical_output_sha256
    ):
        raise ManifestValidationError("historical_output_manifest_binding_mismatch")


def validate_resolution_row(
    row: dict[str, Any], repo_root: Path | None = None
) -> None:
    status = str(row.get("canonical_resolution_status", ""))
    confidence = str(row.get("resolution_confidence", ""))
    if status not in ALLOWED_RESOLUTION_STATUSES:
        raise ManifestValidationError(f"invalid_resolution_status:{status}")
    if confidence not in ALLOWED_CONFIDENCE_VALUES:
        raise ManifestValidationError(f"invalid_resolution_confidence:{confidence}")
    if confidence not in ALLOWED_CONFIDENCE_BY_STATUS[status]:
        raise ManifestValidationError(
            f"invalid_resolution_status_confidence_pairing:{status}:{confidence}"
        )
    evidence_type = str(row.get("resolution_evidence_type", ""))
    if status != "corrected" and evidence_type == "content_hash_verified_manifest":
        raise ManifestValidationError("current_state_hash_not_historical_output_proof")
    if status != "corrected" and evidence_type in CORRECTION_EVIDENCE_TYPES:
        raise ManifestValidationError(
            "correction_evidence_type_on_noncorrected_status"
        )
    encoded_classes = manifest_conflict_classes(row)
    flagged_classes = manifest_conflict_flag_classes(row)
    if encoded_classes != flagged_classes | ({"duplicate_content"} & encoded_classes):
        raise ManifestValidationError("manifest_conflict_class_flag_mismatch")
    classes = encoded_classes | flagged_classes
    lineage_classes = classes - {"duplicate_content"}
    derived_lineage_classes = derive_lineage_classes_from_row(row)
    if derived_lineage_classes != lineage_classes:
        missing = sorted(derived_lineage_classes - lineage_classes)
        unexpected = sorted(lineage_classes - derived_lineage_classes)
        raise ManifestValidationError(
            "manifest_observed_expectation_conflict_mismatch:"
            f"missing={','.join(missing) or 'none'}:"
            f"unexpected={','.join(unexpected) or 'none'}"
        )
    if classes and not str(row.get("conflict_evidence_basis", "")).strip():
        raise ManifestValidationError("conflict_without_evidence_basis")
    lineage_fields = {
        "source_label_conflict": ("observed_source", "canonical_source"),
        "model_backbone_label_conflict": ("observed_model", "canonical_model"),
        "backbone_path_conflict": ("observed_backbone", "canonical_backbone"),
    }
    for conflict_class, (observed_field, canonical_field) in lineage_fields.items():
        if conflict_class in lineage_classes:
            continue
        if str(row.get(canonical_field, "")) != str(row.get(observed_field, "")):
            raise ManifestValidationError(
                f"nonconflicting_canonical_value_changed:{canonical_field}"
            )
    if status == "confirmed" and lineage_classes:
        raise ManifestValidationError("confirmed_with_lineage_conflict")
    if status == "confirmed" and str(
        row.get("expected_backbone_from_path", "")
    ) and not str(row.get("observed_backbone", "")):
        raise ManifestValidationError("confirmed_without_observed_backbone")
    if status == "legacy-optional":
        if lineage_classes:
            raise ManifestValidationError("legacy_optional_has_conflict")
        if str(row.get("observed_backbone", "")) or str(
            row.get("canonical_backbone", "")
        ):
            raise ManifestValidationError(
                "legacy_optional_without_missing_optional_backbone"
            )
        if (
            str(row.get("resolution_evidence_type", ""))
            != "optional_field_absent_without_contradiction"
        ):
            raise ManifestValidationError("legacy_optional_without_optional_evidence")
        expected_backbone = str(row.get("expected_backbone_from_path", ""))
        observed_model = str(row.get("observed_model", ""))
        if (
            not expected_backbone
            or expected_backbone.lower() not in observed_model.lower()
        ):
            raise ManifestValidationError("legacy_optional_has_model_contradiction")
    if status == "unresolved":
        if not lineage_classes:
            raise ManifestValidationError("unresolved_without_lineage_conflict")
        if str(row.get("resolution_evidence_type", "")) != PROPOSED_CANONICAL_EVIDENCE_TYPE:
            raise ManifestValidationError("unresolved_without_proposed_evidence_type")
        notes = str(row.get("resolution_notes", "")).lower()
        if "proposed" not in notes or "not adjudicated" not in notes:
            raise ManifestValidationError("unresolved_without_proposed_canonical_marker")
        if (
            str(row.get("evidence_needed_for_adjudication", ""))
            != UNRESOLVED_ADJUDICATION_EVIDENCE_NEEDED
        ):
            raise ManifestValidationError(
                "unresolved_missing_adjudication_evidence_record_mismatch"
            )
        for field in (
            "generator_config_needed",
            "run_log_needed",
            "source_commit_needed",
            "archive_hash_comparison_needed",
            "human_review_needed",
        ):
            if str(row.get(field, "")).lower() != "true":
                raise ManifestValidationError(
                    f"unresolved_adjudication_need_flag_not_true:{field}"
                )
        if (
            row.get("_archive_specific_evidence_verified") is True
            or row.get("_historical_output_binding_verified") is True
            or row.get("_deterministic_internal_proof_verified") is True
            or bool(row.get("_archive_specific_evidence"))
        ):
            raise ManifestValidationError(
                "unresolved_with_archive_specific_adjudication_evidence"
            )
        proposed_fields = {
            "source_label_conflict": "canonical_source",
            "model_backbone_label_conflict": "canonical_model",
            "backbone_path_conflict": "canonical_backbone",
        }
        observed_fields = {
            "source_label_conflict": "observed_source",
            "model_backbone_label_conflict": "observed_model",
            "backbone_path_conflict": "observed_backbone",
        }
        for conflict_class, field in proposed_fields.items():
            if conflict_class not in lineage_classes:
                continue
            proposed_value = str(row.get(field, ""))
            if not proposed_value:
                raise ManifestValidationError(
                    f"unresolved_missing_proposed_canonical_value:{field}"
                )
            if proposed_value == str(row.get(observed_fields[conflict_class], "")):
                raise ManifestValidationError(
                    f"unresolved_proposed_value_matches_observed:{field}"
                )
    if status == "corrected":
        if not lineage_classes:
            raise ManifestValidationError("corrected_without_lineage_conflict")
        evidence_type = str(row.get("resolution_evidence_type", ""))
        reference = str(row.get("resolution_evidence_reference", ""))
        if evidence_type not in CORRECTION_EVIDENCE_TYPES:
            raise ManifestValidationError("corrected_without_sufficient_evidence_type")
        if not reference:
            raise ManifestValidationError("corrected_without_verified_evidence")
        changed = any(
            str(row.get(canonical, "")) != str(row.get(observed, ""))
            for canonical, observed in (
                ("canonical_source", "observed_source"),
                ("canonical_model", "observed_model"),
                ("canonical_backbone", "observed_backbone"),
            )
        )
        if not changed:
            raise ManifestValidationError("corrected_without_canonical_change")
        validate_archive_specific_correction_evidence(row, repo_root)


def validate_cross_tables(
    manifest_rows: Sequence[dict[str, Any]],
    conflict_rows: Sequence[dict[str, Any]],
) -> None:
    manifest_classes: list[tuple[dict[str, Any], set[str], set[str]]] = []
    manifest_set: set[tuple[str, str, str]] = set()
    for row in manifest_rows:
        encoded = manifest_conflict_classes(row)
        flagged = manifest_conflict_flag_classes(row)
        manifest_classes.append((row, encoded, flagged))
        for item in encoded | flagged:
            manifest_set.add(conflict_key(row, item))

    conflict_keys: list[tuple[str, str, str]] = []
    for row in conflict_rows:
        conflict_class = str(row.get("conflict_class", ""))
        if conflict_class not in CONFLICT_ORDER:
            raise ManifestValidationError(
                f"invalid_conflict_csv_class:{conflict_class}"
            )
        if not str(row.get("evidence_basis", "")).strip():
            raise ManifestValidationError("conflict_csv_without_evidence_basis")
        if not str(row.get("archive_id", "")) or not str(
            row.get("canonical_path", "")
        ):
            raise ManifestValidationError("conflict_csv_missing_identity")
        conflict_keys.append(conflict_key(row, conflict_class))
    conflict_set = set(conflict_keys)
    missing = sorted(manifest_set - conflict_set)
    extra = sorted(conflict_set - manifest_set)
    if missing:
        raise ManifestValidationError(
            "manifest_conflict_missing_from_conflict_csv:" + repr(missing[0])
        )
    if extra:
        raise ManifestValidationError(
            "conflict_csv_row_not_in_manifest:" + repr(extra[0])
        )
    if len(conflict_keys) != len(conflict_set):
        raise ManifestValidationError("duplicate_conflict_csv_issue")
    for row, encoded, flagged in manifest_classes:
        encoded_flags = encoded & set(CONFLICT_FLAG_TO_CLASS.values())
        if flagged - encoded_flags:
            item = sorted(flagged - encoded_flags)[0]
            raise ManifestValidationError(
                "manifest_conflict_flag_not_in_conflict_class:"
                + repr(conflict_key(row, item))
            )
        if encoded_flags - flagged:
            item = sorted(encoded_flags - flagged)[0]
            raise ManifestValidationError(
                "manifest_conflict_class_flag_not_true:"
                + repr(conflict_key(row, item))
            )


def validate_no_absolute_leakage(
    rows: Iterable[dict[str, Any]], repo_root: Path
) -> None:
    root_texts = {
        str(repo_root.resolve()).replace("\\", "/").lower(),
        str(repo_root.resolve()).lower(),
    }
    for row in rows:
        for key, raw_value in row.items():
            if str(key).startswith("_"):
                continue
            value = str(raw_value)
            lowered = value.lower()
            if any(root and root in lowered for root in root_texts):
                raise ManifestValidationError(f"absolute_path_leakage:{key}")
            if re.search(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]", value):
                raise ManifestValidationError(f"absolute_path_leakage:{key}")
            if "\\\\" in value:
                raise ManifestValidationError(f"absolute_path_leakage:{key}")
            if re.search(
                r"(?:^|[\s=|;,])/(?:[^/\s]+)(?:/[^/\s]+)+", value
            ):
                raise ManifestValidationError(f"absolute_path_leakage:{key}")
            if key in {"canonical_path", "relative_path"} and value.startswith("/"):
                raise ManifestValidationError(f"absolute_path_leakage:{key}")


def validate_baselines(
    manifest_rows: Sequence[dict[str, Any]],
    conflict_rows: Sequence[dict[str, Any]],
    fallback_conflicts: int,
    duplicate_groups: Sequence[Sequence[str]],
) -> dict[str, int]:
    if len(manifest_rows) != EXPECTED_ARCHIVE_COUNT:
        raise ManifestValidationError(
            f"manifest_archive_count_mismatch:{len(manifest_rows)}"
        )
    source_ids = {
        str(row["archive_id"])
        for row in manifest_rows
        if row["source_label_conflict"] == "true"
    }
    model_ids = {
        str(row["archive_id"])
        for row in manifest_rows
        if row["model_backbone_label_conflict"] == "true"
    }
    counts = {
        "source_label_conflicts": len(source_ids),
        "explicit_gated_backbone_model_label_conflicts": len(model_ids),
        "backbone_path_conflicts": sum(
            row["backbone_path_conflict"] == "true" for row in manifest_rows
        ),
        "dataset_path_conflicts": sum(
            row["dataset_path_conflict"] == "true" for row in manifest_rows
        ),
        "conflict_set_overlap": len(source_ids & model_ids),
        "duplicate_path_conflicts": sum(
            row["duplicate_path_conflict"] == "true" for row in manifest_rows
        ),
        "duplicate_content_groups": len(duplicate_groups),
        "optional_backbone_legacy_archives": sum(
            not str(row["observed_backbone"]) for row in manifest_rows
        ),
        "gated_fallback_conflicts": fallback_conflicts,
        "malformed_metadata": sum(
            row["metadata_malformed"] == "true" for row in manifest_rows
        ),
    }
    expected = {
        "source_label_conflicts": EXPECTED_SOURCE_CONFLICTS,
        "explicit_gated_backbone_model_label_conflicts": EXPECTED_MODEL_CONFLICTS,
        "backbone_path_conflicts": EXPECTED_BACKBONE_PATH_CONFLICTS,
        "dataset_path_conflicts": EXPECTED_DATASET_PATH_CONFLICTS,
        "conflict_set_overlap": EXPECTED_CONFLICT_OVERLAP,
        "duplicate_path_conflicts": 0,
        "optional_backbone_legacy_archives": EXPECTED_OPTIONAL_BACKBONE_ABSENCES,
        "gated_fallback_conflicts": EXPECTED_FALLBACK_CONFLICTS,
        "malformed_metadata": 0,
    }
    for key, expected_value in expected.items():
        if counts[key] != expected_value:
            raise ManifestValidationError(
                f"known_count_mismatch:{key}:expected={expected_value}:observed={counts[key]}"
            )
    resolution_counts = Counter(
        str(row["canonical_resolution_status"]) for row in manifest_rows
    )
    for status, expected_value in EXPECTED_RESOLUTION_COUNTS.items():
        if resolution_counts.get(status, 0) != expected_value:
            raise ManifestValidationError(
                "resolution_count_mismatch:"
                f"{status}:expected={expected_value}:observed={resolution_counts.get(status, 0)}"
            )
    duplicate_content_issues = sum(len(group) for group in duplicate_groups)
    expected_conflict_rows = (
        EXPECTED_SOURCE_CONFLICTS
        + EXPECTED_MODEL_CONFLICTS
        + duplicate_content_issues
    )
    if len(conflict_rows) != expected_conflict_rows:
        raise ManifestValidationError(
            "conflict_csv_count_mismatch:"
            f"expected={expected_conflict_rows}:observed={len(conflict_rows)}"
        )
    return counts


def render_csv(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(columns),
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return stream.getvalue().encode("utf-8")


def inventory_fingerprint(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["canonical_path"])):
        digest.update(str(row["canonical_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["content_sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["file_size_bytes"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def render_report(
    summary: dict[str, Any],
    family_counts: dict[str, int],
    dataset_counts: dict[str, int],
    duplicate_groups: Sequence[Sequence[str]],
    manifest_sha256: str,
    conflicts_sha256: str,
) -> str:
    family_lines = [
        f"| `{family}` | {count} |" for family, count in family_counts.items()
    ]
    dataset_lines = [
        f"| `{dataset}` | {count} |" for dataset, count in dataset_counts.items()
    ]
    conflicts = summary["conflict_counts"]
    resolutions = summary["resolution_counts"]
    evidence = summary["evidence_availability"]
    duplicate_note = (
        "No duplicate-content groups were found."
        if not duplicate_groups
        else (
            f"{len(duplicate_groups)} duplicate-content groups were found. Distinct paths "
            "with identical bytes are recorded as possible copied or deterministic equivalents, "
            "not as evidence of a scientific defect."
        )
    )
    lines = [
        "# Paired-acquisition provenance validation report",
        "",
        "## Scope and status",
        "",
        (
            "Status: **PASS** for deterministic inventory and validation in the present "
            "local artifact workspace; all metadata-lineage conflicts remain unresolved."
        ),
        "",
        (
            "This is a read-only metadata-lineage and evidence-reconciliation audit. It does "
            "not run training, rewrite source archives, modify experiment outputs, or change "
            "scientific claims."
        ),
        "",
        "## Inventory",
        "",
        f"- Total archives: {summary['total_archives']}",
        f"- Total raw bytes: {summary['total_bytes']}",
        f"- Archives with metadata_json: {summary['archives_with_metadata_json']}",
        (
            "- Archives without optional explicit backbone metadata: "
            f"{conflicts['optional_backbone_legacy_archives']}"
        ),
        f"- Malformed metadata count: {conflicts['malformed_metadata']}",
        f"- Duplicate canonical path count: {conflicts['duplicate_path_conflicts']}",
        f"- Duplicate content group count: {conflicts['duplicate_content_groups']}",
        f"- Unique content hashes: {summary['unique_content_hashes']}",
        "",
        "### Family counts",
        "",
        "| Archive family | Count |",
        "|---|---:|",
        *family_lines,
        "",
        "### Dataset counts",
        "",
        "| Dataset | Count |",
        "|---|---:|",
        *dataset_lines,
        "",
        "## Conflict counts",
        "",
        f"- Source-label conflicts: {conflicts['source_label_conflicts']}",
        (
            "- Explicit gated backbone/model-label conflicts: "
            f"{conflicts['explicit_gated_backbone_model_label_conflicts']}"
        ),
        f"- Backbone/path conflicts: {conflicts['backbone_path_conflicts']}",
        f"- Dataset/path conflicts: {conflicts['dataset_path_conflicts']}",
        f"- Conflict-set overlap: {conflicts['conflict_set_overlap']}",
        f"- Optional-backbone fallback conflicts: {conflicts['gated_fallback_conflicts']}",
        "",
        "These are metadata-lineage findings. They do not establish that a different dataset "
        "or backbone generated the features and do not determine scientific validity.",
        "",
        "## Resolution counts",
        "",
        f"- confirmed: {resolutions['confirmed']}",
        f"- corrected: {resolutions['corrected']}",
        f"- unresolved: {resolutions['unresolved']}",
        f"- legacy-optional: {resolutions['legacy-optional']}",
        "",
        (
            "The 350 conflict rows remain unresolved. Their conflicting canonical field carries "
            "a medium-confidence proposed value derived from path/family expectations, internal "
            "metadata, and compatible group-level run evidence; it is not an adjudicated "
            "correction."
        ),
        (
            "The 226 optional-backbone absences are counted separately: 200 occur on rows "
            "with an unresolved source-label conflict, while 26 are optional-only and therefore "
            "receive the archive-level legacy-optional status."
        ),
        (
            "The 50 confirmed rows have complete current observed-metadata agreement with "
            "applicable lineage expectations; confirmed does not assert historical byte origin."
        ),
        "",
        "## Evidence availability",
        "",
        (
            "- Archives with structured embedded configuration: "
            f"{evidence['structured_embedded_configs']}"
        ),
        (
            "- Archives with group/family evidence references: "
            f"{evidence['group_evidence_reference_archives']}"
        ),
        (
            "- Archives with associated family-level run logs: "
            f"{evidence['associated_family_level_run_logs']}"
        ),
        (
            "- Archives with family-level source/result commit associations: "
            f"{evidence['family_level_source_result_commit_associations']}"
        ),
        (
            "- Conflict archives with compatible per-run text records: "
            f"{evidence['conflict_archives_with_compatible_per_run_records']}"
        ),
        (
            "- Conflict archives with aggregate-only text records: "
            f"{evidence['conflict_archives_with_aggregate_only_records']}"
        ),
        (
            "- Corrected rows with archive-specific adjudication evidence: "
            f"{evidence['corrected_rows_with_archive_specific_adjudication_evidence']}"
        ),
        (
            "- Conflict archives lacking adjudication evidence: "
            f"{evidence['conflict_archives_lacking_adjudication_evidence']}"
        ),
        (
            "- Unresolved rows with a proposed canonical conflict value: "
            f"{evidence['proposed_canonical_value_rows']}"
        ),
        (
            "- Archives with current-state content hashes: "
            f"{evidence['current_state_content_hashes']}"
        ),
        (
            "- Archives with historical cryptographic output binding: "
            f"{evidence['archives_with_historical_cryptographic_output_binding']}"
        ),
        "",
        (
            "None of the 426 source NPZ paths is Git-tracked. Their public availability is "
            "not established by this package. Deterministic checking requires a workspace "
            "where the same archives and referenced Git objects are separately available."
        ),
        "",
        "## Duplicate-content assessment",
        "",
        duplicate_note,
        "",
        "## Deterministic fingerprints",
        "",
        f"- Archive inventory fingerprint: `{summary['inventory_fingerprint_sha256']}`",
        f"- provenance_manifest.csv SHA-256: `{manifest_sha256}`",
        f"- provenance_conflicts.csv SHA-256: `{conflicts_sha256}`",
        "",
        "## Limitations",
        "",
        (
            "1. Path and family expectations are lineage-derived proposals, not adjudicated "
            "canonical values."
        ),
        (
            "2. Family-level code and run records do not uniquely bind present bytes to "
            "historical invocations."
        ),
        "3. Present hashes fingerprint current local files only.",
        "4. Public availability of the 426 source archives is not established.",
        (
            "5. Metadata reconciliation does not modify metrics, source archives, or prior "
            "scientific claims."
        ),
        (
            "6. Conflict findings do not establish scientific invalidity, causal provenance, "
            "clinical relevance, or an incorrect historical model or dataset."
        ),
        "7. Crossed-preparation and crossed-site attribution remains future work.",
        "",
    ]
    return "\n".join(lines)


def build(repo_root: Path = DEFAULT_REPO_ROOT) -> BuildResult:
    repo_root = repo_root.resolve(strict=True)
    verified_families = verify_evidence_specs(repo_root)
    discovered = discover_archives(repo_root)
    observations = inspect_archives(repo_root, discovered)
    duplicate_groups = duplicate_content_groups(observations)
    manifest_rows, fallback_conflicts = make_manifest_rows(
        observations, verified_families, duplicate_groups
    )
    conflict_rows = make_conflict_rows(manifest_rows)
    validate_identity_rows(manifest_rows)
    for row in manifest_rows:
        validate_resolution_row(row, repo_root)
    validate_cross_tables(manifest_rows, conflict_rows)
    validate_no_absolute_leakage(manifest_rows, repo_root)
    validate_no_absolute_leakage(conflict_rows, repo_root)
    conflict_counts = validate_baselines(
        manifest_rows, conflict_rows, fallback_conflicts, duplicate_groups
    )
    family_counts = {
        family.family_id: sum(
            row["archive_family"] == family.family_id for row in manifest_rows
        )
        for family in FAMILIES
    }
    dataset_counts = dict(
        sorted(Counter(str(row["dataset"]) for row in manifest_rows).items())
    )
    resolution_counts = {
        status: sum(
            row["canonical_resolution_status"] == status for row in manifest_rows
        )
        for status in ("confirmed", "corrected", "unresolved", "legacy-optional")
    }
    structured_configs = sum(
        isinstance(observation.metadata.get("config"), dict)
        for observation in observations
    )
    archives_with_evidence = sum(
        observation.family.evidence is not None for observation in observations
    )
    lineage_conflict_rows = [
        row
        for row in manifest_rows
        if manifest_conflict_classes(row) - {"duplicate_content"}
    ]
    compatible_per_run_conflicts = sum(
        row.get("_evidence_record_scope") == "compatible_per_run"
        for row in lineage_conflict_rows
    )
    aggregate_only_conflicts = sum(
        row.get("_evidence_record_scope") == "aggregate_only"
        for row in lineage_conflict_rows
    )
    archive_specific_conflicts = sum(
        row["canonical_resolution_status"] == "corrected"
        for row in lineage_conflict_rows
    )
    proposed_canonical_rows = sum(
        row["canonical_resolution_status"] == "unresolved"
        for row in lineage_conflict_rows
    )
    historical_output_hash_bindings = sum(
        row["canonical_resolution_status"] == "corrected"
        and bool(
            dict(row.get("_archive_specific_evidence", {})).get(
                "historical_output_sha256", ""
            )
        )
        for row in manifest_rows
    )
    evidence_baselines = {
        "group_evidence_reference_archives": archives_with_evidence,
        "conflict_archives_with_compatible_per_run_records": (
            compatible_per_run_conflicts
        ),
        "conflict_archives_with_aggregate_only_records": aggregate_only_conflicts,
    }
    expected_evidence_baselines = {
        "group_evidence_reference_archives": EXPECTED_GROUP_EVIDENCE_ARCHIVES,
        "conflict_archives_with_compatible_per_run_records": (
            EXPECTED_CONFLICT_COMPATIBLE_PER_RUN_RECORDS
        ),
        "conflict_archives_with_aggregate_only_records": (
            EXPECTED_CONFLICT_AGGREGATE_ONLY_RECORDS
        ),
    }
    if evidence_baselines != expected_evidence_baselines:
        raise ManifestValidationError(
            "evidence_availability_baseline_mismatch:"
            f"expected={json.dumps(expected_evidence_baselines, sort_keys=True)}:"
            f"observed={json.dumps(evidence_baselines, sort_keys=True)}"
        )
    total_bytes = sum(int(row["file_size_bytes"]) for row in manifest_rows)
    summary: dict[str, Any] = {
        "audit_id": "paired_acquisition_provenance_manifest_v1",
        "status": "passed",
        "total_archives": len(manifest_rows),
        "total_bytes": total_bytes,
        "archives_with_metadata_json": len(manifest_rows),
        "unique_content_hashes": len(
            {str(row["content_sha256"]) for row in manifest_rows}
        ),
        "family_counts": family_counts,
        "dataset_counts": dataset_counts,
        "conflict_counts": conflict_counts,
        "resolution_counts": resolution_counts,
        "evidence_availability": {
            "structured_embedded_configs": structured_configs,
            **evidence_baselines,
            "associated_family_level_run_logs": archives_with_evidence,
            "family_level_source_result_commit_associations": (
                archives_with_evidence
            ),
            "conflict_archives_with_archive_specific_adjudication": (
                archive_specific_conflicts
            ),
            "corrected_rows_with_archive_specific_adjudication_evidence": (
                archive_specific_conflicts
            ),
            "conflicts_lacking_archive_specific_adjudication": (
                len(lineage_conflict_rows) - archive_specific_conflicts
            ),
            "conflict_archives_lacking_adjudication_evidence": (
                len(lineage_conflict_rows) - archive_specific_conflicts
            ),
            "proposed_canonical_value_rows": proposed_canonical_rows,
            "unresolved_rows_with_proposed_canonical_values": (
                proposed_canonical_rows
            ),
            "current_state_content_hashes": len(manifest_rows),
            "historical_output_hash_bindings": historical_output_hash_bindings,
            "archives_with_historical_cryptographic_output_binding": (
                historical_output_hash_bindings
            ),
        },
        "inventory_fingerprint_sha256": inventory_fingerprint(manifest_rows),
        "execution_boundary": {
            "training_run": False,
            "feature_payload_loaded": False,
            "source_archives_modified": False,
            "experiment_outputs_modified": False,
        },
    }
    manifest_bytes = render_csv(manifest_rows, MANIFEST_COLUMNS)
    conflict_bytes = render_csv(conflict_rows, CONFLICT_COLUMNS)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    conflict_sha = hashlib.sha256(conflict_bytes).hexdigest()
    report_text = render_report(
        summary,
        family_counts,
        dataset_counts,
        duplicate_groups,
        manifest_sha,
        conflict_sha,
    )
    report_bytes = report_text.encode("utf-8")
    summary["output_sha256"] = {
        MANIFEST_FILE.name: manifest_sha,
        CONFLICT_FILE.name: conflict_sha,
        REPORT_FILE.name: hashlib.sha256(report_bytes).hexdigest(),
    }
    outputs = {
        MANIFEST_FILE.name: manifest_bytes,
        CONFLICT_FILE.name: conflict_bytes,
        REPORT_FILE.name: report_bytes,
    }
    for name, content in outputs.items():
        if str(repo_root).encode("utf-8") in content or str(repo_root).replace(
            "\\", "/"
        ).encode("utf-8") in content:
            raise ManifestValidationError(f"absolute_path_leakage:rendered:{name}")
    return BuildResult(
        manifest_rows=manifest_rows,
        conflict_rows=conflict_rows,
        family_counts=family_counts,
        dataset_counts=dataset_counts,
        duplicate_content_groups=duplicate_groups,
        summary=summary,
        outputs=outputs,
    )


def atomic_write(path: Path, content: bytes) -> None:
    if path.parent.resolve() != PACKAGE_DIR.resolve():
        raise ManifestValidationError(f"write_outside_package_blocked:{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_outputs(result: BuildResult) -> None:
    for name in (MANIFEST_FILE.name, CONFLICT_FILE.name, REPORT_FILE.name):
        atomic_write(PACKAGE_DIR / name, result.outputs[name])


def check_outputs(result: BuildResult) -> None:
    mismatches: list[str] = []
    for name, expected in result.outputs.items():
        path = PACKAGE_DIR / name
        if not path.is_file():
            mismatches.append(f"missing:{name}")
        elif path.read_bytes() != expected:
            mismatches.append(f"content:{name}")
    if mismatches:
        raise ManifestValidationError("checked_output_mismatch:" + ";".join(mismatches))


def encode_test_npy_text(values: Sequence[str]) -> bytes:
    width = max(1, *(len(value) for value in values))
    shape: tuple[int, ...] = () if len(values) == 1 else (len(values),)
    header = repr({"descr": f"<U{width}", "fortran_order": False, "shape": shape})
    padding = (-((10 + len(header) + 1) % 16)) % 16
    header_bytes = (header + " " * padding + "\n").encode("latin1")
    payload = bytearray()
    for value in values:
        encoded = value.encode("utf-32-le")
        payload.extend(encoded)
        payload.extend(b"\0" * (width * 4 - len(encoded)))
    return (
        b"\x93NUMPY"
        + bytes((1, 0))
        + struct.pack("<H", len(header_bytes))
        + header_bytes
        + bytes(payload)
    )


def write_test_npz(path: Path, metadata_values: Sequence[str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("metadata_json.npy", encode_test_npy_text(metadata_values))


def expect_validation_error(
    name: str, category: str, operation: Callable[[], None]
) -> str:
    try:
        operation()
    except ManifestValidationError as exc:
        if category not in str(exc):
            raise AssertionError(
                f"{name}: expected category {category!r}, observed {str(exc)!r}"
            ) from exc
        return name
    raise AssertionError(f"{name}: expected ManifestValidationError")


def minimal_resolution_row(**updates: Any) -> dict[str, Any]:
    binding_updates = dict(updates.pop("_binding_updates", {}))
    canonical_path = "results/example/fold_0/true_pairs_seed_1/archive.npz"
    content_sha256 = "b" * 64
    archive_id_value = archive_id(canonical_path, content_sha256)
    binding: dict[str, Any] = {
        "archive_id": archive_id_value,
        "canonical_path": canonical_path,
        "content_sha256": content_sha256,
        "archive_family": "example_phikon_true_pair",
        "dataset": "scorpion",
        "fold": 0,
        "seed": 1,
        "condition": "true_pairs",
        "variant": "projected",
        "evaluation_split": "test",
        "canonical_source": "SCORPION DINOv2 projection experiment",
        "canonical_model": "phikon_pathoalign",
        "canonical_backbone": "phikon",
        "commit": "1" * 40,
        "generator_path": "experiments/example/generator.py",
        "config_identifier": "evidence/config.json",
        "log_record_identifier": "evidence/run_log.json",
        "run_identifier": "run:fold=0:seed=1:condition=true_pairs",
        "historical_output_sha256": "b" * 64,
        "verified_manifest_reference": "evidence/verified_manifest.json",
        "internal_proof_identifier": "",
    }
    binding.update(binding_updates)
    row: dict[str, Any] = {
        "archive_id": archive_id_value,
        "canonical_path": canonical_path,
        "content_sha256": content_sha256,
        "archive_family": "example_phikon_true_pair",
        "dataset": "scorpion",
        "fold": 0,
        "seed": 1,
        "condition": "true_pairs",
        "variant": "projected",
        "evaluation_split": "test",
        "observed_source": "SCORPION DINOv2 projection experiment",
        "observed_model": "dinov2_pathoalign",
        "observed_backbone": "phikon",
        "_observed_dataset": "scorpion",
        "_expected_source_token": "scorpion",
        "_family_projected": True,
        "_requires_explicit_backbone": True,
        "expected_source_from_path": "SCORPION",
        "expected_backbone_from_path": "phikon",
        "expected_dataset_from_path": "scorpion",
        "expected_condition_from_path": "true_pairs",
        "expected_variant_from_path": "projected",
        "canonical_source": "SCORPION DINOv2 projection experiment",
        "canonical_model": "phikon_pathoalign",
        "canonical_backbone": "phikon",
        "canonical_resolution_status": "corrected",
        "resolution_confidence": "medium",
        "resolution_evidence_type": "source_commit_and_code_path",
        "resolution_evidence_reference": archive_binding_reference(binding),
        "resolution_notes": "Archive-specific evidence adjudicates the label.",
        "conflict_class": "model_backbone_label_conflict",
        "conflict_evidence_basis": "explicit backbone versus model label",
        "source_label_conflict": "false",
        "model_backbone_label_conflict": "true",
        "backbone_path_conflict": "false",
        "dataset_path_conflict": "false",
        "duplicate_path_conflict": "false",
        "_evidence_verified": True,
        "_archive_specific_evidence_verified": True,
        "_referenced_objects_available": True,
        "_referenced_commit_available": True,
        "_referenced_config_log_available": True,
        "_historical_output_binding_verified": False,
        "_deterministic_internal_proof_verified": False,
        "_evidence_candidate_count": 1,
        "_archive_specific_evidence": binding,
    }
    row.update(updates)
    return row


def minimal_unresolved_row(**updates: Any) -> dict[str, Any]:
    row = minimal_resolution_row(
        canonical_resolution_status="unresolved",
        resolution_confidence="medium",
        resolution_evidence_type=PROPOSED_CANONICAL_EVIDENCE_TYPE,
        resolution_evidence_reference=(
            "family_level_reconstructed_lineage_evidence|"
            "1111111111111111111111111111111111111111:experiments/example/generator.py"
        ),
        resolution_notes=(
            "The conflicting field contains a proposed canonical value; it is not "
            "adjudicated because archive-specific binding is absent."
        ),
        evidence_needed_for_adjudication=(
            UNRESOLVED_ADJUDICATION_EVIDENCE_NEEDED
        ),
        generator_config_needed="true",
        run_log_needed="true",
        source_commit_needed="true",
        archive_hash_comparison_needed="true",
        human_review_needed="true",
        _archive_specific_evidence_verified=False,
        _historical_output_binding_verified=False,
        _deterministic_internal_proof_verified=False,
        _archive_specific_evidence={},
    )
    row.update(updates)
    return row


def create_test_evidence_repository(root: Path) -> tuple[Path, dict[str, Any]]:
    repo = root / "evidence-repository"
    evidence_dir = repo / "evidence"
    generator_dir = repo / "experiments" / "example"
    evidence_dir.mkdir(parents=True)
    generator_dir.mkdir(parents=True)
    seed_row = minimal_resolution_row()
    seed_binding = dict(seed_row["_archive_specific_evidence"])
    record_fields = (
        *ARCHIVE_BINDING_ROW_FIELDS,
        "generator_path",
        "config_identifier",
        "log_record_identifier",
        "run_identifier",
        "historical_output_sha256",
        "internal_proof_identifier",
    )
    record = {field: seed_binding[field] for field in record_fields}
    (generator_dir / "generator.py").write_text(
        "# Deterministic temporary correction-evidence fixture.\n",
        encoding="utf-8",
    )
    evidence_payloads = {
        "config.json": {"records": [record]},
        "ambiguous_config.json": {"records": [record, record]},
        "run_log.json": {"records": [record]},
        "verified_manifest.json": {"records": [record]},
    }
    for name, payload in evidence_payloads.items():
        (evidence_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def run_git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "temporary_git_fixture_failed:"
                + completed.stderr.decode("utf-8", errors="replace")
            )
        return completed.stdout

    run_git("init", "--quiet")
    fixture_paths = (
        "evidence/ambiguous_config.json",
        "evidence/config.json",
        "evidence/run_log.json",
        "evidence/verified_manifest.json",
        "experiments/example/generator.py",
    )
    run_git("add", "--", *fixture_paths)
    run_git(
        "-c",
        "user.name=Provenance Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "Create correction evidence fixture",
    )
    commit = run_git("rev-parse", "HEAD").decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise AssertionError("temporary_git_fixture_commit_invalid")
    return repo, {
        "commit": commit,
        "generator_path": "experiments/example/generator.py",
        "config_identifier": "evidence/config.json",
        "log_record_identifier": "evidence/run_log.json",
        "run_identifier": seed_binding["run_identifier"],
        "historical_output_sha256": seed_binding["historical_output_sha256"],
        "verified_manifest_reference": "evidence/verified_manifest.json",
        "internal_proof_identifier": "",
    }


def load_test_json_fixture(
    root: Path, name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    path = root / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise AssertionError(f"test_fixture_not_mapping:{name}")
    return loaded


def run_self_tests() -> dict[str, Any]:
    passed: list[str] = []
    behavior_passed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="provenance-manifest-tests-") as directory:
        root = Path(directory)
        evidence_repo, evidence_binding = create_test_evidence_repository(root)

        def correction_row(**updates: Any) -> dict[str, Any]:
            binding_updates = dict(evidence_binding)
            binding_updates.update(dict(updates.pop("_binding_updates", {})))
            return minimal_resolution_row(
                _binding_updates=binding_updates, **updates
            )

        duplicate_path_rows = [
            {"archive_id": "a", "canonical_path": "results/a.npz"},
            {"archive_id": "b", "canonical_path": "results/a.npz"},
        ]
        passed.append(
            expect_validation_error(
                "duplicate_canonical_path",
                "duplicate_canonical_path",
                lambda: validate_identity_rows(duplicate_path_rows),
            )
        )
        duplicate_id_rows = [
            {"archive_id": "a", "canonical_path": "results/a.npz"},
            {"archive_id": "a", "canonical_path": "results/b.npz"},
        ]
        passed.append(
            expect_validation_error(
                "duplicate_archive_id",
                "duplicate_archive_id",
                lambda: validate_identity_rows(duplicate_id_rows),
            )
        )
        mutable = root / "mutable.bin"
        mutable.write_bytes(b"before")
        passed.append(
            expect_validation_error(
                "archive_mutation_between_hash_passes",
                "archive_mutated_between_hash_passes",
                lambda: hash_file_twice(
                    mutable, between_passes=lambda: mutable.write_bytes(b"after")
                ),
            )
        )
        malformed = root / "malformed.npz"
        write_test_npz(malformed, ["{"])
        passed.append(
            expect_validation_error(
                "malformed_metadata_json",
                "metadata_json_malformed",
                lambda: parse_metadata_json(malformed),
            )
        )
        multiple = root / "multiple.npz"
        write_test_npz(multiple, ["{}", "{}"])
        passed.append(
            expect_validation_error(
                "multiple_metadata_json_records",
                "metadata_json_record_count_invalid",
                lambda: parse_metadata_json(multiple),
            )
        )
        passed.append(
            expect_validation_error(
                "invalid_resolution_status",
                "invalid_resolution_status",
                lambda: validate_resolution_row(
                    minimal_resolution_row(canonical_resolution_status="invented")
                ),
            )
        )
        passed.append(
            expect_validation_error(
                "corrected_without_evidence",
                "corrected_without_verified_evidence",
                lambda: validate_resolution_row(
                    minimal_resolution_row(resolution_evidence_reference="")
                ),
            )
        )
        passed.append(
            expect_validation_error(
                "corrected_without_lineage_conflict",
                "corrected_without_lineage_conflict",
                lambda: validate_resolution_row(
                    minimal_resolution_row(
                        conflict_class="none",
                        conflict_evidence_basis="",
                        model_backbone_label_conflict="false",
                        observed_model="phikon_pathoalign",
                        canonical_model="phikon_pathoalign",
                    )
                ),
            )
        )
        passed.append(
            expect_validation_error(
                "legacy_optional_counted_as_conflict",
                "legacy_optional_has_conflict",
                lambda: validate_resolution_row(
                    minimal_resolution_row(
                        canonical_resolution_status="legacy-optional",
                        resolution_confidence="not_applicable",
                        resolution_evidence_type=(
                            "optional_field_absent_without_contradiction"
                        ),
                    )
                ),
            )
        )
        passed.append(
            expect_validation_error(
                "legacy_optional_with_backbone_present",
                "legacy_optional_without_missing_optional_backbone",
                lambda: validate_resolution_row(
                    minimal_resolution_row(
                        canonical_resolution_status="legacy-optional",
                        resolution_confidence="not_applicable",
                        resolution_evidence_type=(
                            "optional_field_absent_without_contradiction"
                        ),
                        conflict_class="none",
                        conflict_evidence_basis="",
                        model_backbone_label_conflict="false",
                        observed_model="phikon_pathoalign",
                        canonical_model="phikon_pathoalign",
                    )
                ),
            )
        )
        passed.append(
            expect_validation_error(
                "confirmed_with_lineage_conflict",
                "confirmed_with_lineage_conflict",
                lambda: validate_resolution_row(
                    minimal_resolution_row(
                        canonical_resolution_status="confirmed",
                        resolution_evidence_type="observed_metadata_agreement",
                    )
                ),
            )
        )
        passed.append(
            expect_validation_error(
                "unresolved_without_proposed_marker",
                "unresolved_without_proposed_canonical_marker",
                lambda: validate_resolution_row(
                    minimal_resolution_row(
                        canonical_resolution_status="unresolved",
                        resolution_confidence="medium",
                        resolution_evidence_type=(
                            PROPOSED_CANONICAL_EVIDENCE_TYPE
                        ),
                        resolution_notes="",
                    )
                ),
            )
        )
        path_only = load_test_json_fixture(
            root,
            "corrected_path_only_evidence",
            minimal_resolution_row(
                resolution_evidence_type="path_inference",
                resolution_evidence_reference="results/example/archive.npz",
            ),
        )
        passed.append(
            expect_validation_error(
                "corrected_path_only_evidence",
                "corrected_without_sufficient_evidence_type",
                lambda: validate_resolution_row(path_only),
            )
        )
        generic_reference = load_test_json_fixture(
            root,
            "corrected_generic_evidence_reference",
            minimal_resolution_row(
                resolution_evidence_reference="frozen generator commit and run log"
            ),
        )
        passed.append(
            expect_validation_error(
                "corrected_generic_evidence_reference",
                "corrected_evidence_reference_not_archive_specific",
                lambda: validate_resolution_row(generic_reference),
            )
        )
        source_without_run_binding = load_test_json_fixture(
            root,
            "source_commit_without_run_binding",
            minimal_resolution_row(
                _binding_updates={
                    "config_identifier": "",
                    "log_record_identifier": "",
                    "run_identifier": "",
                }
            ),
        )
        passed.append(
            expect_validation_error(
                "source_commit_without_run_binding",
                "corrected_missing_archive_specific_evidence:config_identifier",
                lambda: validate_resolution_row(source_without_run_binding),
            )
        )
        ambiguous_configs = load_test_json_fixture(
            root,
            "ambiguous_generator_configs",
            correction_row(
                _binding_updates={
                    "config_identifier": "evidence/ambiguous_config.json"
                }
            ),
        )
        passed.append(
            expect_validation_error(
                "ambiguous_generator_configs",
                "corrected_ambiguous_evidence_candidates:2",
                lambda: validate_resolution_row(ambiguous_configs, evidence_repo),
            )
        )
        missing_run_identity = load_test_json_fixture(
            root,
            "missing_fold_seed_variant_binding",
            minimal_resolution_row(
                fold="",
                seed="",
                condition="",
                variant="",
                expected_condition_from_path="",
                expected_variant_from_path="",
                _binding_updates={
                    "fold": "",
                    "seed": "",
                    "condition": "",
                    "variant": "",
                },
            ),
        )
        passed.append(
            expect_validation_error(
                "missing_fold_seed_variant_binding",
                "corrected_missing_archive_identity_binding:fold",
                lambda: validate_resolution_row(missing_run_identity),
            )
        )
        for binding_field in ARCHIVE_BINDING_ROW_FIELDS:
            missing_binding_row = minimal_resolution_row()
            missing_binding_row[binding_field] = ""
            missing_binding = load_test_json_fixture(
                root,
                f"missing_required_archive_binding_{binding_field}",
                missing_binding_row,
            )
            expect_validation_error(
                f"missing_required_archive_binding_{binding_field}",
                f"corrected_missing_archive_identity_binding:{binding_field}",
                lambda missing_binding=missing_binding: (
                    validate_archive_specific_correction_evidence(
                        missing_binding, evidence_repo
                    )
                ),
            )
        passed.append("missing_required_archive_binding_fields")
        confirmed_contradiction = load_test_json_fixture(
            root,
            "confirmed_hidden_backbone_contradiction",
            minimal_resolution_row(
                canonical_resolution_status="confirmed",
                resolution_evidence_type="observed_metadata_agreement",
                conflict_class="none",
                conflict_evidence_basis="",
                model_backbone_label_conflict="false",
                expected_backbone_from_path="resnet50",
            ),
        )
        passed.append(
            expect_validation_error(
                "confirmed_hidden_backbone_contradiction",
                "manifest_observed_expectation_conflict_mismatch",
                lambda: validate_resolution_row(confirmed_contradiction),
            )
        )
        legacy_contradiction = load_test_json_fixture(
            root,
            "legacy_hidden_model_contradiction",
            minimal_resolution_row(
                canonical_resolution_status="legacy-optional",
                resolution_confidence="not_applicable",
                resolution_evidence_type=(
                    "optional_field_absent_without_contradiction"
                ),
                conflict_class="none",
                conflict_evidence_basis="",
                model_backbone_label_conflict="false",
                observed_model="dinov2_pathoalign",
                canonical_model="dinov2_pathoalign",
                observed_backbone="",
                canonical_backbone="",
            ),
        )
        passed.append(
            expect_validation_error(
                "legacy_hidden_model_contradiction",
                "legacy_optional_has_model_contradiction",
                lambda: validate_resolution_row(legacy_contradiction),
            )
        )
        unavailable_commit = load_test_json_fixture(
            root,
            "unavailable_referenced_commit",
            correction_row(_binding_updates={"commit": "1" * 40}),
        )
        passed.append(
            expect_validation_error(
                "unavailable_referenced_commit",
                "corrected_referenced_commit_unavailable",
                lambda: validate_resolution_row(unavailable_commit, evidence_repo),
            )
        )
        for field, evidence_class in (
            ("config_identifier", "config"),
            ("log_record_identifier", "log"),
        ):
            unavailable_config_log = load_test_json_fixture(
                root,
                f"unavailable_referenced_{evidence_class}",
                correction_row(
                    _binding_updates={field: f"evidence/missing-{evidence_class}.json"}
                ),
            )
            expect_validation_error(
                f"unavailable_referenced_{evidence_class}",
                f"corrected_referenced_{evidence_class}_unavailable",
                lambda unavailable_config_log=unavailable_config_log: (
                    validate_resolution_row(unavailable_config_log, evidence_repo)
                ),
            )
        passed.append("unavailable_referenced_config_log")
        current_hash_only = load_test_json_fixture(
            root,
            "current_hash_as_historical_proof",
            correction_row(
                resolution_evidence_type="content_hash_verified_manifest",
                _binding_updates={
                    "historical_output_sha256": "b" * 64,
                    "verified_manifest_reference": "",
                },
            ),
        )
        passed.append(
            expect_validation_error(
                "current_hash_as_historical_proof",
                "current_state_hash_not_historical_output_proof",
                lambda: validate_resolution_row(current_hash_only, evidence_repo),
            )
        )
        current_hash_misrepresented = load_test_json_fixture(
            root,
            "current_hash_misrepresented_as_historical_proof",
            minimal_resolution_row(
                canonical_resolution_status="confirmed",
                resolution_confidence="medium",
                resolution_evidence_type="content_hash_verified_manifest",
                conflict_class="none",
                conflict_evidence_basis="",
                model_backbone_label_conflict="false",
                observed_model="phikon_pathoalign",
                canonical_model="phikon_pathoalign",
            ),
        )
        passed.append(
            expect_validation_error(
                "current_hash_misrepresented_as_historical_proof",
                "current_state_hash_not_historical_output_proof",
                lambda: validate_resolution_row(current_hash_misrepresented),
            )
        )
        verified_correction = load_test_json_fixture(
            root,
            "verified_archive_specific_correction",
            correction_row(),
        )
        validate_resolution_row(verified_correction, evidence_repo)
        behavior_passed.append("verified_archive_specific_correction")
        forged_flags = load_test_json_fixture(
            root,
            "forged_internal_verification_flags",
            minimal_resolution_row(
                _archive_specific_evidence_verified=True,
                _referenced_objects_available=True,
                _referenced_commit_available=True,
                _referenced_config_log_available=True,
                _historical_output_binding_verified=True,
                _evidence_candidate_count=1,
            ),
        )
        passed.append(
            expect_validation_error(
                "forged_internal_verification_flags",
                "corrected_referenced_commit_unavailable",
                lambda: validate_resolution_row(forged_flags, evidence_repo),
            )
        )
        for correction_type in (
            "generator_config",
            "generator_config_and_run_log",
            "run_log_with_output_hash",
            "source_commit_and_code_path",
        ):
            proofless_correction = load_test_json_fixture(
                root,
                f"proofless_{correction_type}",
                correction_row(
                    resolution_evidence_type=correction_type,
                    _binding_updates={
                        "historical_output_sha256": "",
                        "verified_manifest_reference": "",
                    },
                ),
            )
            expect_validation_error(
                f"proofless_{correction_type}",
                "corrected_without_historical_or_internal_proof",
                lambda proofless_correction=proofless_correction: (
                    validate_resolution_row(proofless_correction, evidence_repo)
                ),
            )
        passed.append("correction_types_without_historical_or_internal_proof")
        unresolved_proposal = load_test_json_fixture(
            root,
            "unresolved_proposed_canonical_value_preserved",
            minimal_unresolved_row(),
        )
        validate_resolution_row(unresolved_proposal)
        if unresolved_proposal["canonical_model"] != "phikon_pathoalign":
            raise AssertionError("unresolved_proposed_canonical_value_not_preserved")
        behavior_passed.append("unresolved_proposed_canonical_value_preserved")
        unresolved_missing_evidence = load_test_json_fixture(
            root,
            "unresolved_missing_adjudication_evidence_recorded",
            minimal_unresolved_row(),
        )
        validate_resolution_row(unresolved_missing_evidence)
        behavior_passed.append("unresolved_missing_adjudication_evidence_recorded")
        deceptive_evidence_text = load_test_json_fixture(
            root,
            "unresolved_deceptive_adjudication_evidence_text",
            minimal_unresolved_row(
                evidence_needed_for_adjudication=(
                    UNRESOLVED_ADJUDICATION_EVIDENCE_NEEDED
                    + "; all listed evidence is already present"
                )
            ),
        )
        expect_validation_error(
            "unresolved_deceptive_adjudication_evidence_text",
            "unresolved_missing_adjudication_evidence_record_mismatch",
            lambda: validate_resolution_row(deceptive_evidence_text),
        )
        for needed_field in (
            "generator_config_needed",
            "run_log_needed",
            "source_commit_needed",
            "archive_hash_comparison_needed",
            "human_review_needed",
        ):
            false_need_flag = load_test_json_fixture(
                root,
                f"unresolved_false_need_flag_{needed_field}",
                minimal_unresolved_row(**{needed_field: "false"}),
            )
            expect_validation_error(
                f"unresolved_false_need_flag_{needed_field}",
                f"unresolved_adjudication_need_flag_not_true:{needed_field}",
                lambda false_need_flag=false_need_flag: validate_resolution_row(
                    false_need_flag
                ),
            )
        unresolved_with_adjudication = load_test_json_fixture(
            root,
            "unresolved_with_archive_specific_adjudication",
            minimal_unresolved_row(_archive_specific_evidence_verified=True),
        )
        expect_validation_error(
            "unresolved_with_archive_specific_adjudication",
            "unresolved_with_archive_specific_adjudication_evidence",
            lambda: validate_resolution_row(unresolved_with_adjudication),
        )
        passed.append("unresolved_adjudication_requirements_fail_closed")
        orphan_conflict_fixture = load_test_json_fixture(
            root,
            "conflict_row_absent_manifest_fixture",
            {
                "manifest": [],
                "conflicts": [
                    {
                        "archive_id": "a",
                        "canonical_path": "results/a.npz",
                        "conflict_class": "source_label_conflict",
                        "evidence_basis": "test evidence",
                    }
                ],
            },
        )
        passed.append(
            expect_validation_error(
                "fixture_conflict_row_absent_manifest",
                "conflict_csv_row_not_in_manifest",
                lambda: validate_cross_tables(
                    orphan_conflict_fixture["manifest"],
                    orphan_conflict_fixture["conflicts"],
                ),
            )
        )
        missing_conflict_fixture = load_test_json_fixture(
            root,
            "corrected_row_absent_conflict_csv_fixture",
            {"manifest": [minimal_resolution_row()], "conflicts": []},
        )
        passed.append(
            expect_validation_error(
                "fixture_corrected_row_absent_conflict_csv",
                "manifest_conflict_missing_from_conflict_csv",
                lambda: validate_cross_tables(
                    missing_conflict_fixture["manifest"],
                    missing_conflict_fixture["conflicts"],
                ),
            )
        )
        invalid_pairing = load_test_json_fixture(
            root,
            "invalid_corrected_confidence_pairing",
            minimal_resolution_row(resolution_confidence="not_applicable"),
        )
        passed.append(
            expect_validation_error(
                "invalid_corrected_confidence_pairing",
                "invalid_resolution_status_confidence_pairing:corrected:not_applicable",
                lambda: validate_resolution_row(invalid_pairing),
            )
        )
        manifest_without_issue = [
            {
                "archive_id": "a",
                "canonical_path": "results/a.npz",
                "conflict_class": "none",
            }
        ]
        conflict_with_extra = [
            {
                "archive_id": "a",
                "canonical_path": "results/a.npz",
                "conflict_class": "source_label_conflict",
                "evidence_basis": "test evidence",
            }
        ]
        passed.append(
            expect_validation_error(
                "conflict_csv_row_absent_from_manifest",
                "conflict_csv_row_not_in_manifest",
                lambda: validate_cross_tables(
                    manifest_without_issue, conflict_with_extra
                ),
            )
        )
        manifest_with_issue = [
            {
                "archive_id": "a",
                "canonical_path": "results/a.npz",
                "conflict_class": "none",
                "source_label_conflict": "true",
            }
        ]
        passed.append(
            expect_validation_error(
                "manifest_conflict_without_csv_row",
                "manifest_conflict_missing_from_conflict_csv",
                lambda: validate_cross_tables(manifest_with_issue, []),
            )
        )
        manifest_encoded_issue = [
            {
                "archive_id": "a",
                "canonical_path": "results/a.npz",
                "conflict_class": "source_label_conflict",
                "source_label_conflict": "true",
            }
        ]
        passed.append(
            expect_validation_error(
                "conflict_csv_without_evidence_basis",
                "conflict_csv_without_evidence_basis",
                lambda: validate_cross_tables(
                    manifest_encoded_issue,
                    [
                        {
                            "archive_id": "a",
                            "canonical_path": "results/a.npz",
                            "conflict_class": "source_label_conflict",
                            "evidence_basis": "",
                        }
                    ],
                ),
            )
        )
        passed.append(
            expect_validation_error(
                "conflict_csv_path_mismatch",
                "manifest_conflict_missing_from_conflict_csv",
                lambda: validate_cross_tables(
                    manifest_encoded_issue,
                    [
                        {
                            "archive_id": "a",
                            "canonical_path": "results/wrong.npz",
                            "conflict_class": "source_label_conflict",
                            "evidence_basis": "test evidence",
                        }
                    ],
                ),
            )
        )
        passed.append(
            expect_validation_error(
                "manifest_flag_class_mismatch",
                "manifest_conflict_flag_not_in_conflict_class",
                lambda: validate_cross_tables(
                    manifest_with_issue,
                    [
                        {
                            "archive_id": "a",
                            "canonical_path": "results/a.npz",
                            "conflict_class": "source_label_conflict",
                            "evidence_basis": "test evidence",
                        }
                    ],
                ),
            )
        )
        absolute_values = {
            "machine_specific_absolute_path_leakage": (
                "canonical_path",
                "C:/Users/example/results/archive.npz",
            ),
            "embedded_windows_absolute_path_leakage": (
                "resolution_notes",
                "prefix=C:/Users/example/archive.npz",
            ),
            "embedded_posix_absolute_path_leakage": (
                "resolution_notes",
                "prefix /home/example/archive.npz",
            ),
            "embedded_unc_absolute_path_leakage": (
                "resolution_notes",
                r"prefix \\server\share\archive.npz",
            ),
        }
        for test_name, (field, value) in absolute_values.items():
            passed.append(
                expect_validation_error(
                    test_name,
                    "absolute_path_leakage",
                    lambda field=field, value=value: validate_no_absolute_leakage(
                        [{"archive_id": "a", field: value}], root
                    ),
                )
            )
        duplicate_a = root / "duplicate-a.bin"
        duplicate_b = root / "duplicate-b.bin"
        duplicate_a.write_bytes(b"same")
        duplicate_b.write_bytes(b"same")
        duplicate_observations: list[ArchiveObservation] = []
        for path in (duplicate_a, duplicate_b):
            snapshot = hash_file_twice(path)
            canonical_path = f"fixtures/{path.name}"
            duplicate_observations.append(
                ArchiveObservation(
                    family=FAMILIES[0],
                    path=path,
                    canonical_path=canonical_path,
                    relative_path=canonical_path,
                    content_sha256=snapshot.sha256,
                    file_size_bytes=snapshot.size,
                    metadata_text='{"model":"dinov2_feature_extractor"}',
                    metadata={"model": "dinov2_feature_extractor"},
                    metadata_keys="model",
                    metadata_sha256=hashlib.sha256(
                        b'{"model":"dinov2_feature_extractor"}'
                    ).hexdigest(),
                    fold=None,
                    seed=None,
                    condition="",
                    variant="",
                    evaluation_split="",
                )
            )
        groups = duplicate_content_groups(duplicate_observations)
        if groups != [["fixtures/duplicate-a.bin", "fixtures/duplicate-b.bin"]]:
            raise AssertionError("duplicate_content_behavior_not_detected")
        duplicate_manifest_rows, _ = make_manifest_rows(
            duplicate_observations,
            verified_families=set(),
            duplicate_groups=groups,
        )
        duplicate_conflict_rows = make_conflict_rows(duplicate_manifest_rows)
        validate_identity_rows(duplicate_manifest_rows)
        for row in duplicate_manifest_rows:
            validate_resolution_row(row)
        validate_cross_tables(duplicate_manifest_rows, duplicate_conflict_rows)
        if any(
            row["conflict_class"] != "duplicate_content"
            or row["canonical_resolution_status"] != "legacy-optional"
            for row in duplicate_manifest_rows
        ):
            raise AssertionError("duplicate_content_resolution_not_recorded")
        if len(duplicate_conflict_rows) != 2:
            raise AssertionError("duplicate_content_conflict_rows_not_recorded")
        behavior_passed.append("duplicate_content_under_distinct_paths")
    if root.exists():
        raise AssertionError("temporary_fixture_cleanup_failed")
    required_adversarial_tests = (
        "corrected_generic_evidence_reference",
        "source_commit_without_run_binding",
        "ambiguous_generator_configs",
        "missing_required_archive_binding_fields",
        "current_hash_as_historical_proof",
        "invalid_corrected_confidence_pairing",
        "confirmed_hidden_backbone_contradiction",
        "legacy_hidden_model_contradiction",
        "unavailable_referenced_commit",
        "unavailable_referenced_config_log",
        "unresolved_proposed_canonical_value_preserved",
        "unresolved_missing_adjudication_evidence_recorded",
        "current_hash_misrepresented_as_historical_proof",
    )
    all_passed = passed + behavior_passed
    missing_required_tests = sorted(set(required_adversarial_tests) - set(all_passed))
    if missing_required_tests:
        raise AssertionError(
            "required_adversarial_tests_missing:" + ",".join(missing_required_tests)
        )
    return {
        "status": "passed",
        "fail_closed_tests_passed": len(passed),
        "fail_closed_tests_total": 39,
        "behavior_tests_passed": len(behavior_passed),
        "behavior_tests_total": 4,
        "required_adversarial_tests_passed": len(required_adversarial_tests),
        "required_adversarial_tests_total": 13,
        "required_adversarial_tests": list(required_adversarial_tests),
        "tests": all_passed,
        "temporary_fixtures_removed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--format", choices=("json",))
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        print(json.dumps(run_self_tests(), indent=2, sort_keys=True))
        return 0
    result = build(args.repo_root)
    if args.format == "json":
        print(json.dumps(result.summary, indent=2, sort_keys=True))
        return 0
    if args.check:
        check_outputs(result)
        print(
            "PROVENANCE_MANIFEST_CHECK_PASS "
            f"archives={result.summary['total_archives']} "
            f"fingerprint={result.summary['inventory_fingerprint_sha256']}"
        )
        return 0
    write_outputs(result)
    print(
        "PROVENANCE_MANIFEST_WRITE_PASS "
        f"archives={result.summary['total_archives']} "
        f"fingerprint={result.summary['inventory_fingerprint_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestValidationError as exc:
        print(f"PROVENANCE_MANIFEST_VALIDATION_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
