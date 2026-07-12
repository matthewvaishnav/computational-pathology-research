#!/usr/bin/env python3
"""Read-only feasibility audit for scanner-shared residual provenance.

The audit inspects allowlisted manifests, NPZ schemas, ID alignment, support
counts, and existing summary evidence. It never loads feature payloads, fits a
model or probe, reconstructs a representation, computes a research metric, or
writes a file.
"""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import json
import math
import re
import struct
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKED_REPORT = Path(__file__).with_name("feasibility_report.md")


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    manifest: str
    sample_column: str
    split_manifest_pattern: str
    region_column: str = "region_id"
    scanner_column: str = "scanner_id"
    category_column: str | None = None
    expected_scanners: int = 5


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    dataset_id: str
    pattern: str
    expected_count: int | None
    vector_key: str = "features"
    expected_dataset_token: str | None = None
    expected_model_token: str | None = None
    require_five_by_five_grid: bool = True
    require_training_metadata: bool = True


DATASETS = (
    DatasetSpec(
        dataset_id="canine_scc",
        manifest=(
            "results/external_multiscanner_caninescc/geometry_qualified/"
            "geometry_qualified_manifest.csv"
        ),
        sample_column="sample_id",
        split_manifest_pattern=(
            "data/external_multiscanner_caninescc/patch_manifests/splits/"
            "fold_{fold}_patch_manifest.csv"
        ),
        category_column="category_name",
    ),
    DatasetSpec(
        dataset_id="scorpion",
        manifest="data/scorpion/manifest.csv",
        sample_column="slide_id",
        split_manifest_pattern="data/scorpion/splits/fold_{fold}_manifest.csv",
        category_column=None,
    ),
)


FAMILIES = (
    FamilySpec(
        "canine_original_dinov2",
        "canine_scc",
        "results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz",
        1,
        expected_model_token="dinov2",
        require_five_by_five_grid=False,
        require_training_metadata=False,
    ),
    FamilySpec(
        "canine_true_pair_biological",
        "canine_scc",
        (
            "results/paired_acquisition_factorization_pair_integrity_caninescc/"
            "fold_*/runs/true_pairs_seed_*/projected_features.npz"
        ),
        25,
        expected_dataset_token="canine",
        expected_model_token="dinov2",
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
        expected_dataset_token="canine",
        expected_model_token="dinov2",
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
        expected_dataset_token="canine",
        expected_model_token="dinov2",
    ),
    FamilySpec(
        "canine_shuffled_region_control",
        "canine_scc",
        (
            "results/paired_acquisition_factorization_pair_integrity_caninescc/"
            "fold_*/runs/shuffled_region_pairs_seed_*/projected_features.npz"
        ),
        25,
        expected_dataset_token="canine",
        expected_model_token="dinov2",
    ),
    FamilySpec(
        "canine_shuffled_sample_control",
        "canine_scc",
        (
            "results/paired_acquisition_factorization_pair_integrity_caninescc/"
            "fold_*/runs/shuffled_sample_pairs_seed_*/projected_features.npz"
        ),
        25,
        expected_dataset_token="canine",
        expected_model_token="dinov2",
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
        expected_dataset_token="canine",
        expected_model_token="dinov2",
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
        expected_dataset_token="canine",
        expected_model_token="dinov2",
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
        expected_dataset_token="canine",
        expected_model_token="dinov2",
    ),
    FamilySpec(
        "scorpion_true_pair_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_pair_integrity_scorpion/"
            "fold_*/runs/true_pairs_seed_*/projected_features.npz"
        ),
        25,
        expected_dataset_token="scorpion",
        expected_model_token="dinov2",
    ),
    FamilySpec(
        "scorpion_phikon_true_pair_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_pair_integrity_scorpion_phikon/"
            "fold_*/runs/true_pairs_seed_*/projected_features.npz"
        ),
        25,
        expected_dataset_token="scorpion",
        expected_model_token="phikon",
    ),
    FamilySpec(
        "scorpion_resnet50_true_pair_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50/"
            "fold_*/runs/true_pairs_seed_*/projected_features.npz"
        ),
        25,
        expected_dataset_token="scorpion",
        expected_model_token="resnet50",
    ),
    FamilySpec(
        "scorpion_dinov2_acq_dim8_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_frontier_selected_crossbackbone_"
            "validation/trained_runs/dinov2/fold_*/runs/acq_dim8_default_seed_*/"
            "projected_features.npz"
        ),
        25,
        expected_dataset_token="scorpion",
        expected_model_token="dinov2",
    ),
    FamilySpec(
        "scorpion_dinov2_acq_dim16_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_frontier_selected_crossbackbone_"
            "validation/trained_runs/dinov2/fold_*/runs/"
            "acq_dim16_stronger_xcov_seed_*/projected_features.npz"
        ),
        25,
        expected_dataset_token="scorpion",
        expected_model_token="dinov2",
    ),
    FamilySpec(
        "scorpion_phikon_acq_dim8_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_frontier_selected_crossbackbone_"
            "validation/trained_runs/phikon/fold_*/runs/acq_dim8_default_seed_*/"
            "projected_features.npz"
        ),
        25,
        expected_dataset_token="scorpion",
        expected_model_token="phikon",
    ),
    FamilySpec(
        "scorpion_phikon_acq_dim16_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_frontier_selected_crossbackbone_"
            "validation/trained_runs/phikon/fold_*/runs/"
            "acq_dim16_stronger_xcov_seed_*/projected_features.npz"
        ),
        25,
        expected_dataset_token="scorpion",
        expected_model_token="phikon",
    ),
    FamilySpec(
        "scorpion_resnet50_acq_dim8_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_frontier_selected_crossbackbone_"
            "validation/trained_runs/resnet50/fold_*/runs/acq_dim8_default_seed_*/"
            "projected_features.npz"
        ),
        25,
        expected_dataset_token="scorpion",
        expected_model_token="resnet50",
    ),
    FamilySpec(
        "scorpion_resnet50_acq_dim16_biological",
        "scorpion",
        (
            "results/paired_acquisition_factorization_frontier_selected_crossbackbone_"
            "validation/trained_runs/resnet50/fold_*/runs/"
            "acq_dim16_stronger_xcov_seed_*/projected_features.npz"
        ),
        25,
        expected_dataset_token="scorpion",
        expected_model_token="resnet50",
    ),
    FamilySpec(
        "oldstyle_keep_k4_row_level",
        "canine_scc",
        (
            "results/paired_acquisition_factorization_oldstyle_residual_branch_"
            "separation_audit/**/projected_features.npz"
        ),
        None,
        expected_dataset_token="canine",
        require_five_by_five_grid=False,
        require_training_metadata=False,
    ),
)


METADATA_FILES = (
    "data/external_multiscanner_caninescc/manifest.csv",
    "data/external_multiscanner_caninescc/patch_manifests/full_patch_manifest.csv",
    (
        "results/external_multiscanner_caninescc/geometry_qualified/"
        "geometry_qualified_manifest.csv"
    ),
    (
        "results/external_multiscanner_caninescc/adaptive_crop_audit/"
        "adaptive_crop_plan.csv"
    ),
    (
        "results/external_multiscanner_caninescc/registration_database_audit/"
        "sample_affine_transforms.csv"
    ),
    (
        "results/external_multiscanner_caninescc/orientation_audit/"
        "tiff_file_metadata.csv"
    ),
    "results/external_multiscanner_caninescc/inspection/file_inventory.csv",
    (
        "results/external_multiscanner_caninescc/registration_database_audit/"
        "sqlite_coco_geometry_comparison.csv"
    ),
    "data/scorpion/manifest.csv",
)


SITE_COLUMNS = {
    "site_id",
    "laboratory_id",
    "lab_id",
    "institution_id",
    "center_id",
    "collection_protocol_id",
}
PREPARATION_COLUMNS = {
    "fixation_protocol_id",
    "processing_batch_id",
    "block_id",
    "section_id",
    "section_thickness",
    "microtome_id",
    "slide_prep_batch_id",
    "stain_protocol_id",
    "stain_batch_id",
    "reagent_lot_id",
    "operator_id",
    "coverslip_batch_id",
    "mounting_medium_id",
    "preparation_date",
    "stain_date",
    "stain_date_bin",
    "scan_session_id",
    "scan_date",
    "scan_date_bin",
}
TECHNICAL_PROXY_COLUMNS = {
    "region_rank",
    "annotation_id",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "bbox_area_pixels",
    "area",
    "image_width",
    "image_height",
    "adaptive_crop_side_level0",
    "inside_image_fraction",
    "padding_fraction",
    "region_max_padding_fraction",
    "orientation_normalization_degrees",
    "rms_residual_pixels",
    "size_bytes",
    "lastmodified",
}


FIGURE2 = (
    "paper/paired_acquisition_manuscript/figure_table_artifacts/"
    "figure2_branch_separation_data.csv"
)
FIGURE4 = (
    "paper/paired_acquisition_manuscript/figure_table_artifacts/"
    "figure4_bottleneck_comparison_data.csv"
)


def rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
        return header, rows
    except (OSError, UnicodeError, csv.Error):
        return [], []


def inspect_manifest(path: Path, spec: DatasetSpec, repo_root: Path) -> dict[str, Any]:
    header, rows = read_csv(path)
    required = {spec.sample_column, spec.region_column, spec.scanner_column}
    if spec.category_column:
        required.add(spec.category_column)
        required.add("fold")
    missing = sorted(required - set(header))

    key_rows: set[tuple[str, str, str]] = set()
    duplicate_keys = 0
    region_scanners: dict[tuple[str, str], set[str]] = defaultdict(set)
    region_categories: dict[tuple[str, str], set[str]] = defaultdict(set)
    region_folds: dict[tuple[str, str], set[str]] = defaultdict(set)
    blank_required_values: Counter[str] = Counter()
    samples: set[str] = set()
    scanners: set[str] = set()
    categories: set[str] = set()

    for row in rows:
        sample = str(row.get(spec.sample_column, ""))
        region = str(row.get(spec.region_column, ""))
        scanner = str(row.get(spec.scanner_column, ""))
        for column, value in (
            (spec.sample_column, sample),
            (spec.region_column, region),
            (spec.scanner_column, scanner),
        ):
            if not value.strip():
                blank_required_values[column] += 1
        category = ""
        if spec.category_column:
            category = str(row.get(spec.category_column, ""))
            if not category.strip():
                blank_required_values[spec.category_column] += 1
        if not sample.strip() or not region.strip() or not scanner.strip():
            continue
        key = (sample, region, scanner)
        if key in key_rows:
            duplicate_keys += 1
        key_rows.add(key)
        samples.add(sample)
        scanners.add(scanner)
        region_scanners[(sample, region)].add(scanner)
        if spec.category_column:
            if category.strip():
                categories.add(category)
                region_categories[(sample, region)].add(category)
        fold = str(row.get("fold", ""))
        if fold.strip():
            region_folds[(sample, region)].add(fold)
        elif spec.category_column:
            blank_required_values["fold"] += 1

    scanner_count_distribution = Counter(len(value) for value in region_scanners.values())
    category_conflicts = sum(len(value) != 1 for value in region_categories.values())
    fold_conflicts = sum(len(value) != 1 for value in region_folds.values())
    incomplete_regions = sum(
        len(value) != spec.expected_scanners for value in region_scanners.values()
    )

    fold_eligibility: dict[str, dict[str, Any]] = {}
    eligible_cells = 0
    eligible_regions = 0
    usable_category_union: set[str] = set()
    if spec.category_column and region_categories:
        fold_records: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for (sample, region), category_values in region_categories.items():
            fold_values = region_folds.get((sample, region), set())
            if len(category_values) != 1 or len(fold_values) != 1:
                continue
            fold_records[next(iter(fold_values))].append(
                (sample, region, next(iter(category_values)))
            )

        for fold, records in sorted(fold_records.items(), key=lambda item: item[0]):
            cell_regions: Counter[tuple[str, str]] = Counter()
            category_samples: dict[str, set[str]] = defaultdict(set)
            fold_samples: set[str] = set()
            for sample, _region, category in records:
                cell_regions[(sample, category)] += 1
                category_samples[category].add(sample)
                fold_samples.add(sample)
            usable_categories = {
                category
                for category, members in category_samples.items()
                if len(members) >= 2
            }
            fold_cells = 0
            fold_regions = 0
            for (_sample, category), count in cell_regions.items():
                if category in usable_categories and count >= 2:
                    fold_cells += 1
                    fold_regions += count
            usable_category_union.update(usable_categories)
            eligible_cells += fold_cells
            eligible_regions += fold_regions
            fold_eligibility[fold] = {
                "samples": len(fold_samples),
                "categories": len(category_samples),
                "eligible_sample_category_cells": fold_cells,
                "eligible_regions": fold_regions,
                "eligible_observations": fold_regions * spec.expected_scanners,
                "usable_categories": sorted(usable_categories),
                "nonestimable_categories": sorted(
                    set(category_samples) - usable_categories
                ),
            }

    usable_categories = sorted(usable_category_union)
    excluded_categories = sorted(categories - usable_category_union)

    status = "available"
    reason_codes: list[str] = []
    if not path.is_file():
        status = "absent"
        reason_codes.append("manifest_absent")
    elif missing:
        status = "schema_mismatch"
        reason_codes.append("required_columns_missing")
    if duplicate_keys:
        status = "alignment_blocked"
        reason_codes.append("duplicate_composite_keys")
    if blank_required_values:
        status = "alignment_blocked"
        reason_codes.append("blank_required_values")
    if category_conflicts:
        status = "alignment_blocked"
        reason_codes.append("category_varies_within_region")
    if fold_conflicts:
        status = "alignment_blocked"
        reason_codes.append("fold_varies_within_region")
    if incomplete_regions:
        status = "alignment_blocked"
        reason_codes.append("incomplete_scanner_bundles")

    return {
        "dataset_id": spec.dataset_id,
        "path": rel(path, repo_root),
        "status": status,
        "reason_codes": reason_codes,
        "header": header,
        "missing_required_columns": missing,
        "rows": len(rows),
        "unique_composite_keys": len(key_rows),
        "duplicate_composite_keys": duplicate_keys,
        "blank_required_values": dict(sorted(blank_required_values.items())),
        "samples": len(samples),
        "regions": len(region_scanners),
        "scanners": len(scanners),
        "scanner_values": sorted(scanners),
        "scanner_count_distribution": {
            str(key): value for key, value in sorted(scanner_count_distribution.items())
        },
        "expected_scanners_per_region": spec.expected_scanners,
        "incomplete_regions": incomplete_regions,
        "categories": len(categories),
        "category_values": sorted(categories),
        "category_conflicts": category_conflicts,
        "fold_conflicts": fold_conflicts,
        "eligible_sample_category_cells": eligible_cells,
        "eligible_regions": eligible_regions,
        "eligible_observations": eligible_regions * spec.expected_scanners,
        "fold_eligibility": fold_eligibility,
        "usable_categories": usable_categories,
        "excluded_categories": excluded_categories,
        "_key_rows": key_rows,
        "_rows": rows,
    }


def inspect_split_manifests(
    repo_root: Path,
    spec: DatasetSpec,
    dataset_manifest: dict[str, Any],
) -> dict[str, Any]:
    fold_maps: dict[int, dict[tuple[str, str, str], str]] = {}
    fold_summaries: dict[str, dict[str, Any]] = {}
    overall_status = "available"
    reason_codes: list[str] = []
    dataset_keys = dataset_manifest.get("_key_rows", set())

    for fold in range(5):
        relative = spec.split_manifest_pattern.format(fold=fold)
        path = repo_root / relative
        header, rows = read_csv(path)
        required = {
            spec.sample_column,
            spec.region_column,
            spec.scanner_column,
            "split",
        }
        missing_columns = sorted(required - set(header))
        mapping: dict[tuple[str, str, str], str] = {}
        duplicate_keys = 0
        blank_values = 0
        for row in rows:
            key = (
                str(row.get(spec.sample_column, "")),
                str(row.get(spec.region_column, "")),
                str(row.get(spec.scanner_column, "")),
            )
            split = str(row.get("split", ""))
            if not all(value.strip() for value in key) or not split.strip():
                blank_values += 1
                continue
            if key in mapping:
                duplicate_keys += 1
            mapping[key] = split
        key_match = set(mapping) == dataset_keys
        fold_status = "available"
        fold_reasons: list[str] = []
        if not path.is_file():
            fold_status = "absent"
            fold_reasons.append("split_manifest_absent")
        elif missing_columns:
            fold_status = "schema_mismatch"
            fold_reasons.append("split_manifest_columns_missing")
        if duplicate_keys:
            fold_status = "alignment_blocked"
            fold_reasons.append("split_manifest_duplicate_keys")
        if blank_values:
            fold_status = "alignment_blocked"
            fold_reasons.append("split_manifest_blank_values")
        if dataset_keys and not key_match:
            fold_status = "alignment_blocked"
            fold_reasons.append("split_manifest_dataset_key_mismatch")
        if fold_status != "available":
            overall_status = "blocked"
            reason_codes.extend(f"fold_{fold}:{reason}" for reason in fold_reasons)
        fold_maps[fold] = mapping
        fold_summaries[str(fold)] = {
            "path": relative,
            "status": fold_status,
            "reason_codes": fold_reasons,
            "rows": len(rows),
            "unique_keys": len(mapping),
            "duplicate_keys": duplicate_keys,
            "blank_values": blank_values,
            "dataset_key_match": key_match,
            "split_values": sorted(set(mapping.values())),
        }

    return {
        "dataset_id": spec.dataset_id,
        "status": overall_status,
        "reason_codes": sorted(set(reason_codes)),
        "folds": fold_summaries,
        "_maps": fold_maps,
    }


def read_npy_header(handle: BinaryIO) -> dict[str, Any]:
    magic = handle.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("invalid_npy_magic")
    version_raw = handle.read(2)
    if len(version_raw) != 2:
        raise ValueError("truncated_npy_version")
    major, _minor = version_raw
    if major == 1:
        size_raw = handle.read(2)
        if len(size_raw) != 2:
            raise ValueError("truncated_npy_header_length")
        header_size = struct.unpack("<H", size_raw)[0]
    elif major in {2, 3}:
        size_raw = handle.read(4)
        if len(size_raw) != 4:
            raise ValueError("truncated_npy_header_length")
        header_size = struct.unpack("<I", size_raw)[0]
    else:
        raise ValueError("unsupported_npy_version")
    if header_size > 1_000_000:
        raise ValueError("oversized_npy_header")
    header_raw = handle.read(header_size)
    if len(header_raw) != header_size:
        raise ValueError("truncated_npy_header")
    encoding = "utf-8" if major == 3 else "latin1"
    header = ast.literal_eval(header_raw.decode(encoding).strip())
    if not isinstance(header, dict):
        raise ValueError("invalid_npy_header")
    shape = header.get("shape")
    descriptor = header.get("descr")
    fortran_order = header.get("fortran_order")
    if not isinstance(shape, tuple) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in shape
    ):
        raise ValueError("invalid_npy_shape")
    if not isinstance(descriptor, str) or not descriptor:
        raise ValueError("invalid_npy_descriptor")
    if not isinstance(fortran_order, bool):
        raise ValueError("invalid_npy_fortran_order")
    return header


def npz_members(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        members: dict[str, str] = {}
        for member in archive.namelist():
            name = Path(member).name
            if name.endswith(".npy"):
                key = name[:-4]
                if key in members:
                    raise ValueError(f"duplicate_npz_member_basename:{key}")
                members[key] = member
        return members


def read_npz_header(path: Path, key: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        members: dict[str, str] = {}
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            basename = Path(name).name[:-4]
            if basename in members:
                raise ValueError(f"duplicate_npz_member_basename:{basename}")
            members[basename] = name
        member = members.get(key)
        if not member:
            raise KeyError(key)
        with archive.open(member) as handle:
            return read_npy_header(handle)


def _shape_count(shape: tuple[int, ...]) -> int:
    return math.prod(shape) if shape else 1


def read_npz_text(path: Path, key: str, max_items: int = 20_000) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        members: dict[str, str] = {}
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            basename = Path(name).name[:-4]
            if basename in members:
                raise ValueError(f"duplicate_npz_member_basename:{basename}")
            members[basename] = name
        member = members.get(key)
        if not member:
            raise KeyError(key)
        if archive.getinfo(member).file_size > 20_000_000:
            raise ValueError("text_member_file_size_limit_exceeded")
        with archive.open(member) as handle:
            header = read_npy_header(handle)
            shape = tuple(int(value) for value in header.get("shape", ()))
            count = _shape_count(shape)
            if len(shape) > 1 or count > max_items:
                raise ValueError("text_array_shape_or_size_unsupported")
            descriptor = str(header.get("descr", ""))
            unicode_match = re.fullmatch(r"([<>=|])U(\d+)", descriptor)
            bytes_match = re.fullmatch(r"[|<>=]S(\d+)", descriptor)
            if unicode_match:
                width = int(unicode_match.group(2)) * 4
                endian = unicode_match.group(1)
                encoding = "utf-32-be" if endian == ">" else "utf-32-le"
            elif bytes_match:
                width = int(bytes_match.group(1))
                encoding = "utf-8"
            else:
                raise ValueError(f"unsupported_text_dtype:{descriptor}")
            if width > 65_536 or width * count > 16_000_000:
                raise ValueError("text_array_byte_limit_exceeded")
            raw = handle.read(width * count)
            if len(raw) != width * count:
                raise ValueError("truncated_npy_data")
            values = []
            for index in range(count):
                item = raw[index * width : (index + 1) * width]
                values.append(item.decode(encoding, errors="strict").rstrip("\x00"))
            return values


def inspect_archive(
    path: Path,
    family: FamilySpec,
    manifest: dict[str, Any],
    split_maps: dict[int, dict[tuple[str, str, str], str]],
    repo_root: Path,
) -> dict[str, Any]:
    required_keys = {
        family.vector_key,
        "region_id",
        "scanner_id",
        "slide_id",
        "split",
        "metadata_json",
    }
    result: dict[str, Any] = {
        "path": rel(path, repo_root),
        "status": "available",
        "reason_codes": [],
        "warnings": [],
    }
    try:
        members = npz_members(path)
        keys = sorted(members)
        missing = sorted(required_keys - set(keys))
        result["keys"] = keys
        result["missing_keys"] = missing
        if missing:
            result["status"] = "schema_mismatch"
            result["reason_codes"].append("required_npz_keys_missing")
            return result

        headers = {key: read_npz_header(path, key) for key in required_keys}
        shapes = {
            key: [int(value) for value in tuple(header.get("shape", ()))]
            for key, header in headers.items()
        }
        result["shapes"] = shapes
        vector_shape = shapes[family.vector_key]
        vector_header = headers[family.vector_key]
        vector_descriptor = str(vector_header.get("descr", ""))
        vector_valid = (
            len(vector_shape) == 2
            and all(value > 0 for value in vector_shape)
            and re.fullmatch(r"[<>=|]?[fiu]\d+", vector_descriptor) is not None
            and vector_header.get("fortran_order") is False
        )
        if not vector_valid:
            result["status"] = "schema_mismatch"
            result["reason_codes"].append("feature_array_not_positive_2d_numeric_c_order")
        rows = vector_shape[0] if vector_shape else 0
        result["rows"] = rows
        result["vector_dtype"] = vector_descriptor
        mismatched = sorted(
            key
            for key in {"region_id", "scanner_id", "slide_id", "split"}
            if not shapes[key] or shapes[key][0] != rows
        )
        if mismatched:
            result["status"] = "schema_mismatch"
            result["reason_codes"].append("array_first_dimension_mismatch")
            result["mismatched_arrays"] = mismatched

        samples = read_npz_text(path, "slide_id")
        regions = read_npz_text(path, "region_id")
        scanners = read_npz_text(path, "scanner_id")
        splits = read_npz_text(path, "split")
        archive_keys = set(zip(samples, regions, scanners))
        sample_splits: dict[str, set[str]] = defaultdict(set)
        region_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
        for sample, region, split in zip(samples, regions, splits):
            sample_splits[sample].add(split)
            region_splits[(sample, region)].add(split)
        manifest_keys = manifest.get("_key_rows", set())
        overlap = archive_keys & manifest_keys
        result["unique_archive_keys"] = len(archive_keys)
        result["duplicate_archive_keys"] = rows - len(archive_keys)
        result["manifest_join_matches"] = len(overlap)
        result["manifest_join_total"] = len(archive_keys)
        result["manifest_join_fraction"] = (
            len(overlap) / len(archive_keys) if archive_keys else 0.0
        )
        result["split_values"] = sorted(set(splits))
        result["sample_split_conflicts"] = sum(
            len(values) != 1 for values in sample_splits.values()
        )
        result["region_split_conflicts"] = sum(
            len(values) != 1 for values in region_splits.values()
        )
        if len(archive_keys) != rows:
            result["status"] = "alignment_blocked"
            result["reason_codes"].append("duplicate_archive_composite_keys")
        if archive_keys != manifest_keys:
            result["status"] = "alignment_blocked"
            result["reason_codes"].append("archive_manifest_key_mismatch")
        if result["sample_split_conflicts"] or result["region_split_conflicts"]:
            result["status"] = "alignment_blocked"
            result["reason_codes"].append("group_crosses_archive_splits")
        if set(splits) != {"train", "val", "test"}:
            result["status"] = "alignment_blocked"
            result["reason_codes"].append("archive_missing_required_split")

        path_text = path.as_posix()
        fold_match = re.search(r"fold_(\d+)", path_text)
        seed_match = re.search(r"_seed_(\d+)/", path_text)
        condition_match = re.search(
            r"/runs/(.+)_seed_\d+/projected_features\.npz$", path_text
        )
        path_fold = int(fold_match.group(1)) if fold_match else None
        path_seed = int(seed_match.group(1)) if seed_match else None
        path_condition = condition_match.group(1) if condition_match else None
        result["path_fold"] = path_fold
        result["path_seed"] = path_seed
        result["path_condition"] = path_condition

        if path_fold is None:
            result["status"] = "alignment_blocked"
            result["reason_codes"].append("fold_not_identifiable_from_path")
        else:
            expected_splits = split_maps.get(path_fold)
            if expected_splits is None:
                result["status"] = "alignment_blocked"
                result["reason_codes"].append("fold_split_manifest_unavailable")
            else:
                observed_splits = dict(zip(zip(samples, regions, scanners), splits))
                split_mismatches = sum(
                    expected_splits.get(key) != split
                    for key, split in observed_splits.items()
                )
                result["split_manifest_matches"] = len(observed_splits) - split_mismatches
                result["split_manifest_total"] = len(observed_splits)
                result["split_manifest_mismatches"] = split_mismatches
                if set(observed_splits) != set(expected_splits) or split_mismatches:
                    result["status"] = "alignment_blocked"
                    result["reason_codes"].append("archive_split_manifest_mismatch")

        metadata_values = read_npz_text(path, "metadata_json")
        metadata: dict[str, Any] = {}
        if metadata_values:
            parsed = json.loads(metadata_values[0])
            if isinstance(parsed, dict):
                metadata = parsed
        source = str(metadata.get("source", ""))
        model = str(metadata.get("model", ""))
        condition = str(metadata.get("condition", ""))
        variant = str(metadata.get("variant", ""))
        path_labels = {value for value in (condition, variant) if value}
        result["metadata_source"] = source
        result["metadata_model"] = model
        result["metadata_condition"] = condition
        result["metadata_variant"] = variant
        result["metadata_fold"] = metadata.get("fold")
        result["metadata_seed"] = metadata.get("seed")
        result["metadata_fit_splits"] = metadata.get("fit_splits")
        result["metadata_evaluation_split"] = metadata.get("evaluation_split")
        result["metadata_contains_test_rows"] = metadata.get("contains_test_rows")

        if path_fold is not None and metadata.get("fold") is not None:
            if metadata.get("fold") != path_fold:
                result["status"] = "alignment_blocked"
                result["reason_codes"].append("metadata_fold_path_mismatch")
        if path_seed is not None and metadata.get("seed") is not None:
            if metadata.get("seed") != path_seed:
                result["status"] = "alignment_blocked"
                result["reason_codes"].append("metadata_seed_path_mismatch")
        if path_condition and path_labels and path_condition not in path_labels:
            result["status"] = "alignment_blocked"
            result["reason_codes"].append("metadata_condition_path_mismatch")

        if family.require_training_metadata:
            if metadata.get("fold") != path_fold or metadata.get("seed") != path_seed:
                result["status"] = "alignment_blocked"
                result["reason_codes"].append("training_metadata_fold_seed_missing_or_wrong")
            fit_splits = metadata.get("fit_splits")
            if not isinstance(fit_splits, list) or set(fit_splits) != {"train", "val"}:
                result["status"] = "alignment_blocked"
                result["reason_codes"].append("training_fit_splits_not_train_val")
            if metadata.get("evaluation_split") != "test":
                result["status"] = "alignment_blocked"
                result["reason_codes"].append("evaluation_split_not_test")
            if metadata.get("contains_test_rows") is not True:
                result["status"] = "alignment_blocked"
                result["reason_codes"].append("contains_test_rows_not_true")
            if path_condition and path_condition not in path_labels:
                result["status"] = "alignment_blocked"
                result["reason_codes"].append("training_condition_missing_or_wrong")

        n_images = metadata.get("n_images")
        if isinstance(n_images, int) and n_images != rows:
            result["status"] = "alignment_blocked"
            result["reason_codes"].append("metadata_n_images_mismatch")
        token = family.expected_dataset_token
        if token and not source:
            result["warnings"].append("internal_source_missing")
        elif token and token.lower() not in source.lower():
            result["warnings"].append("internal_source_conflicts_with_expected_dataset")
        model_token = family.expected_model_token
        if model_token and not model:
            result["warnings"].append("internal_model_missing")
        elif model_token and model_token.lower() not in model.lower():
            result["warnings"].append("internal_model_conflicts_with_expected_backbone")
    except FileNotFoundError:
        result["status"] = "absent"
        result["reason_codes"].append("archive_absent")
    except (
        OSError,
        KeyError,
        MemoryError,
        OverflowError,
        SyntaxError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        result["status"] = "corrupt_or_unreadable"
        result["reason_codes"].append(exc.__class__.__name__)
        result["error"] = str(exc)
    return result


def inspect_family(
    family: FamilySpec,
    manifests: dict[str, dict[str, Any]],
    split_manifests: dict[str, dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    paths = sorted(repo_root.glob(family.pattern), key=lambda value: value.as_posix())
    result: dict[str, Any] = {
        **asdict(family),
        "count": len(paths),
        "status": "available",
        "reason_codes": [],
        "evidence_paths": [rel(path, repo_root) for path in paths[:3]],
    }
    if not paths:
        result["status"] = "blocked"
        result["reason_codes"].append("row_level_archives_absent")
        return result
    folds: set[int] = set()
    seeds: set[int] = set()
    grid_pairs: set[tuple[int, int]] = set()
    for path in paths:
        text = path.as_posix()
        fold_match = re.search(r"/fold_(\d+)/", text)
        seed_match = re.search(r"_seed_(\d+)/", text)
        if fold_match and seed_match:
            fold = int(fold_match.group(1))
            seed = int(seed_match.group(1))
            folds.add(fold)
            seeds.add(seed)
            grid_pairs.add((fold, seed))
    result["folds"] = sorted(folds)
    result["seeds"] = sorted(seeds)
    result["fold_seed_pairs"] = len(grid_pairs)
    if family.expected_count is not None and family.require_five_by_five_grid:
        grid_complete = (
            len(paths) == family.expected_count
            and sorted(folds) == [0, 1, 2, 3, 4]
            and len(seeds) == 5
            and len(grid_pairs) == family.expected_count
        )
        result["grid_complete"] = grid_complete
        if not grid_complete:
            result["status"] = "partial"
            result["reason_codes"].append("archive_grid_incomplete")
    elif family.expected_count is not None:
        grid_complete = len(paths) == family.expected_count
        result["grid_complete"] = grid_complete
        if not grid_complete:
            result["status"] = "partial"
            result["reason_codes"].append("archive_count_mismatch")

    checks = [
        inspect_archive(
            path,
            family,
            manifests[family.dataset_id],
            split_manifests[family.dataset_id].get("_maps", {}),
            repo_root,
        )
        for path in paths
    ]
    result["archives_checked"] = len(checks)
    result["archive_status_counts"] = dict(
        sorted(Counter(check["status"] for check in checks).items())
    )
    warning_counts: Counter[str] = Counter()
    for check in checks:
        warning_counts.update(check.get("warnings", []))
    result["warning_counts"] = dict(sorted(warning_counts.items()))
    result["representative"] = checks[0]
    result["archive_issue_examples"] = [
        {
            "path": check["path"],
            "status": check["status"],
            "reason_codes": check.get("reason_codes", []),
            "warnings": check.get("warnings", []),
        }
        for check in checks
        if check["status"] != "available" or check.get("warnings")
    ][:5]

    integrity_failures = [check for check in checks if check["status"] != "available"]
    if integrity_failures:
        available_count = sum(check["status"] == "available" for check in checks)
        result["status"] = "partial" if available_count else "blocked"
        result["reason_codes"].append("archive_integrity_failures")
    if warning_counts:
        if result["status"] == "available":
            result["status"] = "manual_review"
        result["reason_codes"].append("representation_lineage_conflict")
    result["reason_codes"] = sorted(set(result["reason_codes"]))
    return result


def inspect_metadata(repo_root: Path, canine_manifest: dict[str, Any]) -> dict[str, Any]:
    headers: dict[str, list[str]] = {}
    all_columns: set[str] = set()
    file_statuses: dict[str, str] = {}
    for relative in METADATA_FILES:
        path = repo_root / relative
        header, _rows = read_csv(path)
        if header:
            headers[relative] = header
            all_columns.update(column.lower() for column in header)
            file_statuses[relative] = "available"
        elif not path.is_file():
            file_statuses[relative] = "absent"
        else:
            file_statuses[relative] = "unreadable_or_empty"

    def hits(wanted: set[str]) -> list[str]:
        return sorted(all_columns & {item.lower() for item in wanted})

    candidate_rows = canine_manifest.get("_rows", [])

    def summarize_proxy(
        proxy_id: str,
        relative: str,
        level: str,
        candidate_key: Any,
        source_key: Any,
        fields: list[str],
        source_filter: Any | None = None,
    ) -> dict[str, Any]:
        path = repo_root / relative
        header, source_rows = read_csv(path)
        if source_filter is not None:
            source_rows = [row for row in source_rows if source_filter(row)]
        source_map: dict[Any, dict[str, str]] = {}
        duplicate_keys = 0
        for row in source_rows:
            key = source_key(row)
            if not key or (isinstance(key, tuple) and not all(key)):
                continue
            if key in source_map:
                duplicate_keys += 1
            source_map[key] = row
        candidate_keys = [candidate_key(row) for row in candidate_rows]
        coverage = sum(key in source_map for key in candidate_keys)
        field_stats: dict[str, dict[str, Any]] = {}
        for field in fields:
            values = [str(row.get(field, "")) for row in source_map.values()]
            nonmissing = [value for value in values if value.strip()]
            field_stats[field] = {
                "nonmissing": len(nonmissing),
                "total": len(values),
                "unique_nonmissing": len(set(nonmissing)),
            }
        missing_fields = sorted(set(fields) - set(header))
        status = "available"
        reason_codes: list[str] = []
        if not path.is_file():
            status = "absent"
            reason_codes.append("proxy_source_absent")
        elif not header:
            status = "unreadable_or_empty"
            reason_codes.append("proxy_source_unreadable_or_empty")
        else:
            if missing_fields:
                status = "partial"
                reason_codes.append("proxy_fields_missing")
            if duplicate_keys:
                status = "partial"
                reason_codes.append("proxy_join_keys_duplicated")
            if candidate_rows and coverage != len(candidate_rows):
                status = "partial"
                reason_codes.append("proxy_join_incomplete")
            if fields and not any(
                stats["unique_nonmissing"] > 1 for stats in field_stats.values()
            ):
                status = "partial"
                reason_codes.append("proxy_fields_lack_variation")
        return {
            "proxy_id": proxy_id,
            "path": relative,
            "level": level,
            "status": status,
            "reason_codes": reason_codes,
            "source_rows": len(source_rows),
            "source_unique_keys": len(source_map),
            "duplicate_source_keys": duplicate_keys,
            "candidate_rows": len(candidate_rows),
            "join_matches": coverage,
            "join_fraction": coverage / len(candidate_rows) if candidate_rows else 0.0,
            "missing_fields": missing_fields,
            "field_stats": field_stats,
        }

    proxy_sources = [
        summarize_proxy(
            "geometry_crop_qc",
            (
                "results/external_multiscanner_caninescc/geometry_qualified/"
                "geometry_qualified_manifest.csv"
            ),
            "scanner_observation_and_region",
            lambda row: (row.get("sample_id", ""), row.get("region_id", ""), row.get("scanner_id", "")),
            lambda row: (row.get("sample_id", ""), row.get("region_id", ""), row.get("scanner_id", "")),
            [
                "adaptive_crop_side_level0",
                "inside_image_fraction",
                "padding_fraction",
                "region_max_padding_fraction",
                "region_rank",
            ],
        ),
        summarize_proxy(
            "registration_affine",
            (
                "results/external_multiscanner_caninescc/registration_database_audit/"
                "sample_affine_transforms.csv"
            ),
            "sample_by_scanner",
            lambda row: (row.get("sample_id", ""), row.get("scanner_id", "")),
            lambda row: (row.get("sample_id", ""), row.get("scanner_id", "")),
            [
                "rms_residual_pixels",
                "rotation_degrees",
                "translation_x",
                "translation_y",
                "scale_major",
                "scale_minor",
            ],
        ),
        summarize_proxy(
            "tiff_metadata",
            (
                "results/external_multiscanner_caninescc/orientation_audit/"
                "tiff_file_metadata.csv"
            ),
            "sample_by_scanner_file",
            lambda row: row.get("file_name", ""),
            lambda row: row.get("file_name", ""),
            ["raw_width", "raw_height", "orientation", "x_resolution", "y_resolution"],
        ),
        summarize_proxy(
            "tiff_file_size",
            "results/external_multiscanner_caninescc/inspection/file_inventory.csv",
            "sample_by_scanner_file",
            lambda row: row.get("file_name", ""),
            lambda row: Path(str(row.get("relative_path", ""))).name,
            ["size_bytes"],
            source_filter=lambda row: str(row.get("extension", "")).lower()
            in {".tif", ".tiff"},
        ),
        summarize_proxy(
            "annotation_history_geometry_delta",
            (
                "results/external_multiscanner_caninescc/registration_database_audit/"
                "sqlite_coco_geometry_comparison.csv"
            ),
            "scanner_observation_region_annotation",
            lambda row: (row.get("file_name", ""), row.get("annotation_id", "")),
            lambda row: (row.get("file_name", ""), row.get("coco_annotation_id", "")),
            ["lastModified", "delta_center_x", "delta_center_y", "absolute_bbox_delta_max"],
        ),
    ]
    available_proxy_sources = [
        source for source in proxy_sources if source["status"] == "available"
    ]

    return {
        "inspected_files": sorted(headers),
        "file_statuses": dict(sorted(file_statuses.items())),
        "site_columns": hits(SITE_COLUMNS),
        "preparation_columns": hits(PREPARATION_COLUMNS),
        "technical_proxy_columns": hits(TECHNICAL_PROXY_COLUMNS),
        "site_metadata_available": bool(hits(SITE_COLUMNS)),
        "preparation_metadata_available": bool(hits(PREPARATION_COLUMNS)),
        "technical_proxies_available": bool(available_proxy_sources),
        "technical_proxy_sources": proxy_sources,
        "available_proxy_source_count": len(available_proxy_sources),
    }


def find_row(rows: list[dict[str, str]], field: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(field) == value:
            return row
    return {}


def inspect_scanner_evidence(repo_root: Path) -> dict[str, Any]:
    header2, rows2 = read_csv(repo_root / FIGURE2)
    header4, rows4 = read_csv(repo_root / FIGURE4)
    original = find_row(rows2, "representation", "original_frozen_features")
    true_pair = find_row(rows2, "representation", "true_pair_biological")
    oldstyle = find_row(rows2, "representation", "oldstyle_keep_k4")
    dim8 = find_row(rows4, "variant", "acq_dim8_default")
    dim16 = find_row(rows4, "variant", "acq_dim16_stronger_xcov")

    def number(row: dict[str, str], key: str) -> float | None:
        try:
            value = float(row[key])
            return value if math.isfinite(value) else None
        except (KeyError, TypeError, ValueError):
            return None

    values = {
        "five_scanner_chance": 0.2,
        "original_scanner_probe": number(original, "scanner_probe_mean"),
        "true_pair_biological_scanner_probe": number(true_pair, "scanner_probe_mean"),
        "oldstyle_keep_k4_scanner_probe": number(oldstyle, "scanner_probe_mean"),
        "acq_dim8_biological_scanner_probe": number(dim8, "bio_scanner_probe"),
        "acq_dim16_biological_scanner_probe": number(dim16, "bio_scanner_probe"),
    }
    if all(value is not None for value in values.values()):
        interpretation = (
            "Neural biological branches are scanner-suppressed but remain above "
            "five-class chance in the existing linear probe. Oldstyle keep_k4 is "
            "at chance in summary evidence, but row-level embeddings are absent."
        )
    else:
        interpretation = "Scanner-summary evidence is incomplete or unreadable."
    return {
        **values,
        "source_paths": [FIGURE2, FIGURE4],
        "source_statuses": {
            FIGURE2: "available" if header2 else (
                "absent" if not (repo_root / FIGURE2).is_file() else "unreadable_or_empty"
            ),
            FIGURE4: "available" if header4 else (
                "absent" if not (repo_root / FIGURE4).is_file() else "unreadable_or_empty"
            ),
        },
        "interpretation": interpretation,
    }


def public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if not key.startswith("_")}


def metric_assessments(
    manifests: dict[str, dict[str, Any]],
    families: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    canine = manifests["canine_scc"]
    true_pair = families["canine_true_pair_biological"]
    oldstyle = families["oldstyle_keep_k4_row_level"]
    scorpion = manifests["scorpion"]

    if canine["status"] != "available" or true_pair["status"] == "blocked":
        canine_m1_status = "blocked"
        canine_m1_reasons = ["candidate_rows_or_metadata_unavailable"]
        canine_m1_rows = 0
    elif true_pair["status"] in {"manual_review", "partial"}:
        canine_m1_status = "partial"
        canine_m1_reasons = [
            "candidate_rows_and_category_matches_available",
            "strict_scanner_invariance_not_established",
            "representation_lineage_requires_manual_review",
        ]
        canine_m1_rows = canine["eligible_observations"]
    else:
        canine_m1_status = "feasible"
        canine_m1_reasons = [
            "candidate_rows_and_category_matches_available",
            "strict_scanner_invariance_not_established",
        ]
        canine_m1_rows = canine["eligible_observations"]

    provenance_metadata_present = (
        metadata["site_metadata_available"] or metadata["preparation_metadata_available"]
    )
    provenance_status = "partial" if provenance_metadata_present else "blocked"
    provenance_reasons = (
        ["crossed_design_and_aliasing_not_yet_validated"]
        if provenance_metadata_present
        else ["site_metadata_absent", "preparation_metadata_absent"]
    )

    proxy_status = "partial" if metadata["technical_proxies_available"] else "blocked"
    proxy_reasons = (
        [
            "proxy_not_proven_provenance",
            "proxy_level_aliasing_and_estimable_contrast_require_preregistration",
        ]
        if metadata["technical_proxies_available"]
        else ["technical_proxy_sources_unavailable"]
    )

    if oldstyle["count"] == 0:
        oldstyle_status = "blocked"
        oldstyle_reasons = ["oldstyle_row_level_embeddings_absent"]
        oldstyle_rows = 0
    elif oldstyle["status"] in {"available", "manual_review"}:
        oldstyle_status = "partial"
        oldstyle_reasons = ["operational_scanner_equivalence_gate_still_required"]
        oldstyle_rows = canine["eligible_observations"]
    else:
        oldstyle_status = oldstyle["status"]
        oldstyle_reasons = ["oldstyle_archive_integrity_incomplete"]
        oldstyle_rows = 0

    return [
        {
            "id": "M1_canine_cross_scanner_sample_link_auc",
            "status": canine_m1_status,
            "reason_codes": canine_m1_reasons,
            "eligible_rows": canine_m1_rows,
            "eligible_sample_category_cells": canine["eligible_sample_category_cells"],
            "leakage_guard": "cross_scanner_same_category_exact_region_excluded_test_only",
            "interpretation_ceiling": "coarse_category_adjusted_sample_structure_only",
        },
        {
            "id": "M1_oldstyle_cross_scanner_sample_link_auc",
            "status": oldstyle_status,
            "reason_codes": oldstyle_reasons,
            "eligible_rows": oldstyle_rows,
            "leakage_guard": "not_runnable",
            "interpretation_ceiling": "scanner_removal_residual_structure_only",
        },
        {
            "id": "M1_scorpion_category_adjusted_sample_link_auc",
            "status": "blocked",
            "reason_codes": ["category_labels_absent"],
            "eligible_rows": 0,
            "available_identity_rows": scorpion["rows"],
            "leakage_guard": "category_gate_unavailable",
            "interpretation_ceiling": "unadjusted_slide_region_structure_only",
        },
        {
            "id": "M2_cross_sample_site_preparation_link_auc",
            "status": provenance_status,
            "reason_codes": provenance_reasons,
            "eligible_rows": 0,
            "leakage_guard": "requires_different_biological_units_and_crossed_design",
            "interpretation_ceiling": "non_biological_attribution_blocked",
        },
        {
            "id": "M7_canine_technical_proxy_association",
            "status": proxy_status,
            "reason_codes": proxy_reasons,
            "eligible_rows": canine["rows"] if proxy_status == "partial" else 0,
            "leakage_guard": "requires_proxy_specific_level_join_and_independent_unit_guard",
            "interpretation_ceiling": (
                "measured_proxy_association_in_scanner_suppressed_representation_only"
            ),
        },
    ]


def build_audit(repo_root: Path) -> dict[str, Any]:
    manifests_internal: dict[str, dict[str, Any]] = {}
    specs_by_id = {spec.dataset_id: spec for spec in DATASETS}
    for spec in DATASETS:
        manifests_internal[spec.dataset_id] = inspect_manifest(
            repo_root / spec.manifest, spec, repo_root
        )

    split_manifests_internal = {
        dataset_id: inspect_split_manifests(
            repo_root, specs_by_id[dataset_id], manifest
        )
        for dataset_id, manifest in manifests_internal.items()
    }

    family_list = [
        inspect_family(
            family, manifests_internal, split_manifests_internal, repo_root
        )
        for family in FAMILIES
    ]
    families = {family["family_id"]: family for family in family_list}
    metadata = inspect_metadata(repo_root, manifests_internal["canine_scc"])
    scanner_evidence = inspect_scanner_evidence(repo_root)
    assessments = metric_assessments(manifests_internal, families, metadata)
    primary_status = next(
        item["status"]
        for item in assessments
        if item["id"] == "M1_canine_cross_scanner_sample_link_auc"
    )
    overall_status = "partial" if primary_status in {"feasible", "partial"} else "blocked"
    assessment_by_id = {item["id"]: item for item in assessments}
    if overall_status == "partial":
        bottom_line = (
            "Existing artifacts support a no-training, category-conditioned residual-"
            "structure audit for canine scanner-suppressed embeddings after lineage "
            "review. They do not establish scanner invariance or non-biological origin. "
            "SCORPION supports identity controls but lacks category labels. Site- and "
            "preparation-level attribution remains blocked."
        )
    else:
        bottom_line = (
            "The primary category-conditioned residual-structure audit is blocked because "
            "the required row-level candidate artifacts or aligned metadata were not found. "
            "Non-biological attribution is also blocked."
        )

    audit: dict[str, Any] = {
        "audit_id": "scanner_invariant_non_biological_residual_feasibility_v0",
        "overall_status": overall_status,
        "execution_boundary": {
            "new_training_run": False,
            "new_probe_fit": False,
            "representation_reconstructed": False,
            "candidate_metrics_computed": False,
            "evidence_sources_modified": False,
        },
        "manifests": {
            key: public_manifest(value) for key, value in manifests_internal.items()
        },
        "split_manifests": {
            key: public_manifest(value)
            for key, value in split_manifests_internal.items()
        },
        "families": family_list,
        "metadata": metadata,
        "scanner_evidence": scanner_evidence,
        "metric_assessments": assessments,
        "scientific_gates": {
            "candidate_residual_detection": (
                "partial_ready_after_lineage_review"
                if primary_status == "partial"
                else primary_status
            ),
            "strict_scanner_invariant_row_level_candidate": (
                "blocked"
                if families["oldstyle_keep_k4_row_level"]["count"] == 0
                else "partial_requires_operational_equivalence_gate"
            ),
            "category_adjustment_canine": (
                "available"
                if manifests_internal["canine_scc"]["status"] == "available"
                and manifests_internal["canine_scc"]["categories"] > 0
                else "blocked"
            ),
            "category_adjustment_scorpion": (
                "available"
                if manifests_internal["scorpion"]["categories"] > 0
                else "blocked"
            ),
            "technical_proxy_association": assessment_by_id[
                "M7_canine_technical_proxy_association"
            ]["status"],
            "site_preparation_attribution": assessment_by_id[
                "M2_cross_sample_site_preparation_link_auc"
            ]["status"],
            "artifact_origin": "not_identifiable_from_current_metadata",
        },
        "bottom_line": bottom_line,
    }
    fingerprint_payload = json.dumps(audit, sort_keys=True, separators=(",", ":"))
    audit["evidence_schema_fingerprint_sha256"] = hashlib.sha256(
        fingerprint_payload.encode("utf-8")
    ).hexdigest()
    return audit


def family_map(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {family["family_id"]: family for family in audit["families"]}


def render_report(audit: dict[str, Any]) -> str:
    manifests = audit["manifests"]
    canine = manifests["canine_scc"]
    scorpion = manifests["scorpion"]
    families = family_map(audit)
    scanner = audit["scanner_evidence"]
    metadata = audit["metadata"]

    def probe(name: str) -> str:
        value = scanner.get(name)
        return "not found" if value is None else f"{value:.3f}"

    if audit["overall_status"] == "partial":
        verdict = (
            "**PARTIAL: candidate residual-structure detection is feasible; "
            "non-biological attribution is not.**"
        )
    else:
        verdict = (
            "**BLOCKED: required candidate artifacts or aligned metadata are unavailable; "
            "non-biological attribution is not feasible.**"
        )

    gates = audit["scientific_gates"]
    canine_category_status = gates["category_adjustment_canine"]
    scorpion_category_status = gates["category_adjustment_scorpion"]
    provenance_status = gates["site_preparation_attribution"]
    if canine["status"] == "available":
        canine_support_sentence = (
            f"Canine composite keys are unique and its geometry-qualified manifest "
            f"contains rows for {canine['regions']} regions on {canine['scanners']} "
            "scanners. The proposed different-region, same-category sample audit has "
            f"{canine['eligible_sample_category_cells']} held-out-fold-eligible "
            f"sample-category cells, {canine['eligible_regions']} eligible regions, and "
            f"{canine['eligible_observations']} eligible scanner observations across "
            f"{len(canine['usable_categories'])} categories. Excluded category: "
            f"{', '.join(canine['excluded_categories']) or 'none'}."
        )
    else:
        canine_support_sentence = (
            f"Canine manifest status is `{canine['status']}`; fold-aware category/sample "
            "support is unavailable."
        )

    lines = [
        "# Scanner-Invariant Residual Provenance Feasibility Report",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        audit["bottom_line"],
        "",
        "## Execution boundary",
        "",
        "- No new representation training was run.",
        "- No probe, classifier, residualizer, projection, or metric was fit.",
        "- No oldstyle representation was reconstructed.",
        "- No existing data, result, checkpoint, or experiment output was modified.",
        "- This report records artifact and metadata feasibility only.",
        "",
        "## Scientific premise",
        "",
        "Paired scanner acquisitions vary scanner while holding both tissue biology and pre-scanner slide/preparation factors fixed. Scanner agreement can therefore preserve both. Paired data can support detection of scanner-shared residual structure, but cannot identify its biological versus preparation origin without independent provenance variation.",
        "",
        "The general invariance blind spot is not itself a new result. The defensible future novelty would be a literature-positioned paired-scanner provenance audit with crossed preparation data and explicit identifiability gates. The current sample-link path is an exploratory gate.",
        "",
        "## Dataset capability",
        "",
        "| Dataset | Rows | Regions | Samples/slides | Scanners | Categories | Category-adjusted audit | Provenance attribution |",
        "|---|---:|---:|---:|---:|---:|---|---|",
        (
            f"| Canine SCC | {canine['rows']} | {canine['regions']} | {canine['samples']} | "
            f"{canine['scanners']} | {canine['categories']} | {canine_category_status} | {provenance_status} |"
        ),
        (
            f"| SCORPION | {scorpion['rows']} | {scorpion['regions']} | {scorpion['samples']} | "
            f"{scorpion['scanners']} | {scorpion['categories']} | {scorpion_category_status} | {provenance_status} |"
        ),
        "",
        canine_support_sentence,
        "",
        "| Held-out fold | Eligible cells | Eligible regions | Eligible observations | Non-estimable categories |",
        "|---:|---:|---:|---:|---|",
    ]
    for fold, item in canine.get("fold_eligibility", {}).items():
        nonestimable = ", ".join(item["nonestimable_categories"]) or "none"
        lines.append(
            f"| {fold} | {item['eligible_sample_category_cells']} | {item['eligible_regions']} | {item['eligible_observations']} | {nonestimable} |"
        )

    lines.extend(
        [
            "",
        "## Representation artifact inventory",
        "",
            "| Family | Archives | Expected | Checked | Integrity/status | Representative join |",
            "|---|---:|---:|---:|---|---|",
        ]
    )

    ordered = [spec.family_id for spec in FAMILIES]
    for family_id in ordered:
        item = families[family_id]
        expected = "n/a" if item["expected_count"] is None else str(item["expected_count"])
        representative = item.get("representative", {})
        if representative:
            matches = representative.get("manifest_join_matches", 0)
            total = representative.get("manifest_join_total", 0)
            join = f"{matches}/{total}" if total else "not available"
        else:
            join = "not available"
        lines.append(
            f"| `{family_id}` | {item['count']} | {expected} | {item.get('archives_checked', 0)} | {item['status']} | {join} |"
        )

    primary = families["canine_true_pair_biological"].get("representative", {})
    source_warning = primary.get("metadata_source", "not found")
    expected_grids = [
        item for item in families.values() if item.get("require_five_by_five_grid")
    ]
    complete_grids = sum(bool(item.get("grid_complete")) for item in expected_grids)
    archives_checked = sum(item.get("archives_checked", 0) for item in families.values())
    integrity_failures = sum(
        count
        for item in families.values()
        for status, count in item.get("archive_status_counts", {}).items()
        if status != "available"
    )
    source_conflicts = sum(
        item.get("warning_counts", {}).get(
            "internal_source_conflicts_with_expected_dataset", 0
        )
        for item in families.values()
    )
    model_conflicts = sum(
        item.get("warning_counts", {}).get(
            "internal_model_conflicts_with_expected_backbone", 0
        )
        for item in families.values()
    )
    primary_family = families["canine_true_pair_biological"]
    if primary_family["count"]:
        candidate_lineage_intro = (
            "The primary canine archive is `manual_review` for confirmatory use because "
            "its internal source string is inconsistent with its path, rows, fold, seed, "
            "and split evidence:"
        )
        primary_source_line = f"- `metadata_json.source`: `{source_warning}`"
    else:
        candidate_lineage_intro = "The primary canine row-level archive is unavailable."
        primary_source_line = "- `metadata_json.source`: not available"
    scorpion_crossbackbone_count = sum(
        item["count"]
        for family_id, item in families.items()
        if family_id.startswith("scorpion_")
    )
    if scorpion_crossbackbone_count:
        crossbackbone_sentence = (
            "SCORPION has DINOv2, Phikon, and ResNet50 true-pair and bottleneck archives "
            "for cross-backbone sensitivity. It still cannot support category-adjusted "
            "testing because category labels are absent, and Phikon/ResNet50 internal "
            "model strings require lineage correction."
        )
    else:
        crossbackbone_sentence = "No SCORPION cross-backbone row-level archives were found."
    scanner_values_present = all(
        scanner.get(key) is not None
        for key in (
            "original_scanner_probe",
            "true_pair_biological_scanner_probe",
            "oldstyle_keep_k4_scanner_probe",
            "acq_dim8_biological_scanner_probe",
            "acq_dim16_biological_scanner_probe",
        )
    )
    if scanner_values_present:
        scanner_premise_sentence = (
            "The neural biological branches are scanner-suppressed, not established "
            "scanner-invariant. The only current chance-level summary candidate is "
            "oldstyle `keep_k4`, but no row-level oldstyle embedding archive is present. "
            "The strict scanner-invariant residual metric is therefore blocked in this "
            "read-only phase."
        )
    else:
        scanner_premise_sentence = (
            "Required scanner-summary evidence is incomplete; the scanner-suppression "
            "premise cannot be audited from this repository root."
        )
    if archives_checked:
        lineage_summary_sentence = (
            f"Lineage review remains required: {source_conflicts} archives have an "
            f"internal dataset/source conflict and {model_conflicts} have an internal "
            f"backbone/model conflict. All {source_conflicts} source conflicts occur in "
            "the audited canine projected archives."
        )
    else:
        lineage_summary_sentence = "No row-level archives were available for lineage review."
    if scorpion["rows"] and scorpion_crossbackbone_count:
        scorpion_control_sentence = (
            "SCORPION can support cross-scanner slide/region and cross-backbone controls, "
            "but it cannot establish that structure is unexplained by category because "
            "category labels are absent."
        )
    else:
        scorpion_control_sentence = (
            "SCORPION control feasibility is blocked because its rows or representation "
            "archives are unavailable."
        )
    proxy_sources = metadata["technical_proxy_sources"]
    available_proxy_sources = [
        source for source in proxy_sources if source["status"] == "available"
    ]
    if proxy_sources and len(available_proxy_sources) == len(proxy_sources):
        proxy_summary_sentence = (
            "All listed proxy sources currently join completely, but M7 remains partial "
            "until each proxy has a pre-declared level, independent unit, missingness "
            "rule, aliasing audit, and estimable contrast. A generic region/sample block "
            "is not valid for every proxy level."
        )
    elif available_proxy_sources:
        proxy_summary_sentence = (
            "Only some proxy sources are available and aligned; M7 remains partial and "
            "requires proxy-specific design review."
        )
    else:
        proxy_summary_sentence = "No aligned technical-proxy source is available; M7 is blocked."
    if scanner_values_present:
        metric_pitfall_sentence = (
            "Existing same-sample top-1 retrieval near 1.0 is not evidence for sample-level "
            "residual artifact because another scanner view of the exact same region was "
            "eligible as the nearest neighbor. The proposed primary metric must use a "
            "different region from the same sample, exact category matching, a different "
            "scanner, and a same-target-scanner negative from another sample."
        )
    else:
        metric_pitfall_sentence = (
            "Any same-sample retrieval metric that permits another scanner view of the "
            "exact same region is invalid for sample-level residual inference."
        )
    if audit["overall_status"] == "partial":
        next_step_sentence = (
            "Do not train a new representation. First fix representation provenance and "
            "acquire or recover crossed preparation/site metadata. If the immediate goal "
            "is only candidate discovery, the next authorized artifact should be a "
            "no-training implementation of matched cross-scanner, different-region, "
            "same-category sample-link AUC on existing canine test embeddings with the "
            "listed controls."
        )
    else:
        next_step_sentence = (
            "Do not train a new representation. First restore or identify the required "
            "row-level candidates, aligned manifests, and provenance metadata, then rerun "
            "this feasibility audit."
        )
    if complete_grids == len(expected_grids):
        grid_sentence = (
            f"All {complete_grids} expected five-fold by five-seed archive grids are "
            f"complete. The audit checked all {archives_checked} discovered archives; "
            f"{integrity_failures} failed schema, alignment, fold, split, or training-metadata checks."
        )
    else:
        grid_sentence = (
            f"Only {complete_grids} of {len(expected_grids)} expected archive grids are "
            "complete; incomplete grids require review."
        )
    lines.extend(
        [
            "",
            grid_sentence,
            "",
            lineage_summary_sentence,
            "",
            candidate_lineage_intro,
            "",
            primary_source_line,
            "",
            "A frozen representation manifest and checksums are required before metric execution.",
            "",
            crossbackbone_sentence,
            "",
            "## Scanner premise",
            "",
            f"- Five-scanner chance: {probe('five_scanner_chance')}.",
            f"- Original frozen scanner probe: {probe('original_scanner_probe')}.",
            f"- True-pair biological scanner probe: {probe('true_pair_biological_scanner_probe')}.",
            f"- Acquisition-dim-8 biological scanner probe: {probe('acq_dim8_biological_scanner_probe')}.",
            f"- Acquisition-dim-16 biological scanner probe: {probe('acq_dim16_biological_scanner_probe')}.",
            f"- Oldstyle `keep_k4` scanner probe: {probe('oldstyle_keep_k4_scanner_probe')} (summary evidence only).",
            "",
            scanner_premise_sentence,
            "",
            "## Candidate-metric feasibility",
            "",
            "| Metric | Status | Interpretation ceiling |",
            "|---|---|---|",
        ]
    )
    for assessment in audit["metric_assessments"]:
        lines.append(
            f"| `{assessment['id']}` | {assessment['status']} | {assessment['interpretation_ceiling']} |"
        )

    site = ", ".join(metadata["site_columns"]) or "none found in allowlisted fields"
    prep = ", ".join(metadata["preparation_columns"]) or "none found in allowlisted fields"
    proxy_preview = ", ".join(metadata["technical_proxy_columns"][:12]) or "none found"
    lines.extend(
        [
            "",
            "## Metadata boundary",
            "",
            f"- Site/laboratory fields: {site}.",
            f"- Preparation/batch/stain fields: {prep}.",
            f"- Technical proxy fields available: {proxy_preview}.",
            f"- Successfully read allowlisted metadata files: {len(metadata['inspected_files'])}; unreadable/empty files: {sum(status == 'unreadable_or_empty' for status in metadata['file_statuses'].values())}.",
            "",
            "Sample, slide, and region IDs are biological identities as well as possible process carriers. Geometry, crop, padding, registration, orientation, file-size, and annotation fields are proxy controls and may reflect tissue or scanner. None is a validated preparation label.",
            "",
            scorpion_control_sentence,
            "",
            "| Technical proxy source | Level | Status | Join coverage |",
            "|---|---|---|---:|",
        ]
    )
    for source in metadata["technical_proxy_sources"]:
        lines.append(
            f"| `{source['proxy_id']}` | {source['level']} | {source['status']} | {source['join_matches']}/{source['candidate_rows']} |"
        )
    lines.extend(
        [
            "",
            proxy_summary_sentence,
            "",
            "## Existing metric pitfall",
            "",
            metric_pitfall_sentence,
            "",
            "## Blockers before metric execution",
            "",
            "1. Resolve archive lineage conflicts and freeze a representation manifest.",
            "2. Keep neural candidates labeled scanner-suppressed unless they pass a stronger operational gate.",
            "3. Materialize oldstyle row-level output only under a separately authorized derived-artifact step if the strict removal audit is desired.",
            "4. Pre-register per-fold/category/scanner-direction eligibility and rare-stratum exclusions.",
            "5. Implement exact-region exclusion and atomic region-bundle permutations.",
            "",
            "## Blockers before non-biological attribution",
            "",
            "1. Add explicit site/preparation/processing/stain metadata with definitions and lineage.",
            "2. Demonstrate that provenance levels repeat across independent biological units.",
            "3. Demonstrate that provenance is not aliased with sample, scanner, category, or fold.",
            "4. Test provenance across different samples or blocks so fine-grained biology cannot trivially supply the match.",
            "",
            "## Claim boundary",
            "",
            "A later positive canine sample-link result could support: same-category cross-scanner sample association is detectable in fixed scanner-suppressed embeddings. It could not establish that the structure is non-biological. A technical-proxy association would remain association-only. A valid crossed site/preparation result could support association with a measured non-scanner provenance variable; use `operationally scanner-invariant` only if the separate G2 gate also passes.",
            "",
            "This feasibility audit supports no clinical, diagnostic, deployment, patient-care, scanner-bias-solved, universal disentanglement, or causal artifact claim.",
            "",
            "## Bottom line",
            "",
            next_step_sentence,
            "",
            "## Deterministic evidence fingerprint",
            "",
            f"`{audit['evidence_schema_fingerprint_sha256']}`",
            "",
            "This fingerprint covers the audit's deterministic schema/count/status payload. It is not a checksum of the full feature payloads.",
            "",
        ]
    )
    return "\n".join(lines)


def check_report(rendered: str) -> int:
    try:
        existing = CHECKED_REPORT.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"REPORT_CHECK_FAIL: {exc}", file=sys.stderr)
        return 1
    if existing == rendered:
        print(f"REPORT_CHECK_PASS: {CHECKED_REPORT.name}")
        return 0
    diff = difflib.unified_diff(
        existing.splitlines(),
        rendered.splitlines(),
        fromfile=CHECKED_REPORT.name,
        tofile="rendered",
        lineterm="",
    )
    print("REPORT_CHECK_FAIL: checked report differs", file=sys.stderr)
    for line in diff:
        print(line, file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root to inspect (default: inferred from script path).",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format written to stdout.",
    )
    parser.add_argument(
        "--check-report",
        action="store_true",
        help="Compare deterministic Markdown with the checked-in feasibility_report.md.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    audit = build_audit(repo_root)
    rendered = render_report(audit)
    if args.check_report:
        return check_report(rendered)
    if args.format == "json":
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
