#!/usr/bin/env python3
"""Read-only feasibility audit for the paired scanner counterfactual benchmark."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "paired_scanner_counterfactual_benchmark_feasibility"
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
WSI_SUFFIXES = {".svs", ".ndpi", ".mrxs", ".scn", ".bif", ".tif", ".tiff"}
ARRAY_SUFFIXES = {".npz", ".npy"}
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}

SCANNER_COLUMNS = {"scanner_id", "scanner", "scanner_label", "scanner_name"}
PAIR_COLUMNS = {
    "region_id",
    "pair_id",
    "pair_group",
    "pair_identifier",
    "sample_region_id",
    "sample_id",
}
CATEGORY_COLUMNS = {
    "category",
    "category_id",
    "category_name",
    "tissue",
    "tissue_label",
    "class",
    "class_label",
}
COORD_COLUMNS = {
    "bbox",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "bbox_center_x",
    "bbox_center_y",
    "crop_x0_level0",
    "crop_y0_level0",
    "coordinatex",
    "coordinatey",
    "coordinate_x",
    "coordinate_y",
}
REGISTRATION_COLUMNS = {
    "correspondence_basis",
    "orientation_normalization_degrees",
    "rms_residual_pixels_median",
    "rms_residual_pixels_q05",
    "rms_residual_pixels_q95",
    "scanner_affine_summary",
    "registration_keyword_columns",
}
REGISTRATION_QC_COLUMNS = {
    "registration_confidence",
    "confidence",
    "inside_image_fraction",
    "padding_fraction",
    "region_max_padding_fraction",
    "rms_residual_pixels_median",
}
PATH_COLUMNS = {"path", "patch_path", "image_path", "file_name", "source_filename"}


def rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def status(flag: bool) -> str:
    return "available" if flag else "not_found"


def join_evidence(items: list[str], limit: int = 4) -> str:
    clean = [item for item in items if item]
    if not clean:
        return "not_found"
    shown = clean[:limit]
    suffix = "" if len(clean) <= limit else f"; +{len(clean) - limit} more"
    return "; ".join(shown) + suffix


def read_csv_preview(path: Path, rows: int = 5) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            preview = []
            for _, row in zip(range(rows), reader):
                preview.append({key: str(value) for key, value in row.items()})
            return header, preview
    except (OSError, UnicodeError, csv.Error):
        return [], []


def collect_json_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            full = f"{prefix}.{key_text}" if prefix else key_text
            keys.add(full)
            keys.update(collect_json_keys(nested, full))
    elif isinstance(value, list):
        for item in value[:20]:
            keys.update(collect_json_keys(item, prefix))
    return keys


def read_json_keys(path: Path) -> set[str]:
    try:
        if path.stat().st_size > 8_000_000:
            return set()
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
        return collect_json_keys(data)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return set()


def read_npz_keys(path: Path) -> set[str]:
    if path.suffix.lower() != ".npz":
        return set()
    try:
        with zipfile.ZipFile(path) as archive:
            return {Path(name).name for name in archive.namelist()}
    except (OSError, zipfile.BadZipFile):
        return set()


def dataset_roots(dataset: str) -> list[Path]:
    results = REPO_ROOT / "results"
    data = REPO_ROOT / "data"
    roots: list[Path] = []

    if dataset == "external_multiscanner_caninescc":
        exact = [
            data / "external_multiscanner_caninescc",
            results / "external_multiscanner_caninescc",
            results / "paired_acquisition_factorization_acquisition_bottleneck_separation_frontier",
        ]
        roots.extend(exact)
        tokens = ("caninescc",)
    elif dataset == "scorpion":
        exact = [
            data / "scorpion",
            results / "scorpion",
            results / "paired_acquisition_factorization_frontier_selected_crossbackbone_validation",
        ]
        roots.extend(exact)
        tokens = ("scorpion",)
    else:
        tokens = (dataset,)

    if results.exists():
        for child in results.iterdir():
            if child.is_dir() and any(token in child.name.lower() for token in tokens):
                roots.append(child)

    seen: set[Path] = set()
    deduped = []
    for root in roots:
        resolved = root.resolve()
        if root.exists() and resolved not in seen:
            deduped.append(root)
            seen.add(resolved)
    return deduped


def iter_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            for path in root.rglob("*"):
                if path.is_file():
                    resolved = path.resolve()
                    if resolved not in seen:
                        files.append(path)
                        seen.add(resolved)
        except OSError:
            continue
    return files


def inspect_checkpoints(paths: list[Path], limit: int = 5) -> tuple[str, list[str]]:
    if not paths:
        return "not_found", []
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        return "checkpoint_found_uninspected", [
            f"{rel(paths[0])} (torch unavailable: {exc.__class__.__name__})"
        ]

    evidence: list[str] = []
    for path in paths[:limit]:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - checkpoint format dependent
            evidence.append(f"{rel(path)} (inspection_failed: {exc.__class__.__name__})")
            continue
        key_names: set[str] = set()
        if isinstance(checkpoint, dict):
            for key, value in checkpoint.items():
                key_names.add(str(key))
                if isinstance(value, dict):
                    key_names.update(str(nested_key) for nested_key in value.keys())
        if any("decoder" in key.lower() for key in key_names):
            evidence.append(f"{rel(path)} (decoder keys found)")
    if evidence:
        return "available", evidence
    return "checkpoint_found_no_decoder_keys", [rel(path) for path in paths[:limit]]


def has_any(columns: set[str], wanted: set[str]) -> bool:
    return bool(columns & wanted)


def evidence_for_columns(
    csv_headers: dict[Path, list[str]],
    json_keys: dict[Path, set[str]],
    wanted: set[str],
) -> list[str]:
    evidence: list[str] = []
    wanted_lower = {item.lower() for item in wanted}
    for path, header in csv_headers.items():
        hits = sorted({column for column in header if column.lower() in wanted_lower})
        if hits:
            evidence.append(f"{rel(path)} columns={','.join(hits)}")
    for path, keys in json_keys.items():
        hits = sorted({key for key in keys if key.split(".")[-1].lower() in wanted_lower})
        if hits:
            evidence.append(f"{rel(path)} keys={','.join(hits[:6])}")
    return evidence


def path_column_has_suffix(
    previews: dict[Path, list[dict[str, str]]],
    columns: set[str],
    suffixes: set[str],
) -> list[str]:
    evidence: list[str] = []
    columns_lower = {column.lower() for column in columns}
    for path, rows in previews.items():
        for row in rows:
            for key, value in row.items():
                if key.lower() in columns_lower and Path(value).suffix.lower() in suffixes:
                    evidence.append(f"{rel(path)} {key}={value}")
                    break
            if evidence and evidence[-1].startswith(rel(path)):
                break
    return evidence


def existing_external_paths(
    previews: dict[Path, list[dict[str, str]]],
    json_keys: dict[Path, set[str]],
    roots: list[Path],
) -> list[str]:
    evidence: list[str] = []
    # This audit records path metadata but does not require files outside the repo.
    for root in roots:
        if root.name == "scorpion":
            summary = root / "manifest_summary.json"
            if summary.exists():
                evidence.append(f"{rel(summary)}")
    for path, rows in previews.items():
        for row in rows:
            value = row.get("path") or row.get("image_path") or row.get("patch_path")
            if value and Path(value).suffix.lower() in IMAGE_SUFFIXES:
                evidence.append(f"{rel(path)} path={value}")
                break
    for path, keys in json_keys.items():
        lowered = {key.lower() for key in keys}
        if "data_root" in {key.split(".")[-1] for key in lowered}:
            evidence.append(f"{rel(path)} data_root metadata")
    return evidence


def audit_dataset(dataset: str) -> dict[str, Any]:
    roots = dataset_roots(dataset)
    files = iter_files(roots)

    csv_headers: dict[Path, list[str]] = {}
    csv_previews: dict[Path, list[dict[str, str]]] = {}
    json_keys: dict[Path, set[str]] = {}
    npz_keys: dict[Path, set[str]] = {}
    suffix_counts: dict[str, int] = defaultdict(int)

    for path in files:
        suffix = path.suffix.lower()
        suffix_counts[suffix] += 1
        if suffix == ".csv":
            header, preview = read_csv_preview(path)
            csv_headers[path] = header
            csv_previews[path] = preview
        elif suffix == ".json":
            json_keys[path] = read_json_keys(path)
        elif suffix == ".npz":
            npz_keys[path] = read_npz_keys(path)

    all_columns = {
        column.lower()
        for header in csv_headers.values()
        for column in header
    }
    all_json_leaf_keys = {
        key.split(".")[-1].lower()
        for keys in json_keys.values()
        for key in keys
    }
    all_npz_keys = {
        key.lower()
        for keys in npz_keys.values()
        for key in keys
    }

    scanner_evidence = evidence_for_columns(csv_headers, json_keys, SCANNER_COLUMNS)
    if "scanner_id.npy" in all_npz_keys:
        scanner_evidence.extend(
            f"{rel(path)} arrays=scanner_id.npy"
            for path, keys in npz_keys.items()
            if "scanner_id.npy" in {key.lower() for key in keys}
        )

    pair_evidence = evidence_for_columns(csv_headers, json_keys, PAIR_COLUMNS)
    if "region_id.npy" in all_npz_keys:
        pair_evidence.extend(
            f"{rel(path)} arrays=region_id.npy"
            for path, keys in npz_keys.items()
            if "region_id.npy" in {key.lower() for key in keys}
        )

    category_evidence = evidence_for_columns(csv_headers, json_keys, CATEGORY_COLUMNS)

    feature_npz = [
        path for path, keys in npz_keys.items()
        if "features.npy" in {key.lower() for key in keys}
        and "projected_features" not in path.name.lower()
    ]
    feature_json = [
        path for path, keys in json_keys.items()
        if "feature_archive" in {key.split(".")[-1].lower() for key in keys}
        or "output" in {key.split(".")[-1].lower() for key in keys}
        and "features" in path.as_posix().lower()
    ]
    frozen_feature_evidence = [rel(path) for path in feature_npz[:8]]
    frozen_feature_evidence.extend(rel(path) for path in feature_json[:8])

    projected_archives = [
        path for path, keys in npz_keys.items()
        if {"features.npy", "acquisition_features.npy"}.issubset(
            {key.lower() for key in keys}
        )
    ]
    biological_evidence = [
        f"{rel(path)} arrays=features.npy" for path in projected_archives[:8]
    ]
    acquisition_evidence = [
        f"{rel(path)} arrays=acquisition_features.npy" for path in projected_archives[:8]
    ]

    checkpoint_paths = [
        path for path in files
        if path.suffix.lower() in CHECKPOINT_SUFFIXES
        and "runs" in path.as_posix().lower()
    ]
    decoder_status, decoder_evidence = inspect_checkpoints(checkpoint_paths)

    pair_assignment_files = [
        path for path in files
        if path.suffix.lower() == ".csv" and "pair_assignments" in path.as_posix().lower()
    ]
    swap_metadata_evidence = [rel(path) for path in pair_assignment_files[:8]]

    actual_patch_images = [
        path for path in files
        if path.suffix.lower() in IMAGE_SUFFIXES and "patch" in path.as_posix().lower()
    ]
    manifest_patch_paths = path_column_has_suffix(csv_previews, PATH_COLUMNS, IMAGE_SUFFIXES)
    patch_image_evidence = [rel(path) for path in actual_patch_images[:8]]
    patch_image_evidence.extend(manifest_patch_paths[:8])

    actual_wsi_files = [
        path for path in files
        if path.suffix.lower() in WSI_SUFFIXES
        and "patch" not in path.as_posix().lower()
    ]
    wsi_metadata = path_column_has_suffix(csv_previews, {"file_name", "wsi_path"}, WSI_SUFFIXES)
    wsi_evidence = [rel(path) for path in actual_wsi_files[:8]]
    wsi_evidence.extend(f"metadata_only: {item}" for item in wsi_metadata[:8])

    coord_evidence = evidence_for_columns(csv_headers, json_keys, COORD_COLUMNS)
    registration_evidence = evidence_for_columns(
        csv_headers, json_keys, REGISTRATION_COLUMNS
    )
    registration_evidence.extend(
        rel(path)
        for path in files[:]
        if "registration" in path.as_posix().lower()
    )
    registration_evidence = list(dict.fromkeys(registration_evidence))

    registration_qc_evidence = evidence_for_columns(
        csv_headers, json_keys, REGISTRATION_QC_COLUMNS
    )

    scanner_available = bool(scanner_evidence)
    pair_available = bool(pair_evidence)
    category_available = bool(category_evidence)
    feature_available = bool(frozen_feature_evidence)
    biological_available = bool(biological_evidence)
    acquisition_available = bool(acquisition_evidence)
    decoder_available = decoder_status == "available"
    swap_metadata_available = bool(swap_metadata_evidence)
    patch_images_available = bool(patch_image_evidence)
    wsi_available = bool(actual_wsi_files)
    wsi_metadata_only = bool(wsi_metadata) and not wsi_available
    coords_available = bool(coord_evidence)
    registration_available = bool(registration_evidence)
    registration_qc_available = bool(registration_qc_evidence)

    feature_feasible = scanner_available and pair_available and feature_available
    decoder_feasible = (
        feature_feasible
        and biological_available
        and acquisition_available
        and decoder_available
        and swap_metadata_available
    )
    pixel_prereqs = (
        scanner_available
        and pair_available
        and patch_images_available
        and coords_available
        and registration_available
    )
    if pixel_prereqs and registration_qc_available:
        pixel_status = "candidate_requires_registration_qc_review"
    elif pixel_prereqs:
        pixel_status = "partial_missing_registration_confidence"
    else:
        pixel_status = "future_only"

    return {
        "dataset": dataset,
        "roots": roots,
        "files": files,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "scanner_ids": status(scanner_available),
        "scanner_evidence": scanner_evidence,
        "paired_region_ids": status(pair_available),
        "pair_evidence": pair_evidence,
        "category_labels": status(category_available),
        "category_evidence": category_evidence,
        "frozen_feature_arrays": status(feature_available),
        "feature_evidence": frozen_feature_evidence,
        "biological_branch_embeddings": status(biological_available),
        "biological_evidence": biological_evidence,
        "acquisition_branch_embeddings": status(acquisition_available),
        "acquisition_evidence": acquisition_evidence,
        "decoder_weights": decoder_status,
        "decoder_evidence": decoder_evidence,
        "swap_metadata": status(swap_metadata_available),
        "swap_metadata_evidence": swap_metadata_evidence,
        "patch_image_paths": status(patch_images_available),
        "patch_image_evidence": patch_image_evidence,
        "raw_wsi_paths": "available" if wsi_available else (
            "metadata_only" if wsi_metadata_only else "not_found"
        ),
        "wsi_evidence": wsi_evidence,
        "patch_coordinates": status(coords_available),
        "coordinate_evidence": coord_evidence,
        "registration_metadata": status(registration_available),
        "registration_evidence": registration_evidence,
        "registration_confidence_or_qc": status(registration_qc_available),
        "registration_qc_evidence": registration_qc_evidence,
        "feature_space_feasible": "yes" if feature_feasible else "no",
        "decoder_space_feasible": "yes" if decoder_feasible else "no",
        "pixel_space_feasible_now": pixel_status,
        "category_label_anchor": "yes" if category_available else "no",
        "pair_retrieval_only_anchor": "yes" if pair_available and not category_available else "no",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def requirement_rows(audits: list[dict[str, Any]], layer: str) -> list[dict[str, str]]:
    if layer == "feature":
        requirements = [
            ("scanner IDs", "scanner_ids", "scanner_evidence"),
            ("paired region IDs", "paired_region_ids", "pair_evidence"),
            ("frozen feature arrays", "frozen_feature_arrays", "feature_evidence"),
            ("category labels", "category_labels", "category_evidence"),
        ]
    elif layer == "decoder":
        requirements = [
            ("biological branch embeddings", "biological_branch_embeddings", "biological_evidence"),
            ("acquisition branch embeddings", "acquisition_branch_embeddings", "acquisition_evidence"),
            ("decoder/composition weights", "decoder_weights", "decoder_evidence"),
            ("paired swap metadata", "swap_metadata", "swap_metadata_evidence"),
            ("scanner IDs", "scanner_ids", "scanner_evidence"),
            ("paired region IDs", "paired_region_ids", "pair_evidence"),
        ]
    else:
        requirements = [
            ("patch image paths", "patch_image_paths", "patch_image_evidence"),
            ("raw WSI paths", "raw_wsi_paths", "wsi_evidence"),
            ("scanner-specific paired acquisitions", "scanner_ids", "scanner_evidence"),
            ("local region correspondence", "paired_region_ids", "pair_evidence"),
            ("patch coordinates", "patch_coordinates", "coordinate_evidence"),
            ("registration metadata", "registration_metadata", "registration_evidence"),
            ("registration confidence/QC", "registration_confidence_or_qc", "registration_qc_evidence"),
            ("category labels for preservation", "category_labels", "category_evidence"),
        ]

    rows = []
    for audit in audits:
        for requirement, status_key, evidence_key in requirements:
            rows.append(
                {
                    "dataset": audit["dataset"],
                    "layer": layer,
                    "requirement": requirement,
                    "status": audit[status_key],
                    "evidence": join_evidence(audit[evidence_key]),
                }
            )
    return rows


def missing_rows(audits: list[dict[str, Any]]) -> list[dict[str, str]]:
    required = {
        "feature": [
            ("scanner IDs", "scanner_ids"),
            ("paired region IDs", "paired_region_ids"),
            ("frozen feature arrays", "frozen_feature_arrays"),
        ],
        "decoder": [
            ("biological branch embeddings", "biological_branch_embeddings"),
            ("acquisition branch embeddings", "acquisition_branch_embeddings"),
            ("decoder/composition weights", "decoder_weights"),
            ("paired swap metadata", "swap_metadata"),
        ],
        "pixel": [
            ("patch image paths", "patch_image_paths"),
            ("patch coordinates", "patch_coordinates"),
            ("registration metadata", "registration_metadata"),
            ("registration confidence/QC", "registration_confidence_or_qc"),
        ],
    }
    why = {
        "scanner IDs": "needed to define source and target acquisition",
        "paired region IDs": "needed to define observed counterfactual pairs",
        "frozen feature arrays": "needed for feature-space metrics",
        "biological branch embeddings": "needed for branch recombination",
        "acquisition branch embeddings": "needed for target acquisition branch",
        "decoder/composition weights": "needed for decoded feature composition",
        "paired swap metadata": "needed to define source/target swaps",
        "patch image paths": "needed for pixel reconstruction targets",
        "patch coordinates": "needed to align local regions",
        "registration metadata": "needed to justify pixel comparisons",
        "registration confidence/QC": "needed to reject misregistered patches",
    }
    rows = []
    for audit in audits:
        for layer, items in required.items():
            for requirement, key in items:
                value = audit[key]
                if value not in {"available", "yes"}:
                    rows.append(
                        {
                            "dataset": audit["dataset"],
                            "layer": layer,
                            "missing_requirement": requirement,
                            "status": value,
                            "why_it_matters": why[requirement],
                        }
                    )
        if audit["pixel_space_feasible_now"] != "ready":
            rows.append(
                {
                    "dataset": audit["dataset"],
                    "layer": "pixel",
                    "missing_requirement": "validated pixel reconstruction readiness",
                    "status": audit["pixel_space_feasible_now"],
                    "why_it_matters": (
                        "pixel metrics require explicit acceptance of paired image, "
                        "coordinate, registration, and QC evidence"
                    ),
                }
            )
    return rows


def write_report(path: Path, audits: list[dict[str, Any]]) -> None:
    feature_now = [a["dataset"] for a in audits if a["feature_space_feasible"] == "yes"]
    decoder_now = [a["dataset"] for a in audits if a["decoder_space_feasible"] == "yes"]
    pixel_future = [
        f"{a['dataset']} ({a['pixel_space_feasible_now']})"
        for a in audits
        if a["pixel_space_feasible_now"] != "ready"
    ]
    category_anchors = [a["dataset"] for a in audits if a["category_label_anchor"] == "yes"]
    retrieval_only = [a["dataset"] for a in audits if a["pair_retrieval_only_anchor"] == "yes"]

    lines = [
        "# Paired Scanner Counterfactual Benchmark v0 Feasibility Report",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Benchmark layers feasible now",
        "",
        f"- Feature-space: {', '.join(feature_now) if feature_now else 'none found'}.",
        f"- Decoder-space: {', '.join(decoder_now) if decoder_now else 'none found'}.",
        "- Pixel-space: no dataset is marked ready by this audit; candidates require explicit registration/QC review before reconstruction metrics.",
        "",
        "## Existing artifact support",
        "",
    ]

    for audit in audits:
        lines.extend(
            [
                f"### {audit['dataset']}",
                "",
                f"- Scanner IDs: {audit['scanner_ids']} ({join_evidence(audit['scanner_evidence'], 2)}).",
                f"- Paired region IDs: {audit['paired_region_ids']} ({join_evidence(audit['pair_evidence'], 2)}).",
                f"- Category labels: {audit['category_labels']} ({join_evidence(audit['category_evidence'], 2)}).",
                f"- Frozen feature arrays: {audit['frozen_feature_arrays']} ({join_evidence(audit['feature_evidence'], 2)}).",
                f"- Branch embeddings: biological={audit['biological_branch_embeddings']}, acquisition={audit['acquisition_branch_embeddings']}.",
                f"- Decoder/composition weights: {audit['decoder_weights']} ({join_evidence(audit['decoder_evidence'], 2)}).",
                f"- Patch images: {audit['patch_image_paths']} ({join_evidence(audit['patch_image_evidence'], 2)}).",
                f"- WSI paths: {audit['raw_wsi_paths']} ({join_evidence(audit['wsi_evidence'], 2)}).",
                f"- Coordinates: {audit['patch_coordinates']} ({join_evidence(audit['coordinate_evidence'], 2)}).",
                f"- Registration metadata: {audit['registration_metadata']} ({join_evidence(audit['registration_evidence'], 2)}).",
                f"- Registration confidence/QC: {audit['registration_confidence_or_qc']} ({join_evidence(audit['registration_qc_evidence'], 2)}).",
                f"- Pixel-space status: {audit['pixel_space_feasible_now']}.",
                "",
            ]
        )

    lines.extend(
        [
            "## Pixel-space requirements",
            "",
            "Pixel-space reconstruction requires paired image paths, scanner-specific paired acquisitions, local region correspondence, patch coordinates, and registration confidence/QC rules. The audit treats patch files or path metadata as insufficient by themselves; registration/QC evidence is required before pixel metrics are reported.",
            "",
            "## Category-label anchors",
            "",
            f"- {', '.join(category_anchors) if category_anchors else 'none found'}.",
            "",
            "## Pair/tissue-retrieval-only anchors",
            "",
            f"- {', '.join(retrieval_only) if retrieval_only else 'none found'}.",
            "",
            "## Explicitly unsupported claims",
            "",
            "This audit does not support clinical validation, diagnostic performance, deployment, patient-care readiness, FDA readiness, HIPAA readiness, scanner bias solved, universal disentanglement proven, pixel-level acquisition modeling proven, factorization proven, scanner-free representation, breakthrough claims, perfect causal factorization, or solves scanner bias claims.",
            "",
            "## Bottom line",
            "",
            "Feature-space and decoder-space benchmark v0 are feasible for datasets with available artifacts in the capability matrix. Pixel-space reconstruction remains future work until registration/QC readiness is explicitly validated.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for audit outputs.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["external_multiscanner_caninescc", "scorpion"]
    audits = [audit_dataset(dataset) for dataset in datasets]

    matrix_rows = []
    for audit in audits:
        matrix_rows.append(
            {
                "dataset": audit["dataset"],
                "scanner_ids": audit["scanner_ids"],
                "paired_region_ids": audit["paired_region_ids"],
                "category_labels": audit["category_labels"],
                "frozen_feature_arrays": audit["frozen_feature_arrays"],
                "biological_branch_embeddings": audit["biological_branch_embeddings"],
                "acquisition_branch_embeddings": audit["acquisition_branch_embeddings"],
                "decoder_weights": audit["decoder_weights"],
                "swap_metadata": audit["swap_metadata"],
                "patch_image_paths": audit["patch_image_paths"],
                "raw_wsi_paths": audit["raw_wsi_paths"],
                "patch_coordinates": audit["patch_coordinates"],
                "registration_metadata": audit["registration_metadata"],
                "registration_confidence_or_qc": audit["registration_confidence_or_qc"],
                "feature_space_feasible": audit["feature_space_feasible"],
                "decoder_space_feasible": audit["decoder_space_feasible"],
                "pixel_space_feasible_now": audit["pixel_space_feasible_now"],
                "category_label_anchor": audit["category_label_anchor"],
                "pair_retrieval_only_anchor": audit["pair_retrieval_only_anchor"],
                "evidence_summary": (
                    f"roots={join_evidence([rel(root) for root in audit['roots']], 6)}; "
                    f"files={len(audit['files'])}; suffix_counts={audit['suffix_counts']}"
                ),
            }
        )

    write_csv(
        output_dir / "dataset_capability_matrix.csv",
        matrix_rows,
        [
            "dataset",
            "scanner_ids",
            "paired_region_ids",
            "category_labels",
            "frozen_feature_arrays",
            "biological_branch_embeddings",
            "acquisition_branch_embeddings",
            "decoder_weights",
            "swap_metadata",
            "patch_image_paths",
            "raw_wsi_paths",
            "patch_coordinates",
            "registration_metadata",
            "registration_confidence_or_qc",
            "feature_space_feasible",
            "decoder_space_feasible",
            "pixel_space_feasible_now",
            "category_label_anchor",
            "pair_retrieval_only_anchor",
            "evidence_summary",
        ],
    )

    write_csv(
        output_dir / "feature_space_capability.csv",
        requirement_rows(audits, "feature"),
        ["dataset", "layer", "requirement", "status", "evidence"],
    )
    write_csv(
        output_dir / "decoder_space_capability.csv",
        requirement_rows(audits, "decoder"),
        ["dataset", "layer", "requirement", "status", "evidence"],
    )
    write_csv(
        output_dir / "pixel_space_capability.csv",
        requirement_rows(audits, "pixel"),
        ["dataset", "layer", "requirement", "status", "evidence"],
    )
    write_csv(
        output_dir / "missing_requirements.csv",
        missing_rows(audits),
        ["dataset", "layer", "missing_requirement", "status", "why_it_matters"],
    )
    write_report(output_dir / "benchmark_v0_feasibility_report.md", audits)

    run_log = [
        "paired_scanner_counterfactual feasibility audit",
        f"timestamp={dt.datetime.now().isoformat(timespec='seconds')}",
        f"repo_root={REPO_ROOT}",
        f"output_dir={output_dir}",
        f"datasets={','.join(datasets)}",
        "mode=read_only_input_scan",
        "training_invoked=false",
        "result_files_modified=false",
    ]
    for audit in audits:
        run_log.append(
            f"{audit['dataset']}: feature={audit['feature_space_feasible']} "
            f"decoder={audit['decoder_space_feasible']} "
            f"pixel={audit['pixel_space_feasible_now']}"
        )
    (output_dir / "run_log.txt").write_text("\n".join(run_log) + "\n", encoding="utf-8")

    print(f"Wrote audit outputs to {rel(output_dir)}")
    for audit in audits:
        print(
            f"{audit['dataset']}: feature={audit['feature_space_feasible']} "
            f"decoder={audit['decoder_space_feasible']} "
            f"pixel={audit['pixel_space_feasible_now']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
