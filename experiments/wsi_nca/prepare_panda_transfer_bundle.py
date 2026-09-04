#!/usr/bin/env python3
"""Create and validate portable PANDA coordinate-feature bundles for WSI-NCA.

The tracked PANDA manifest contains Windows-local absolute feature paths.  This
utility copies a deterministic, optionally stratified cohort into a self-contained
bundle whose manifest uses relative paths.  Every selected HDF5 is fully read before
copying and checked again by the complementary validator.

Create a 300-slide transfer cohort on the machine that owns the feature files::

    python experiments/wsi_nca/prepare_panda_transfer_bundle.py create \
        --source-root D:\\panda\\features_phikon \
        --out-dir D:\\panda\\transfer\\panda_wsi_nca_300 \
        --limit 300 --stratified --seed 42

Validate the transferred directory on the destination machine::

    python experiments/wsi_nca/prepare_panda_transfer_bundle.py validate \
        --bundle-dir /data/panda_wsi_nca_300
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import h5py
import numpy as np
import pandas as pd

DEFAULT_MANIFEST = "results/panda_manifest/panda_phikon_manifest.csv"
DEFAULT_EXCLUDE_CSV = "results/panda_attention_mil_baseline/unreadable_features.csv"
REQUIRED_MANIFEST_COLUMNS = {"image_id", "isup_grade", "feature_path", "valid"}
CHECKSUM_FILE = "checksums.sha256"
BUNDLE_MANIFEST = "manifest.csv"
BUNDLE_SUMMARY = "bundle_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable PANDA WSI-NCA bundle utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a verified portable bundle")
    create.add_argument("--manifest", default=DEFAULT_MANIFEST)
    create.add_argument("--exclude-csv", default=DEFAULT_EXCLUDE_CSV)
    create.add_argument(
        "--source-root",
        default=None,
        help=(
            "Directory containing the HDF5 files. When set, replaces the directory part "
            "of Windows-local feature_path values while preserving each filename."
        ),
    )
    create.add_argument("--out-dir", required=True)
    create.add_argument("--limit", type=int, default=None)
    create.add_argument(
        "--stratified",
        action="store_true",
        help="Select --limit slides proportionally within ISUP grades",
    )
    create.add_argument("--seed", type=int, default=42)

    validate = subparsers.add_parser("validate", help="Validate an existing bundle")
    validate.add_argument("--bundle-dir", required=True)
    validate.add_argument(
        "--source-manifest",
        default=None,
        help="Optional original manifest whose hash must match bundle_summary.json",
    )
    validate.add_argument(
        "--report",
        default=None,
        help="Optional path for a machine-readable validation report",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_valid_column(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _clean_optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any, field: str, image_id: str) -> int | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    numeric = float(value)
    integer = int(numeric)
    if numeric != integer:
        raise ValueError(f"Non-integral {field} for {image_id}: {value}")
    return integer


def _optional_shape(value: Any, field: str, image_id: str) -> tuple[int, ...] | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid {field} for {image_id}: {value}") from exc
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Invalid {field} for {image_id}: {value}")
    return tuple(int(item) for item in parsed)


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    frame = pd.read_csv(path, dtype={"image_id": str})
    missing = REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    frame["image_id"] = frame["image_id"].astype(str)
    if frame["image_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["image_id"].duplicated(), "image_id"].unique())
        raise ValueError(f"Manifest contains duplicate image_id values: {duplicates[:10]}")
    frame["isup_grade"] = pd.to_numeric(frame["isup_grade"], errors="raise").astype(int)
    invalid_grades = sorted(set(frame.loc[~frame["isup_grade"].between(0, 5), "isup_grade"]))
    if invalid_grades:
        raise ValueError(f"ISUP grades must be in [0, 5], found: {invalid_grades}")
    return frame


def read_excluded_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Unreadable-feature exclusion CSV not found: {path}")
    frame = pd.read_csv(path, dtype={"image_id": str})
    if "image_id" not in frame.columns:
        raise ValueError(f"Exclude CSV missing image_id column: {path}")
    ids = frame["image_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"Exclude CSV contains duplicate image_id values: {path}")
    return set(ids)


def partition_manifest(
    frame: pd.DataFrame, excluded_ids: set[str]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    unknown_exclusions = sorted(excluded_ids - set(frame["image_id"]))
    if unknown_exclusions:
        raise ValueError(
            "Exclude CSV contains IDs absent from the source manifest: "
            f"{unknown_exclusions[:10]}"
        )

    valid = parse_valid_column(frame["valid"])
    eligible_indices: list[int] = []
    excluded: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        image_id = str(row["image_id"])
        feature_path = _clean_optional_text(row["feature_path"])
        if feature_path is None:
            reason = "missing_feature_path"
        elif not bool(valid.loc[index]):
            reason = "manifest_invalid"
        elif image_id in excluded_ids:
            reason = "known_unreadable"
        else:
            eligible_indices.append(index)
            continue

        item: dict[str, Any] = {"image_id": image_id, "reason": reason}
        error_message = _clean_optional_text(row.get("error_message"))
        if error_message is not None:
            item["error_message"] = error_message
        excluded.append(item)

    return frame.loc[eligible_indices].copy(), excluded


def _stratified_allocations(frame: pd.DataFrame, limit: int) -> dict[int, int]:
    counts = frame.groupby("isup_grade", sort=True).size().astype(int)
    raw = counts.astype(float) * float(limit) / float(len(frame))
    allocations = np.floor(raw).astype(int)
    remaining = limit - int(allocations.sum())
    order = sorted(counts.index, key=lambda grade: (-(raw[grade] - allocations[grade]), grade))
    while remaining:
        progressed = False
        for grade in order:
            if allocations[grade] < counts[grade]:
                allocations[grade] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Unable to allocate requested stratified sample")
    return {int(grade): int(count) for grade, count in allocations.items()}


def select_rows(
    frame: pd.DataFrame,
    limit: int | None,
    stratified: bool,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    if limit is None or limit >= len(frame):
        selected = frame.copy()
    else:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        if stratified:
            allocations = _stratified_allocations(frame, limit)
            rng = np.random.default_rng(seed)
            selected_indices: list[int] = []
            for grade in sorted(allocations):
                indices = np.asarray(
                    sorted(frame.index[frame["isup_grade"] == grade].tolist()), dtype=np.int64
                )
                selected_indices.extend(rng.permutation(indices)[: allocations[grade]].tolist())
            selected = frame.loc[selected_indices].copy()
        else:
            selected = frame.head(limit).copy()

    selected = selected.sort_values("image_id", kind="stable").reset_index(drop=True)
    selected_ids = set(selected["image_id"])
    not_selected = [
        {"image_id": str(image_id), "reason": "not_selected_by_limit"}
        for image_id in frame.loc[~frame["image_id"].isin(selected_ids), "image_id"]
    ]
    return selected, not_selected


def resolve_source_path(row: pd.Series, source_root: Path | None) -> Path:
    image_id = str(row["image_id"])
    raw = _clean_optional_text(row["feature_path"])
    if raw is None:
        raise ValueError(f"Selected row has no feature_path: {image_id}")

    filename = PureWindowsPath(raw.replace("/", "\\")).name
    if Path(filename).stem != image_id:
        raise ValueError(f"feature_path filename does not match image_id={image_id}: {filename}")
    if source_root is not None:
        return source_root / filename

    path = Path(raw)
    if PureWindowsPath(raw).is_absolute() and not path.is_absolute():
        return path
    return path


def inspect_coordinate_bag(path: Path, row: pd.Series) -> dict[str, Any]:
    image_id = str(row["image_id"])
    if not path.exists():
        raise FileNotFoundError(
            f"Selected HDF5 is missing for image_id={image_id}: {path}. "
            "Pass --source-root to remap the tracked Windows-local paths."
        )
    if path.is_symlink():
        raise ValueError(f"Selected HDF5 must not be a symbolic link: {path}")

    try:
        with h5py.File(path, "r") as handle:
            if "features" not in handle:
                raise KeyError("missing HDF5 dataset 'features'")
            if "coordinates" not in handle:
                raise KeyError("missing HDF5 dataset 'coordinates'")
            features = handle["features"][:]
            coordinates = handle["coordinates"][:]
            slide_id = handle.attrs.get("slide_id")
    except Exception as exc:
        raise OSError(f"Unreadable HDF5 image_id={image_id}, path={path}: {exc}") from exc

    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError(f"Invalid features for {image_id}: shape={features.shape}")
    if coordinates.shape != (features.shape[0], 2):
        raise ValueError(
            f"Coordinate/feature mismatch for {image_id}: "
            f"features={features.shape}, coordinates={coordinates.shape}"
        )
    if not np.issubdtype(features.dtype, np.number) or not np.isfinite(features).all():
        raise ValueError(f"Features must be finite numeric values for {image_id}")
    if not np.issubdtype(coordinates.dtype, np.number) or not np.isfinite(coordinates).all():
        raise ValueError(f"Coordinates must be finite numeric values for {image_id}")
    if slide_id is not None:
        if isinstance(slide_id, bytes):
            slide_id = slide_id.decode("utf-8")
        if str(slide_id) != image_id:
            raise ValueError(
                f"HDF5 slide_id attribute mismatch: manifest={image_id}, HDF5={slide_id}"
            )

    expected_num_patches = _optional_int(row.get("num_patches"), "num_patches", image_id)
    expected_feature_dim = _optional_int(row.get("feature_dim"), "feature_dim", image_id)
    expected_feature_shape = _optional_shape(row.get("feature_shape"), "feature_shape", image_id)
    expected_coordinate_shape = _optional_shape(
        row.get("coordinate_shape"), "coordinate_shape", image_id
    )
    comparisons = {
        "num_patches": (expected_num_patches, int(features.shape[0])),
        "feature_dim": (expected_feature_dim, int(features.shape[1])),
        "feature_shape": (expected_feature_shape, tuple(features.shape)),
        "coordinate_shape": (expected_coordinate_shape, tuple(coordinates.shape)),
    }
    for field, (expected, observed) in comparisons.items():
        if expected is not None and expected != observed:
            raise ValueError(
                f"Manifest/HDF5 {field} mismatch for {image_id}: "
                f"manifest={expected}, HDF5={observed}"
            )

    return {
        "num_patches": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "feature_shape": tuple(int(item) for item in features.shape),
        "coordinate_shape": tuple(int(item) for item in coordinates.shape),
        "feature_dtype": str(features.dtype),
        "coordinate_dtype": str(coordinates.dtype),
    }


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(int(grade)): int(count)
        for grade, count in frame["isup_grade"].value_counts().sort_index().items()
    }


def write_checksums(bundle_dir: Path, relative_paths: list[str]) -> None:
    lines = []
    for relative_path in sorted(relative_paths):
        digest = sha256_file(bundle_dir / Path(PurePosixPath(relative_path)))
        lines.append(f"{digest}  {relative_path}")
    (bundle_dir / CHECKSUM_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_bundle_path(bundle_dir: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe bundle-relative path: {relative_path}")
    candidate = bundle_dir / Path(pure)
    if candidate.is_symlink():
        raise ValueError(f"Bundle path must not be a symbolic link: {relative_path}")
    path = candidate.resolve()
    root = bundle_dir.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path escapes bundle directory: {relative_path}")
    return path


def read_checksums(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Checksum file not found: {path}")
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, relative_path = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed checksum line {line_number}: {line}") from exc
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Invalid SHA256 on checksum line {line_number}: {digest}")
        if relative_path in checksums:
            raise ValueError(f"Duplicate checksum path: {relative_path}")
        checksums[relative_path] = digest
    return checksums


def validate_bundle(
    bundle_dir: Path,
    source_manifest: Path | None = None,
) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / BUNDLE_MANIFEST
    summary_path = bundle_dir / BUNDLE_SUMMARY
    if not summary_path.exists():
        raise FileNotFoundError(f"Bundle summary not found: {summary_path}")

    frame = read_manifest(manifest_path)
    if not parse_valid_column(frame["valid"]).all():
        raise ValueError("Portable manifest contains rows not marked valid")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checksums = read_checksums(bundle_dir / CHECKSUM_FILE)

    feature_paths = [str(value) for value in frame["feature_path"]]
    if len(feature_paths) != len(set(feature_paths)):
        raise ValueError("Portable manifest contains duplicate feature_path values")
    expected_paths = {BUNDLE_MANIFEST, BUNDLE_SUMMARY, *feature_paths}
    if set(checksums) != expected_paths:
        missing = sorted(expected_paths - set(checksums))
        extra = sorted(set(checksums) - expected_paths)
        raise ValueError(f"Checksum inventory mismatch: missing={missing}, extra={extra}")

    actual_hdf5 = {
        path.relative_to(bundle_dir).as_posix() for path in (bundle_dir / "features").glob("*.h5")
    }
    if actual_hdf5 != set(feature_paths):
        missing = sorted(set(feature_paths) - actual_hdf5)
        extra = sorted(actual_hdf5 - set(feature_paths))
        raise ValueError(f"HDF5 inventory mismatch: missing={missing}, extra={extra}")

    for relative_path, expected_digest in checksums.items():
        path = _safe_bundle_path(bundle_dir, relative_path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Checksummed bundle path is not a regular file: {relative_path}")
        observed_digest = sha256_file(path)
        if observed_digest != expected_digest:
            raise ValueError(
                f"SHA256 mismatch for {relative_path}: "
                f"expected={expected_digest}, observed={observed_digest}"
            )

    feature_dims: set[int] = set()
    total_patches = 0
    for _, row in frame.iterrows():
        relative_path = str(row["feature_path"])
        if PurePosixPath(relative_path).parent != PurePosixPath("features"):
            raise ValueError(f"Feature path must be directly under features/: {relative_path}")
        path = _safe_bundle_path(bundle_dir, relative_path)
        metadata = inspect_coordinate_bag(path, row)
        feature_dims.add(int(metadata["feature_dim"]))
        total_patches += int(metadata["num_patches"])

    if len(feature_dims) != 1:
        raise ValueError(f"Bundle feature dimensions are inconsistent: {sorted(feature_dims)}")
    included_ids = frame["image_id"].astype(str).tolist()
    if summary.get("included_slide_ids") != included_ids:
        raise ValueError("bundle_summary.json included_slide_ids do not match manifest order")
    if int(summary.get("selected_rows", -1)) != len(frame):
        raise ValueError("bundle_summary.json selected_rows does not match manifest")
    excluded_slides = summary.get("excluded_slides")
    if not isinstance(excluded_slides, list) or any(
        not isinstance(item, dict) or "image_id" not in item or "reason" not in item
        for item in excluded_slides
    ):
        raise ValueError("bundle_summary.json excluded_slides is malformed")
    excluded_ids = [str(item["image_id"]) for item in excluded_slides]
    if len(excluded_ids) != len(set(excluded_ids)):
        raise ValueError("bundle_summary.json contains duplicate excluded slide IDs")
    if set(included_ids) & set(excluded_ids):
        raise ValueError("bundle_summary.json includes a slide in both included and excluded sets")
    if int(summary.get("excluded_rows", -1)) != len(excluded_ids):
        raise ValueError("bundle_summary.json excluded_rows does not match excluded_slides")
    if int(summary.get("source_rows", -1)) != len(included_ids) + len(excluded_ids):
        raise ValueError("bundle_summary.json source row accounting is inconsistent")
    if summary.get("bundle_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("bundle_summary.json manifest hash does not match manifest.csv")
    if int(summary.get("feature_dim", -1)) != next(iter(feature_dims)):
        raise ValueError("bundle_summary.json feature_dim does not match HDF5 data")
    if int(summary.get("total_patches", -1)) != total_patches:
        raise ValueError("bundle_summary.json total_patches does not match HDF5 data")
    if source_manifest is not None:
        observed_source_hash = sha256_file(source_manifest)
        if summary.get("source_manifest_sha256") != observed_source_hash:
            raise ValueError(
                "Source manifest hash mismatch: "
                f"expected={summary.get('source_manifest_sha256')}, "
                f"observed={observed_source_hash}"
            )
        source_frame = read_manifest(source_manifest)
        if len(source_frame) != int(summary["source_rows"]):
            raise ValueError("Source manifest row count does not match bundle_summary.json")
        source_ids = set(source_frame["image_id"].astype(str))
        if source_ids != set(included_ids) | set(excluded_ids):
            raise ValueError("Source manifest IDs do not match included plus excluded bundle IDs")
        source_labels = source_frame.set_index("image_id")["isup_grade"].astype(int)
        observed_labels = frame.set_index("image_id")["isup_grade"].astype(int)
        expected_labels = source_labels.loc[observed_labels.index]
        if not observed_labels.equals(expected_labels):
            raise ValueError("Portable manifest labels do not match the source manifest")

    return {
        "status": "valid",
        "bundle_dir": str(bundle_dir),
        "slides": len(frame),
        "feature_dim": next(iter(feature_dims)),
        "total_patches": total_patches,
        "label_counts": _label_counts(frame),
        "files_verified": len(checksums),
        "manifest_sha256": sha256_file(manifest_path),
    }


def create_bundle(
    manifest_path: Path,
    exclude_path: Path,
    output_dir: Path,
    source_root: Path | None,
    limit: int | None,
    stratified: bool,
    seed: int,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    exclude_path = exclude_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if source_root is not None:
        source_root = source_root.resolve()
        if not source_root.is_dir():
            raise NotADirectoryError(f"Source HDF5 root not found: {source_root}")

    frame = read_manifest(manifest_path)
    excluded_ids = read_excluded_ids(exclude_path)
    eligible, excluded = partition_manifest(frame, excluded_ids)
    selected, not_selected = select_rows(eligible, limit, stratified, seed)
    excluded.extend(not_selected)
    excluded = sorted(excluded, key=lambda item: item["image_id"])

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent)))
    try:
        features_dir = temp_dir / "features"
        features_dir.mkdir()
        rows: list[dict[str, Any]] = []
        included_ids: list[str] = []
        feature_dims: set[int] = set()
        total_patches = 0
        feature_relative_paths: list[str] = []

        for position, (_, row) in enumerate(selected.iterrows(), start=1):
            image_id = str(row["image_id"])
            source_path = resolve_source_path(row, source_root)
            metadata = inspect_coordinate_bag(source_path, row)
            feature_dims.add(int(metadata["feature_dim"]))
            total_patches += int(metadata["num_patches"])

            relative_path = f"features/{image_id}.h5"
            destination = temp_dir / Path(PurePosixPath(relative_path))
            shutil.copyfile(source_path, destination)
            source_digest = sha256_file(source_path)
            destination_digest = sha256_file(destination)
            if source_digest != destination_digest:
                raise OSError(
                    f"Byte-copy verification failed for {image_id}: "
                    f"source={source_digest}, destination={destination_digest}"
                )

            output_row = row.to_dict()
            output_row.update(
                {
                    "feature_path": relative_path,
                    "feature_shape": str(metadata["feature_shape"]),
                    "coordinate_shape": str(metadata["coordinate_shape"]),
                    "num_patches": metadata["num_patches"],
                    "feature_dim": metadata["feature_dim"],
                    "file_size_bytes": destination.stat().st_size,
                    "valid": True,
                    "error_message": "",
                }
            )
            rows.append(output_row)
            included_ids.append(image_id)
            feature_relative_paths.append(relative_path)
            if position % 25 == 0 or position == len(selected):
                print(f"verified and copied {position}/{len(selected)} HDF5 files", flush=True)

        if not rows:
            raise ValueError("Selection produced an empty bundle")
        if len(feature_dims) != 1:
            raise ValueError(
                f"Selected HDF5 feature dimensions are inconsistent: {sorted(feature_dims)}"
            )

        output_frame = pd.DataFrame(rows, columns=frame.columns)
        output_frame.to_csv(temp_dir / BUNDLE_MANIFEST, index=False)
        summary = {
            "schema_version": 1,
            "status": "portable PANDA WSI-NCA coordinate-feature bundle",
            "bundle_name": output_dir.name,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_rows": len(frame),
            "manifest_eligible_rows": len(eligible),
            "selected_rows": len(output_frame),
            "excluded_rows": len(excluded),
            "source_root_remap": str(source_root) if source_root is not None else None,
            "exclude_csv": str(exclude_path),
            "exclude_csv_sha256": sha256_file(exclude_path),
            "selection": {
                "limit": limit,
                "stratified_by": "isup_grade" if stratified else None,
                "seed": int(seed),
            },
            "eligible_label_counts": _label_counts(eligible),
            "included_label_counts": _label_counts(output_frame),
            "feature_dim": next(iter(feature_dims)),
            "total_patches": total_patches,
            "hdf5_contract": {
                "features": "rank-2 numeric array with at least two rows",
                "coordinates": "rank-2 array with shape (num_patches, 2)",
                "copy_semantics": "byte-for-byte SHA256-verified copy",
            },
            "included_slide_ids": included_ids,
            "excluded_slides": excluded,
        }
        summary["bundle_manifest_sha256"] = sha256_file(temp_dir / BUNDLE_MANIFEST)
        (temp_dir / BUNDLE_SUMMARY).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        write_checksums(
            temp_dir,
            [BUNDLE_MANIFEST, BUNDLE_SUMMARY, *feature_relative_paths],
        )

        validation = validate_bundle(temp_dir, source_manifest=manifest_path)
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir)
        raise

    validation["bundle_dir"] = str(output_dir)
    validation["source_rows"] = len(frame)
    validation["eligible_rows"] = len(eligible)
    validation["excluded_rows"] = len(excluded)
    return validation


def main() -> None:
    args = parse_args()
    if args.command == "create":
        result = create_bundle(
            manifest_path=Path(args.manifest),
            exclude_path=Path(args.exclude_csv),
            output_dir=Path(args.out_dir),
            source_root=Path(args.source_root) if args.source_root else None,
            limit=args.limit,
            stratified=bool(args.stratified),
            seed=args.seed,
        )
    else:
        result = validate_bundle(
            Path(args.bundle_dir),
            source_manifest=Path(args.source_manifest) if args.source_manifest else None,
        )
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
