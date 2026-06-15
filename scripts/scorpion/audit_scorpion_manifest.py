#!/usr/bin/env python3
"""Audit a SCORPION paired-scanner manifest before any model training.

The audit is intentionally strict about the scientific grouping structure:

- one row per image patch;
- 48 original slides;
- 10 aligned tissue regions per slide;
- five scanner views per tissue region;
- 480 tissue regions and 2,400 patches in the complete release;
- no original slide may appear in more than one split.

The input manifest must contain:

    slide_id, region_id, scanner_id, path

An optional ``split`` column is strongly recommended. Paths may be absolute or
relative to ``--data-root``.

Example
-------
python scripts/scorpion/audit_scorpion_manifest.py \
    --manifest data/scorpion/manifest.csv \
    --data-root data/scorpion \
    --out-dir results/scorpion/audit \
    --strict-release-counts
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


EXPECTED_SCANNERS = ("AT2", "GT450", "DP200", "P1000", "B300")
REQUIRED_COLUMNS = ("slide_id", "region_id", "scanner_id", "path")


class AuditError(ValueError):
    """Raised when a manifest violates the paired-scanner design."""


def load_manifest(path: Path) -> pd.DataFrame:
    """Load CSV or Parquet while preserving identifiers as strings."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, dtype=str)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        for column in frame.columns:
            frame[column] = frame[column].astype(str)
    else:
        raise AuditError("Manifest must be CSV or Parquet.")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise AuditError(f"Manifest is missing required columns: {missing}")

    if frame.empty:
        raise AuditError("Manifest contains no rows.")

    frame = frame.copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()
        if (frame[column] == "").any():
            raise AuditError(f"Column {column!r} contains empty values.")

    if "split" in frame.columns:
        frame["split"] = frame["split"].astype(str).str.strip()

    return frame


def normalize_scanner(value: str) -> str:
    """Normalize common scanner spellings to the paper's short identifiers."""
    compact = "".join(character for character in value.upper() if character.isalnum())
    aliases = {
        "AT2": "AT2",
        "APERIOAT2": "AT2",
        "LEICAAPERIOAT2": "AT2",
        "GT450": "GT450",
        "APERIOGT450": "GT450",
        "LEICAAPERIOGT450": "GT450",
        "DP200": "DP200",
        "VENTANADP200": "DP200",
        "ROCHEVENTANADP200": "DP200",
        "P1000": "P1000",
        "PANNORAMIC1000": "P1000",
        "3DHISTECHP1000": "P1000",
        "B300": "B300",
        "UFSB300": "B300",
        "PHILIPSUFSB300": "B300",
    }
    return aliases.get(compact, value.strip())


def resolve_paths(frame: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Add an absolute resolved_path column without mutating source paths."""
    resolved: list[str] = []
    for raw in frame["path"]:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = data_root / candidate
        resolved.append(str(candidate.resolve()))
    result = frame.copy()
    result["resolved_path"] = resolved
    return result


def validate_unique_rows(frame: pd.DataFrame) -> None:
    """Reject duplicate scanner views and paths."""
    pair_columns = ["slide_id", "region_id", "scanner_id"]
    duplicated_pairs = frame[frame.duplicated(pair_columns, keep=False)]
    if not duplicated_pairs.empty:
        examples = duplicated_pairs[pair_columns].head(10).to_dict("records")
        raise AuditError(f"Duplicate slide/region/scanner rows found: {examples}")

    duplicated_paths = frame[frame.duplicated(["resolved_path"], keep=False)]
    if not duplicated_paths.empty:
        examples = duplicated_paths["resolved_path"].head(10).tolist()
        raise AuditError(f"The same file path appears more than once: {examples}")


def validate_scanners(frame: pd.DataFrame) -> None:
    """Require the five scanners specified by the SCORPION paper."""
    observed = set(frame["scanner_id"])
    expected = set(EXPECTED_SCANNERS)
    if observed != expected:
        raise AuditError(
            "Scanner set mismatch. "
            f"Expected {sorted(expected)}, observed {sorted(observed)}."
        )


def validate_region_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Require exactly one view from every scanner for every tissue region."""
    region_summary = (
        frame.groupby(["slide_id", "region_id"], dropna=False)
        .agg(
            n_rows=("scanner_id", "size"),
            n_scanners=("scanner_id", "nunique"),
            scanners=("scanner_id", lambda values: ",".join(sorted(set(values)))),
        )
        .reset_index()
    )

    expected_scanner_string = ",".join(sorted(EXPECTED_SCANNERS))
    invalid = region_summary[
        (region_summary["n_rows"] != len(EXPECTED_SCANNERS))
        | (region_summary["n_scanners"] != len(EXPECTED_SCANNERS))
        | (region_summary["scanners"] != expected_scanner_string)
    ]
    if not invalid.empty:
        examples = invalid.head(10).to_dict("records")
        raise AuditError(f"Incomplete or malformed five-scanner regions: {examples}")

    return region_summary


def validate_slide_structure(
    frame: pd.DataFrame,
    region_summary: pd.DataFrame,
    strict_release_counts: bool,
) -> pd.DataFrame:
    """Validate region ownership, slide grouping, and optional release counts."""
    region_to_slides = frame.groupby("region_id")["slide_id"].nunique()
    reused = region_to_slides[region_to_slides != 1]
    if not reused.empty:
        raise AuditError(
            "A region_id maps to multiple slide_id values: "
            f"{reused.head(10).to_dict()}"
        )

    slide_summary = (
        region_summary.groupby("slide_id", dropna=False)
        .agg(n_regions=("region_id", "nunique"), n_patches=("n_rows", "sum"))
        .reset_index()
    )

    if strict_release_counts:
        expected = {
            "rows": 2400,
            "slides": 48,
            "regions": 480,
            "regions_per_slide": 10,
        }
        observed = {
            "rows": int(len(frame)),
            "slides": int(frame["slide_id"].nunique()),
            "regions": int(region_summary.shape[0]),
        }
        mismatches = {
            key: (expected[key], observed[key])
            for key in observed
            if expected[key] != observed[key]
        }
        bad_slides = slide_summary[slide_summary["n_regions"] != expected["regions_per_slide"]]
        if mismatches or not bad_slides.empty:
            raise AuditError(
                "Release-count validation failed. "
                f"Count mismatches={mismatches}; "
                f"bad slide examples={bad_slides.head(10).to_dict('records')}"
            )

    return slide_summary


def validate_split_leakage(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject original slides or tissue regions appearing in multiple splits."""
    if "split" not in frame.columns:
        return pd.DataFrame(columns=["split", "n_slides", "n_regions", "n_patches"])

    empty = frame["split"].isin({"", "nan", "None"})
    if empty.any():
        raise AuditError("The split column contains empty values.")

    slide_split_counts = frame.groupby("slide_id")["split"].nunique()
    leaking_slides = slide_split_counts[slide_split_counts > 1]
    if not leaking_slides.empty:
        raise AuditError(
            "Original-slide leakage across splits: "
            f"{leaking_slides.head(20).to_dict()}"
        )

    region_split_counts = frame.groupby(["slide_id", "region_id"])["split"].nunique()
    leaking_regions = region_split_counts[region_split_counts > 1]
    if not leaking_regions.empty:
        raise AuditError(
            "Tissue-region leakage across splits: "
            f"{leaking_regions.head(20).to_dict()}"
        )

    return (
        frame.groupby("split", dropna=False)
        .agg(
            n_slides=("slide_id", "nunique"),
            n_regions=("region_id", "nunique"),
            n_patches=("path", "size"),
        )
        .reset_index()
    )


def validate_files(frame: pd.DataFrame, check_images: bool) -> pd.DataFrame:
    """Check file existence and optionally read image dimensions with Pillow."""
    rows: list[dict[str, object]] = []
    image_module = None
    if check_images:
        try:
            from PIL import Image
        except ImportError as exc:
            raise AuditError(
                "--check-images requires Pillow: python -m pip install pillow"
            ) from exc
        image_module = Image

    for row in frame.itertuples(index=False):
        path = Path(row.resolved_path)
        record: dict[str, object] = {
            "slide_id": row.slide_id,
            "region_id": row.region_id,
            "scanner_id": row.scanner_id,
            "resolved_path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else -1,
        }
        if path.is_file() and image_module is not None:
            try:
                with image_module.open(path) as image:
                    record["width"] = int(image.width)
                    record["height"] = int(image.height)
                    record["mode"] = str(image.mode)
                    image.verify()
                record["image_ok"] = True
            except Exception as exc:  # noqa: BLE001 - audit must report corrupt files
                record["image_ok"] = False
                record["image_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(record)

    audit = pd.DataFrame(rows)
    missing = audit[~audit["exists"]]
    if not missing.empty:
        raise AuditError(
            f"Missing {len(missing)} files. Examples: "
            f"{missing['resolved_path'].head(10).tolist()}"
        )
    if check_images and "image_ok" in audit and not audit["image_ok"].all():
        bad = audit[~audit["image_ok"]]
        raise AuditError(
            f"Unreadable images found: {bad.head(10).to_dict('records')}"
        )
    return audit


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def add_checksums(file_audit: pd.DataFrame) -> pd.DataFrame:
    """Compute checksums after the cheaper structural checks pass."""
    result = file_audit.copy()
    result["sha256"] = [sha256_file(Path(path)) for path in result["resolved_path"]]
    duplicate_hashes = result[result.duplicated(["sha256"], keep=False)]
    if not duplicate_hashes.empty:
        examples = duplicate_hashes[
            ["slide_id", "region_id", "scanner_id", "resolved_path", "sha256"]
        ].head(20)
        raise AuditError(
            "Duplicate image bytes found across manifest rows: "
            f"{examples.to_dict('records')}"
        )
    return result


def scanner_pair_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Count complete aligned comparisons for all ten scanner pairs."""
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(["slide_id", "region_id"], sort=False)
    for scanner_a, scanner_b in itertools.combinations(EXPECTED_SCANNERS, 2):
        n_complete = 0
        for _, group in grouped:
            scanners = set(group["scanner_id"])
            n_complete += int(scanner_a in scanners and scanner_b in scanners)
        rows.append(
            {
                "scanner_a": scanner_a,
                "scanner_b": scanner_b,
                "n_aligned_regions": n_complete,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    frame: pd.DataFrame,
    region_summary: pd.DataFrame,
    slide_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    strict_release_counts: bool,
) -> None:
    """Write a compact, reviewable Markdown evidence report."""
    lines = [
        "# SCORPION manifest audit",
        "",
        "## Status",
        "",
        "**PASS** — paired-scanner structure and leakage checks completed.",
        "",
        "## Counts",
        "",
        f"- Patches: {len(frame):,}",
        f"- Original slides: {frame['slide_id'].nunique():,}",
        f"- Tissue regions: {len(region_summary):,}",
        f"- Scanners: {', '.join(EXPECTED_SCANNERS)}",
        f"- Strict release counts enforced: {strict_release_counts}",
        "",
        "## Split summary",
        "",
        split_summary.to_markdown(index=False) if not split_summary.empty else "No split column was present.",
        "",
        "## Slide summary",
        "",
        slide_summary.to_markdown(index=False),
        "",
        "## Scanner-pair coverage",
        "",
        pair_summary.to_markdown(index=False),
        "",
        "## Leakage boundary",
        "",
        "The original source slide is the split and resampling unit. All ten regions and all five scanner views from a slide must remain in the same partition.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    manifest: Path,
    data_root: Path,
    out_dir: Path,
    strict_release_counts: bool,
    check_images: bool,
    checksums: bool,
) -> None:
    """Run the complete audit and write artifacts only after validation passes."""
    frame = load_manifest(manifest)
    frame["scanner_id"] = frame["scanner_id"].map(normalize_scanner)
    frame = resolve_paths(frame, data_root)

    validate_unique_rows(frame)
    validate_scanners(frame)
    region_summary = validate_region_groups(frame)
    slide_summary = validate_slide_structure(frame, region_summary, strict_release_counts)
    split_summary = validate_split_leakage(frame)
    file_audit = validate_files(frame, check_images)
    if checksums:
        file_audit = add_checksums(file_audit)
    pair_summary = scanner_pair_summary(frame)

    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "validated_manifest.csv", index=False)
    region_summary.to_csv(out_dir / "region_summary.csv", index=False)
    slide_summary.to_csv(out_dir / "slide_summary.csv", index=False)
    split_summary.to_csv(out_dir / "split_summary.csv", index=False)
    pair_summary.to_csv(out_dir / "scanner_pair_coverage.csv", index=False)
    file_audit.to_csv(out_dir / "file_audit.csv", index=False)

    summary = {
        "status": "pass",
        "n_patches": int(len(frame)),
        "n_slides": int(frame["slide_id"].nunique()),
        "n_regions": int(len(region_summary)),
        "scanners": list(EXPECTED_SCANNERS),
        "strict_release_counts": bool(strict_release_counts),
        "checked_images": bool(check_images),
        "computed_checksums": bool(checksums),
    }
    (out_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        out_dir / "scorpion_manifest_audit.md",
        frame,
        region_summary,
        slide_summary,
        split_summary,
        pair_summary,
        strict_release_counts,
    )

    print("SCORPION MANIFEST AUDIT PASSED")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Artifacts: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("results/scorpion/audit"))
    parser.add_argument("--strict-release-counts", action="store_true")
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--checksums", action="store_true")
    args = parser.parse_args()

    try:
        run_audit(
            manifest=args.manifest,
            data_root=args.data_root,
            out_dir=args.out_dir,
            strict_release_counts=args.strict_release_counts,
            check_images=args.check_images,
            checksums=args.checksums,
        )
    except AuditError as exc:
        raise SystemExit(f"SCORPION MANIFEST AUDIT FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
