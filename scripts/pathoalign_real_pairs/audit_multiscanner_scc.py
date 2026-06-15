#!/usr/bin/env python3
"""Audit the Multi-Scanner SCC dataset for PathoAlign real-pairs experiments.

The public dataset contains the same 44 canine cutaneous squamous-cell
carcinoma specimens scanned on five devices. This script performs no model
training. It verifies the pairing structure, freezes specimen-level splits,
and writes manifests that downstream PathoAlign experiments must consume.

Expected scanners:
    CS2, NZ210, NZ20, P1000, GT450

Example:
    python scripts/pathoalign_real_pairs/audit_multiscanner_scc.py \
        --data-root D:/datasets/multiscanner_scc \
        --output-dir results/pathoalign_real_pairs/audit
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


IMAGE_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".svs",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".bif",
}
ANNOTATION_EXTENSIONS = {".xml", ".json", ".geojson"}

# Canonical names follow the scanner abbreviations used in the dataset paper.
SCANNER_ALIASES: dict[str, tuple[str, ...]] = {
    "CS2": (
        "cs2",
        "scanscopecs2",
        "scan_scope_cs2",
        "aperiocs2",
        "aperio_cs2",
    ),
    "NZ210": (
        "nz210",
        "s210",
        "nanozoomers210",
        "nanozoomer_s210",
    ),
    "NZ20": (
        "nz20",
        "nz2.0",
        "nz2_0",
        "2.0ht",
        "2_0ht",
        "nanozoomer20",
        "nanozoomer_20",
    ),
    "P1000": (
        "p1000",
        "pannoramic1000",
        "pannoramic_1000",
    ),
    "GT450": (
        "gt450",
        "aperiogt450",
        "aperio_gt450",
    ),
}

EXPECTED_SCANNERS = tuple(SCANNER_ALIASES)
EXPECTED_SPECIMENS = 44
EXPECTED_IMAGES = EXPECTED_SPECIMENS * len(EXPECTED_SCANNERS)
DEFAULT_SEED = 20260615
DEFAULT_SPLIT_COUNTS = {"train": 30, "validation": 5, "test": 9}


@dataclass(frozen=True)
class AuditFile:
    path: Path
    relative_path: str
    kind: str
    scanner: str | None
    specimen_id: str | None
    size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pathoalign_real_pairs/audit"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write diagnostic manifests even when the complete 44 x 5 dataset is absent.",
    )
    return parser.parse_args()


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def detect_scanner(path: Path) -> str | None:
    candidates = [path.name, *path.parts]
    normalized_candidates = [normalize_token(item) for item in candidates]

    matches: list[str] = []
    for canonical, aliases in SCANNER_ALIASES.items():
        normalized_aliases = {normalize_token(alias) for alias in aliases}
        if any(
            alias and alias in candidate
            for candidate in normalized_candidates
            for alias in normalized_aliases
        ):
            matches.append(canonical)

    unique = sorted(set(matches))
    if len(unique) > 1:
        raise ValueError(
            f"Ambiguous scanner identity for {path}: matched {unique}. "
            "Rename the file/folder or extend SCANNER_ALIASES explicitly."
        )
    return unique[0] if unique else None


def remove_scanner_tokens(value: str) -> str:
    cleaned = value
    aliases = sorted(
        {alias for values in SCANNER_ALIASES.values() for alias in values},
        key=len,
        reverse=True,
    )
    for alias in aliases:
        cleaned = re.sub(re.escape(alias), "_", cleaned, flags=re.IGNORECASE)
    return cleaned


def infer_specimen_id(path: Path, data_root: Path, scanner: str | None) -> str | None:
    """Infer specimen identity while supporting scanner-folder layouts.

    Common valid layouts include:
        root/CS2/specimen_001.tif
        root/specimen_001/CS2.tif
        root/specimen_001_GT450.tif
    """
    relative = path.relative_to(data_root)
    stem = remove_scanner_tokens(path.stem)
    stem = re.sub(
        r"(?i)(annotation|annotations|polygon|polygons|mask|labels?|registered|wsi)",
        "_",
        stem,
    )
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")

    generic_stems = {"", "image", "slide", "scan", "data"}
    if normalize_token(stem) not in {normalize_token(x) for x in generic_stems}:
        return stem.lower()

    # If the filename is only the scanner name, the specimen folder is usually
    # the nearest parent that is not itself a scanner folder.
    for parent_name in reversed(relative.parent.parts):
        if detect_scanner(Path(parent_name)) is None:
            candidate = re.sub(r"[^A-Za-z0-9]+", "_", parent_name).strip("_")
            if candidate:
                return candidate.lower()
    return None


def classify_file(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in ANNOTATION_EXTENSIONS:
        return "annotation"
    return None


def discover_files(data_root: Path) -> list[AuditFile]:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)

    records: list[AuditFile] = []
    for path in sorted(item for item in data_root.rglob("*") if item.is_file()):
        kind = classify_file(path)
        if kind is None:
            continue
        scanner = detect_scanner(path)
        specimen_id = infer_specimen_id(path, data_root, scanner)
        records.append(
            AuditFile(
                path=path,
                relative_path=path.relative_to(data_root).as_posix(),
                kind=kind,
                scanner=scanner,
                specimen_id=specimen_id,
                size_bytes=path.stat().st_size,
            )
        )
    return records


def stable_hash(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def freeze_splits(
    specimen_ids: Iterable[str],
    seed: int,
    allow_partial: bool,
) -> pd.DataFrame:
    specimens = sorted(set(specimen_ids), key=lambda item: stable_hash(item, seed))
    if not specimens:
        return pd.DataFrame(columns=["specimen_id", "split", "split_seed"])

    if len(specimens) == EXPECTED_SPECIMENS:
        counts = DEFAULT_SPLIT_COUNTS
    elif allow_partial:
        n = len(specimens)
        train = max(1, round(0.68 * n))
        validation = max(1, round(0.11 * n)) if n >= 3 else 0
        if train + validation >= n:
            validation = max(0, n - train - 1)
        counts = {
            "train": train,
            "validation": validation,
            "test": n - train - validation,
        }
    else:
        raise RuntimeError(
            f"Expected {EXPECTED_SPECIMENS} complete specimens, found {len(specimens)}. "
            "Use --allow-partial only for inspection, never for the confirmatory run."
        )

    rows: list[dict[str, object]] = []
    cursor = 0
    for split in ("train", "validation", "test"):
        for specimen_id in specimens[cursor : cursor + counts[split]]:
            rows.append(
                {
                    "specimen_id": specimen_id,
                    "split": split,
                    "split_seed": seed,
                }
            )
        cursor += counts[split]
    return pd.DataFrame(rows)


def build_image_manifest(records: list[AuditFile]) -> pd.DataFrame:
    rows = [
        {
            "specimen_id": record.specimen_id,
            "scanner": record.scanner,
            "relative_path": record.relative_path,
            "size_bytes": record.size_bytes,
            "extension": record.path.suffix.lower(),
        }
        for record in records
        if record.kind == "image"
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "specimen_id",
            "scanner",
            "relative_path",
            "size_bytes",
            "extension",
        ],
    )


def build_annotation_manifest(records: list[AuditFile]) -> pd.DataFrame:
    rows = [
        {
            "specimen_id": record.specimen_id,
            "scanner": record.scanner,
            "relative_path": record.relative_path,
            "size_bytes": record.size_bytes,
            "extension": record.path.suffix.lower(),
        }
        for record in records
        if record.kind == "annotation"
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "specimen_id",
            "scanner",
            "relative_path",
            "size_bytes",
            "extension",
        ],
    )


def validate_images(images: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    if images.empty:
        return pd.DataFrame(), ["No supported whole-slide image files were found."]

    unresolved = images[images["scanner"].isna() | images["specimen_id"].isna()]
    if not unresolved.empty:
        errors.append(
            f"{len(unresolved)} image files have unresolved scanner or specimen identity."
        )

    resolved = images.dropna(subset=["scanner", "specimen_id"]).copy()
    duplicates = (
        resolved.groupby(["specimen_id", "scanner"], as_index=False)
        .size()
        .query("size != 1")
    )
    if not duplicates.empty:
        errors.append(
            f"{len(duplicates)} specimen/scanner cells contain duplicate image files."
        )

    presence = (
        resolved.assign(present=1)
        .pivot_table(
            index="specimen_id",
            columns="scanner",
            values="present",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=EXPECTED_SCANNERS, fill_value=0)
        .reset_index()
    )
    presence["scanner_count"] = presence[list(EXPECTED_SCANNERS)].sum(axis=1)
    presence["complete_pair_set"] = presence["scanner_count"] == len(EXPECTED_SCANNERS)
    presence["missing_scanners"] = presence.apply(
        lambda row: ",".join(
            scanner for scanner in EXPECTED_SCANNERS if int(row[scanner]) == 0
        ),
        axis=1,
    )

    incomplete = presence[~presence["complete_pair_set"]]
    if not incomplete.empty:
        errors.append(
            f"{len(incomplete)} specimens do not have exactly one scan from all five scanners."
        )

    unexpected = sorted(set(resolved["scanner"]) - set(EXPECTED_SCANNERS))
    if unexpected:
        errors.append(f"Unexpected scanner identities: {unexpected}")

    return presence, errors


def build_pair_manifest(
    images: pd.DataFrame,
    presence: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    if images.empty or presence.empty:
        return pd.DataFrame()

    complete_ids = set(
        presence.loc[presence["complete_pair_set"], "specimen_id"].astype(str)
    )
    resolved = images[
        images["specimen_id"].astype(str).isin(complete_ids)
        & images["scanner"].notna()
    ].copy()
    path_lookup = {
        (str(row.specimen_id), str(row.scanner)): str(row.relative_path)
        for row in resolved.itertuples(index=False)
    }
    split_lookup = dict(zip(splits["specimen_id"], splits["split"]))

    rows: list[dict[str, object]] = []
    for specimen_id in sorted(complete_ids):
        for scanner_a, scanner_b in itertools.combinations(EXPECTED_SCANNERS, 2):
            rows.append(
                {
                    "specimen_id": specimen_id,
                    "split": split_lookup[specimen_id],
                    "scanner_a": scanner_a,
                    "scanner_b": scanner_b,
                    "image_a": path_lookup[(specimen_id, scanner_a)],
                    "image_b": path_lookup[(specimen_id, scanner_b)],
                    "pair_id": f"{specimen_id}__{scanner_a}__{scanner_b}",
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    images: pd.DataFrame,
    annotations: pd.DataFrame,
    presence: pd.DataFrame,
    splits: pd.DataFrame,
    pairs: pd.DataFrame,
    errors: list[str],
    allow_partial: bool,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    images.merge(splits, on="specimen_id", how="left").to_csv(
        output_dir / "dataset_manifest.csv", index=False
    )
    annotations.to_csv(output_dir / "annotation_manifest.csv", index=False)
    presence.to_csv(output_dir / "specimen_scanner_matrix.csv", index=False)
    splits.to_csv(output_dir / "split_manifest.csv", index=False)
    pairs.to_csv(output_dir / "pair_manifest.csv", index=False)

    scanner_counts = (
        images.groupby("scanner", dropna=False, as_index=False)
        .agg(image_count=("relative_path", "count"), total_bytes=("size_bytes", "sum"))
        .sort_values("scanner", na_position="last")
    )
    scanner_counts.to_csv(output_dir / "scanner_counts.csv", index=False)

    complete_specimens = (
        int(presence["complete_pair_set"].sum()) if not presence.empty else 0
    )
    summary = {
        "status": "pass" if not errors else "fail",
        "allow_partial": allow_partial,
        "expected_specimens": EXPECTED_SPECIMENS,
        "expected_scanners": list(EXPECTED_SCANNERS),
        "expected_images": EXPECTED_IMAGES,
        "discovered_images": int(len(images)),
        "discovered_annotations": int(len(annotations)),
        "resolved_specimens": int(len(presence)),
        "complete_specimens": complete_specimens,
        "scanner_pair_rows": int(len(pairs)),
        "expected_pair_rows": EXPECTED_SPECIMENS * 10,
        "split_seed": seed,
        "split_counts": splits["split"].value_counts().sort_index().to_dict(),
        "errors": errors,
        "confirmatory_ready": (
            not errors
            and len(images) == EXPECTED_IMAGES
            and complete_specimens == EXPECTED_SPECIMENS
            and len(pairs) == EXPECTED_SPECIMENS * 10
        ),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    records = discover_files(args.data_root)
    images = build_image_manifest(records)
    annotations = build_annotation_manifest(records)
    presence, errors = validate_images(images)

    complete_ids = (
        presence.loc[presence["complete_pair_set"], "specimen_id"].astype(str).tolist()
        if not presence.empty
        else []
    )
    splits = freeze_splits(complete_ids, args.seed, args.allow_partial)
    pairs = build_pair_manifest(images, presence, splits)

    if not args.allow_partial:
        if len(images) != EXPECTED_IMAGES:
            errors.append(
                f"Expected {EXPECTED_IMAGES} images, found {len(images)}."
            )
        if len(complete_ids) != EXPECTED_SPECIMENS:
            errors.append(
                f"Expected {EXPECTED_SPECIMENS} complete specimens, found {len(complete_ids)}."
            )
        if len(pairs) != EXPECTED_SPECIMENS * 10:
            errors.append(
                f"Expected {EXPECTED_SPECIMENS * 10} scanner pairs, found {len(pairs)}."
            )

    errors = sorted(set(errors))
    write_outputs(
        output_dir=args.output_dir,
        images=images,
        annotations=annotations,
        presence=presence,
        splits=splits,
        pairs=pairs,
        errors=errors,
        allow_partial=args.allow_partial,
        seed=args.seed,
    )

    print(f"Images: {len(images)}")
    print(f"Complete specimens: {len(complete_ids)}")
    print(f"Scanner pairs: {len(pairs)}")
    print(f"Audit status: {'PASS' if not errors else 'FAIL'}")
    print(f"Wrote: {args.output_dir}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        if not args.allow_partial:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
