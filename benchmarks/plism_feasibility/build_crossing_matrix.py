#!/usr/bin/env python3
"""Normalize a PLISM image-list CSV and audit the executed stain x scanner crossing.

The script is intentionally image-free. It consumes a provenance-enriched PLISM
image-list CSV and writes deterministic metadata products before embeddings are
generated. It fails closed unless a parent slide/section identifier is present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_NORMALIZED = PACKAGE_DIR / "normalized_observations.csv"
DEFAULT_MATRIX = PACKAGE_DIR / "crossing_matrix.csv"
DEFAULT_REPORT = PACKAGE_DIR / "crossing_report.md"
SCHEMA_VERSION = "plism_crossing_matrix_v2"

CANONICAL_COLUMNS = (
    "observation_id",
    "parent_id",
    "registered_group_id",
    "tissue_type",
    "stain_type",
    "scanner_domain",
    "coordinate",
    "image_path",
    "comparison_unit",
)

HEADER_ALIASES = {
    "parent_id": {
        "parent id", "parent_id", "slide id", "slide_id", "section id",
        "section_id", "wsi id", "wsi_id", "archive group id", "archive_group_id",
    },
    "tissue_type": {"tissue type", "tissue_type", "tissue"},
    "stain_type": {"stain type", "stain_type", "stain"},
    "scanner_domain": {"device type", "device_type", "device", "scanner", "scanner type"},
    "coordinate": {"coordinate", "coordinates", "coord"},
    "image_path": {"image path", "image_path", "path", "relative path"},
}

PATH_RE = re.compile(
    r"^(?P<folder_stain>[^/\\]+)_(?P<folder_device>[^/\\]+)[/\\]"
    r"(?P<file_stain>[^/\\]+)_(?P<file_device>[^/\\]+)_"
    r"(?P<x>-?\d+)_(?P<y>-?\d+)\.png$",
    re.IGNORECASE,
)


class CrossingError(Exception):
    """Fail-closed input or design validation error."""


def stable_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_header(value: str) -> str:
    return " ".join(value.strip().lower().replace("-", " ").split())


def resolve_headers(fieldnames: Iterable[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise CrossingError("missing_headers")
    normalized = {normalize_header(name): name for name in fieldnames}
    resolved: dict[str, str] = {}
    for canonical, aliases in HEADER_ALIASES.items():
        matches = [normalized[a] for a in aliases if a in normalized]
        if len(matches) != 1:
            raise CrossingError(f"missing_or_ambiguous_header:{canonical}")
        resolved[canonical] = matches[0]
    return resolved


def clean(value: Any, field: str, row_number: int) -> str:
    if not isinstance(value, str):
        raise CrossingError(f"non_string:{field}:row={row_number}")
    result = value.strip()
    if not result:
        raise CrossingError(f"empty:{field}:row={row_number}")
    return result


def canonical_coordinate(value: str) -> str:
    numbers = re.findall(r"-?\d+", value)
    if len(numbers) < 2:
        raise CrossingError(f"invalid_coordinate:{value}")
    return "_".join(numbers)


def parse_path(path: str) -> dict[str, str] | None:
    normalized = path.replace("\\", "/").lstrip("./")
    match = PATH_RE.match(normalized)
    if not match:
        return None
    groups = match.groupdict()
    if groups["folder_stain"] != groups["file_stain"]:
        raise CrossingError(f"path_stain_contradiction:{path}")
    if groups["folder_device"] != groups["file_device"]:
        raise CrossingError(f"path_device_contradiction:{path}")
    return {
        "stain_type": groups["file_stain"],
        "scanner_domain": groups["file_device"],
        "coordinate": f"{groups['x']}_{groups['y']}",
    }


def normalize_rows(text: str) -> list[dict[str, str]]:
    if text.startswith("\ufeff"):
        raise CrossingError("utf8_bom_not_allowed")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    headers = resolve_headers(reader.fieldnames)
    rows: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    seen_identity: set[tuple[str, str, str]] = set()

    for row_number, raw in enumerate(reader, start=2):
        parent_id = clean(raw.get(headers["parent_id"]), "parent_id", row_number)
        tissue = clean(raw.get(headers["tissue_type"]), "tissue_type", row_number)
        stain = clean(raw.get(headers["stain_type"]), "stain_type", row_number)
        scanner = clean(raw.get(headers["scanner_domain"]), "scanner_domain", row_number)
        coordinate = canonical_coordinate(
            clean(raw.get(headers["coordinate"]), "coordinate", row_number)
        )
        image_path = clean(
            raw.get(headers["image_path"]), "image_path", row_number
        ).replace("\\", "/")

        parsed = parse_path(image_path)
        if parsed:
            if parsed["stain_type"].casefold() != stain.casefold():
                raise CrossingError(f"csv_path_stain_contradiction:row={row_number}")
            if parsed["scanner_domain"].casefold() != scanner.casefold():
                raise CrossingError(f"csv_path_device_contradiction:row={row_number}")
            parsed_coord = canonical_coordinate(parsed["coordinate"])
            if not coordinate.startswith(parsed_coord) and not parsed_coord.startswith(coordinate):
                raise CrossingError(f"csv_path_coordinate_contradiction:row={row_number}")

        registered_group_id = f"{parent_id}|{coordinate}"
        identity = (registered_group_id, stain, scanner)
        if image_path in seen_paths:
            raise CrossingError(f"duplicate_image_path:{image_path}")
        if identity in seen_identity:
            raise CrossingError(f"duplicate_group_stain_scanner:{identity}")
        seen_paths.add(image_path)
        seen_identity.add(identity)

        observation_key = "|".join((*identity, image_path))
        rows.append({
            "observation_id": sha256_text(observation_key)[:20],
            "parent_id": parent_id,
            "registered_group_id": registered_group_id,
            "tissue_type": tissue,
            "stain_type": stain,
            "scanner_domain": scanner,
            "coordinate": coordinate,
            "image_path": image_path,
            "comparison_unit": "same_section_within_stain__serial_section_across_stain",
        })

    if not rows:
        raise CrossingError("empty_csv")
    return sorted(
        rows,
        key=lambda r: (
            r["parent_id"],
            r["registered_group_id"],
            r["stain_type"],
            r["scanner_domain"],
            r["image_path"],
        ),
    )


def build_matrix(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    stains = sorted({r["stain_type"] for r in rows})
    scanners = sorted({r["scanner_domain"] for r in rows})
    groups = sorted({r["registered_group_id"] for r in rows})
    parents = sorted({r["parent_id"] for r in rows})
    tissues = sorted({r["tissue_type"] for r in rows})

    counts: Counter[tuple[str, str]] = Counter(
        (r["stain_type"], r["scanner_domain"]) for r in rows
    )
    group_cells: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        group_cells[row["registered_group_id"]].add(
            (row["stain_type"], row["scanner_domain"])
        )

    matrix = [
        {
            "stain_type": stain,
            "scanner_domain": scanner,
            "observation_count": str(counts[(stain, scanner)]),
            "cell_present": "yes" if counts[(stain, scanner)] else "no",
        }
        for stain in stains
        for scanner in scanners
    ]

    expected_cells = {(s, d) for s in stains for d in scanners}
    complete_groups = [g for g, cells in group_cells.items() if cells == expected_cells]
    missing_by_group = {
        g: [f"{s}|{d}" for s, d in sorted(expected_cells - cells)]
        for g, cells in sorted(group_cells.items())
        if cells != expected_cells
    }

    observed_pairs = sum(1 for value in counts.values() if value > 0)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "observation_count": len(rows),
        "parent_count": len(parents),
        "registered_group_count": len(groups),
        "tissue_count": len(tissues),
        "stain_count": len(stains),
        "scanner_count": len(scanners),
        "observed_stain_scanner_cells": observed_pairs,
        "possible_stain_scanner_cells": len(expected_cells),
        "globally_complete_crossing": observed_pairs == len(expected_cells),
        "fully_complete_registered_group_count": len(complete_groups),
        "fully_complete_registered_group_fraction": len(complete_groups) / len(groups),
        "stains": stains,
        "scanners": scanners,
        "tissues": tissues,
        "missing_cells_by_registered_group": missing_by_group,
        "claim_boundaries": [
            "parent_id must be a verified slide, section, WSI, or archive-group identifier",
            "scanner comparisons are same-section only within a fixed stain and registered group",
            "stain comparisons are serial-section correspondences rather than identical-section counterfactuals",
            "parent and registered-group identifiers must not cross train and test splits",
            "scanner domain labels are not verified physical device identifiers",
        ],
    }
    summary["audit_fingerprint"] = sha256_text(stable_json(summary))
    return matrix, summary


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: dict[str, Any]) -> str:
    complete = summary["fully_complete_registered_group_count"]
    total = summary["registered_group_count"]
    lines = [
        "# PLISM executed crossing report",
        "",
        f"Audit fingerprint: `{summary['audit_fingerprint']}`",
        "",
        "## Inventory",
        "",
        f"- Observations: **{summary['observation_count']}**",
        f"- Provenance parents: **{summary['parent_count']}**",
        f"- Registered groups: **{total}**",
        f"- Tissues: **{summary['tissue_count']}**",
        f"- Stains: **{summary['stain_count']}**",
        f"- Scanner domains: **{summary['scanner_count']}**",
        f"- Observed stain × scanner cells: **{summary['observed_stain_scanner_cells']} / {summary['possible_stain_scanner_cells']}**",
        f"- Fully complete registered groups: **{complete} / {total}**",
        "",
        "## Decision",
        "",
    ]
    if complete:
        lines.append(
            "At least one provenance-bounded registered group has complete stain × "
            "scanner coverage and can enter a leakage-safe feasibility analysis."
        )
    else:
        lines.append(
            "No provenance-bounded registered group has complete stain × scanner "
            "coverage. Use a declared restricted crossing."
        )
    lines.extend(["", "## Claim boundaries", ""])
    lines.extend(f"- {item}" for item in summary["claim_boundaries"])
    lines.extend(["", "## Missingness", ""])
    lines.append(
        "Registered groups with at least one missing cell: "
        f"**{len(summary['missing_cells_by_registered_group'])}**"
    )
    return "\n".join(lines).rstrip() + "\n"


def process(input_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = normalize_rows(input_path.read_text(encoding="utf-8"))
    matrix, summary = build_matrix(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / DEFAULT_NORMALIZED.name, CANONICAL_COLUMNS, rows)
    write_csv(
        output_dir / DEFAULT_MATRIX.name,
        ("stain_type", "scanner_domain", "observation_count", "cell_present"),
        matrix,
    )
    (output_dir / DEFAULT_REPORT.name).write_text(
        render_report(summary), encoding="utf-8"
    )
    (output_dir / "crossing_summary.json").write_text(
        stable_json(summary), encoding="utf-8"
    )
    return summary


def run_self_tests() -> dict[str, Any]:
    fixture = """Slide ID,Tissue Type,Stain Type,Device Type,Coordinate,Image Path
slide-a,Liver,GV,S1,1000_500,GV_S1/GV_S1_1000_500.png
slide-a,Liver,GV,S2,1000_500,GV_S2/GV_S2_1000_500.png
slide-a,Liver,GMH,S1,1000_500,GMH_S1/GMH_S1_1000_500.png
slide-a,Liver,GMH,S2,1000_500,GMH_S2/GMH_S2_1000_500.png
"""
    rows = normalize_rows(fixture)
    _, summary = build_matrix(rows)
    assert summary["observation_count"] == 4
    assert summary["parent_count"] == 1
    assert summary["stain_count"] == 2
    assert summary["scanner_count"] == 2
    assert summary["fully_complete_registered_group_count"] == 1

    missing_parent = fixture.replace("Slide ID,", "", 1)
    missing_parent = "\n".join(
        line.split(",", 1)[1] if "," in line else line
        for line in missing_parent.splitlines()
    )
    try:
        normalize_rows(missing_parent)
    except CrossingError as exc:
        assert "missing_or_ambiguous_header:parent_id" in str(exc)
    else:
        raise AssertionError("input without provenance parent was accepted")

    adversarial = fixture + (
        "slide-b,Liver,GV,S1,1000_500,other/GV_S1/GV_S1_1000_500.png\n"
    )
    adversarial_rows = normalize_rows(adversarial)
    assert len({r["registered_group_id"] for r in adversarial_rows}) == 2

    duplicate = fixture + (
        "slide-a,Liver,GV,S1,1000_500,copy.png\n"
    )
    try:
        normalize_rows(duplicate)
    except CrossingError as exc:
        assert "duplicate_group_stain_scanner" in str(exc)
    else:
        raise AssertionError("duplicate identity was accepted")

    contradiction = fixture.replace(
        "GV_S1/GV_S1_1000_500.png", "GV_S2/GV_S2_1000_500.png", 1
    )
    try:
        normalize_rows(contradiction)
    except CrossingError as exc:
        assert "csv_path_device_contradiction" in str(exc)
    else:
        raise AssertionError("path contradiction was accepted")

    with tempfile.TemporaryDirectory(prefix="plism-crossing-") as directory:
        source = Path(directory) / "input.csv"
        source.write_text(fixture, encoding="utf-8")
        generated = process(source, Path(directory) / "out")
        assert generated["audit_fingerprint"]
        assert (Path(directory) / "out" / "crossing_report.md").exists()

    return {"status": "passed", "tests": 6, "schema_version": SCHEMA_VERSION}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Provenance-enriched PLISM image-list CSV")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    try:
        if args.self_test:
            print(stable_json(run_self_tests()), end="")
            return 0
        if args.input is None:
            parser.error("--input is required unless --self-test is used")
        print(stable_json(process(args.input, args.output_dir)), end="")
        return 0
    except (CrossingError, OSError, UnicodeError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
