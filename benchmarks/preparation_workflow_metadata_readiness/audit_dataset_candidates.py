#!/usr/bin/env python3
"""Deterministic metadata-readiness audit for preparation/workflow studies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PACKAGE_DIR / "dataset_candidate_registry.csv"
DEFAULT_REPORT = PACKAGE_DIR / "readiness_report.md"
AUDIT_ID = "preparation_workflow_metadata_readiness_v1"

REQUIRED_COLUMNS = (
    "dataset_id", "dataset_name", "dataset_version", "source_type",
    "access_status", "license_status", "biological_unit_id_available",
    "block_id_available", "section_id_available", "preparation_condition_available",
    "preparation_batch_available", "scanner_device_id_available",
    "scanner_model_available", "scan_batch_available", "site_available",
    "post_preparation_workflow_available", "acquisition_order_available",
    "section_order_available", "section_distance_available",
    "paired_same_section_scans_available", "matched_serial_sections_available",
    "raw_wsi_available", "patch_coordinates_available",
    "immutable_source_id_available", "metadata_source", "evidence_status", "notes",
)
AVAILABILITY_FIELDS = REQUIRED_COLUMNS[4:24]
ALLOWED_AVAILABILITY = {"yes", "no", "partial", "unknown", "not_applicable"}
ALLOWED_EVIDENCE = {
    "verified_primary_source", "verified_repository_metadata",
    "reported_unverified", "inferred", "unknown",
}
VERIFIED_EVIDENCE = {"verified_primary_source", "verified_repository_metadata"}

PREPARATION_REQUIREMENTS = {
    "biological_unit_id_available": "missing_biological_anchor",
    "block_id_available": "missing_block_identity",
    "section_id_available": "missing_section_identity",
    "preparation_condition_available": "missing_preparation_condition",
    "preparation_batch_available": "missing_preparation_batch",
    "scanner_device_id_available": "missing_scanner_identity",
    "scan_batch_available": "missing_scan_batch",
    "matched_serial_sections_available": "missing_serial_section_relationship",
}
SCANNER_REQUIREMENTS = {
    "section_id_available": "missing_section_identity",
    "preparation_condition_available": "missing_preparation_condition",
    "scanner_device_id_available": "missing_scanner_identity",
    "paired_same_section_scans_available": "missing_same_section_scanner_pairing",
    "scan_batch_available": "missing_scan_batch",
    "immutable_source_id_available": "missing_immutable_source_provenance",
}
WORKFLOW_REQUIREMENTS = {
    "section_id_available": "missing_section_identity",
    "preparation_condition_available": "missing_preparation_condition",
    "scanner_device_id_available": "missing_scanner_identity",
    "post_preparation_workflow_available": "missing_workflow_definition",
    "acquisition_order_available": "missing_acquisition_order",
    "scan_batch_available": "missing_scan_batch",
    "paired_same_section_scans_available": "missing_same_section_scanner_pairing",
}
BOUNDARIES = (
    "This is a metadata feasibility result, not an experimental result.",
    "No candidate is confirmatory-ready unless every required field is explicit and verified.",
    "Scanner suppression is not evidence of biological validity.",
    "Absence of metadata is not evidence that the underlying factor was absent.",
    "Inferred site or scanner labels are not process provenance.",
)


class RegistryError(Exception):
    """Fail-closed registry validation error."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def source_label(path: Path) -> str:
    return path.name if path.resolve() == DEFAULT_INPUT.resolve() else str(path)


def load_registry(path: Path) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise RegistryError("utf8_bom_not_allowed")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
        raise RegistryError("invalid_headers")
    rows = [dict(row) for row in reader]
    if not rows:
        raise RegistryError("empty_registry")
    validate_rows(rows)
    return rows, sha256_bytes(raw)


def validate_rows(rows: list[dict[str, str]]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        for field in REQUIRED_COLUMNS:
            if field == "notes":
                continue
            value = row[field]
            if not value or value != value.strip():
                raise RegistryError(f"empty_or_noncanonical:{field}:row={index}")
        dataset_id = row["dataset_id"]
        if dataset_id in seen:
            raise RegistryError(f"duplicate_dataset_id:{dataset_id}")
        seen.add(dataset_id)
        for field in AVAILABILITY_FIELDS:
            if row[field] not in ALLOWED_AVAILABILITY:
                raise RegistryError(f"invalid_enum:{field}:{row[field]}")
        if row["evidence_status"] not in ALLOWED_EVIDENCE:
            raise RegistryError(f"invalid_evidence_status:{row['evidence_status']}")
        if row["paired_same_section_scans_available"] == "yes" and row["section_id_available"] != "yes":
            raise RegistryError("contradiction:paired_scans_without_section_identity")
        if row["matched_serial_sections_available"] == "yes" and (
            row["block_id_available"] != "yes" or row["section_id_available"] != "yes"
        ):
            raise RegistryError("contradiction:serial_sections_without_block_and_section")
        if row["post_preparation_workflow_available"] == "yes" and "site_only" in row["notes"]:
            raise RegistryError("contradiction:site_only_claimed_as_workflow")


def contrast_result(row: dict[str, str], requirements: dict[str, str], kind: str) -> dict[str, Any]:
    blockers = {reason for field, reason in requirements.items() if row[field] != "yes"}
    if kind == "workflow" and row["site_available"] == "yes" and row["post_preparation_workflow_available"] != "yes":
        blockers.add("site_not_equivalent_to_workflow")
    if "factor_nesting_unresolved" in row["notes"]:
        blockers.add("factor_nesting_unresolved")
    if row["evidence_status"] in {"inferred", "unknown"}:
        blockers.add("metadata_inferred_not_verified")
    verified = row["evidence_status"] in VERIFIED_EVIDENCE
    if not blockers and verified:
        status = "confirmatory_design_candidate"
    elif any(row[field] in {"yes", "partial"} for field in requirements):
        status = "candidate_discovery"
    else:
        status = "descriptive_inventory"
    return {"status": status, "blocking_reasons": sorted(blockers)}


def analyze_row(row: dict[str, str]) -> dict[str, Any]:
    preparation = contrast_result(row, PREPARATION_REQUIREMENTS, "preparation")
    scanner = contrast_result(row, SCANNER_REQUIREMENTS, "scanner")
    workflow = contrast_result(row, WORKFLOW_REQUIREMENTS, "workflow")
    global_blockers: set[str] = set()
    if row["access_status"] != "yes" or row["license_status"] != "yes":
        global_blockers.add("access_or_license_unresolved")
    if row["evidence_status"] in {"inferred", "unknown"}:
        global_blockers.add("metadata_inferred_not_verified")
    if "factor_nesting_unresolved" in row["notes"]:
        global_blockers.add("factor_nesting_unresolved")
    for result in (preparation, scanner, workflow):
        result["blocking_reasons"] = sorted(set(result["blocking_reasons"]) | global_blockers)
        if result["status"] == "confirmatory_design_candidate" and global_blockers:
            result["status"] = "candidate_discovery"
    statuses = {preparation["status"], scanner["status"], workflow["status"]}
    if "confirmatory_design_candidate" in statuses:
        tier = "confirmatory_design_candidate"
    elif "candidate_discovery" in statuses:
        tier = "candidate_discovery"
    else:
        tier = "descriptive_inventory"
    blockers = sorted(
        global_blockers
        | set(preparation["blocking_reasons"])
        | set(scanner["blocking_reasons"])
        | set(workflow["blocking_reasons"])
    )
    return {
        "dataset_id": row["dataset_id"],
        "dataset_name": row["dataset_name"],
        "descriptive_readiness_tier": tier,
        "preparation_contrast": preparation,
        "scanner_contrast": scanner,
        "workflow_contrast": workflow,
        "blocking_reasons": blockers,
        "nonblocking_limitations": [
            "metadata feasibility is not statistical power",
            "absence of metadata is not evidence that the factor was absent",
        ],
        "verified_fields": sorted(field for field in AVAILABILITY_FIELDS if row[field] == "yes"),
        "inferred_fields": sorted(field for field in AVAILABILITY_FIELDS if row[field] == "partial"),
        "unknown_fields": sorted(field for field in AVAILABILITY_FIELDS if row[field] == "unknown"),
        "exact_evidence_source": row["metadata_source"],
        "evidence_status": row["evidence_status"],
        "recommended_next_action": (
            "Proceed to a contrast-specific crossed-design audit before analysis."
            if tier == "confirmatory_design_candidate"
            else "Resolve the listed provenance and access gaps; do not treat inferred labels as process provenance."
        ),
    }


def analyze_registry(rows: list[dict[str, str]], input_sha256: str, source: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "audit_id": AUDIT_ID,
        "source": source,
        "input_sha256": input_sha256,
        "candidate_count": len(rows),
        "candidates": [analyze_row(row) for row in sorted(rows, key=lambda item: item["dataset_id"])],
        "boundaries": list(BOUNDARIES),
    }
    payload["audit_fingerprint"] = sha256_bytes(stable_json(payload).encode("utf-8"))
    return payload


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Preparation/workflow metadata readiness report", "",
        f"Audit ID: `{result['audit_id']}`",
        f"Input SHA-256: `{result['input_sha256']}`",
        f"Audit fingerprint: `{result['audit_fingerprint']}`", "",
        "## Boundaries", "",
    ]
    lines.extend(f"- {item}" for item in result["boundaries"])
    lines.extend(["", "## Candidate results", ""])
    for candidate in result["candidates"]:
        lines.extend([
            f"### {candidate['dataset_name']} (`{candidate['dataset_id']}`)", "",
            f"- Overall tier: **{candidate['descriptive_readiness_tier']}**",
            f"- Preparation contrast: **{candidate['preparation_contrast']['status']}**",
            f"- Scanner contrast: **{candidate['scanner_contrast']['status']}**",
            f"- Workflow contrast: **{candidate['workflow_contrast']['status']}**",
            f"- Evidence: `{candidate['evidence_status']}` from `{candidate['exact_evidence_source']}`",
            "- Blocking reasons: " + (", ".join(f"`{item}`" for item in candidate["blocking_reasons"]) or "none"),
            "- Recommended next action: " + candidate["recommended_next_action"], "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def write_fixture(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def base_test_row() -> dict[str, str]:
    row = {field: "yes" for field in REQUIRED_COLUMNS}
    row.update({
        "dataset_id": "TEST", "dataset_name": "Test dataset", "dataset_version": "v1",
        "source_type": "test_fixture", "metadata_source": "fixture",
        "evidence_status": "verified_primary_source", "notes": "prospective_crossing_verified",
    })
    return row


def expect_error(rows: list[dict[str, str]], code: str) -> None:
    with tempfile.TemporaryDirectory(prefix="metadata-readiness-") as directory:
        path = Path(directory) / "fixture.csv"
        write_fixture(path, rows)
        try:
            load_registry(path)
        except RegistryError as exc:
            if code not in str(exc):
                raise AssertionError(f"expected {code}, got {exc}") from exc
        else:
            raise AssertionError(f"expected error containing {code}")


def run_self_tests() -> dict[str, Any]:
    passed: list[str] = []
    row = base_test_row(); expect_error([row, dict(row)], "duplicate_dataset_id"); passed.append("duplicate_dataset_id_fails")
    row = base_test_row(); row["site_available"] = "maybe"; expect_error([row], "invalid_enum"); passed.append("invalid_enum_fails")
    row = base_test_row(); row["block_id_available"] = "unknown"; assert analyze_row(row)["preparation_contrast"]["status"] != "confirmatory_design_candidate"; passed.append("confirmatory_with_unknown_required_field_fails")
    row = base_test_row(); row["evidence_status"] = "inferred"; assert analyze_row(row)["descriptive_readiness_tier"] != "confirmatory_design_candidate"; passed.append("inferred_metadata_cannot_support_confirmatory")
    row = base_test_row(); row["post_preparation_workflow_available"] = "no"; row["site_available"] = "yes"; assert "site_not_equivalent_to_workflow" in analyze_row(row)["workflow_contrast"]["blocking_reasons"]; passed.append("site_only_does_not_satisfy_workflow")
    row = base_test_row(); row["scanner_device_id_available"] = "no"; row["scanner_model_available"] = "yes"; assert "missing_scanner_identity" in analyze_row(row)["scanner_contrast"]["blocking_reasons"]; passed.append("scanner_model_without_device_pairing_is_not_same_section_support")
    row = base_test_row(); row["immutable_source_id_available"] = "no"; assert "missing_immutable_source_provenance" in analyze_row(row)["scanner_contrast"]["blocking_reasons"]; passed.append("missing_immutable_source_id_blocks_scanner_confirmatory")
    row = base_test_row(); row["block_id_available"] = "no"; row["matched_serial_sections_available"] = "no"; assert "missing_block_identity" in analyze_row(row)["preparation_contrast"]["blocking_reasons"]; passed.append("serial_sections_without_block_identity_block_preparation")
    row = base_test_row(); row["notes"] = "factor_nesting_unresolved"; assert "factor_nesting_unresolved" in analyze_row(row)["blocking_reasons"]; passed.append("preparation_nested_in_scanner_is_not_confirmatory")
    row = base_test_row(); row["license_status"] = "unknown"; assert "access_or_license_unresolved" in analyze_row(row)["blocking_reasons"]; passed.append("access_or_license_unknown_blocks_confirmatory")
    row = base_test_row(); row["preparation_batch_available"] = "partial"; assert analyze_row(row)["preparation_contrast"]["status"] == "candidate_discovery"; passed.append("candidate_discovery_allows_partial_nonconfirmatory_metadata")
    row = base_test_row(); digest = sha256_bytes(b"fixture"); assert stable_json(analyze_registry([row], digest, "fixture")) == stable_json(analyze_registry([row], digest, "fixture")); passed.append("deterministic_output")
    original = DEFAULT_REPORT.read_bytes() if DEFAULT_REPORT.exists() else b""
    with tempfile.TemporaryDirectory(prefix="metadata-readiness-custom-") as directory:
        path = Path(directory) / "fixture.csv"; write_fixture(path, [base_test_row()]); loaded, digest = load_registry(path); analyze_registry(loaded, digest, source_label(path))
    after = DEFAULT_REPORT.read_bytes() if DEFAULT_REPORT.exists() else b""; assert original == after; passed.append("custom_input_does_not_modify_checked_report")
    holder: Path | None = None
    with tempfile.TemporaryDirectory(prefix="metadata-readiness-cleanup-") as directory:
        holder = Path(directory); (holder / "temp.txt").write_text("x", encoding="utf-8")
    assert holder is not None and not holder.exists(); passed.append("temporary_fixtures_removed")
    return {"passed": len(passed), "total": 14, "tests": passed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        print(stable_json(run_self_tests()), end="")
        return 0
    rows, digest = load_registry(args.input)
    result = analyze_registry(rows, digest, source_label(args.input))
    if args.format == "json":
        print(stable_json(result), end="")
        return 0
    report = render_report(result)
    if args.check_report:
        if not DEFAULT_REPORT.exists() or DEFAULT_REPORT.read_text(encoding="utf-8") != report:
            raise SystemExit("checked report differs from deterministic regeneration")
        print("PASS: checked report is deterministic")
        return 0
    if args.input.resolve() == DEFAULT_INPUT.resolve():
        DEFAULT_REPORT.write_text(report, encoding="utf-8", newline="\n")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
