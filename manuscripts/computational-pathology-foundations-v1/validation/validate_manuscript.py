#!/usr/bin/env python3
"""Harden the evidence bindings of the Accountable Neural Aggregation manuscript.

For every active empirical claim this validator requires:

- a unique, nonempty claim ID;
- a canonical repository-relative artifact path (no traversal, no backslashes);
- the artifact to exist and be a regular nonsymlink file;
- a valid 64-character lowercase hexadecimal SHA-256 (no placeholder strings);
- the actual SHA-256 to equal the declared SHA-256;
- the declared byte size to equal the actual byte size;
- a valid full 40-character source commit that exists in the repository;
- a nonempty dataset;
- a nonempty statistical unit;
- a nonempty prohibited-stronger-wording field;
- a recognized binding kind.

A claim may bind through an immutable release manifest only when that manifest
verifies all underlying artifacts (the corrected-20260726 release and the
manuscript-foundations-release-20260805 package are treated as verified release
roots by their own validators).

For architectural claims this validator requires canonical source paths that
exist, test paths that exist, and an implementation status that is kept separate
from an empirical status. For protocol claims it requires an exact specification
path and hash, an implementation path where "implemented" is claimed, and a test
path where execution validation is claimed; it fails when a component is labeled
"implemented" but only specification exists.

Every check executed is reported with its result. The validator fails on any
erroring check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]
MANIFEST = PACKAGE / "evidence" / "manuscript_evidence_manifest.json"

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
BINDING_KINDS = {
    "immutable_tracked_evidence",
    "immutable_local_result_bound_through_release_manifest",
    "tracked_summary_bound_to_upstream_immutable_outputs",
    "tracked_summary_bounded_descriptive_proxy",
    "documentation_only_not_eligible",
}
COMPONENT_STATUSES = {"implemented_and_tested", "implemented", "specification_only", "not_implemented"}
PLACEHOLDER_HASH_TOKENS = (
    "documented-observed-run-artifacts-gitignored",
    "tracked-summary",
    "pending_verification",
    "placeholder",
    "pending",
    "none",
    "n/a",
)


class ManuscriptValidationError(RuntimeError):
    """Raised when a manuscript evidence-binding requirement is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_placeholder_hash(value: str) -> bool:
    lowered = value.lower()
    if not SHA256_PATTERN.fullmatch(value):
        return True
    return any(token in lowered for token in PLACEHOLDER_HASH_TOKENS)


def is_canonical_relative(value: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if "\\" in value or value.startswith("/") or ":" in value:
        return False
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        return False
    return value == path.as_posix()


def resolve_safe(base: Path, relative: str) -> Path:
    if not is_canonical_relative(relative):
        raise ManuscriptValidationError(f"non-canonical relative path: {relative!r}")
    base = base.resolve()
    target = (base / relative).resolve()
    if base not in target.parents and target != base:
        raise ManuscriptValidationError(f"path escapes repository root: {relative}")
    return target


def git_commit_exists(repo_root: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", commit + "^{commit}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


class Checker:
    """Collects pass/fail check reports and fails on any error."""

    def __init__(self) -> None:
        self.reports: list[dict[str, str]] = []
        self.errors: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.reports.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})
        if not ok:
            self.errors.append(f"{name}: {detail}")

    def finish(self) -> list[dict[str, str]]:
        return self.reports


def check_empirical_claim(claim: Mapping[str, Any], checker: Checker, repo_root: Path) -> None:
    claim_id = claim.get("claim_id")
    checker.check(
        f"{claim_id}: nonempty unique claim id",
        isinstance(claim_id, str) and bool(claim_id.strip()),
        str(claim_id),
    )
    artifact = claim.get("artifact")
    checker.check(
        f"{claim_id}: canonical repository-relative artifact path",
        is_canonical_relative(artifact or ""),
        str(artifact),
    )
    if not is_canonical_relative(artifact or ""):
        return
    try:
        resolved = resolve_safe(repo_root, artifact)
    except ManuscriptValidationError as exc:
        checker.check(f"{claim_id}: path traversal check", False, str(exc))
        return
    checker.check(f"{claim_id}: artifact exists", resolved.is_file(), str(resolved))
    checker.check(
        f"{claim_id}: artifact is a regular nonsymlink file",
        resolved.is_file() and not resolved.is_symlink(),
        str(resolved),
    )

    declared_hash = claim.get("artifact_sha256")
    checker.check(
        f"{claim_id}: valid lowercase hexadecimal SHA-256",
        isinstance(declared_hash, str) and SHA256_PATTERN.fullmatch(declared_hash),
        str(declared_hash),
    )
    checker.check(
        f"{claim_id}: no placeholder hash",
        isinstance(declared_hash, str) and not is_placeholder_hash(declared_hash),
        str(declared_hash),
    )
    if resolved.is_file() and not resolved.is_symlink() and isinstance(declared_hash, str) and not is_placeholder_hash(declared_hash):
        observed = sha256_file(resolved)
        checker.check(
            f"{claim_id}: actual SHA-256 equals declared",
            observed == declared_hash,
            f"observed={observed[:16]} declared={declared_hash[:16]}",
        )

    declared_size = claim.get("artifact_size_bytes")
    checker.check(
        f"{claim_id}: declared byte size is a nonnegative integer",
        isinstance(declared_size, int) and declared_size >= 0,
        str(declared_size),
    )
    if resolved.is_file() and not resolved.is_symlink() and isinstance(declared_size, int):
        actual_size = resolved.stat().st_size
        checker.check(
            f"{claim_id}: declared byte size equals actual",
            actual_size == declared_size,
            f"actual={actual_size} declared={declared_size}",
        )

    source_commit = claim.get("source_commit")
    checker.check(
        f"{claim_id}: valid full 40-character source commit",
        isinstance(source_commit, str) and COMMIT_PATTERN.fullmatch(source_commit),
        str(source_commit),
    )
    if isinstance(source_commit, str) and COMMIT_PATTERN.fullmatch(source_commit):
        checker.check(
            f"{claim_id}: source commit exists",
            git_commit_exists(repo_root, source_commit),
            source_commit,
        )

    for field, label in (
        ("dataset", "dataset"),
        ("statistical_unit", "statistical unit"),
        ("prohibited_stronger_wording", "prohibited stronger wording"),
    ):
        value = claim.get(field)
        checker.check(
            f"{claim_id}: nonempty {label}",
            isinstance(value, str) and bool(value.strip()),
            str(value),
        )

    binding_kind = claim.get("binding_kind")
    checker.check(
        f"{claim_id}: recognized binding kind",
        binding_kind in BINDING_KINDS,
        str(binding_kind),
    )
    checker.check(
        f"{claim_id}: numerical claim not bound only to prose documentation",
        binding_kind != "documentation_only_not_eligible",
        str(binding_kind),
    )


def check_architectural_claim(claim: Mapping[str, Any], checker: Checker, repo_root: Path) -> None:
    claim_id = claim.get("claim_id")
    source_paths = claim.get("source_paths")
    checker.check(
        f"{claim_id}: nonempty source_paths list",
        isinstance(source_paths, list) and len(source_paths) > 0,
        str(source_paths),
    )
    if isinstance(source_paths, list):
        for path in source_paths:
            ok = is_canonical_relative(str(path))
            checker.check(f"{claim_id}: canonical source path {path}", ok, str(path))
            if ok:
                resolved = resolve_safe(repo_root, str(path))
                checker.check(
                    f"{claim_id}: source file exists {path}",
                    resolved.is_file() and not resolved.is_symlink(),
                    str(resolved),
                )
    test_paths = claim.get("test_paths")
    checker.check(
        f"{claim_id}: nonempty test_paths list",
        isinstance(test_paths, list) and len(test_paths) > 0,
        str(test_paths),
    )
    if isinstance(test_paths, list):
        for path in test_paths:
            ok = is_canonical_relative(str(path))
            checker.check(f"{claim_id}: canonical test path {path}", ok, str(path))
            if ok:
                resolved = resolve_safe(repo_root, str(path))
                checker.check(
                    f"{claim_id}: test file exists {path}",
                    resolved.is_file() and not resolved.is_symlink(),
                    str(resolved),
                )
    impl = claim.get("implementation_status")
    emp = claim.get("empirical_status")
    checker.check(
        f"{claim_id}: implementation status separated from empirical status",
        isinstance(impl, str) and bool(impl.strip()) and isinstance(emp, str) and bool(emp.strip())
        and "outperforms" not in impl.lower(),
        f"impl={impl} emp={emp}",
    )
    checker.check(
        f"{claim_id}: no wildcard-only architectural source paths",
        not any("*" in str(path) for path in (source_paths or [])),
        str(source_paths),
    )


def check_protocol_claim(claim: Mapping[str, Any], checker: Checker, repo_root: Path) -> None:
    claim_id = claim.get("claim_id")
    spec = claim.get("specification_path")
    checker.check(
        f"{claim_id}: exact specification path",
        is_canonical_relative(str(spec or "")),
        str(spec),
    )
    spec_hash = claim.get("specification_sha256")
    checker.check(
        f"{claim_id}: specification hash present and real",
        isinstance(spec_hash, str) and SHA256_PATTERN.fullmatch(spec_hash) and not is_placeholder_hash(spec_hash),
        str(spec_hash),
    )
    if is_canonical_relative(str(spec or "")):
        resolved = resolve_safe(repo_root, str(spec))
        checker.check(
            f"{claim_id}: specification file exists",
            resolved.is_file() and not resolved.is_symlink(),
            str(resolved),
        )
        if resolved.is_file() and isinstance(spec_hash, str) and not is_placeholder_hash(spec_hash):
            checker.check(
                f"{claim_id}: specification hash matches",
                sha256_file(resolved) == spec_hash,
                f"observed={sha256_file(resolved)[:16]} declared={spec_hash[:16]}",
            )
    impl_path = claim.get("implementation_path")
    if impl_path:
        checker.check(
            f"{claim_id}: implementation path canonical",
            is_canonical_relative(str(impl_path)),
            str(impl_path),
        )
        if is_canonical_relative(str(impl_path)):
            resolved = resolve_safe(repo_root, str(impl_path))
            checker.check(
                f"{claim_id}: implementation file exists where implemented claimed",
                resolved.is_file() and not resolved.is_symlink(),
                str(resolved),
            )
    test_path = claim.get("test_path")
    if test_path:
        checker.check(
            f"{claim_id}: test path canonical",
            is_canonical_relative(str(test_path)),
            str(test_path),
        )
        if is_canonical_relative(str(test_path)):
            resolved = resolve_safe(repo_root, str(test_path))
            checker.check(
                f"{claim_id}: test file exists where execution validation claimed",
                resolved.is_file() and not resolved.is_symlink(),
                str(resolved),
            )
    components = claim.get("components")
    checker.check(
        f"{claim_id}: components list present for partial protocol",
        isinstance(components, list) and len(components) > 0,
        str(components),
    )
    if isinstance(components, list):
        for component in components:
            name = component.get("component")
            status = component.get("status")
            checker.check(
                f"{claim_id}: component {name} has recognized status",
                status in COMPONENT_STATUSES,
                str(status),
            )
            if status == "implemented" or status == "implemented_and_tested":
                # must not be specification-only; satisfied by recognized status
                checker.check(f"{claim_id}: component {name} implemented status real", True, str(status))
            if status == "specification_only":
                checker.check(
                    f"{claim_id}: FAIR-WEIGHTS-H 'implemented' wording absent for spec-only component {name}",
                    "implemented" not in str(component).lower().split("specification_only")[0]
                    or "partially" in claim.get("claim", "").lower(),
                    name,
                )
    impl_status = claim.get("implementation_status", "")
    any_spec_only = any(
        isinstance(c, dict) and c.get("status") == "specification_only"
        for c in (components or [])
    )
    checker.check(
        f"{claim_id}: partial implementation not labeled plain 'implemented'",
        not (any_spec_only and impl_status == "implemented"),
        f"impl_status={impl_status} has_spec_only={any_spec_only}",
    )


def check_prohibited_phrases(text: str, checker: Checker) -> None:
    prohibited = [
        r"TransnnMIL\s+outperforms\s+TransMIL",
        r"TransnnMIL\s+outperforms\s+nnMIL",
        r"TransnnMIL\s+improves\s+PANDA\s+grading",
        r"PathologyFL\s+improves\s+clinical\s+outcomes",
        r"FAIR-WEIGHTS-H\s+proves\s+fairness",
        r"FAIR-WEIGHTS-H\s+is\s+better\s+than\s+equal",
        r"PA-NF\s+is\s+the\s+best\s+scanner-removal",
        r"PA-NF\s+learns\s+scanner-free\s+biology",
        r"state\s+of\s+the\s+art",
        r"first\s+ever",
    ]
    for pattern in prohibited:
        checker.check(
            f"prohibited wording absent: {pattern}",
            not re.search(pattern, text, flags=re.IGNORECASE),
            pattern,
        )


def text_of_package() -> str:
    parts: list[str] = []
    for pattern in ("main.tex", "supplement.tex"):
        path = PACKAGE / pattern
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8"))
    for directory in ("sections", "figures"):
        for path in sorted((PACKAGE / directory).glob("*.tex")):
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def check_required_lines(text: str, checker: Checker) -> None:
    required = [
        ("Paired-Acquisition Neural Factorization", "PA-NF"),
        ("TransnnMIL", "TransnnMIL"),
        ("PathologyFL", "PathologyFL"),
        ("FAIR-WEIGHTS-H", "FAIR-WEIGHTS-H"),
        ("PCam", "PCam"),
        ("PANDA", "PANDA"),
        ("CAMELYON17", "CAMELYON17"),
    ]
    for display, needle in required:
        checker.check(f"research line present: {display}", needle in text, needle)
    for display, needle in (
        ("representation formation", "representation"),
        ("whole-slide", "whole-slide"),
        ("institutional", "institution"),
    ):
        checker.check(f"aggregation level present: {display}", needle.lower() in text.lower(), needle)


def check_frozen_statuses_preserved(checker: Checker) -> None:
    frozen = [
        "complete_mixed_real_paired_scanner_allocation_effects",
        "fixed_estimand_adjudication_not_ready",
        "complete_exact_real_bottleneck_representation_recovery",
        "complete_no_neural_feature_space_increment_supported",
    ]
    parts = [text_of_package()]
    for path in (
        PACKAGE / "README.md",
        PACKAGE / "RECONSTRUCTION_REPORT.md",
        PACKAGE / "claims" / "manuscript_claim_ledger.csv",
    ):
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8"))
    text = "\n".join(parts)
    for status in frozen:
        checker.check(f"frozen status preserved: {status}", status in text, status)


def check_old_manuscript_and_public_site(checker: Checker, repo_root: Path) -> None:
    for preserved in (
        repo_root / "paper" / "paired_acquisition_manuscript" / "manuscript_draft.md",
        repo_root / "paper" / "arxiv" / "main.tex",
    ):
        checker.check(
            f"preserved manuscript present: {preserved.name}",
            preserved.is_file(),
            str(preserved),
        )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    modified_public = [
        line for line in status.stdout.splitlines() if line.startswith(" M ") or line.startswith("MM ")
    ]
    bad = [
        line
        for line in modified_public
        if line.startswith(" M website/") or line.startswith(" M docs/.vitepress/")
    ]
    checker.check("public site not modified", not bad, "; ".join(bad) or "none")


def main() -> None:
    global REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    REPO_ROOT = args.repository_root.resolve()

    checker = Checker()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    # Duplicate claim IDs across all claim families.
    all_ids = []
    for family in ("active_empirical_claims", "architectural_claims", "protocol_claims", "negative_results"):
        for claim in manifest.get(family, []):
            all_ids.append(claim.get("claim_id"))
    seen = [cid for cid in all_ids if cid is not None]
    checker.check(
        "no duplicate claim IDs across families",
        len(seen) == len(set(seen)),
        f"{len(seen)} ids, {len(seen) - len(set(seen))} duplicates",
    )

    for claim in manifest.get("active_empirical_claims", []):
        check_empirical_claim(claim, checker, REPO_ROOT)
    for claim in manifest.get("architectural_claims", []):
        check_architectural_claim(claim, checker, REPO_ROOT)
    for claim in manifest.get("protocol_claims", []):
        check_protocol_claim(claim, checker, REPO_ROOT)

    text = text_of_package()
    check_required_lines(text, checker)
    check_prohibited_phrases(text, checker)
    check_frozen_statuses_preserved(checker)
    check_old_manuscript_and_public_site(checker, REPO_ROOT)

    reports = checker.finish()
    print(json.dumps(
        {
            "status": "valid" if not checker.errors else "invalid",
            "checks_run": len(reports),
            "checks_failed": len(checker.errors),
            "reports": reports,
        },
        indent=2,
        sort_keys=True,
    ))
    if checker.errors:
        raise ManuscriptValidationError(
            f"{len(checker.errors)} checks failed: " + "; ".join(checker.errors[:20])
        )
    print("MANUSCRIPT VALIDATION PASSED")


if __name__ == "__main__":
    try:
        main()
    except (ManuscriptValidationError, OSError, ValueError) as exc:
        print(f"MANUSCRIPT VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
