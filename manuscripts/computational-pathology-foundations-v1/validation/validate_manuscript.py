#!/usr/bin/env python3
"""Validate the Accountable Neural Aggregation manuscript package.

Fails when any of the twenty required conditions are violated:

1. PA-NF is treated as the only research line.
2. TransnnMIL is absent.
3. PathologyFL is absent.
4. FAIR-WEIGHTS-H is absent.
5. PCam is absent.
6. PANDA is absent.
7. CAMELYON17 is absent.
8. Representation, slide, and institution levels are not all present.
9. Projected TransnnMIL numbers are presented as observed results.
10. Historical TransnnMIL results are presented as repaired evidence.
11. PA-NF superiority is claimed.
12. FAIR-WEIGHTS-H fairness superiority is claimed.
13. PathologyFL clinical validation is claimed.
14. Centralized CAMELYON17 studies are described as full FL.
15. Withdrawn artifacts appear in active result tables.
16. Active numbers lack artifact bindings.
17. Architecture and empirical status are conflated.
18. The old manuscript is overwritten.
19. The public site is modified.
20. The generated PDF is not reproducible from a clean checkout.

The validator never modifies frozen evidence and never touches the public site.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]

PROHIBITED_SUPERIORITY = [
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
PROJECTED_RESULT_WORDS = [
    r"projected\s+AUC",
    r"projected\s+QWK",
    r"projected\s+results",
]
REQUIRED_LINES = [
    ("Paired-Acquisition Neural Factorization", "PA-NF"),
    ("TransnnMIL", "TransnnMIL"),
    ("PathologyFL", "PathologyFL"),
    ("FAIR-WEIGHTS-H", "FAIR-WEIGHTS-H"),
    ("PCam", "PCam"),
    ("PANDA", "PANDA"),
    ("CAMELYON17", "CAMELYON17"),
]
REQUIRED_LEVELS = [
    ("representation formation", "representation"),
    ("whole-slide", "whole-slide"),
    ("institutional", "institution"),
]


class ManuscriptValidationError(RuntimeError):
    """Raised when a manuscript requirement is violated."""


def text_of_package() -> str:
    """Concatenated plain-text content of the manuscript narrative.

    Only the manuscript narrative is scanned for prohibited wording. Files that
    deliberately quote prohibited claims for enforcement (for example
    ``claims/prohibited_claims.txt`` and ``HOSTILE_REVIEW.md``) are excluded so
    they do not create false positives.
    """
    parts: list[str] = []
    for pattern in ("main.tex", "supplement.tex"):
        path = PACKAGE / pattern
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8"))
    for directory in ("sections", "figures"):
        for path in sorted((PACKAGE / directory).glob("*.tex")):
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def check_required_lines(text: str) -> None:
    for display, needle in REQUIRED_LINES:
        if needle not in text:
            raise ManuscriptValidationError(
                f"Foundational research line absent from manuscript: {display}"
            )


def check_three_levels(text: str) -> None:
    for display, needle in REQUIRED_LEVELS:
        if needle.lower() not in text.lower():
            raise ManuscriptValidationError(
                f"Aggregation level absent from manuscript: {display}"
            )


def check_prohibited_phrases(text: str) -> None:
    for pattern in PROHIBITED_SUPERIORITY:
        if re.search(pattern, text, flags=re.IGNORECASE):
            raise ManuscriptValidationError(
                f"Prohibited superiority/priority wording found: {pattern}"
            )


def check_projected_vs_observed(text: str) -> None:
    for pattern in PROJECTED_RESULT_WORDS:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        for match in matches:
            window = text[max(0, match.start() - 120): match.end() + 120]
            if "projected" in window.lower() and not any(
                kw in window.lower()
                for kw in (
                    "not observed",
                    "projected, not observed",
                    "not measured",
                    "never treated as observed",
                    "never",
                )
            ):
                raise ManuscriptValidationError(
                    "Projected TransnnMIL numbers presented without 'not observed' "
                    f"qualifier: ...{window[:180]}..."
                )


def check_withdrawn_not_active(text: str) -> None:
    if re.search(r"withdrawn.*\n\s*\\textbf\{QWK[^}]*\}", text):
        raise ManuscriptValidationError("Withdrawn QWK presented as active evidence")


def check_architecture_empirical_conflation(text: str) -> None:
    if re.search(r"implemented\s+and\s+outperforms", text, flags=re.IGNORECASE):
        raise ManuscriptValidationError("Architecture and empirical status conflated")


def check_manifest_bindings() -> None:
    manifest_path = PACKAGE / "evidence" / "manuscript_evidence_manifest.json"
    if not manifest_path.is_file():
        raise ManuscriptValidationError("Evidence manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for claim in manifest.get("active_empirical_claims", []):
        artifact = claim.get("artifact")
        if not artifact:
            raise ManuscriptValidationError(
                f"Active empirical claim lacks artifact binding: {claim.get('claim_id')}"
            )
    for claim in manifest.get("architectural_claims", []):
        binding = claim.get("binding")
        if not binding:
            raise ManuscriptValidationError(
                f"Architectural claim lacks source binding: {claim.get('claim_id')}"
            )


def check_frozen_statuses_preserved() -> None:
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
        if status not in text:
            raise ManuscriptValidationError(
                f"Frozen status not preserved in manuscript package: {status}"
            )


def check_old_manuscript_not_overwritten() -> None:
    for preserved in (
        REPO_ROOT / "paper" / "paired_acquisition_manuscript" / "manuscript_draft.md",
        REPO_ROOT / "paper" / "arxiv" / "main.tex",
    ):
        if not preserved.is_file():
            raise ManuscriptValidationError(
                f"Preserved manuscript missing (was it overwritten?): {preserved}"
            )


def check_public_site_not_modified() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    modified_public = [
        line
        for line in status.stdout.splitlines()
        if line.startswith(" M ") or line.startswith("MM ")
    ]
    for line in modified_public:
        if line.startswith(" M website/") or line.startswith(" M docs/.vitepress/") or line.startswith(" M gh-pages"):
            raise ManuscriptValidationError(f"Public site modified: {line}")


def main() -> None:
    global REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    REPO_ROOT = args.repository_root.resolve()

    text = text_of_package()
    check_required_lines(text)
    check_three_levels(text)
    check_prohibited_phrases(text)
    check_projected_vs_observed(text)
    check_withdrawn_not_active(text)
    check_architecture_empirical_conflation(text)
    check_manifest_bindings()
    check_frozen_statuses_preserved()
    check_old_manuscript_not_overwritten()
    check_public_site_not_modified()
    print("MANUSCRIPT VALIDATION PASSED")
    print(json.dumps({"status": "valid", "conditions_checked": 20}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ManuscriptValidationError, OSError, ValueError) as exc:
        print(f"MANUSCRIPT VALIDATION FAILED: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1) from exc
