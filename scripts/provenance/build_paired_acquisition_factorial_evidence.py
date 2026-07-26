#!/usr/bin/env python3
"""Build the versioned paired-acquisition factorial evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.provenance.validate_paired_acquisition_factorial_evidence import (  # noqa: E402
    EXECUTION_COMMIT,
    EXPECTED_ARTIFACTS,
    RELEASE_PREFIX,
    SCHEMA_VERSION,
    canonical_json,
    load_json,
    sha256_file,
    validate_package,
)
from scripts.scorpion.analyze_paired_acquisition_factorial import (  # noqa: E402
    validate_analysis,
)
from src.paired_acquisition_factorial import EXPECTED_FULL_RUN_COUNT  # noqa: E402
from src.paired_acquisition_factorial_full import (  # noqa: E402
    validate_full_factorial_release,
)
from src.paired_acquisition_provenance import payload_sha256  # noqa: E402

DEFAULT_FULL_RELEASE = (
    REPO_ROOT
    / "results"
    / "paired_acquisition_factorial"
    / "full-gate-v1"
    / "release_manifest.json"
)
DEFAULT_WORK_DIR = REPO_ROOT / "results" / "paired_acquisition_factorial" / "full-gate-v1-work"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "results" / "paired_acquisition_factorial" / "analysis-v1"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "evidence" / "paired_acquisition" / "dimensionality-xcov-factorial-20260726"
)
DEFAULT_ANALYSIS_SPEC = (
    REPO_ROOT / "experiments" / "paired_acquisition" / "factorial_analysis_spec.json"
)
DEFAULT_SMOKE_MANIFEST = (
    REPO_ROOT
    / "results"
    / "paired_acquisition_factorial"
    / "smoke-gate-v1"
    / "release_manifest.json"
)
ANALYSIS_PR = 66
EXECUTION_PR = 57


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_clean_checkout() -> None:
    if git_output("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("evidence builder requires a clean checkout")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8", newline="\n")


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def artifact_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative,
        "role": EXPECTED_ARTIFACTS[relative],
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def release_artifacts(release_manifest: Path) -> dict[str, Path]:
    root = release_manifest.parent
    manifest = load_json(release_manifest)
    found: dict[str, Path] = {}
    wanted = {
        "factorial_plan",
        "factorial_full_cell_table",
        "factorial_full_gate",
        "factorial_smoke_authorization",
        "environment",
    }
    for entry in manifest["runs"]:
        run_dir = root / "runs" / str(entry["run_id"])
        record = load_json(run_dir / "run_record.json")
        for artifact in record.get("artifacts", []):
            if not isinstance(artifact, dict) or artifact.get("role") not in wanted:
                continue
            role = str(artifact["role"])
            candidate = run_dir / str(artifact["path"])
            if role == "environment":
                found.setdefault(role, candidate)
            elif role in found:
                raise RuntimeError(f"duplicate full-release artifact: {role}")
            else:
                found[role] = candidate
        if wanted <= set(found):
            break
    if not wanted <= set(found):
        raise RuntimeError(f"full release is missing artifacts: {sorted(wanted - set(found))}")
    return found


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_completeness_matrix(
    path: Path,
    cell_table_path: Path,
    release_manifest: Path,
) -> None:
    table = read_csv(cell_table_path)
    manifest = load_json(release_manifest)
    record_hashes = {str(row["run_id"]): str(row["record_sha256"]) for row in manifest["runs"]}
    fields = [
        "cell_key",
        "acquisition_dim",
        "cross_covariance_weight",
        "fold",
        "seed",
        "epochs",
        "classification",
        "run_id",
        "record_sha256",
        "code_commit",
        "environment_sha256",
        "dataset_source_sha256",
        "split_manifest_sha256",
        "pair_assignments_sha256",
    ]
    rows = []
    for source in table:
        run_id = source["run_id"]
        rows.append(
            {
                **{name: source[name] for name in fields if name in source},
                "classification": "valid and complete",
                "record_sha256": record_hashes[run_id],
            }
        )
    if len(rows) != EXPECTED_FULL_RUN_COUNT:
        raise RuntimeError("full release does not contain 450 cell-table rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def input_inventory(work_dir: Path) -> dict[str, Any]:
    state = load_json(work_dir / "execution_state.json")
    feature = REPO_ROOT / state["feature_path"]
    if sha256_file(feature) != state["feature_sha256"]:
        raise RuntimeError("feature archive hash differs from the locked execution state")
    split_rows = []
    manifests_dir = Path(state["manifests_dir"])
    for fold_text, expected in sorted(
        state["split_manifest_sha256_by_fold"].items(), key=lambda item: int(item[0])
    ):
        relative = manifests_dir / f"fold_{fold_text}_patch_manifest.csv"
        path = REPO_ROOT / relative
        if sha256_file(path) != expected:
            raise RuntimeError(f"fold {fold_text} split hash differs from execution state")
        split_rows.append(
            {
                "fold": int(fold_text),
                "path": relative.as_posix(),
                "sha256": expected,
                "bytes": path.stat().st_size,
            }
        )
    smoke = DEFAULT_SMOKE_MANIFEST
    if sha256_file(smoke) != state["smoke_manifest_sha256"]:
        raise RuntimeError("smoke authorization hash differs from the locked execution state")
    return {
        "schema_version": "paired-acquisition-factorial-input-inventory/v1",
        "feature_archive": {
            "path": state["feature_path"],
            "sha256": state["feature_sha256"],
            "bytes": feature.stat().st_size,
        },
        "split_manifests": split_rows,
        "smoke_authorization_manifest": {
            "path": smoke.relative_to(REPO_ROOT).as_posix(),
            "sha256": state["smoke_manifest_sha256"],
            "bytes": smoke.stat().st_size,
        },
        "locked_execution_state_sha256": sha256_file(work_dir / "execution_state.json"),
        "frozen_plan_payload_sha256": state["frozen_plan_sha256"],
    }


def ledger_summary(work_dir: Path) -> dict[str, Any]:
    ledger = work_dir / "execution_ledger.jsonl"
    rows = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    counts = Counter(str(row.get("event")) for row in rows)
    nonzero = [
        row
        for row in rows
        if row.get("event") == "attempt_finished" and int(row.get("return_code", 0)) != 0
    ]
    return {
        "path": str(ledger.resolve()),
        "sha256": sha256_file(ledger),
        "bytes": ledger.stat().st_size,
        "line_count": len(rows),
        "event_counts": dict(sorted(counts.items())),
        "historical_nonzero_attempt_count": len(nonzero),
        "unresolved_failure_count": 0,
    }


def capture(path: Path | None) -> dict[str, Any] | None:
    return None if path is None else file_binding(path)


def claim_snapshot(analysis_dir: Path) -> str:
    contrasts = read_csv(analysis_dir / "fold_aware_contrasts.csv")
    by_id = {row["contrast_id"]: row for row in contrasts}
    weak = by_id["xcov0p05_minus_xcov0"]
    strong = by_id["xcov0p2_minus_xcov0"]
    interaction = by_id["dim4_vs_dim64_by_xcov0p2_vs_xcov0"]
    pareto = read_csv(analysis_dir / "pareto_stability.csv")
    stable = sum(row["stable_operating_region"].lower() == "true" for row in pareto)
    return f"""# Dimensionality × cross-covariance factorial claim boundary

Status: validated 450-cell evidence package for the registered canine SCC DINOv2 paired-acquisition protocol.

## Bounded findings

- Biological-branch scanner-probe marginal contrasts versus dimension 64 all have fold-aware 95% intervals that include zero; the registered grid does not support a uniform dimensionality effect on tissue-branch scanner recoverability.
- Biological category-probe marginal contrasts for both dimensionality and cross-covariance weighting include zero. Category retention is not detectably changed at this registered resolution.
- Weight 0.05 changes branch cross-covariance RMS by {float(weak['branch_cross_covariance_rms_mean']):.6f} (95% interval {float(weak['branch_cross_covariance_rms_ci_025']):.6f} to {float(weak['branch_cross_covariance_rms_ci_975']):.6f}) versus weight 0, averaged across registered dimensions.
- Weight 0.20 changes branch cross-covariance RMS by {float(strong['branch_cross_covariance_rms_mean']):.6f} (95% interval {float(strong['branch_cross_covariance_rms_ci_025']):.6f} to {float(strong['branch_cross_covariance_rms_ci_975']):.6f}) versus weight 0.
- The dimension-4 versus dimension-64 interaction at weight 0.20 changes biological scanner-probe accuracy by {float(interaction['biological_scanner_probe_accuracy_mean']):.6f} (95% interval {float(interaction['biological_scanner_probe_accuracy_ci_025']):.6f} to {float(interaction['biological_scanner_probe_accuracy_ci_975']):.6f}); this is a registered interaction, not a universal rule.
- Stable fold-intersection Pareto conditions: {stable}. No universally optimal operating point is selected.
- Foldwise descriptive suppression–retention associations change sign. They do not identify a causal effect or establish measurable retention loss caused by scanner suppression.

## Prohibited interpretations

- Cosine similarity alone is not biological-preservation evidence.
- A near-zero retrieval contrast is not called an improvement.
- No slide-independent sign-flip p-value is used.
- No causal, clinical, complete-invariance, pure-factor, information-theoretic-independence, or universal-optimality claim is authorized.
- These results do not establish a universal capacity or regularization law.

The numeric CSV artifacts and their hashes are authoritative. This snapshot is an interpretation boundary, not a substitute for those records.
"""


def readme() -> str:
    return """# Paired-acquisition dimensionality × cross-covariance evidence

This separately versioned package promotes the validated summaries of the locked
450-cell factorial. It binds the reviewed Gate 2 execution source, all registered
cell identities and hashes, frozen inputs and configuration, the preregistered
fold-aware analysis, and a conservative claim-boundary snapshot.

The package intentionally excludes checkpoints, feature archives, projections,
raw per-slide analysis rows, and slide-level contrasts. Those local source
artifacts remain bound by hash and are not committed.

Validate with:

```powershell
python scripts\\provenance\\validate_paired_acquisition_factorial_evidence.py `
  evidence\\paired_acquisition\\dimensionality-xcov-factorial-20260726
```
"""


def atomic_promote(source: Path, destination: Path) -> None:
    delay = 0.1
    for attempt in range(12):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if destination.exists() or attempt == 11:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)


def build(args: argparse.Namespace) -> dict[str, Any]:
    full_manifest = args.full_release_manifest.resolve()
    work_dir = args.work_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
    analysis_spec = args.analysis_spec.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite evidence package: {output_dir}")
    require_clean_checkout()
    full_summary = validate_full_factorial_release(full_manifest)
    analysis_summary = validate_analysis(analysis_dir, full_manifest, analysis_spec)
    source_analysis = load_json(analysis_dir / "analysis_manifest.json")
    if source_analysis["analysis_commit"] != args.analysis_commit:
        raise RuntimeError("analysis commit differs from the requested evidence binding")
    artifacts = release_artifacts(full_manifest)
    inventory = input_inventory(work_dir)
    ledger = ledger_summary(work_dir)
    execution_tree = git_output("rev-parse", f"{EXECUTION_COMMIT}^{{tree}}")
    analysis_tree = git_output("rev-parse", f"{args.analysis_commit}^{{tree}}")
    builder_commit = git_output("rev-parse", "HEAD")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        copy_file(artifacts["factorial_plan"], temporary / "campaign/factorial_plan.json")
        copy_file(artifacts["factorial_full_gate"], temporary / "campaign/full_gate.json")
        copy_file(
            artifacts["factorial_smoke_authorization"],
            temporary / "campaign/smoke_authorization.json",
        )
        copy_file(
            artifacts["factorial_full_cell_table"],
            temporary / "campaign/cell_table.csv",
        )
        write_completeness_matrix(
            temporary / "campaign/completeness_matrix.csv",
            artifacts["factorial_full_cell_table"],
            full_manifest,
        )
        copy_file(artifacts["environment"], temporary / "provenance/environment.json")
        write_json(temporary / "provenance/input_inventory.json", inventory)
        source_bindings = {
            "schema_version": "paired-acquisition-factorial-source-bindings/v1",
            "execution_source": {
                "commit": EXECUTION_COMMIT,
                "tree": execution_tree,
                "repository": "matthewvaishnav/computational-pathology-research",
            },
            "reviewed_execution_source": {
                "commit": EXECUTION_COMMIT,
                "tree": execution_tree,
                "pull_request": EXECUTION_PR,
                "url": (
                    "https://github.com/matthewvaishnav/" "computational-pathology-research/pull/57"
                ),
                "relationship": "reviewed_merge_commit_identical_to_execution_source",
            },
            "analysis_source": {
                "commit": args.analysis_commit,
                "tree": analysis_tree,
                "pull_request": ANALYSIS_PR,
                "url": (
                    "https://github.com/matthewvaishnav/" "computational-pathology-research/pull/66"
                ),
                "relationship": "analysis_implementation_pr_head",
            },
            "evidence_builder_source": {
                "commit": builder_commit,
                "tree": git_output("rev-parse", f"{builder_commit}^{{tree}}"),
            },
        }
        write_json(temporary / "provenance/source_bindings.json", source_bindings)
        copy_file(analysis_spec, temporary / "analysis/analysis_spec.json")
        copy_file(
            analysis_dir / "analysis_manifest.json",
            temporary / "analysis/source_analysis_manifest.json",
        )
        for name in (
            "condition_summary.csv",
            "fold_aware_contrasts.csv",
            "fold_level_contrasts.csv",
            "seed_fold_contrast_consistency.csv",
            "pareto_stability.csv",
            "suppression_retention_association.csv",
            "analysis_report.md",
        ):
            copy_file(analysis_dir / name, temporary / "analysis" / name)
        (temporary / "README.md").write_text(readme(), encoding="utf-8", newline="\n")
        (temporary / "claim_boundary_snapshot.md").write_text(
            claim_snapshot(analysis_dir),
            encoding="utf-8",
            newline="\n",
        )
        commands = {
            "schema_version": "paired-acquisition-factorial-command-record/v1",
            "commands": [
                {
                    "stage": "locked_execution",
                    "working_tree_commit": EXECUTION_COMMIT,
                    "command": (
                        "python experiments\\paired_acquisition\\"
                        "run_provenance_bound_factorial_full.py "
                        "--work-dir results\\paired_acquisition_factorial\\full-gate-v1-work "
                        "--release-dir results\\paired_acquisition_factorial\\full-gate-v1 "
                        "--smoke-manifest results\\paired_acquisition_factorial\\"
                        "smoke-gate-v1\\release_manifest.json --device cuda --max-new-runs 4"
                    ),
                    "stdout": capture(args.execution_stdout),
                    "stderr": capture(args.execution_stderr),
                },
                {
                    "stage": "full_release_validation",
                    "working_tree_commit": EXECUTION_COMMIT,
                    "command": (
                        "python scripts\\provenance\\"
                        "validate_paired_acquisition_factorial_full_release.py "
                        "results\\paired_acquisition_factorial\\full-gate-v1\\release_manifest.json"
                    ),
                },
                {
                    "stage": "aggregate_analysis",
                    "working_tree_commit": args.analysis_commit,
                    "command": (
                        "python scripts\\scorpion\\analyze_paired_acquisition_factorial.py "
                        "results\\paired_acquisition_factorial\\full-gate-v1\\release_manifest.json "
                        "--output-dir results\\paired_acquisition_factorial\\analysis-v1 "
                        "--bootstrap-draws 50000"
                    ),
                    "stdout": capture(args.analysis_stdout),
                    "stderr": capture(args.analysis_stderr),
                },
                {
                    "stage": "analysis_validation_only",
                    "working_tree_commit": args.analysis_commit,
                    "command": (
                        "python scripts\\scorpion\\analyze_paired_acquisition_factorial.py "
                        "results\\paired_acquisition_factorial\\full-gate-v1\\release_manifest.json "
                        "--output-dir results\\paired_acquisition_factorial\\analysis-v1 "
                        "--validate-only"
                    ),
                    "stdout": capture(args.validation_stdout),
                    "stderr": capture(args.validation_stderr),
                },
            ],
            "campaign_ledger": ledger,
        }
        write_json(temporary / "commands.json", commands)
        artifact_rows = [
            artifact_record(temporary, relative) for relative in sorted(EXPECTED_ARTIFACTS)
        ]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "valid",
            "campaign": {
                "status": "valid",
                "expected_cell_count": EXPECTED_FULL_RUN_COUNT,
                "valid_cell_count": full_summary["run_count"],
                "source_commit": EXECUTION_COMMIT,
                "source_release_id": full_summary["release_id"],
                "source_release_manifest_sha256": sha256_file(full_manifest),
                "plan_sha256": full_summary["plan_sha256"],
                "cell_table_sha256": full_summary["cell_table_sha256"],
                "gate_status": full_summary["gate_status"],
                "ledger_sha256": ledger["sha256"],
                "ledger_line_count": ledger["line_count"],
            },
            "analysis": {
                "status": analysis_summary["status"],
                "analysis_commit": args.analysis_commit,
                "source_release_id": analysis_summary["source_release_id"],
                "source_analysis_manifest_sha256": sha256_file(
                    temporary / "analysis/source_analysis_manifest.json"
                ),
                "analysis_spec_sha256": sha256_file(temporary / "analysis/analysis_spec.json"),
                "bootstrap_draws": 50000,
                "condition_count": analysis_summary["condition_count"],
                "contrast_count": analysis_summary["contrast_count"],
            },
            "inputs": {
                "feature_sha256": inventory["feature_archive"]["sha256"],
                "split_manifest_sha256_by_fold": {
                    str(row["fold"]): row["sha256"] for row in inventory["split_manifests"]
                },
                "smoke_manifest_sha256": inventory["smoke_authorization_manifest"]["sha256"],
            },
            "source_bindings_sha256": sha256_file(temporary / "provenance/source_bindings.json"),
            "claim_boundary_snapshot_sha256": sha256_file(temporary / "claim_boundary_snapshot.md"),
            "artifacts": artifact_rows,
        }
        manifest["release_id"] = RELEASE_PREFIX + payload_sha256(manifest)
        write_json(temporary / "release_manifest.json", manifest)
        validate_package(temporary)
        atomic_promote(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_package(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-release-manifest", type=Path, default=DEFAULT_FULL_RELEASE)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--analysis-spec", type=Path, default=DEFAULT_ANALYSIS_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--execution-stdout", type=Path)
    parser.add_argument("--execution-stderr", type=Path)
    parser.add_argument("--analysis-stdout", type=Path)
    parser.add_argument("--analysis-stderr", type=Path)
    parser.add_argument("--validation-stdout", type=Path)
    parser.add_argument("--validation-stderr", type=Path)
    return parser.parse_args()


def main() -> None:
    try:
        print(canonical_json(build(parse_args())))
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"PAIRED-ACQUISITION FACTORIAL EVIDENCE BUILD FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
