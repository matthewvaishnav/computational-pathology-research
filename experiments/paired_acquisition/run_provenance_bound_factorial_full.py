#!/usr/bin/env python3
"""Execute the locked 450-run paired-acquisition factorial with resumable state."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paired_acquisition_factorial import (  # noqa: E402
    EXPECTED_FULL_RUN_COUNT,
    FULL_EPOCHS,
    factorial_plan,
)
from src.paired_acquisition_factorial_full import (  # noqa: E402
    assemble_full_release,
    expected_full_cells,
    inspect_full_cell_release,
    smoke_authorization,
)
from src.paired_acquisition_provenance import (  # noqa: E402
    ProvenanceValidationError,
    payload_sha256,
    sha256_file,
)
from src.paired_acquisition_release_writer import current_git_commit  # noqa: E402

CELL_PRODUCER = (
    REPO_ROOT / "experiments" / "paired_acquisition" / "run_provenance_bound_bottleneck_cell.py"
)
DEFAULT_FEATURE_PATH = Path(
    "results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz"
)
DEFAULT_MANIFESTS_DIR = Path("data/external_multiscanner_caninescc/patch_manifests/splits")
STATE_SCHEMA_VERSION = "paired-acquisition-factorial-full-state/v1"
LEDGER_SCHEMA_VERSION = "paired-acquisition-factorial-full-ledger/v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def require_clean_checkout() -> None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ProvenanceValidationError("unable to verify the Gate 2 working tree") from exc
    if result.stdout.strip():
        changed = result.stdout.strip().splitlines()
        raise ProvenanceValidationError(
            "Gate 2 checkout is not clean; commit or remove changes before running: "
            + ", ".join(changed[:5])
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--smoke-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--manifests-dir", type=Path, default=DEFAULT_MANIFESTS_DIR)
    parser.add_argument(
        "--max-new-runs",
        type=int,
        help="Operational batch limit; rerun the identical command to resume the frozen grid",
    )
    parser.add_argument(
        "--code-commit",
        help="Optional assertion for HEAD; the orchestrator refuses a different commit",
    )
    args = parser.parse_args()
    if args.max_new_runs is not None and args.max_new_runs <= 0:
        parser.error("--max-new-runs must be positive")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvenanceValidationError(f"invalid Gate 2 state JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProvenanceValidationError(f"Gate 2 state must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_ledger(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _attempt_number(ledger_path: Path, key: str) -> int:
    if not ledger_path.is_file():
        return 1
    count = 0
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProvenanceValidationError("Gate 2 execution ledger is corrupted") from exc
        if event.get("event") == "attempt_started" and event.get("cell_key") == key:
            count += 1
    return count + 1


def expected_state(
    *,
    args: argparse.Namespace,
    code_commit: str,
    feature_path: Path,
    manifests_dir: Path,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    if not feature_path.is_file():
        raise ProvenanceValidationError(f"missing real feature archive: {feature_path}")
    split_hashes: dict[str, str] = {}
    for fold in range(5):
        manifest = manifests_dir / f"fold_{fold}_patch_manifest.csv"
        if not manifest.is_file():
            raise ProvenanceValidationError(f"missing real split manifest: {manifest}")
        split_hashes[str(fold)] = sha256_file(manifest)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "code_commit": code_commit,
        "frozen_plan_sha256": payload_sha256(factorial_plan()),
        "expected_run_count": EXPECTED_FULL_RUN_COUNT,
        "epochs": FULL_EPOCHS,
        "device": str(args.device),
        "feature_path": display_path(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "manifests_dir": display_path(manifests_dir),
        "split_manifest_sha256_by_fold": split_hashes,
        "smoke_release_id": authorization["smoke_release_id"],
        "smoke_manifest_sha256": authorization["source_manifest_sha256"],
        "claim_boundary": (
            "This state authorizes only the locked Gate 2 execution; changing it "
            "requires a new preregistration."
        ),
    }


def establish_state(path: Path, expected: Mapping[str, Any]) -> None:
    if path.exists():
        observed = _read_json(path)
        if observed != dict(expected):
            raise ProvenanceValidationError(
                "Gate 2 execution state differs from this invocation; do not retune "
                "or change inputs"
            )
        return
    _atomic_json(path, expected)


def cell_command(
    *,
    cell: Mapping[str, Any],
    release_dir: Path,
    args: argparse.Namespace,
    code_commit: str,
    feature_path: Path,
    manifests_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(CELL_PRODUCER),
        "--release-dir",
        str(release_dir),
        "--acquisition-dim",
        str(cell["acquisition_dim"]),
        "--cross-covariance-weight",
        str(cell["cross_covariance_weight"]),
        "--fold",
        str(cell["fold"]),
        "--seed",
        str(cell["seed"]),
        "--epochs",
        str(FULL_EPOCHS),
        "--region-batch-size",
        "32",
        "--learning-rate",
        "0.0003",
        "--weight-decay",
        "0.0001",
        "--device",
        str(args.device),
        "--feature-path",
        str(feature_path),
        "--manifests-dir",
        str(manifests_dir),
        "--code-commit",
        code_commit,
    ]


def _verify_existing(
    *,
    release_dir: Path,
    cell: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    inspected = inspect_full_cell_release(release_dir)
    key = str(cell["cell_key"])
    if inspected["cell_key"] != key:
        raise ProvenanceValidationError(
            f"existing cell directory contains {inspected['cell_key']}, expected {key}"
        )
    if inspected["code_commit"] != state["code_commit"]:
        raise ProvenanceValidationError(f"existing cell {key} was produced by another commit")
    if inspected["dataset_source_sha256"] != state["feature_sha256"]:
        raise ProvenanceValidationError(f"existing cell {key} used another feature archive")
    expected_split = state["split_manifest_sha256_by_fold"][str(cell["fold"])]
    if inspected["split_manifest_sha256"] != expected_split:
        raise ProvenanceValidationError(f"existing cell {key} used another split manifest")
    return inspected


def _run_cell(
    *,
    command: list[str],
    key: str,
    attempt: int,
    log_path: Path,
    ledger_path: Path,
) -> None:
    event_base = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "cell_key": key,
        "attempt": attempt,
    }
    _append_ledger(
        ledger_path,
        {
            **event_base,
            "event": "attempt_started",
            "timestamp": utc_now(),
            "command": command,
            "log_path": str(log_path),
        },
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise ProvenanceValidationError(f"unable to capture producer output for {key}")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    _append_ledger(
        ledger_path,
        {
            **event_base,
            "event": "attempt_finished" if return_code == 0 else "attempt_failed",
            "timestamp": utc_now(),
            "return_code": return_code,
            "log_path": str(log_path),
        },
    )
    if return_code != 0:
        raise ProvenanceValidationError(
            f"full-factorial cell failed: {key} (exit {return_code}); retained log: {log_path}"
        )


def main() -> None:
    args = parse_args()
    release_dir = resolve_repo_path(args.release_dir)
    work_dir = resolve_repo_path(args.work_dir)
    smoke_manifest = resolve_repo_path(args.smoke_manifest)
    feature_path = resolve_repo_path(args.feature_path)
    manifests_dir = resolve_repo_path(args.manifests_dir)
    if release_dir.exists():
        raise ProvenanceValidationError(
            f"refusing to overwrite existing output path: {release_dir}"
        )
    if (
        release_dir == work_dir
        or release_dir in work_dir.parents
        or work_dir in release_dir.parents
    ):
        raise ProvenanceValidationError(
            "--work-dir and --release-dir must be separate sibling paths"
        )

    require_clean_checkout()
    code_commit = current_git_commit(REPO_ROOT)
    if args.code_commit is not None and args.code_commit != code_commit:
        raise ProvenanceValidationError(
            f"--code-commit {args.code_commit} does not match checked-out HEAD {code_commit}"
        )
    authorization = smoke_authorization(smoke_manifest)
    state = expected_state(
        args=args,
        code_commit=code_commit,
        feature_path=feature_path,
        manifests_dir=manifests_dir,
        authorization=authorization,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "execution_state.json"
    ledger_path = work_dir / "execution_ledger.jsonl"
    establish_state(state_path, state)

    cells = expected_full_cells()
    if len(cells) != EXPECTED_FULL_RUN_COUNT:
        raise ProvenanceValidationError("frozen full-factorial enumeration is incomplete")
    cell_root = work_dir / "cells"
    logs_root = work_dir / "attempt_logs"
    completed = 0
    new_runs = 0
    cell_release_dirs: list[Path] = []
    for index, cell in enumerate(cells, start=1):
        key = str(cell["cell_key"])
        cell_release = cell_root / key
        if cell_release.exists():
            _verify_existing(release_dir=cell_release, cell=cell, state=state)
            completed += 1
            cell_release_dirs.append(cell_release)
            print(f"[{index}/{EXPECTED_FULL_RUN_COUNT}] validated existing {key}", flush=True)
            continue
        if args.max_new_runs is not None and new_runs >= args.max_new_runs:
            print(
                json.dumps(
                    {
                        "status": "incomplete",
                        "completed_run_count": completed,
                        "expected_run_count": EXPECTED_FULL_RUN_COUNT,
                        "work_dir": str(work_dir),
                        "next_cell": key,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        attempt = _attempt_number(ledger_path, key)
        log_path = logs_root / key / f"attempt-{attempt:03d}.log"
        print(
            f"[{index}/{EXPECTED_FULL_RUN_COUNT}] executing {key} attempt {attempt}",
            flush=True,
        )
        command = cell_command(
            cell=cell,
            release_dir=cell_release,
            args=args,
            code_commit=code_commit,
            feature_path=feature_path,
            manifests_dir=manifests_dir,
        )
        _run_cell(
            command=command,
            key=key,
            attempt=attempt,
            log_path=log_path,
            ledger_path=ledger_path,
        )
        inspected = _verify_existing(release_dir=cell_release, cell=cell, state=state)
        _append_ledger(
            ledger_path,
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "event": "cell_validated",
                "timestamp": utc_now(),
                "cell_key": key,
                "attempt": attempt,
                "run_id": inspected["run_id"],
                "record_sha256": inspected["record_sha256"],
            },
        )
        completed += 1
        new_runs += 1
        cell_release_dirs.append(cell_release)

    require_clean_checkout()
    if current_git_commit(REPO_ROOT) != code_commit:
        raise ProvenanceValidationError("Gate 2 producer HEAD changed during execution")
    summary = assemble_full_release(cell_release_dirs, release_dir, smoke_manifest)
    _append_ledger(
        ledger_path,
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "event": "aggregate_release_published",
            "timestamp": utc_now(),
            "release_id": summary["release_id"],
            "manifest_path": summary["manifest_path"],
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, ProvenanceValidationError) as exc:
        print(f"PAIRED-ACQUISITION FACTORIAL FULL RUN FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
