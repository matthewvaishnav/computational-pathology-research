#!/usr/bin/env python3
"""Execute and assemble the complete provenance-bound factorial smoke grid.

Gate 1 is deliberately all-or-nothing. This orchestrator launches the existing
real one-cell producer for every preregistered dimension × cross-covariance cell,
validates each release immediately, and publishes one aggregate release only
after all 18 cells are complete, finite, non-colliding, and provenance-valid.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paired_acquisition_factorial import (  # noqa: E402
    EXPECTED_SMOKE_CELL_COUNT,
    assemble_smoke_release,
    expected_smoke_cells,
    inspect_single_cell_release,
)
from src.paired_acquisition_provenance import ProvenanceValidationError  # noqa: E402
from src.paired_acquisition_release_writer import current_git_commit  # noqa: E402

CELL_PRODUCER = (
    REPO_ROOT / "experiments" / "paired_acquisition" / "run_provenance_bound_bottleneck_cell.py"
)
DEFAULT_FEATURE_PATH = Path(
    "results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz"
)
DEFAULT_MANIFESTS_DIR = Path("data/external_multiscanner_caninescc/patch_manifests/splits")


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


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
        raise ProvenanceValidationError(
            "unable to verify the factorial producer working tree"
        ) from exc
    if result.stdout.strip():
        changed = result.stdout.strip().splitlines()
        preview = ", ".join(changed[:5])
        raise ProvenanceValidationError(
            "factorial producer checkout is not clean; commit or remove changes before running: "
            + preview
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--manifests-dir", type=Path, default=DEFAULT_MANIFESTS_DIR)
    parser.add_argument(
        "--code-commit",
        help="Optional assertion for HEAD; the orchestrator refuses a different commit",
    )
    return parser.parse_args()


def cell_command(
    *,
    cell: dict[str, object],
    release_dir: Path,
    args: argparse.Namespace,
    code_commit: str,
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
        str(cell["epochs"]),
        "--region-batch-size",
        "32",
        "--learning-rate",
        "0.0003",
        "--weight-decay",
        "0.0001",
        "--device",
        str(args.device),
        "--feature-path",
        str(resolve_repo_path(args.feature_path)),
        "--manifests-dir",
        str(resolve_repo_path(args.manifests_dir)),
        "--code-commit",
        code_commit,
    ]


def main() -> None:
    args = parse_args()
    release_dir = resolve_repo_path(args.release_dir)
    if release_dir.exists():
        raise ProvenanceValidationError(
            f"refusing to overwrite existing output path: {release_dir}"
        )

    require_clean_checkout()
    code_commit = current_git_commit(REPO_ROOT)
    if args.code_commit is not None and args.code_commit != code_commit:
        raise ProvenanceValidationError(
            f"--code-commit {args.code_commit} does not match checked-out HEAD {code_commit}"
        )

    cells = expected_smoke_cells()
    if len(cells) != EXPECTED_SMOKE_CELL_COUNT:
        raise ProvenanceValidationError("frozen smoke-cell enumeration is incomplete")

    with tempfile.TemporaryDirectory(prefix="paired-acquisition-factorial-smoke-") as temporary:
        work_root = Path(temporary)
        cell_release_dirs = []
        observed_keys = []
        for index, cell in enumerate(cells, start=1):
            key = str(cell["cell_key"])
            cell_release = work_root / key
            print(f"[{index}/{EXPECTED_SMOKE_CELL_COUNT}] executing {key}", flush=True)
            command = cell_command(
                cell=cell,
                release_dir=cell_release,
                args=args,
                code_commit=code_commit,
            )
            try:
                subprocess.run(command, cwd=REPO_ROOT, check=True)
            except subprocess.CalledProcessError as exc:
                raise ProvenanceValidationError(
                    f"factorial smoke cell failed: {key} (exit {exc.returncode})"
                ) from exc
            inspected = inspect_single_cell_release(cell_release)
            if inspected["cell_key"] != key:
                raise ProvenanceValidationError(
                    f"factorial producer returned the wrong cell: expected {key}, "
                    f"got {inspected['cell_key']}"
                )
            cell_release_dirs.append(cell_release)
            observed_keys.append(key)

        if observed_keys != [str(cell["cell_key"]) for cell in cells]:
            raise ProvenanceValidationError("factorial smoke execution order changed")
        require_clean_checkout()
        if current_git_commit(REPO_ROOT) != code_commit:
            raise ProvenanceValidationError(
                "factorial producer HEAD changed while Gate 1 was running"
            )
        summary = assemble_smoke_release(cell_release_dirs, release_dir)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, ProvenanceValidationError) as exc:
        print(f"PAIRED-ACQUISITION FACTORIAL SMOKE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
