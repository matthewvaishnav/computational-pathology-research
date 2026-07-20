#!/usr/bin/env python3
"""Create or validate the deterministic paired-acquisition provenance smoke release."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paired_acquisition_provenance import (
    ProvenanceValidationError,
    create_smoke_release,
    validate_release,
)

DEFAULT_OUTPUT = Path("benchmarks/paired_acquisition_provenance_release/smoke-v1")


def current_commit() -> str:
    """Resolve the checked-out commit without silently accepting an unknown value."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ProvenanceValidationError("unable to resolve the current Git commit") from exc
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--code-commit", help="Exact producing commit; defaults to HEAD")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the existing release without writing files",
    )
    args = parser.parse_args()

    if args.check:
        summary = validate_release(args.out_dir / "release_manifest.json")
    else:
        summary = create_smoke_release(args.out_dir, args.code_commit or current_commit())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ProvenanceValidationError) as exc:
        print(f"PAIRED-ACQUISITION PROVENANCE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
