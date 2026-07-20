#!/usr/bin/env python3
"""Validate a complete paired-acquisition factorial Gate 1 release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paired_acquisition_factorial import validate_factorial_release  # noqa: E402
from src.paired_acquisition_provenance import ProvenanceValidationError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to release_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_factorial_release(args.manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ProvenanceValidationError) as exc:
        print(f"FACTORIAL RELEASE INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
