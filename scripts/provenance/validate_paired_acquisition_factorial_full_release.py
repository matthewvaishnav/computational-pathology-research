#!/usr/bin/env python3
"""Validate a completed provenance-bound paired-acquisition Gate 2 release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paired_acquisition_factorial_full import (  # noqa: E402
    validate_full_factorial_release,
)
from src.paired_acquisition_provenance import ProvenanceValidationError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    summary = validate_full_factorial_release(args.manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, ProvenanceValidationError) as exc:
        print(f"PAIRED-ACQUISITION FACTORIAL FULL VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
