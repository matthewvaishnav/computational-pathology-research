#!/usr/bin/env python3
"""Validate a forward paired-acquisition provenance release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paired_acquisition_provenance import ProvenanceValidationError, validate_release

DEFAULT_MANIFEST = Path(
    "benchmarks/paired_acquisition_provenance_release/smoke-v1/release_manifest.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(validate_release(args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ProvenanceValidationError) as exc:
        print(f"PAIRED-ACQUISITION PROVENANCE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
