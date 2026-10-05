#!/usr/bin/env python3
"""Pre-outcome RT1 validator with semantic manifest identity repair."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion.pa_nf_v5_rt1_manifest_binding import validate_manifest_semantics
from experiments.scorpion.run_pathoalign_projection import load_archive
from scripts.scorpion import validate_pa_nf_v5_rt1_scorpion_translation as base_validator


def fail(message: str) -> None:
    raise SystemExit(f"PA-NF V5 RT1 REPAIRED VALIDATION FAILED: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-features",
        type=Path,
        default=REPO_ROOT / "results/scorpion/features/fold_0_dinov2_base.npz",
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=REPO_ROOT / "data/scorpion/splits",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=REPO_ROOT / "experiments/scorpion/pa_nf_v5_rt1_scorpion_translation_spec_20261005.json",
    )
    parser.add_argument(
        "--repair-record",
        type=Path,
        default=REPO_ROOT / "experiments/scorpion/pa_nf_v5_rt1_preoutcome_protocol_repair_20261005.json",
    )
    args = parser.parse_args()

    if not args.repair_record.is_file():
        fail(f"missing pre-outcome repair record: {args.repair_record}")
    repair = json.loads(args.repair_record.read_text(encoding="utf-8"))
    if repair.get("status") != "frozen_before_any_rt1_scientific_outcomes":
        fail("repair record is not frozen in the expected pre-outcome state")

    if not args.base_features.is_file():
        fail(f"missing feature archive: {args.base_features}")
    observed_feature_hash = rt1.sha256_file(args.base_features)
    if observed_feature_hash != rt1.EXPECTED_FEATURE_SHA256:
        fail("base feature archive hash mismatch")

    base_features, base_frame, _ = load_archive(args.base_features)
    archival_hashes = dict(rt1.EXPECTED_MANIFEST_SHA256)
    runtime_hashes: dict[int, str] = {}
    reports = []
    for fold in rt1.FOLDS:
        manifest_path = args.manifests_dir / f"fold_{fold}_manifest.csv"
        if not manifest_path.is_file():
            fail(f"missing fold manifest: {manifest_path}")
        runtime_hashes[fold] = rt1.sha256_file(manifest_path)
        try:
            _, _, report = validate_manifest_semantics(
                base_features, base_frame, manifest_path, fold
            )
        except Exception as exc:
            fail(f"fold {fold} semantic identity failed: {exc}")
        report["runtime_sha256"] = runtime_hashes[fold]
        report["archival_sha256"] = archival_hashes[fold]
        report["raw_hash_matches_archival"] = runtime_hashes[fold] == archival_hashes[fold]
        reports.append(report)

    # The original validator contains the remaining frozen architecture, leakage,
    # optimizer-exclusion, parameter-count, roundtrip and transport-independence
    # checks. Patch only its byte-hash expectation after semantic identity passes.
    rt1.EXPECTED_MANIFEST_SHA256 = dict(runtime_hashes)
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(Path(old_argv[0]).resolve()),
            "--spec", str(args.spec),
            "--base-features", str(args.base_features),
            "--manifests-dir", str(args.manifests_dir),
        ]
        base_validator.main()
    finally:
        sys.argv = old_argv
        rt1.EXPECTED_MANIFEST_SHA256 = archival_hashes

    print("PRE-OUTCOME MANIFEST BINDING REPAIR PASSED")
    print("Archival raw hashes remain provenance references; runtime CSVs matched the frozen scientific split semantically.")
    for report in reports:
        print(
            "fold={fold} raw_match={raw_hash_matches_archival} runtime_sha256={runtime_sha256}".format(**report)
        )
    print(f"CUDA CUBLAS_WORKSPACE_CONFIG currently: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}")
    print("No RT1 scientific outcomes were computed by this repaired validator")


if __name__ == "__main__":
    main()
