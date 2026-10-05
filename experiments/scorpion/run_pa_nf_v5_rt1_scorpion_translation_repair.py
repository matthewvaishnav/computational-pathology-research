#!/usr/bin/env python3
"""Run frozen PA-NF v5 RT1 after pre-outcome semantic manifest repair.

No scientific design element is changed. The wrapper replaces byte-identical CSV
acceptance with the frozen semantic identity rule recorded in
pa_nf_v5_rt1_preoutcome_protocol_repair_20261005.json, then delegates all model
training, evaluation, statistics and gates to the original frozen RT1 runner.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion.pa_nf_v5_rt1_manifest_binding import validate_manifest_semantics
from experiments.scorpion.run_pathoalign_projection import load_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-features",
        type=Path,
        default=Path("results/scorpion/features/fold_0_dinov2_base.npz"),
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("data/scorpion/splits"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/pa_nf_v5_rt1_scorpion_translation_20261005"),
    )
    parser.add_argument(
        "--repair-record",
        type=Path,
        default=Path("experiments/scorpion/pa_nf_v5_rt1_preoutcome_protocol_repair_20261005.json"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output_root.exists():
        raise rt1.RT1Error(f"Output root already exists: {args.output_root}")
    if not args.repair_record.is_file():
        raise rt1.RT1Error(f"Missing pre-outcome protocol repair record: {args.repair_record}")
    repair = json.loads(args.repair_record.read_text(encoding="utf-8"))
    if repair.get("status") != "frozen_before_any_rt1_scientific_outcomes":
        raise rt1.RT1Error("Protocol repair record is not frozen pre-outcome")
    if rt1.sha256_file(args.base_features) != rt1.EXPECTED_FEATURE_SHA256:
        raise rt1.RT1Error("Base DINOv2 feature archive SHA-256 does not match frozen RT1 input")

    base_features, base_frame, _ = load_archive(args.base_features)
    archival_hashes = dict(rt1.EXPECTED_MANIFEST_SHA256)
    runtime_hashes: dict[int, str] = {}
    semantic_reports = []
    for fold in rt1.FOLDS:
        manifest_path = args.manifests_dir / f"fold_{fold}_manifest.csv"
        if not manifest_path.is_file():
            raise rt1.RT1Error(f"Missing fold manifest: {manifest_path}")
        runtime_hashes[fold] = rt1.sha256_file(manifest_path)
        try:
            _, _, report = validate_manifest_semantics(
                base_features, base_frame, manifest_path, fold
            )
        except Exception as exc:
            raise rt1.RT1Error(f"Fold {fold} semantic manifest identity failed: {exc}") from exc
        report["runtime_sha256"] = runtime_hashes[fold]
        report["archival_sha256"] = archival_hashes[fold]
        report["raw_hash_matches_archival"] = runtime_hashes[fold] == archival_hashes[fold]
        semantic_reports.append(report)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise rt1.RT1Error("CUDA requested but unavailable")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise rt1.RT1Error(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python for CUDA reproducibility"
        )

    # Delegate all scientific computation unchanged after semantic input identity.
    rt1.EXPECTED_MANIFEST_SHA256 = dict(runtime_hashes)
    try:
        result = rt1.run_experiment(
            args.base_features,
            args.manifests_dir,
            args.output_root,
            device,
            rt1.RT1Config(),
        )
    finally:
        rt1.EXPECTED_MANIFEST_SHA256 = archival_hashes

    binding = {
        "schema_version": "pa-nf-v5-rt1-runtime-manifest-binding/v1",
        "repair_record": str(args.repair_record),
        "archival_manifest_sha256": {str(k): v for k, v in archival_hashes.items()},
        "runtime_manifest_sha256": {str(k): v for k, v in runtime_hashes.items()},
        "semantic_reports": semantic_reports,
        "scientific_design_changed": False,
        "result_sha256": result.get("result_sha256"),
    }
    (args.output_root / "preoutcome_runtime_manifest_binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RT1 PRE-OUTCOME MANIFEST REPAIR RECORD WRITTEN")
    print(f"Manifest binding: {(args.output_root / 'preoutcome_runtime_manifest_binding.json').resolve()}")


if __name__ == "__main__":
    main()
