#!/usr/bin/env python3
"""Run PathoAlign Identity Audit across canine paired-scanner seeds.

This script turns the one-off fold-0 seed-911 audit into a small benchmark
sweep over paired-reference and PathoAlign projected-feature seeds.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

RAW_DINOV2 = Path("results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz")
MANIFEST = Path("results/external_multiscanner_caninescc/geometry_qualified/geometry_qualified_manifest.csv")
RUN_ROOT = Path("results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold/fold_0/runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canine identity-audit seed sweep.")
    parser.add_argument("--out", type=Path, default=Path("tmp/canine_identity_audit_seed_sweep"))
    parser.add_argument("--seeds", default="911,912,913,914,915")
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--block-column", default="sample_id")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[sweep]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def audit_one(args: argparse.Namespace, name: str, npz: Path) -> Path:
    run_dir = args.out / name
    features = run_dir / "features.csv"
    metadata = run_dir / "metadata.csv"
    summary = run_dir / "audit" / "identity_audit_summary.json"
    if args.skip_existing and summary.exists():
        print(f"[sweep] reusing {summary}")
        return summary

    if not npz.exists():
        raise FileNotFoundError(npz)
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    run([
        args.python,
        "scripts/export_identity_audit_npz.py",
        "--npz",
        str(npz),
        "--manifest",
        str(MANIFEST),
        "--out",
        str(run_dir),
    ])
    run([
        args.python,
        "scripts/pathoalign_identity_audit.py",
        "--features",
        str(features),
        "--metadata",
        str(metadata),
        "--out",
        str(run_dir / "audit"),
        "--block-column",
        args.block_column,
        "--n-bootstrap",
        str(args.n_bootstrap),
    ])
    return summary


def load_metrics(name: str, method: str, seed: int | None, summary_path: Path) -> dict[str, Any]:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    probe = s["shortcut_probes"][0]
    region = next(x for x in s["biology_retrieval"] if x["column"] == "region_id")
    sample = next(x for x in s["biology_retrieval"] if x["column"] == "sample_id")
    consistency = next(
        x
        for x in s["cross_acquisition_consistency"]
        if x["biology_column"] == "region_id" and x["shortcut_column"] == "scanner_id"
    )
    collapse = s["collapse"]
    return {
        "name": name,
        "method": method,
        "seed": seed,
        "scanner_probe_accuracy": probe["accuracy"],
        "scanner_probe_balanced_accuracy": probe["balanced_accuracy"],
        "scanner_probe_ci_low": probe.get("accuracy_ci_low"),
        "scanner_probe_ci_high": probe.get("accuracy_ci_high"),
        "random_label_probe_accuracy": probe.get("random_label_accuracy"),
        "region_top1_retrieval": region["top1_same_label_retrieval"],
        "region_top1_ci_low": region.get("top1_ci_low"),
        "region_top1_ci_high": region.get("top1_ci_high"),
        "sample_top1_retrieval": sample["top1_same_label_retrieval"],
        "region_cross_scanner_cosine": consistency["mean_cross_acquisition_cosine"],
        "region_cross_scanner_pairs": consistency["n_cross_acquisition_pairs"],
        "effective_rank": collapse["effective_rank"],
        "zero_variance_dimension_fraction": collapse["zero_variance_dimension_fraction"],
        "n_units": collapse["n_units"],
        "n_features": collapse["n_features"],
        "summary_json": str(summary_path),
    }


def format_float(x: Any, digits: int = 6) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def write_markdown(per_run: pd.DataFrame, grouped: pd.DataFrame, out: Path) -> None:
    lines: list[str] = []
    lines.append("# Canine paired-scanner identity-audit seed sweep")
    lines.append("")
    lines.append("## Group summary")
    lines.append("")
    lines.append("| Method | Runs | Scanner probe ↓ | Region R@1 ↑ | Sample R@1 ↑ | Region cross-scanner cosine ↑/≈ | Effective rank |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in grouped.iterrows():
        lines.append(
            "| {method} | {n} | {probe} | {region} | {sample} | {cosine} | {rank} |".format(
                method=r["method"],
                n=int(r["n_runs"]),
                probe=format_float(r["scanner_probe_accuracy_mean"]),
                region=format_float(r["region_top1_retrieval_mean"]),
                sample=format_float(r["sample_top1_retrieval_mean"]),
                cosine=format_float(r["region_cross_scanner_cosine_mean"]),
                rank=format_float(r["effective_rank_mean"], 3),
            )
        )
    lines.append("")
    lines.append("## Per-run summary")
    lines.append("")
    lines.append("| Run | Method | Seed | Scanner probe ↓ | Random probe | Region R@1 ↑ | Sample R@1 ↑ | Cross-scanner cosine | Effective rank | Zero-var frac |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in per_run.iterrows():
        seed = "" if pd.isna(r["seed"]) else str(int(r["seed"]))
        lines.append(
            "| {name} | {method} | {seed} | {probe} | {rand} | {region} | {sample} | {cosine} | {rank} | {zvf} |".format(
                name=r["name"],
                method=r["method"],
                seed=seed,
                probe=format_float(r["scanner_probe_accuracy"]),
                rand=format_float(r["random_label_probe_accuracy"]),
                region=format_float(r["region_top1_retrieval"]),
                sample=format_float(r["sample_top1_retrieval"]),
                cosine=format_float(r["region_cross_scanner_cosine"]),
                rank=format_float(r["effective_rank"], 3),
                zvf=format_float(r["zero_variance_dimension_fraction"]),
            )
        )
    lines.append("")
    lines.append("## Claim boundary")
    lines.append("")
    lines.append("This is a representation-identifiability benchmark result on an external paired-scanner canine SCC dataset. It is not clinical diagnostic validation.")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    rows = []
    raw_summary = audit_one(args, "raw_dinov2", RAW_DINOV2)
    rows.append(load_metrics("raw_dinov2", "raw_dinov2", None, raw_summary))

    for seed in seeds:
        paired_npz = RUN_ROOT / f"paired_reference_seed_{seed}" / "projected_features.npz"
        pathoalign_npz = RUN_ROOT / f"pathoalign_dep20_seed_{seed}" / "projected_features.npz"
        paired_summary = audit_one(args, f"paired_reference_seed_{seed}", paired_npz)
        pathoalign_summary = audit_one(args, f"pathoalign_seed_{seed}", pathoalign_npz)
        rows.append(load_metrics(f"paired_reference_seed_{seed}", "paired_reference", seed, paired_summary))
        rows.append(load_metrics(f"pathoalign_seed_{seed}", "pathoalign", seed, pathoalign_summary))

    per_run = pd.DataFrame(rows)
    per_run_path = args.out / "canine_identity_audit_seed_sweep_per_run.csv"
    per_run.to_csv(per_run_path, index=False)

    grouped = (
        per_run.groupby("method", dropna=False)
        .agg(
            n_runs=("name", "count"),
            scanner_probe_accuracy_mean=("scanner_probe_accuracy", "mean"),
            scanner_probe_accuracy_std=("scanner_probe_accuracy", "std"),
            random_label_probe_accuracy_mean=("random_label_probe_accuracy", "mean"),
            region_top1_retrieval_mean=("region_top1_retrieval", "mean"),
            region_top1_retrieval_std=("region_top1_retrieval", "std"),
            sample_top1_retrieval_mean=("sample_top1_retrieval", "mean"),
            region_cross_scanner_cosine_mean=("region_cross_scanner_cosine", "mean"),
            region_cross_scanner_cosine_std=("region_cross_scanner_cosine", "std"),
            effective_rank_mean=("effective_rank", "mean"),
            effective_rank_std=("effective_rank", "std"),
            zero_variance_dimension_fraction_mean=("zero_variance_dimension_fraction", "mean"),
        )
        .reset_index()
    )
    order = {"raw_dinov2": 0, "paired_reference": 1, "pathoalign": 2}
    grouped["_order"] = grouped["method"].map(order).fillna(99)
    grouped = grouped.sort_values("_order").drop(columns=["_order"])
    grouped_path = args.out / "canine_identity_audit_seed_sweep_grouped.csv"
    grouped.to_csv(grouped_path, index=False)

    md_path = args.out / "canine_identity_audit_seed_sweep_summary.md"
    write_markdown(per_run, grouped, md_path)

    print(f"[sweep] wrote {per_run_path}")
    print(f"[sweep] wrote {grouped_path}")
    print(f"[sweep] wrote {md_path}")
    print("\nGROUP SUMMARY")
    print(grouped.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
