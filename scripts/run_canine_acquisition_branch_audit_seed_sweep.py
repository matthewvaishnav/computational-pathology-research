#!/usr/bin/env python3
"""Audit PathoAlign biological vs acquisition branches on canine SCC.

This runner checks the stronger separation claim:

    biological branch: scanner identity should be suppressed while biology remains recoverable
    acquisition branch: scanner identity should remain recoverable

It uses the existing NPZ exporter and Identity Audit CLI.
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
    parser = argparse.ArgumentParser(description="Run canine biological/acquisition branch identity-audit sweep.")
    parser.add_argument("--out", type=Path, default=Path("tmp/canine_acquisition_branch_audit_seed_sweep"))
    parser.add_argument("--seeds", default="911,912,913,914,915")
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--block-column", default="sample_id")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[branch-audit]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def audit_npz(
    args: argparse.Namespace,
    name: str,
    npz: Path,
    feature_key: str,
) -> Path:
    run_dir = args.out / name
    features = run_dir / "features.csv"
    metadata = run_dir / "metadata.csv"
    summary = run_dir / "audit" / "identity_audit_summary.json"
    if args.skip_existing and summary.exists():
        print(f"[branch-audit] reusing {summary}")
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
        "--feature-key",
        feature_key,
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


def load_metrics(name: str, representation: str, seed: int | None, summary_path: Path) -> dict[str, Any]:
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
        "representation": representation,
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
    lines.append("# Canine PathoAlign biological vs acquisition branch audit")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This audit tests whether PathoAlign separates acquisition identity from biological identity rather than merely deleting information.")
    lines.append("")
    lines.append("Expected pattern:")
    lines.append("")
    lines.append("- PathoAlign biological branch: lower scanner probe, high biological retrieval, non-collapsed representation.")
    lines.append("- PathoAlign acquisition branch: scanner identity remains recoverable.")
    lines.append("")
    lines.append("## Group summary")
    lines.append("")
    lines.append("| Representation | Runs | Scanner probe ↓/↑ | Random probe | Region R@1 | Sample R@1 | Cross-scanner cosine | Effective rank | Zero-var frac |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in grouped.iterrows():
        lines.append(
            "| {rep} | {n} | {probe} | {rand} | {region} | {sample} | {cosine} | {rank} | {zvf} |".format(
                rep=r["representation"],
                n=int(r["n_runs"]),
                probe=format_float(r["scanner_probe_accuracy_mean"]),
                rand=format_float(r["random_label_probe_accuracy_mean"]),
                region=format_float(r["region_top1_retrieval_mean"]),
                sample=format_float(r["sample_top1_retrieval_mean"]),
                cosine=format_float(r["region_cross_scanner_cosine_mean"]),
                rank=format_float(r["effective_rank_mean"], 3),
                zvf=format_float(r["zero_variance_dimension_fraction_mean"]),
            )
        )
    lines.append("")
    lines.append("## Per-run summary")
    lines.append("")
    lines.append("| Run | Representation | Seed | Scanner probe | Random probe | Region R@1 | Sample R@1 | Cross-scanner cosine | Effective rank | Zero-var frac |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in per_run.iterrows():
        seed = "" if pd.isna(r["seed"]) else str(int(r["seed"]))
        lines.append(
            "| {name} | {rep} | {seed} | {probe} | {rand} | {region} | {sample} | {cosine} | {rank} | {zvf} |".format(
                name=r["name"],
                rep=r["representation"],
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
    lines.append("This audit evaluates representation identity and branch separation on an external paired-scanner canine SCC dataset. It is not clinical diagnostic validation.")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    rows = []
    raw_summary = audit_npz(args, "raw_dinov2_features", RAW_DINOV2, "features")
    rows.append(load_metrics("raw_dinov2_features", "raw_dinov2_features", None, raw_summary))

    for seed in seeds:
        npz = RUN_ROOT / f"pathoalign_dep20_seed_{seed}" / "projected_features.npz"
        bio_summary = audit_npz(args, f"pathoalign_seed_{seed}_biological_features", npz, "features")
        acq_summary = audit_npz(args, f"pathoalign_seed_{seed}_acquisition_features", npz, "acquisition_features")
        rows.append(load_metrics(f"pathoalign_seed_{seed}_biological_features", "pathoalign_biological_features", seed, bio_summary))
        rows.append(load_metrics(f"pathoalign_seed_{seed}_acquisition_features", "pathoalign_acquisition_features", seed, acq_summary))

    per_run = pd.DataFrame(rows)
    per_run_path = args.out / "canine_branch_audit_per_run.csv"
    per_run.to_csv(per_run_path, index=False)

    grouped = (
        per_run.groupby("representation", dropna=False)
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
    order = {
        "raw_dinov2_features": 0,
        "pathoalign_biological_features": 1,
        "pathoalign_acquisition_features": 2,
    }
    grouped["_order"] = grouped["representation"].map(order).fillna(99)
    grouped = grouped.sort_values("_order").drop(columns=["_order"])
    grouped_path = args.out / "canine_branch_audit_grouped.csv"
    grouped.to_csv(grouped_path, index=False)

    md_path = args.out / "canine_branch_audit_summary.md"
    write_markdown(per_run, grouped, md_path)

    print(f"[branch-audit] wrote {per_run_path}")
    print(f"[branch-audit] wrote {grouped_path}")
    print(f"[branch-audit] wrote {md_path}")
    print("\nGROUP SUMMARY")
    print(grouped.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
