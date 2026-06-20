#!/usr/bin/env python3
"""Run the canine PathoAlign Identity Audit across folds and seeds.

This is the paper-grade sweep for the external multi-scanner canine SCC benchmark.
It audits:

- raw DINOv2 features once as the frozen encoder baseline;
- paired-reference projected features across folds and seeds;
- PathoAlign biological projected features across folds and seeds;
- PathoAlign acquisition features across folds and seeds.

The key separation pattern is:

- PathoAlign biological branch: scanner probe low, biological retrieval high;
- PathoAlign acquisition branch: scanner probe high, biological retrieval low.
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
RUN_ROOT = Path("results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canine crossfold identity-audit seed sweep.")
    parser.add_argument("--out", type=Path, default=Path("tmp/canine_identity_audit_crossfold_seed_sweep"))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="911,912,913,914,915")
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--block-column", default="sample_id")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[crossfold-audit]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def audit_npz(
    args: argparse.Namespace,
    name: str,
    npz: Path,
    feature_key: str,
) -> Path | None:
    run_dir = args.out / name
    features = run_dir / "features.csv"
    metadata = run_dir / "metadata.csv"
    summary = run_dir / "audit" / "identity_audit_summary.json"
    if args.skip_existing and summary.exists():
        print(f"[crossfold-audit] reusing {summary}")
        return summary

    if not npz.exists():
        if args.allow_missing:
            print(f"[crossfold-audit] missing, skipping: {npz}")
            return None
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


def load_metrics(
    name: str,
    representation: str,
    fold: int | None,
    seed: int | None,
    summary_path: Path,
) -> dict[str, Any]:
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
        "fold": fold,
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


def add_delta_columns(per_run: pd.DataFrame) -> pd.DataFrame:
    out = per_run.copy()
    raw_rows = out[out["representation"] == "raw_dinov2_features"]
    if raw_rows.empty:
        return out
    raw = raw_rows.iloc[0]
    for metric in [
        "scanner_probe_accuracy",
        "region_top1_retrieval",
        "sample_top1_retrieval",
        "region_cross_scanner_cosine",
        "effective_rank",
    ]:
        out[f"{metric}_delta_vs_raw"] = out[metric] - raw[metric]
    return out


def write_markdown(per_run: pd.DataFrame, grouped: pd.DataFrame, by_fold: pd.DataFrame, out: Path) -> None:
    lines: list[str] = []
    lines.append("# Canine PathoAlign crossfold identity-audit seed sweep")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This audit tests whether the PathoAlign separation pattern holds across folds 0--4 and seeds 911--915 on the external paired-scanner canine SCC benchmark.")
    lines.append("")
    lines.append("Expected pattern:")
    lines.append("")
    lines.append("- PathoAlign biological features: scanner probe low, biological retrieval high, non-collapsed.")
    lines.append("- PathoAlign acquisition features: scanner probe high, biological retrieval low, non-collapsed.")
    lines.append("- Paired-reference features: useful comparator, but not as scanner-suppressed as PathoAlign biological features.")
    lines.append("")
    lines.append("## Group summary")
    lines.append("")
    lines.append("| Representation | Runs | Scanner probe | Random probe | Region R@1 | Sample R@1 | Cross-scanner cosine | Effective rank | Zero-var frac |")
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
    lines.append("## Fold summary")
    lines.append("")
    lines.append("| Fold | Representation | Runs | Scanner probe | Region R@1 | Sample R@1 | Cross-scanner cosine | Effective rank |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for _, r in by_fold.iterrows():
        fold = "" if pd.isna(r["fold"]) else str(int(r["fold"]))
        lines.append(
            "| {fold} | {rep} | {n} | {probe} | {region} | {sample} | {cosine} | {rank} |".format(
                fold=fold,
                rep=r["representation"],
                n=int(r["n_runs"]),
                probe=format_float(r["scanner_probe_accuracy_mean"]),
                region=format_float(r["region_top1_retrieval_mean"]),
                sample=format_float(r["sample_top1_retrieval_mean"]),
                cosine=format_float(r["region_cross_scanner_cosine_mean"]),
                rank=format_float(r["effective_rank_mean"], 3),
            )
        )
    lines.append("")
    lines.append("## Claim boundary")
    lines.append("")
    lines.append("This is a representation-identifiability and branch-separation benchmark result on an external paired-scanner canine SCC dataset. It is not clinical diagnostic validation.")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def grouped_summary(per_run: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        per_run.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("name", "count"),
            scanner_probe_accuracy_mean=("scanner_probe_accuracy", "mean"),
            scanner_probe_accuracy_std=("scanner_probe_accuracy", "std"),
            random_label_probe_accuracy_mean=("random_label_probe_accuracy", "mean"),
            region_top1_retrieval_mean=("region_top1_retrieval", "mean"),
            region_top1_retrieval_std=("region_top1_retrieval", "std"),
            sample_top1_retrieval_mean=("sample_top1_retrieval", "mean"),
            sample_top1_retrieval_std=("sample_top1_retrieval", "std"),
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
        "paired_reference_features": 1,
        "pathoalign_biological_features": 2,
        "pathoalign_acquisition_features": 3,
    }
    grouped["_order"] = grouped["representation"].map(order).fillna(99)
    sort_cols = [c for c in ["fold", "_order"] if c in grouped.columns]
    grouped = grouped.sort_values(sort_cols).drop(columns=["_order"])
    return grouped


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folds = [int(s.strip()) for s in args.folds.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    rows: list[dict[str, Any]] = []
    raw_summary = audit_npz(args, "raw_dinov2_features", RAW_DINOV2, "features")
    if raw_summary is not None:
        rows.append(load_metrics("raw_dinov2_features", "raw_dinov2_features", None, None, raw_summary))

    for fold in folds:
        for seed in seeds:
            fold_root = RUN_ROOT / f"fold_{fold}" / "runs"
            paired_npz = fold_root / f"paired_reference_seed_{seed}" / "projected_features.npz"
            pathoalign_npz = fold_root / f"pathoalign_dep20_seed_{seed}" / "projected_features.npz"

            paired_summary = audit_npz(
                args,
                f"fold_{fold}_paired_reference_seed_{seed}",
                paired_npz,
                "features",
            )
            if paired_summary is not None:
                rows.append(load_metrics(f"fold_{fold}_paired_reference_seed_{seed}", "paired_reference_features", fold, seed, paired_summary))

            bio_summary = audit_npz(
                args,
                f"fold_{fold}_pathoalign_seed_{seed}_biological_features",
                pathoalign_npz,
                "features",
            )
            if bio_summary is not None:
                rows.append(load_metrics(f"fold_{fold}_pathoalign_seed_{seed}_biological_features", "pathoalign_biological_features", fold, seed, bio_summary))

            acq_summary = audit_npz(
                args,
                f"fold_{fold}_pathoalign_seed_{seed}_acquisition_features",
                pathoalign_npz,
                "acquisition_features",
            )
            if acq_summary is not None:
                rows.append(load_metrics(f"fold_{fold}_pathoalign_seed_{seed}_acquisition_features", "pathoalign_acquisition_features", fold, seed, acq_summary))

    per_run = pd.DataFrame(rows)
    if per_run.empty:
        raise RuntimeError("No audit runs completed.")
    per_run = add_delta_columns(per_run)

    per_run_path = args.out / "canine_identity_audit_crossfold_per_run.csv"
    per_run.to_csv(per_run_path, index=False)

    grouped = grouped_summary(per_run, ["representation"])
    grouped_path = args.out / "canine_identity_audit_crossfold_grouped.csv"
    grouped.to_csv(grouped_path, index=False)

    by_fold = grouped_summary(per_run[per_run["fold"].notna()].copy(), ["fold", "representation"])
    by_fold_path = args.out / "canine_identity_audit_crossfold_by_fold.csv"
    by_fold.to_csv(by_fold_path, index=False)

    md_path = args.out / "canine_identity_audit_crossfold_summary.md"
    write_markdown(per_run, grouped, by_fold, md_path)

    print(f"[crossfold-audit] wrote {per_run_path}")
    print(f"[crossfold-audit] wrote {grouped_path}")
    print(f"[crossfold-audit] wrote {by_fold_path}")
    print(f"[crossfold-audit] wrote {md_path}")
    print("\nGROUP SUMMARY")
    print(grouped.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
