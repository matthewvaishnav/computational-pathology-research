#!/usr/bin/env python3
"""
Generate paper-style figures for the dominant-site federated pathology note.

This script intentionally starts with the compact, already-curated detector
artifacts rather than the large per-seed prediction folders.

Outputs:
    figures/dominant-site-figure-3-detector-transfer.png
    figures/dominant-site-figure-4-detector-ablation.png

Example:
    python scripts/figures/make_dominant_site_paper_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required to generate figures") from exc


DEFAULT_TRANSFER_SUMMARY = Path(
    "results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_summary.csv"
)
DEFAULT_DIAGNOSTIC_FREQUENCY = Path(
    "results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_diagnostic_summary/diagnostic_frequency_by_stress.csv"
)
DEFAULT_LEAVE_ONE_OUT = Path(
    "results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_leave_one_out/diagnostic_ablation_headline_35_45.csv"
)
DEFAULT_CALIBRATION = Path(
    "results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity/calibration_sensitivity_headline.csv"
)

KEY_TRANSFER_METRICS = {
    "global_qwk": "Global QWK",
    "macro_f1": "Macro-F1",
    "worst_site_qwk": "Worst-site QWK",
}

DIAGNOSTIC_LABELS = {
    "mean_abs_error_high": "Mean abs.\nordinal error high",
    "worst_site_qwk_low": "Worst-site\nQWK low",
    "global_qwk_low": "Global\nQWK low",
    "severe_error_rate_high": "Severe error\nrate high",
    "site_qwk_spread_high": "Site-QWK\nspread high",
}

VARIANT_LABELS = {
    "only_mean_abs_error_high": "Only mean abs. error",
    "full": "Full detector",
    "minus_site_qwk_spread_high": "Minus site spread",
    "only_global_qwk_low": "Only global QWK",
    "minus_worst_site_qwk_low": "Minus worst-site QWK",
    "only_severe_error_rate_high": "Only severe error",
    "minus_severe_error_rate_high": "Minus severe error",
    "only_worst_site_qwk_low": "Only worst-site QWK",
    "minus_global_qwk_low": "Minus global QWK",
    "minus_mean_abs_error_high": "Minus mean abs. error",
    "only_site_qwk_spread_high": "Only site spread",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-summary", type=Path, default=DEFAULT_TRANSFER_SUMMARY)
    parser.add_argument("--diagnostic-frequency", type=Path, default=DEFAULT_DIAGNOSTIC_FREQUENCY)
    parser.add_argument("--leave-one-out", type=Path, default=DEFAULT_LEAVE_ONE_OUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def metric_rows(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    frame = summary[summary["metric"] == metric].copy()
    frame["noise"] = frame["noise"].astype(int)
    return frame.sort_values("noise")


def make_detector_transfer_figure(summary_path: Path, out_path: Path, dpi: int) -> None:
    summary = pd.read_csv(summary_path)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    axes = axes.flatten()

    global_rows = metric_rows(summary, "global_qwk")
    shifts = global_rows["noise"].to_numpy()
    trigger_rates = global_rows["detector_trigger_rate"].to_numpy() * 100.0

    ax = axes[0]
    ax.bar(shifts.astype(str), trigger_rates)
    ax.set_title("A. Detector trigger rate")
    ax.set_xlabel("Conservative threshold shift (%)")
    ax.set_ylabel("Triggered runs (%)")
    ax.set_ylim(0, max(80, trigger_rates.max() + 10))
    for x, value in enumerate(trigger_rates):
        ax.text(x, value + 2, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    style_axes(ax)

    panel_titles = [
        ("global_qwk", "B. Global QWK delta"),
        ("macro_f1", "C. Macro-F1 delta"),
        ("worst_site_qwk", "D. Worst-site QWK delta"),
    ]

    for ax, (metric, title) in zip(axes[1:], panel_titles):
        rows = metric_rows(summary, metric)
        x = range(len(rows))
        means = rows["mean_delta_detector_vs_clean"].astype(float).to_numpy()
        lows = rows["ci95_low"].astype(float).to_numpy()
        highs = rows["ci95_high"].astype(float).to_numpy()
        yerr = [means - lows, highs - means]
        ax.axhline(0.0, linewidth=1.0)
        ax.errorbar(x, means, yerr=yerr, fmt="o", capsize=4)
        ax.set_xticks(list(x))
        ax.set_xticklabels(rows["noise"].astype(int).astype(str).tolist())
        ax.set_title(title)
        ax.set_xlabel("Conservative threshold shift (%)")
        ax.set_ylabel("Detector - clean baseline")
        style_axes(ax)

    fig.suptitle(
        "Fixed detector transfer to conservative ordinal threshold shift",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_detector_ablation_figure(
    diagnostic_frequency_path: Path,
    leave_one_out_path: Path,
    calibration_path: Path,
    out_path: Path,
    dpi: int,
) -> None:
    freq = pd.read_csv(diagnostic_frequency_path)
    leave_one = pd.read_csv(leave_one_out_path)
    calibration = pd.read_csv(calibration_path)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    axes = axes.flatten()

    ax = axes[0]
    total_freq = freq.groupby("diagnostic", as_index=False)["count"].sum()
    total_freq = total_freq.sort_values("count", ascending=True)
    labels = [DIAGNOSTIC_LABELS.get(item, item) for item in total_freq["diagnostic"]]
    ax.barh(labels, total_freq["count"].astype(float))
    ax.set_title("A. Trigger diagnostic frequency")
    ax.set_xlabel("Count across runs")
    style_axes(ax)

    ax = axes[1]
    plot_variants = [
        "full",
        "minus_mean_abs_error_high",
        "only_mean_abs_error_high",
        "only_global_qwk_low",
        "only_worst_site_qwk_low",
        "only_severe_error_rate_high",
        "only_site_qwk_spread_high",
    ]
    subset = leave_one[leave_one["variant"].isin(plot_variants)].copy()
    subset["variant_order"] = subset["variant"].map({variant: i for i, variant in enumerate(plot_variants)})
    subset = subset.sort_values("variant_order", ascending=False)
    ax.barh(
        [VARIANT_LABELS.get(item, item) for item in subset["variant"]],
        subset["mean_global_qwk_delta_35_45"].astype(float),
    )
    ax.axvline(0.0, linewidth=1.0)
    ax.set_title("B. Leave-one-out / single-family ablation")
    ax.set_xlabel("Mean global QWK delta, 35/45")
    style_axes(ax)

    ax = axes[2]
    counts = calibration["robust_positive_config"].astype(str).str.lower().map({"true": True, "false": False})
    robust_count = int(counts.sum())
    total_count = int(len(calibration))
    ax.bar(["Robust-positive", "Other"], [robust_count, total_count - robust_count])
    ax.set_title("C. Calibration sensitivity")
    ax.set_ylabel("Configurations")
    ax.text(0, robust_count + 0.6, f"{robust_count}/{total_count}", ha="center", va="bottom", fontsize=11)
    style_axes(ax)

    ax = axes[3]
    top = calibration.sort_values("mean_delta_global_qwk_target", ascending=False).head(8).copy()
    top = top.sort_values("mean_delta_global_qwk_target", ascending=True)
    ax.barh(top["config_id"], top["mean_delta_global_qwk_target"].astype(float))
    ax.set_title("D. Top calibration settings")
    ax.set_xlabel("Mean global QWK delta, 35/45")
    style_axes(ax)

    fig.suptitle(
        "Detector interpretability, ablation, and calibration robustness",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    for path in [args.transfer_summary, args.diagnostic_frequency, args.leave_one_out, args.calibration]:
        require_file(path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure3_path = args.out_dir / "dominant-site-figure-3-detector-transfer.png"
    figure4_path = args.out_dir / "dominant-site-figure-4-detector-ablation.png"

    make_detector_transfer_figure(args.transfer_summary, figure3_path, dpi=args.dpi)
    make_detector_ablation_figure(
        args.diagnostic_frequency,
        args.leave_one_out,
        args.calibration,
        figure4_path,
        dpi=args.dpi,
    )

    print(f"Wrote {figure3_path}")
    print(f"Wrote {figure4_path}")


if __name__ == "__main__":
    main()
