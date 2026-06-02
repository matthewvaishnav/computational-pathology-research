#!/usr/bin/env python3
"""
Generate Figure 1 for the dominant-site federated pathology paper.

Output:
    figures/dominant-site-figure-1-problem-schematic.png

Example:
    python scripts/figures/make_dominant_site_schematic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required to generate figures") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fontsize: int = 10,
    linewidth: float = 1.4,
    alpha: float = 0.08,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=linewidth,
        edgecolor="black",
        facecolor="black",
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        color="black",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def make_schematic(out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Sample volume is not the same as site-signal alignment",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )

    ax.text(0.05, 0.88, "A. Standard FedAvg assumption", fontsize=13, fontweight="bold")
    add_box(ax, 0.05, 0.70, 0.16, 0.10, "Client A\nlarge sample count", fontsize=10)
    add_box(ax, 0.05, 0.55, 0.16, 0.10, "Client B\nsmaller sample count", fontsize=10)
    add_box(ax, 0.05, 0.40, 0.16, 0.10, "Client C\nsmaller sample count", fontsize=10)
    add_box(ax, 0.31, 0.55, 0.20, 0.14, "FedAvg\nweights updates by\nsample count", fontsize=11, alpha=0.10)
    add_arrow(ax, (0.21, 0.75), (0.31, 0.63))
    add_arrow(ax, (0.21, 0.60), (0.31, 0.62))
    add_arrow(ax, (0.21, 0.45), (0.31, 0.61))
    ax.text(0.25, 0.75, "more samples\n→ more influence", fontsize=9, ha="center")

    ax.text(0.56, 0.88, "B. Pathology failure mode", fontsize=13, fontweight="bold")
    add_box(ax, 0.56, 0.68, 0.18, 0.12, "Dominant client\nmany samples\nshifted labels", fontsize=10, alpha=0.13)
    add_box(ax, 0.56, 0.48, 0.18, 0.12, "Other clients\nfewer samples\naligned labels", fontsize=10)
    add_box(ax, 0.82, 0.58, 0.14, 0.12, "Shared model\namplifies\nshifted signal", fontsize=10, alpha=0.13)
    add_arrow(ax, (0.74, 0.74), (0.82, 0.65))
    add_arrow(ax, (0.74, 0.54), (0.82, 0.63))
    ax.text(
        0.755,
        0.80,
        "sample-size dominance\ncan diverge from\nvalidation objective",
        fontsize=9,
        ha="center",
    )

    ax.text(0.05, 0.28, "C. Dominance-aware detector switch", fontsize=13, fontweight="bold")
    add_box(ax, 0.05, 0.09, 0.18, 0.12, "Clean-calibrated\nFedAvg diagnostics", fontsize=10)
    add_box(ax, 0.31, 0.09, 0.20, 0.12, "Enough diagnostics\nleave safe range?", fontsize=10, alpha=0.10)
    add_box(ax, 0.60, 0.18, 0.17, 0.11, "No:\nkeep FedAvg", fontsize=10)
    add_box(ax, 0.60, 0.02, 0.17, 0.11, "Yes:\nswitch away from\nsample-size dominance", fontsize=10, alpha=0.13)
    add_box(ax, 0.84, 0.09, 0.12, 0.12, "Auditable\ninfluence\npolicy", fontsize=10, alpha=0.10)
    add_arrow(ax, (0.23, 0.15), (0.31, 0.15))
    add_arrow(ax, (0.51, 0.17), (0.60, 0.23))
    add_arrow(ax, (0.51, 0.12), (0.60, 0.07))
    add_arrow(ax, (0.77, 0.235), (0.84, 0.17))
    add_arrow(ax, (0.77, 0.075), (0.84, 0.13))

    ax.text(
        0.50,
        0.33,
        "Goal: audit the FedAvg assumption 'larger client = more authority' without treating sites as institutional rankings.",
        fontsize=10,
        ha="center",
        style="italic",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "dominant-site-figure-1-problem-schematic.png"
    make_schematic(out_path, dpi=args.dpi)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
