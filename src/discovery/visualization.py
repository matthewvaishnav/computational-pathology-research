"""
Visualization for discovered cancer subtypes.

Kaplan-Meier curves, UMAP embeddings, and summary reports
for communicating discovered subtypes to clinicians.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure

logger = logging.getLogger(__name__)


def kaplan_meier_plot(
    survival_times: np.ndarray,
    events: np.ndarray,
    labels: np.ndarray,
    title: str = "Kaplan-Meier by Discovered Subtype",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> "matplotlib.figure.Figure":
    """
    Plot Kaplan-Meier survival curves with Greenwood confidence intervals.

    One curve per discovered subtype, colored distinctly.
    Adds log-rank p-value annotation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .validation import log_rank_test

    fig, ax = plt.subplots(figsize=figsize)
    groups = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))

    for g, color in zip(groups, colors):
        mask = labels == g
        t = survival_times[mask]
        e = events[mask]
        n = mask.sum()

        # Compute KM estimator
        order = np.argsort(t)
        t_sorted, e_sorted = t[order], e[order]
        times = [0.0]
        survival = [1.0]
        se_list = [0.0]
        S = 1.0
        greenwood_sum = 0.0
        n_at_risk = n

        for i, (ti, ei) in enumerate(zip(t_sorted, e_sorted)):
            if ei == 1:
                S *= (n_at_risk - 1) / n_at_risk
                greenwood_sum += 1 / (n_at_risk * (n_at_risk - 1) + 1e-8)
                times.append(float(ti))
                survival.append(float(S))
                se_list.append(float(S * np.sqrt(greenwood_sum)))
            n_at_risk -= 1

        times_arr = np.array(times)
        surv_arr = np.array(survival)
        se_arr = np.array(se_list)

        ax.step(
            times_arr,
            surv_arr,
            where="post",
            color=color,
            label=f"Subtype {g} (n={n})",
            linewidth=2,
        )
        ax.fill_between(
            times_arr,
            np.clip(surv_arr - 1.96 * se_arr, 0, 1),
            np.clip(surv_arr + 1.96 * se_arr, 0, 1),
            alpha=0.15,
            color=color,
            step="post",
        )

    # Log-rank p-value
    lr = log_rank_test(survival_times, events, labels)
    p_str = f"p={lr['p_value']:.3e}" if lr["p_value"] < 0.001 else f"p={lr['p_value']:.3f}"
    ax.text(
        0.98,
        0.98,
        p_str,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"KM plot saved to {save_path}")

    return fig


def umap_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    clinical_vars: Optional[Dict[str, np.ndarray]] = None,
    title: str = "UMAP of Latent Space",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> "matplotlib.figure.Figure":
    """
    UMAP plot of latent embeddings colored by discovered subtypes.
    Optionally adds panels colored by clinical variables.

    Args:
        embeddings: Latent space embeddings [n, d]
        labels: Subtype labels [n]
        clinical_vars: Optional dict of {name: values} for extra panels
        title: Plot title
        save_path: Save figure to path if provided
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap-learn required. pip install umap-learn>=0.5.0")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info(f"Computing UMAP on {embeddings.shape[0]} samples...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    n_panels = 1 + (len(clinical_vars) if clinical_vars else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(figsize[0] * n_panels, figsize[1]))
    if n_panels == 1:
        axes = [axes]

    # Panel 0: subtype labels
    ax = axes[0]
    groups = np.unique(labels)
    cmap = plt.cm.Set1(np.linspace(0, 1, len(groups)))
    for g, color in zip(groups, cmap):
        mask = labels == g
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color],
            label=f"Subtype {g}",
            alpha=0.7,
            s=15,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=9)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2)

    # Additional clinical variable panels
    if clinical_vars:
        for i, (var_name, var_vals) in enumerate(clinical_vars.items()):
            ax = axes[i + 1]
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=var_vals,
                cmap="coolwarm",
                alpha=0.7,
                s=15,
                edgecolors="none",
            )
            plt.colorbar(sc, ax=ax)
            ax.set_title(f"Colored by: {var_name}")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"UMAP plot saved to {save_path}")

    return fig


def subtype_summary_report(
    features: np.ndarray,
    labels: np.ndarray,
    survival_times: np.ndarray,
    events: np.ndarray,
    latent_embeddings: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    Generate a multi-panel summary figure for discovered subtypes:
    - KM survival curves
    - UMAP embedding (if latent_embeddings provided)
    - Subtype size bar chart
    - Per-subtype event rates
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .validation import log_rank_test

    n_panels = 4 if latent_embeddings is not None else 3
    fig = plt.figure(figsize=(6 * n_panels, 5))

    groups = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))

    # Panel 1: Subtype sizes
    ax1 = fig.add_subplot(1, n_panels, 1)
    sizes = [np.sum(labels == g) for g in groups]
    ax1.bar([f"S{g}" for g in groups], sizes, color=colors)
    ax1.set_title("Subtype Sizes")
    ax1.set_ylabel("N patients")
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Event rates
    ax2 = fig.add_subplot(1, n_panels, 2)
    event_rates = [events[labels == g].mean() for g in groups]
    ax2.bar([f"S{g}" for g in groups], event_rates, color=colors)
    ax2.set_title("Event Rate by Subtype")
    ax2.set_ylabel("Event rate")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: Median survival
    ax3 = fig.add_subplot(1, n_panels, 3)
    medians = []
    for g in groups:
        mask = labels == g
        t_g = survival_times[mask]
        e_g = events[mask]
        # Median = time where KM curve crosses 0.5
        order = np.argsort(t_g)
        t_s, e_s = t_g[order], e_g[order]
        S = 1.0
        n_risk = mask.sum()
        median = float(t_g.max())
        for ti, ei in zip(t_s, e_s):
            if ei == 1:
                S *= (n_risk - 1) / n_risk
            if S <= 0.5:
                median = float(ti)
                break
            n_risk -= 1
        medians.append(median)
    ax3.bar([f"S{g}" for g in groups], medians, color=colors)
    ax3.set_title("Median Survival by Subtype")
    ax3.set_ylabel("Median survival time")
    ax3.grid(axis="y", alpha=0.3)

    # Panel 4: UMAP (if available)
    if latent_embeddings is not None and n_panels == 4:
        try:
            from umap import UMAP

            ax4 = fig.add_subplot(1, n_panels, 4)
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
            coords = reducer.fit_transform(latent_embeddings)
            for g, color in zip(groups, colors):
                mask = labels == g
                ax4.scatter(
                    coords[mask, 0], coords[mask, 1], c=[color], label=f"S{g}", alpha=0.7, s=10
                )
            ax4.set_title("Latent Space (UMAP)")
            ax4.legend(markerscale=2, fontsize=8)
            ax4.set_xlabel("UMAP-1")
            ax4.set_ylabel("UMAP-2")
        except ImportError:
            logger.warning("umap-learn not installed, skipping UMAP panel")

    # Add log-rank p-value to figure title
    lr = log_rank_test(survival_times, events, labels)
    p_str = f"p={lr['p_value']:.3e}" if lr["p_value"] < 0.001 else f"p={lr['p_value']:.3f}"
    fig.suptitle(
        f"Discovered Subtypes (k={len(groups)}, log-rank {p_str})", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Summary report saved to {save_path}")

    return fig
