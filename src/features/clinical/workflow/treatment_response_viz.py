"""
Treatment Response Visualization Module

Visualization functions for treatment response analysis including trajectory plots,
unexpected response analysis, and therapeutic regimen comparisons.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def visualize_treatment_response_trajectory(
    trajectory_data: Dict[str, Any], save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    Visualize treatment response trajectory showing disease evolution during/after therapy.

    Args:
        trajectory_data: Output from analyze_treatment_response_trajectory()
        save_path: Optional path to save the figure

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Treatment Response Trajectory - {trajectory_data['treatment_type'].title()}",
        fontsize=16,
        fontweight="bold",
    )

    scans = trajectory_data["scans"]
    if not scans:
        for ax in axes.flat:
            ax.text(
                0.5,
                0.5,
                "No scan data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
        return fig

    days = [s["days_from_treatment"] for s in scans]

    # Plot 1: Primary disease probability over time
    primary_probs = []
    for scan_data in scans:
        scan = scan_data["scan"]
        prob = scan.disease_probabilities.get(scan.disease_state, 0.0)
        primary_probs.append(prob)

    axes[0, 0].plot(days, primary_probs, "o-", linewidth=2, markersize=6, color="#e74c3c")
    axes[0, 0].axvline(x=0, color="black", linestyle="--", alpha=0.7, label="Treatment")
    axes[0, 0].set_xlabel("Days from Treatment")
    axes[0, 0].set_ylabel("Primary Disease Probability")
    axes[0, 0].set_title("Disease Probability Trajectory")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)

    # Plot 2: Disease state changes
    disease_states = [s["scan"].disease_state for s in scans]
    unique_states = list(set(disease_states))
    state_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_states)))
    state_color_map = dict(zip(unique_states, state_colors))

    for i, (day, state) in enumerate(zip(days, disease_states)):
        axes[0, 1].scatter(day, i, c=[state_color_map[state]], s=100, alpha=0.8)

    axes[0, 1].axvline(x=0, color="black", linestyle="--", alpha=0.7, label="Treatment")
    axes[0, 1].set_xlabel("Days from Treatment")
    axes[0, 1].set_ylabel("Scan Index")
    axes[0, 1].set_title("Disease State Evolution")
    axes[0, 1].legend()

    for state, color in state_color_map.items():
        axes[0, 1].scatter([], [], c=[color], s=100, label=state, alpha=0.8)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 3: Response phases
    phases = trajectory_data.get("response_phases", [])
    if phases:
        phase_names = [p["phase"] for p in phases]
        phase_probs = [p["avg_disease_probability"] for p in phases]
        phase_colors = ["#3498db", "#f39c12", "#2ecc71", "#9b59b6"]

        bars = axes[1, 0].bar(
            phase_names, phase_probs, color=phase_colors[: len(phase_names)], alpha=0.7
        )
        axes[1, 0].set_ylabel("Average Disease Probability")
        axes[1, 0].set_title("Response by Phase")
        axes[1, 0].tick_params(axis="x", rotation=45)

        for bar, phase in zip(bars, phases):
            height = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f'n={phase["num_scans"]}',
                ha="center",
                va="bottom",
            )
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No phase data available",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
            fontsize=12,
        )

    # Plot 4: Probability trajectories for all disease states
    disease_evolution = trajectory_data.get("disease_evolution", {})
    prob_trajectories = disease_evolution.get("probability_trajectories", {})

    if prob_trajectories:
        colors = plt.cm.tab10(np.linspace(0, 1, len(prob_trajectories)))

        for (state, trajectory), color in zip(prob_trajectories.items(), colors):
            traj_days = [point["days_from_treatment"] for point in trajectory]
            traj_probs = [point["probability"] for point in trajectory]
            axes[1, 1].plot(
                traj_days, traj_probs, "o-", label=state, color=color, alpha=0.8, linewidth=2
            )

        axes[1, 1].axvline(x=0, color="black", linestyle="--", alpha=0.7, label="Treatment")
        axes[1, 1].set_xlabel("Days from Treatment")
        axes[1, 1].set_ylabel("Probability")
        axes[1, 1].set_title("All Disease State Trajectories")
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No trajectory data available",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=12,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Trajectory visualization saved to {save_path}")

    return fig


def visualize_unexpected_responses(
    unexpected_cases: List[Dict[str, Any]], save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    Visualize unexpected treatment responses for clinical review.

    Args:
        unexpected_cases: Output from identify_unexpected_responses()
        save_path: Optional path to save the figure

    Returns:
        Matplotlib Figure object
    """
    if not unexpected_cases:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No unexpected responses found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Unexpected Treatment Responses")
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Unexpected Treatment Response Analysis", fontsize=16, fontweight="bold")

    unexpected_types = [case["unexpected_type"] for case in unexpected_cases]
    unexpected_scores = [case["unexpected_score"] for case in unexpected_cases]
    treatment_types = [case["treatment_type"] for case in unexpected_cases]
    response_categories = [case["response_category"] for case in unexpected_cases]

    # Plot 1: Distribution of unexpected response types
    type_counts = {}
    for utype in unexpected_types:
        type_counts[utype] = type_counts.get(utype, 0) + 1

    if type_counts:
        types, counts = zip(*type_counts.items())
        colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
        axes[0, 0].pie(
            counts,
            labels=[t.replace("_", " ").title() for t in types],
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[0, 0].set_title("Unexpected Response Types")

    # Plot 2: Unexpected scores distribution
    axes[0, 1].hist(unexpected_scores, bins=10, alpha=0.7, color="#e74c3c", edgecolor="black")
    axes[0, 1].set_xlabel("Unexpected Score")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Unexpected Scores")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Unexpected responses by treatment type
    treatment_unexpected = {}
    for ttype in treatment_types:
        treatment_unexpected[ttype] = treatment_unexpected.get(ttype, 0) + 1

    if treatment_unexpected:
        ttypes, tcounts = zip(*treatment_unexpected.items())
        axes[1, 0].bar(ttypes, tcounts, alpha=0.7, color="#3498db")
        axes[1, 0].set_xlabel("Treatment Type")
        axes[1, 0].set_ylabel("Number of Unexpected Cases")
        axes[1, 0].set_title("Unexpected Responses by Treatment Type")
        axes[1, 0].tick_params(axis="x", rotation=45)

    # Plot 4: Response categories in unexpected cases
    category_counts = {}
    for category in response_categories:
        category_counts[category] = category_counts.get(category, 0) + 1

    if category_counts:
        categories, ccounts = zip(*category_counts.items())
        colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
        axes[1, 1].bar(
            [c.replace("_", " ").title() for c in categories],
            ccounts,
            color=colors[: len(categories)],
            alpha=0.7,
        )
        axes[1, 1].set_xlabel("Response Category")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Response Categories in Unexpected Cases")
        axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Unexpected responses visualization saved to {save_path}")

    return fig


def visualize_regimen_comparison(
    comparison_results: Dict[str, Any], save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    Visualize comparison across different therapeutic regimens.

    Args:
        comparison_results: Output from compare_therapeutic_regimens()
        save_path: Optional path to save the figure

    Returns:
        Matplotlib Figure object
    """
    regimen_stats = comparison_results.get("regimen_statistics", {})
    regimen_ranking = comparison_results.get("regimen_ranking", [])

    if not regimen_stats:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No regimen data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Therapeutic Regimen Comparison")
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Therapeutic Regimen Comparison", fontsize=16, fontweight="bold")

    regimens = list(regimen_stats.keys())

    # Plot 1: Overall response rates
    response_rates = [stats["overall_response_rate"] for stats in regimen_stats.values()]
    sample_sizes = [stats["sample_size"] for stats in regimen_stats.values()]

    bars = axes[0, 0].bar(regimens, response_rates, alpha=0.7, color="#2ecc71")
    axes[0, 0].set_ylabel("Overall Response Rate")
    axes[0, 0].set_title("Response Rates by Regimen")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].set_ylim(0, 1)

    for bar, size in zip(bars, sample_sizes):
        height = bar.get_height()
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"n={size}",
            ha="center",
            va="bottom",
        )

    # Plot 2: Response category breakdown
    response_categories = ["complete", "partial", "stable", "progressive"]
    category_colors = ["#2ecc71", "#f39c12", "#3498db", "#e74c3c"]

    bottom = np.zeros(len(regimens))
    for i, category in enumerate(response_categories):
        rates = [stats["response_rates"][category] for stats in regimen_stats.values()]
        axes[0, 1].bar(
            regimens,
            rates,
            bottom=bottom,
            label=category.title(),
            color=category_colors[i],
            alpha=0.8,
        )
        bottom += rates

    axes[0, 1].set_ylabel("Response Rate")
    axes[0, 1].set_title("Response Category Breakdown")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)

    # Plot 3: Time to response comparison
    time_means = []
    time_stds = []
    valid_regimens = []

    for regimen, regimen_stat in regimen_stats.items():
        time_stats = regimen_stat.get("time_to_response", {})
        if time_stats:
            time_means.append(time_stats["mean"])
            time_stds.append(time_stats["std"])
            valid_regimens.append(regimen)

    if time_means:
        axes[1, 0].bar(
            valid_regimens, time_means, yerr=time_stds, alpha=0.7, color="#9b59b6", capsize=5
        )
        axes[1, 0].set_ylabel("Days to Response")
        axes[1, 0].set_title("Time to Response by Regimen")
        axes[1, 0].tick_params(axis="x", rotation=45)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No time data available",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
            fontsize=12,
        )

    # Plot 4: Effectiveness ranking
    if regimen_ranking:
        ranking_regimens = [r["regimen"] for r in regimen_ranking]
        effectiveness_scores = [r["effectiveness_score"] for r in regimen_ranking]

        bars = axes[1, 1].barh(ranking_regimens, effectiveness_scores, alpha=0.7, color="#34495e")
        axes[1, 1].set_xlabel("Effectiveness Score")
        axes[1, 1].set_title("Regimen Effectiveness Ranking")

        for bar, score in zip(bars, effectiveness_scores):
            width = bar.get_width()
            axes[1, 1].text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{score:.2f}",
                ha="left",
                va="center",
            )
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No ranking data available",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=12,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Regimen comparison visualization saved to {save_path}")

    return fig
