"""
Timeline visualization utilities for longitudinal patient tracking.

This module provides visualization functions for patient timelines, including
disease progression trajectories, risk score evolution, and treatment events.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)


class TimelineVisualizer:
    """
    Visualizer for patient timelines showing disease progression over time.

    Creates comprehensive timeline visualizations including:
    - Disease state trajectory with confidence scores
    - Risk score evolution for multiple diseases and time horizons
    - Treatment events marked on timeline
    - Significant change highlights

    Args:
        figsize: Figure size (width, height) in inches (default: (14, 10))
        dpi: Figure resolution (default: 100)
        style: Matplotlib style to use (default: 'seaborn-v0_8-darkgrid')

    Example:
        >>> from src.clinical.longitudinal import PatientTimeline
        >>> visualizer = TimelineVisualizer()
        >>>
        >>> # Create timeline with scans
        >>> timeline = PatientTimeline(patient_id="PATIENT_001")
        >>> # ... add scans and treatments ...
        >>>
        >>> # Generate visualization
        >>> fig = visualizer.plot_timeline(timeline)
        >>> fig.savefig("patient_timeline.png")
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 100,
        style: str = "seaborn-v0_8-darkgrid",
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.style = style

        # Color schemes
        self.disease_colors = {
            "benign": "#2ecc71",  # Green
            "grade_1": "#f39c12",  # Orange
            "grade_2": "#e74c3c",  # Red
            "grade_3": "#c0392b",  # Dark red
            "malignant": "#e74c3c",  # Red
            "normal": "#3498db",  # Blue
        }
        self.default_color = "#95a5a6"  # Gray

        self.treatment_colors = {
            "chemotherapy": "#9b59b6",  # Purple
            "surgery": "#e67e22",  # Orange
            "radiation": "#16a085",  # Teal
            "immunotherapy": "#2980b9",  # Blue
        }
        self.default_treatment_color = "#7f8c8d"  # Dark gray

    def plot_timeline(
        self,
        timeline,  # PatientTimeline
        disease_ids: Optional[List[str]] = None,
        show_risk_scores: bool = True,
        show_treatments: bool = True,
        title: Optional[str] = None,
    ) -> Figure:
        """
        Create comprehensive timeline visualization.

        Args:
            timeline: PatientTimeline instance
            disease_ids: Optional list of disease IDs to show risk scores for
            show_risk_scores: Whether to show risk score subplot
            show_treatments: Whether to show treatment events
            title: Optional custom title

        Returns:
            Matplotlib Figure object
        """
        scans = timeline.get_scans()
        treatments = timeline.get_treatments()

        if not scans:
            logger.warning("No scans in timeline, cannot create visualization")
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(
                0.5,
                0.5,
                "No scan data available",
                ha="center",
                va="center",
                fontsize=14,
            )
            return fig

        # Determine number of subplots
        num_subplots = 2  # Disease state + confidence
        if show_risk_scores and any(scan.risk_scores for scan in scans):
            num_subplots += 1

        # Create figure with subplots
        fig, axes = plt.subplots(num_subplots, 1, figsize=self.figsize, dpi=self.dpi, sharex=True)

        if num_subplots == 1:
            axes = [axes]

        # Set title
        if title is None:
            title = f"Patient Timeline (Hash: {timeline.patient_id_hash[:16]}...)"
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Extract data
        scan_dates = [scan.scan_date for scan in scans]
        disease_states = [scan.disease_state for scan in scans]
        confidences = [scan.confidence for scan in scans]

        # Plot 1: Disease state trajectory
        ax_idx = 0
        self._plot_disease_trajectory(
            axes[ax_idx],
            scan_dates,
            disease_states,
            confidences,
            treatments if show_treatments else [],
        )

        # Plot 2: Confidence scores
        ax_idx += 1
        self._plot_confidence_trajectory(
            axes[ax_idx], scan_dates, confidences, treatments if show_treatments else []
        )

        # Plot 3: Risk scores (if requested and available)
        if show_risk_scores and any(scan.risk_scores for scan in scans):
            ax_idx += 1
            self._plot_risk_trajectory(
                axes[ax_idx], scans, disease_ids, treatments if show_treatments else []
            )

        # Format x-axis (dates)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Adjust layout
        plt.tight_layout()

        return fig

    def _plot_disease_trajectory(
        self,
        ax,
        scan_dates: List[datetime],
        disease_states: List[str],
        confidences: List[float],
        treatments: List,  # List[TreatmentEvent]
    ) -> None:
        """Plot disease state trajectory over time."""
        # Create categorical y-axis for disease states
        unique_states = sorted(set(disease_states))
        state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
        state_indices = [state_to_idx[state] for state in disease_states]

        # Plot disease states as line with markers
        colors = [self.disease_colors.get(state, self.default_color) for state in disease_states]

        for i in range(len(scan_dates)):
            ax.scatter(
                scan_dates[i],
                state_indices[i],
                c=colors[i],
                s=200,
                alpha=0.7,
                edgecolors="black",
                linewidths=2,
                zorder=3,
            )

        # Connect points with lines
        if len(scan_dates) > 1:
            ax.plot(
                scan_dates,
                state_indices,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
                zorder=2,
            )

        # Add treatment markers
        if treatments:
            for treatment in treatments:
                ax.axvline(
                    treatment.treatment_date,
                    color=self.treatment_colors.get(
                        treatment.treatment_type, self.default_treatment_color
                    ),
                    linestyle=":",
                    linewidth=2,
                    alpha=0.7,
                    label=f"{treatment.treatment_type}",
                )

        # Configure axis
        ax.set_yticks(range(len(unique_states)))
        ax.set_yticklabels(unique_states)
        ax.set_ylabel("Disease State", fontsize=12, fontweight="bold")
        ax.set_title("Disease State Trajectory", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add legend for treatments if present
        if treatments:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))  # Remove duplicates
            ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)

    def _plot_confidence_trajectory(
        self,
        ax,
        scan_dates: List[datetime],
        confidences: List[float],
        treatments: List,  # List[TreatmentEvent]
    ) -> None:
        """Plot confidence score trajectory over time."""
        # Plot confidence as line with markers
        ax.plot(
            scan_dates,
            confidences,
            marker="o",
            markersize=8,
            linewidth=2,
            color="#3498db",
            label="Confidence",
        )

        # Add confidence threshold lines
        ax.axhline(0.9, color="green", linestyle="--", linewidth=1, alpha=0.5, label="High (0.9)")
        ax.axhline(
            0.7, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="Medium (0.7)"
        )
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Low (0.5)")

        # Add treatment markers
        if treatments:
            for treatment in treatments:
                ax.axvline(
                    treatment.treatment_date,
                    color=self.treatment_colors.get(
                        treatment.treatment_type, self.default_treatment_color
                    ),
                    linestyle=":",
                    linewidth=2,
                    alpha=0.7,
                )

        # Configure axis
        ax.set_ylabel("Confidence Score", fontsize=12, fontweight="bold")
        ax.set_title("Prediction Confidence Over Time", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)

    def _plot_risk_trajectory(
        self,
        ax,
        scans: List,  # List[ScanRecord]
        disease_ids: Optional[List[str]],
        treatments: List,  # List[TreatmentEvent]
    ) -> None:
        """Plot risk score trajectory over time."""
        # Collect risk data
        scan_dates = [scan.scan_date for scan in scans]

        # Determine which diseases to plot
        if disease_ids is None:
            # Find all diseases with risk scores
            all_disease_ids = set()
            for scan in scans:
                all_disease_ids.update(scan.risk_scores.keys())
            disease_ids = sorted(all_disease_ids)

        if not disease_ids:
            ax.text(
                0.5,
                0.5,
                "No risk score data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            return

        # Plot risk scores for each disease (use first time horizon)
        colors = plt.cm.Set2(np.linspace(0, 1, len(disease_ids)))

        for disease_id, color in zip(disease_ids, colors):
            risk_values = []
            dates_with_risk = []

            for scan in scans:
                if disease_id in scan.risk_scores:
                    risk_data = scan.risk_scores[disease_id]
                    # Use first available time horizon
                    if risk_data:
                        first_horizon = list(risk_data.keys())[0]
                        risk_values.append(risk_data[first_horizon])
                        dates_with_risk.append(scan.scan_date)

            if risk_values:
                ax.plot(
                    dates_with_risk,
                    risk_values,
                    marker="s",
                    markersize=6,
                    linewidth=2,
                    color=color,
                    label=disease_id,
                    alpha=0.8,
                )

        # Add treatment markers
        if treatments:
            for treatment in treatments:
                ax.axvline(
                    treatment.treatment_date,
                    color=self.treatment_colors.get(
                        treatment.treatment_type, self.default_treatment_color
                    ),
                    linestyle=":",
                    linewidth=2,
                    alpha=0.7,
                )

        # Configure axis
        ax.set_ylabel("Risk Score", fontsize=12, fontweight="bold")
        ax.set_title("Risk Score Evolution", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=10, ncol=2)

    def plot_progression_summary(
        self,
        progression_metrics: Dict[str, Any],
        title: Optional[str] = None,
    ) -> Figure:
        """
        Create summary visualization of disease progression metrics.

        Args:
            progression_metrics: Output from LongitudinalTracker.compute_progression_metrics()
            title: Optional custom title

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        axes = axes.flatten()

        if title is None:
            title = "Disease Progression Summary"
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Plot 1: Disease state trajectory
        disease_states = progression_metrics["disease_state_trajectory"]
        unique_states = sorted(set(disease_states))
        state_counts = {state: disease_states.count(state) for state in unique_states}

        colors = [self.disease_colors.get(state, self.default_color) for state in unique_states]
        axes[0].bar(range(len(unique_states)), state_counts.values(), color=colors, alpha=0.7)
        axes[0].set_xticks(range(len(unique_states)))
        axes[0].set_xticklabels(unique_states, rotation=45, ha="right")
        axes[0].set_ylabel("Number of Scans")
        axes[0].set_title("Disease State Distribution")
        axes[0].grid(True, alpha=0.3, axis="y")

        # Plot 2: Confidence trajectory
        confidences = progression_metrics["confidence_trajectory"]
        axes[1].plot(range(len(confidences)), confidences, marker="o", linewidth=2, color="#3498db")
        axes[1].axhline(0.7, color="orange", linestyle="--", linewidth=1, alpha=0.5)
        axes[1].set_xlabel("Scan Index")
        axes[1].set_ylabel("Confidence Score")
        axes[1].set_title("Confidence Trajectory")
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Progression events timeline
        progression_events = progression_metrics["progression_events"]
        if progression_events:
            event_indices = [event["scan_index"] for event in progression_events]
            event_labels = [
                f"{event['previous_state']} → {event['current_state']}"
                for event in progression_events
            ]

            axes[2].scatter(
                event_indices, range(len(event_indices)), s=200, alpha=0.7, color="#e74c3c"
            )
            axes[2].set_yticks(range(len(event_indices)))
            axes[2].set_yticklabels(event_labels, fontsize=9)
            axes[2].set_xlabel("Scan Index")
            axes[2].set_title("Progression Events")
            axes[2].grid(True, alpha=0.3, axis="x")
        else:
            axes[2].text(
                0.5,
                0.5,
                "No progression events",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
                fontsize=12,
            )

        # Plot 4: Overall trend summary
        overall_trend = progression_metrics["overall_trend"]
        trend_colors = {
            "improving": "#2ecc71",
            "stable": "#3498db",
            "worsening": "#e74c3c",
            "mixed": "#f39c12",
            "insufficient_data": "#95a5a6",
        }

        axes[3].bar(
            [0],
            [1],
            color=trend_colors.get(overall_trend, "#95a5a6"),
            alpha=0.7,
            width=0.5,
        )
        axes[3].set_xlim(-0.5, 0.5)
        axes[3].set_ylim(0, 1.2)
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        axes[3].text(
            0,
            0.5,
            overall_trend.replace("_", " ").title(),
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )
        axes[3].set_title("Overall Trend")

        plt.tight_layout()
        return fig

    def plot_treatment_response(
        self,
        treatment_response: Dict[str, Any],
        title: Optional[str] = None,
    ) -> Figure:
        """
        Create visualization of treatment response analysis.

        Args:
            treatment_response: Output from LongitudinalTracker.identify_treatment_response()
            title: Optional custom title

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)

        if title is None:
            title = "Treatment Response Analysis"
        fig.suptitle(title, fontsize=16, fontweight="bold")

        baseline_scan = treatment_response["baseline_scan"]
        response_scan = treatment_response["response_scan"]
        response_category = treatment_response["response_category"]

        if baseline_scan is None or response_scan is None:
            axes[0].text(
                0.5,
                0.5,
                "Insufficient scan data",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
                fontsize=12,
            )
            axes[1].text(
                0.5,
                0.5,
                "Insufficient scan data",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
                fontsize=12,
            )
            return fig

        # Plot 1: Disease state comparison
        states = [baseline_scan.disease_state, response_scan.disease_state]
        labels = ["Baseline", "Response"]
        colors = [
            self.disease_colors.get(baseline_scan.disease_state, self.default_color),
            self.disease_colors.get(response_scan.disease_state, self.default_color),
        ]

        axes[0].bar(labels, [1, 1], color=colors, alpha=0.7, width=0.5)
        axes[0].set_ylim(0, 1.2)
        axes[0].set_yticks([])
        axes[0].set_title("Disease State Change")

        # Add state labels on bars
        for i, (label, state) in enumerate(zip(labels, states)):
            axes[0].text(
                i,
                0.5,
                state,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

        # Plot 2: Probability comparison
        baseline_probs = baseline_scan.disease_probabilities
        response_probs = response_scan.disease_probabilities

        # Get all disease states
        all_states = sorted(set(list(baseline_probs.keys()) + list(response_probs.keys())))

        x = np.arange(len(all_states))
        width = 0.35

        baseline_values = [baseline_probs.get(state, 0.0) for state in all_states]
        response_values = [response_probs.get(state, 0.0) for state in all_states]

        axes[1].bar(
            x - width / 2, baseline_values, width, label="Baseline", alpha=0.7, color="#3498db"
        )
        axes[1].bar(
            x + width / 2, response_values, width, label="Response", alpha=0.7, color="#e74c3c"
        )

        axes[1].set_xlabel("Disease State")
        axes[1].set_ylabel("Probability")
        axes[1].set_title("Probability Distribution Comparison")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(all_states, rotation=45, ha="right")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        # Add response category annotation
        response_colors = {
            "complete_response": "#2ecc71",
            "partial_response": "#f39c12",
            "stable_disease": "#3498db",
            "progressive_disease": "#e74c3c",
            "unknown": "#95a5a6",
        }

        fig.text(
            0.5,
            0.02,
            f"Response Category: {response_category.replace('_', ' ').title()}",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=response_colors.get(response_category, "#95a5a6"),
        )

        plt.tight_layout()
        return fig
