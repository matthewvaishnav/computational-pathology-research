"""
Visualization Generator for HistoCore Project Optimization Analysis System.

Creates interactive charts, heatmaps, and graphs for analysis results using
matplotlib and plotly for comprehensive visual reporting.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .models import AnalysisResult

logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """Generates charts and visualizations for analysis results."""

    def __init__(self):
        """Initialize visualization generator."""
        self.colors = {
            "excellent": "#2E8B57",  # Sea Green
            "good": "#32CD32",  # Lime Green
            "needs_improvement": "#FFD700",  # Gold
            "critical": "#DC143C",  # Crimson
        }

    def generate_all_charts(
        self, result: AnalysisResult, output_dir: str = "charts"
    ) -> Dict[str, str]:
        """
        Generate all visualization charts.

        Args:
            result: Analysis result
            output_dir: Directory to save charts

        Returns:
            Dictionary mapping chart names to file paths
        """
        logger.info("Generating all visualization charts...")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        charts = {}

        # Generate static charts (matplotlib)
        if MATPLOTLIB_AVAILABLE:
            charts.update(self._generate_matplotlib_charts(result, output_path))
        else:
            logger.warning("matplotlib not available - static charts disabled")

        # Generate interactive charts (plotly)
        if PLOTLY_AVAILABLE:
            charts.update(self._generate_plotly_charts(result, output_path))
        else:
            logger.warning("plotly not available - interactive charts disabled")

        # Generate fallback text charts if no libraries available
        if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
            charts.update(self._generate_text_charts(result, output_path))

        logger.info(f"Generated {len(charts)} charts in {output_path}")
        return charts

    def _generate_matplotlib_charts(
        self, result: AnalysisResult, output_path: Path
    ) -> Dict[str, str]:
        """Generate static charts using matplotlib."""
        charts = {}

        try:
            # Overall score radar chart
            radar_path = self._create_radar_chart(result, output_path)
            if radar_path:
                charts["radar_chart"] = str(radar_path)

            # Coverage heatmap
            heatmap_path = self._create_coverage_heatmap(result, output_path)
            if heatmap_path:
                charts["coverage_heatmap"] = str(heatmap_path)

            # Complexity distribution histogram
            complexity_path = self._create_complexity_histogram(result, output_path)
            if complexity_path:
                charts["complexity_histogram"] = str(complexity_path)

            # Score bar chart
            bar_path = self._create_score_bar_chart(result, output_path)
            if bar_path:
                charts["score_bar_chart"] = str(bar_path)

        except Exception as e:
            logger.error(f"Matplotlib chart generation failed: {e}")

        return charts

    def _create_radar_chart(self, result: AnalysisResult, output_path: Path) -> Optional[Path]:
        """Create radar chart for dimension scores."""
        try:
            # Dimension scores
            dimensions = [
                "Architecture",
                "Performance",
                "Coverage",
                "Code Quality",
                "Dependencies",
                "Deployment",
                "Security",
                "Scalability",
            ]

            scores = [
                result.architecture.score,
                result.performance.score,
                result.coverage.score,
                result.code_quality.score,
                result.dependencies.score,
                result.deployment.score,
                result.security.score,
                result.scalability.score,
            ]

            # Number of variables
            num_vars = len(dimensions)

            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            scores_plot = scores + scores[:1]  # Complete the circle
            angles += angles[:1]

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

            # Plot data
            ax.plot(angles, scores_plot, "o-", linewidth=2, color="#1f77b4", label="Current Score")
            ax.fill(angles, scores_plot, alpha=0.25, color="#1f77b4")

            # Add target line (70%)
            target = [70] * (num_vars + 1)
            ax.plot(angles, target, "--", linewidth=1, color="green", label="Target (70%)")

            # Fix axis to go from 0 to 100
            ax.set_ylim(0, 100)

            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(dimensions, size=10)

            # Add legend
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

            # Add title
            plt.title(
                f"Analysis Dimension Scores\nOverall: {result.overall_score:.1f}/100",
                size=14,
                weight="bold",
                pad=20,
            )

            # Save
            chart_path = output_path / "radar_chart.png"
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Radar chart saved to {chart_path}")
            return chart_path

        except Exception as e:
            logger.error(f"Radar chart generation failed: {e}")
            return None

    def _create_coverage_heatmap(self, result: AnalysisResult, output_path: Path) -> Optional[Path]:
        """Create coverage heatmap."""
        try:
            # Create sample heatmap data (in real implementation, use actual coverage data)
            dimensions = [
                "Architecture",
                "Performance",
                "Coverage",
                "Code Quality",
                "Dependencies",
                "Deployment",
                "Security",
                "Scalability",
            ]

            scores = np.array(
                [
                    result.architecture.score,
                    result.performance.score,
                    result.coverage.score,
                    result.code_quality.score,
                    result.dependencies.score,
                    result.deployment.score,
                    result.security.score,
                    result.scalability.score,
                ]
            ).reshape(
                2, 4
            )  # 2x4 grid

            fig, ax = plt.subplots(figsize=(12, 6))

            # Create heatmap
            im = ax.imshow(scores, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

            # Set ticks
            ax.set_xticks(np.arange(4))
            ax.set_yticks(np.arange(2))
            ax.set_xticklabels(dimensions[:4])
            ax.set_yticklabels(["Group 1", "Group 2"])

            # Rotate x labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Add text annotations
            for i in range(2):
                for j in range(4):
                    idx = i * 4 + j
                    if idx < len(dimensions):
                        text = ax.text(
                            j,
                            i,
                            f"{scores[i, j]:.1f}",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=12,
                        )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Score", rotation=270, labelpad=20)

            # Title
            ax.set_title("Analysis Dimension Heatmap", fontsize=14, weight="bold")

            # Save
            chart_path = output_path / "coverage_heatmap.png"
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Coverage heatmap saved to {chart_path}")
            return chart_path

        except Exception as e:
            logger.error(f"Coverage heatmap generation failed: {e}")
            return None

    def _create_complexity_histogram(
        self, result: AnalysisResult, output_path: Path
    ) -> Optional[Path]:
        """Create complexity distribution histogram."""
        try:
            # Get complexity data
            complexities = [
                func.get("complexity", 0) for func in result.code_quality.high_complexity_functions
            ]

            if not complexities:
                # Generate sample data if no real data
                complexities = [result.code_quality.average_complexity] * 10

            fig, ax = plt.subplots(figsize=(10, 6))

            # Create histogram
            n, bins, patches = ax.hist(
                complexities, bins=20, color="skyblue", edgecolor="black", alpha=0.7
            )

            # Add threshold line
            ax.axvline(x=10, color="red", linestyle="--", linewidth=2, label="Threshold (10)")

            # Labels
            ax.set_xlabel("Cyclomatic Complexity", fontsize=12)
            ax.set_ylabel("Number of Functions", fontsize=12)
            ax.set_title("Code Complexity Distribution", fontsize=14, weight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            # Save
            chart_path = output_path / "complexity_histogram.png"
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Complexity histogram saved to {chart_path}")
            return chart_path

        except Exception as e:
            logger.error(f"Complexity histogram generation failed: {e}")
            return None

    def _create_score_bar_chart(self, result: AnalysisResult, output_path: Path) -> Optional[Path]:
        """Create bar chart for dimension scores."""
        try:
            dimensions = [
                "Architecture",
                "Performance",
                "Coverage",
                "Code Quality",
                "Dependencies",
                "Deployment",
                "Security",
                "Scalability",
            ]

            scores = [
                result.architecture.score,
                result.performance.score,
                result.coverage.score,
                result.code_quality.score,
                result.dependencies.score,
                result.deployment.score,
                result.security.score,
                result.scalability.score,
            ]

            # Color bars based on score
            colors = ["green" if s >= 70 else "orange" if s >= 50 else "red" for s in scores]

            fig, ax = plt.subplots(figsize=(12, 6))

            # Create bars
            bars = ax.barh(dimensions, scores, color=colors, alpha=0.7, edgecolor="black")

            # Add score labels
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax.text(score + 2, i, f"{score:.1f}", va="center", fontsize=10)

            # Add target line
            ax.axvline(x=70, color="green", linestyle="--", linewidth=2, label="Target (70%)")

            # Labels
            ax.set_xlabel("Score", fontsize=12)
            ax.set_title("Analysis Dimension Scores", fontsize=14, weight="bold")
            ax.set_xlim(0, 105)
            ax.legend()
            ax.grid(axis="x", alpha=0.3)

            # Save
            chart_path = output_path / "score_bar_chart.png"
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Score bar chart saved to {chart_path}")
            return chart_path

        except Exception as e:
            logger.error(f"Score bar chart generation failed: {e}")
            return None

    def _generate_plotly_charts(self, result: AnalysisResult, output_path: Path) -> Dict[str, str]:
        """Generate interactive charts using plotly."""
        charts = {}

        try:
            # Interactive score dashboard
            dashboard_path = self._create_interactive_dashboard(result, output_path)
            if dashboard_path:
                charts["interactive_dashboard"] = str(dashboard_path)

            # Issue priority sunburst
            sunburst_path = self._create_issue_sunburst(result, output_path)
            if sunburst_path:
                charts["issue_sunburst"] = str(sunburst_path)

        except Exception as e:
            logger.error(f"Plotly chart generation failed: {e}")

        return charts

    def _create_interactive_dashboard(
        self, result: AnalysisResult, output_path: Path
    ) -> Optional[Path]:
        """Create interactive dashboard with plotly."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Dimension Scores",
                    "Issue Distribution",
                    "Coverage Metrics",
                    "Security Status",
                ),
                specs=[
                    [{"type": "bar"}, {"type": "pie"}],
                    [{"type": "indicator"}, {"type": "indicator"}],
                ],
            )

            # 1. Dimension scores bar chart
            dimensions = [
                "Architecture",
                "Performance",
                "Coverage",
                "Code Quality",
                "Dependencies",
                "Deployment",
                "Security",
                "Scalability",
            ]
            scores = [
                result.architecture.score,
                result.performance.score,
                result.coverage.score,
                result.code_quality.score,
                result.dependencies.score,
                result.deployment.score,
                result.security.score,
                result.scalability.score,
            ]

            fig.add_trace(
                go.Bar(x=dimensions, y=scores, name="Scores", marker_color="lightblue"),
                row=1,
                col=1,
            )

            # 2. Issue distribution pie chart
            p0_count = sum(1 for i in result.critical_issues if i.priority.value == "P0")
            p1_count = sum(1 for i in result.critical_issues if i.priority.value == "P1")
            p2_count = sum(1 for i in result.critical_issues if i.priority.value == "P2")

            fig.add_trace(
                go.Pie(
                    labels=["P0", "P1", "P2"],
                    values=[p0_count, p1_count, p2_count],
                    marker_colors=["red", "orange", "yellow"],
                ),
                row=1,
                col=2,
            )

            # 3. Coverage indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=result.coverage.line_coverage,
                    title={"text": "Line Coverage (%)"},
                    delta={"reference": 70},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 70], "color": "gray"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 70,
                        },
                    },
                ),
                row=2,
                col=1,
            )

            # 4. Security score indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=result.security.score,
                    title={"text": "Security Score"},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "green"},
                        "steps": [
                            {"range": [0, 40], "color": "red"},
                            {"range": [40, 70], "color": "yellow"},
                            {"range": [70, 100], "color": "lightgreen"},
                        ],
                    },
                ),
                row=2,
                col=2,
            )

            # Update layout
            fig.update_layout(
                title_text="HistoCore Analysis Dashboard", showlegend=False, height=800
            )

            # Save
            chart_path = output_path / "interactive_dashboard.html"
            fig.write_html(str(chart_path))

            logger.info(f"Interactive dashboard saved to {chart_path}")
            return chart_path

        except Exception as e:
            logger.error(f"Interactive dashboard generation failed: {e}")
            return None

    def _create_issue_sunburst(self, result: AnalysisResult, output_path: Path) -> Optional[Path]:
        """Create sunburst chart for issue hierarchy."""
        try:
            # Prepare data
            labels = ["All Issues"]
            parents = [""]
            values = [len(result.critical_issues)]

            # Group by dimension
            dimensions = {}
            for issue in result.critical_issues:
                dim = issue.dimension
                if dim not in dimensions:
                    dimensions[dim] = []
                dimensions[dim].append(issue)

            for dim, issues in dimensions.items():
                labels.append(dim.title())
                parents.append("All Issues")
                values.append(len(issues))

            # Create sunburst
            fig = go.Figure(
                go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total")
            )

            fig.update_layout(title="Issue Distribution by Dimension", height=600)

            # Save
            chart_path = output_path / "issue_sunburst.html"
            fig.write_html(str(chart_path))

            logger.info(f"Issue sunburst saved to {chart_path}")
            return chart_path

        except Exception as e:
            logger.error(f"Issue sunburst generation failed: {e}")
            return None

    def _generate_text_charts(self, result: AnalysisResult, output_path: Path) -> Dict[str, str]:
        """Generate fallback text-based charts."""
        charts = {}

        try:
            # Simple text bar chart
            chart_path = output_path / "text_chart.txt"

            with open(chart_path, "w") as f:
                f.write("HistoCore Analysis Scores\n")
                f.write("=" * 50 + "\n\n")

                dimensions = [
                    ("Architecture", result.architecture.score),
                    ("Performance", result.performance.score),
                    ("Coverage", result.coverage.score),
                    ("Code Quality", result.code_quality.score),
                    ("Dependencies", result.dependencies.score),
                    ("Deployment", result.deployment.score),
                    ("Security", result.security.score),
                    ("Scalability", result.scalability.score),
                ]

                for dim, score in dimensions:
                    bar_length = int(score / 2)  # Scale to 50 chars max
                    bar = "█" * bar_length
                    f.write(f"{dim:15} [{score:5.1f}] {bar}\n")

                f.write("\n" + "=" * 50 + "\n")
                f.write(f"Overall Score: {result.overall_score:.1f}/100\n")

            charts["text_chart"] = str(chart_path)
            logger.info(f"Text chart saved to {chart_path}")

        except Exception as e:
            logger.error(f"Text chart generation failed: {e}")

        return charts
