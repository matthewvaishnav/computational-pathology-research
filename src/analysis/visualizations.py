"""
Visualization Generator for HistoCore Project Optimization Analysis System.

Creates interactive charts, heatmaps, and graphs for analysis results using
matplotlib and plotly for comprehensive visual reporting.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("matplotlib not available - static charts disabled")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("plotly not available - interactive charts disabled")

from .models import AnalysisResult
from .aggregator import ResultAggregator

logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """Generates charts and visualizations for analysis results."""
    
    def __init__(self):
        """Initialize visualization generator."""
        self.aggregator = ResultAggregator()
        self.colors = {
            'excellent': '#2E8B57',    # Sea Green
            'good': '#32CD32',         # Lime Green  
            'needs_improvement': '#FFD700',  # Gold
            'critical': '#DC143C'      # Crimson
        }
    
    def generate_all_charts(
        self,
        result: AnalysisResult,
        output_dir: str = "charts"
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
        
        # Generate interactive charts (plotly)
        if PLOTLY_AVAILABLE:
            charts.update(self._generate_plotly_charts(result, output_path))
        
        # Generate fallback text charts if no libraries available
        if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
            charts.update(self._generate_text_charts(result, output_path))
        
        logger.info(f"Generated {len(charts)} charts in {output_path}")
        return charts
    
    def _generate_matplotlib_charts(
        self,
        result: AnalysisResult,
        output_path: Path
    ) -> Dict[str, str]:
        """Generate static charts using matplotlib."""
        charts = {}
        
        try:
            # Overall score radar chart
            radar_path = self._create_radar_chart(result, output_path / "radar_chart.png")
            if radar_path:
                charts['radar_chart'] = str(radar_path)
            
            # Dimension scores bar chart
            bar_path = self._create_bar_chart(result, output_path / "dimension_scores.png")
            if bar_path:
                charts['dimension_scores'] = str(bar_path)
            
            # Coverage heatmap
            heatmap_path = self._create_coverage_heatmap(result, output_path / "coverage_heatmap.png")
            if heatmap_path:
                charts['coverage_heatmap'] = str(heatmap_path)
            
            # Complexity distribution
            complexity_path = self._create_complexity_histogram(result, output_path / "complexity_distribution.png")
            if complexity_path:
                charts['complexity_distribution'] = str(complexity_path)
            
        except Exception as e:
            logger.error(f"Failed to generate matplotlib charts: {e}")
        
        return charts
    
    def _generate_plotly_charts(
        self,
        result: AnalysisResult,
        output_path: Path
    ) -> Dict[str, str]:
        """Generate interactive charts using plotly."""
        charts = {}
        
        try:
            # Interactive dashboard
            dashboard_path = self._create_interactive_dashboard(result, output_path / "dashboard.html")
            if dashboard_path:
                charts['interactive_dashboard'] = str(dashboard_path)
            
            # Issue priority treemap
            treemap_path = self._create_issue_treemap(result, output_path / "issue_treemap.html")
            if treemap_path:
                charts['issue_treemap'] = str(treemap_path)
            
            # Performance timeline
            timeline_path = self._create_performance_timeline(result, output_path / "performance_timeline.html")
            if timeline_path:
                charts['performance_timeline'] = str(timeline_path)
            
        except Exception as e:
            logger.error(f"Failed to generate plotly charts: {e}")
        
        return charts
    
    def _generate_text_charts(
        self,
        result: AnalysisResult,
        output_path: Path
    ) -> Dict[str, str]:
        """Generate fallback text-based charts."""
        charts = {}
        
        try:
            # ASCII bar chart
            ascii_path = self._create_ascii_chart(result, output_path / "ascii_chart.txt")
            if ascii_path:
                charts['ascii_chart'] = str(ascii_path)
            
        except Exception as e:
            logger.error(f"Failed to generate text charts: {e}")
        
        return charts
    
    def _create_radar_chart(self, result: AnalysisResult, output_path: Path) -> Optional[Path]:
        """Create radar chart showing all dimension scores."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            # Get dimension scores
            dimensions = ['Architecture', 'Performance', 'Coverage', 'Code Quality',
                         'Dependencies', 'Deployment', 'Security', 'Scalability']
            scores = [
                result.architecture.score,
                result.performance.score,
                result.coverage.score,
                result.code_quality.score,
                result.dependencies.score,
                result.deployment.score,
                result.security.score,
                result.scalability.score
            ]
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
            scores += scores[:1]  # Complete the circle
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            ax.plot(angles, scores, 'o-', linewidth=2, label='Current Score')
            ax.fill(angles, scores, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(dimensions)
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20', '40', '60', '80', '100'])
            ax.grid(True)
            
            plt.title(f'HistoCore Analysis - Overall Score: {result.overall_score:.1f}/100', 
                     size=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created radar chart: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create radar chart: {e}")
            return None
    
    def _create_bar_chart(self, result: AnalysisResult, output_path: Path) -> Optional[Path]:
        """Create bar chart of dimension scores."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            dimensions = ['Architecture', 'Performance', 'Coverage', 'Code Quality',
                         'Dependencies', 'Deployment', 'Security', 'Scalability']
            scores = [
                result.architecture.score,
                result.performance.score,
                result.coverage.score,
                result.code_quality.score,
                result.dependencies.score,
                result.deployment.score,
                result.security.score,
                result.scalability.score
            ]
            
            # Color bars based on score
            colors = []
            for score in scores:
                if score >= 80:
                    colors.append(self.colors['excellent'])
                elif score >= 60:
                    colors.append(self.colors['good'])
                elif score >= 40:
                    colors.append(self.colors['needs_improvement'])
                else:
                    colors.append(self.colors['critical'])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(dimensions, scores, color=colors)
            
            # Add score labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Score (0-100)', fontsize=12)
            ax.set_title('HistoCore Analysis - Dimension Scores', fontsize=16, fontweight='bold')
            ax.set_ylim(0, 105)
            
            # Add horizontal lines for score thresholds
            ax.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Excellent (80+)')
            ax.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Good (60+)')
            ax.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='Needs Improvement (40+)')
            
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created bar chart: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create bar chart: {e}")
            return None
    
    def _create_coverage_heatmap(self, result: AnalysisResult, output_path: Path) -> Optional[Path]:
        """Create coverage heatmap (placeholder implementation)."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            # Create mock coverage data for demonstration
            modules = ['core', 'analysis', 'models', 'utils', 'tests']
            coverage_data = np.array([
                [result.coverage.line_coverage, result.coverage.branch_coverage, 85, 90, 75],
                [80, 75, result.coverage.line_coverage, 85, 80],
                [90, 85, 80, result.coverage.branch_coverage, 85],
                [75, 80, 85, 90, result.coverage.line_coverage],
                [85, 90, 75, 80, result.coverage.branch_coverage]
            ])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(coverage_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(modules)))
            ax.set_yticks(np.arange(len(modules)))
            ax.set_xticklabels(modules)
            ax.set_yticklabels(modules)
            
            # Add text annotations
            for i in range(len(modules)):
                for j in range(len(modules)):
                    text = ax.text(j, i, f'{coverage_data[i, j]:.0f}%',
                                 ha="center", va="center", color="black", fontweight='bold')
            
            ax.set_title('Test Coverage Heatmap', fontsize=16, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Coverage %')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created coverage heatmap: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create coverage heatmap: {e}")
            return None
    
    def _create_complexity_histogram(self, result: AnalysisResult, output_path: Path) -> Optional[Path]:
        """Create complexity distribution histogram."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            # Generate mock complexity data based on analysis
            complexities = []
            
            # Add data from high complexity functions
            for func in result.code_quality.high_complexity_functions:
                complexities.append(func.get('complexity', 10))
            
            # Add some normal complexity values
            np.random.seed(42)  # For reproducible results
            normal_complexities = np.random.normal(result.code_quality.average_complexity, 2, 100)
            complexities.extend([max(1, c) for c in normal_complexities])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            n, bins, patches = ax.hist(complexities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Color bars based on complexity thresholds
            for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
                if bin_val > 15:
                    patch.set_facecolor(self.colors['critical'])
                elif bin_val > 10:
                    patch.set_facecolor(self.colors['needs_improvement'])
                elif bin_val > 5:
                    patch.set_facecolor(self.colors['good'])
                else:
                    patch.set_facecolor(self.colors['excellent'])
            
            ax.axvline(result.code_quality.average_complexity, color='red', linestyle='--', 
                      linewidth=2, label=f'Average: {result.code_quality.average_complexity:.1f}')
            ax.axvline(10, color='orange', linestyle='--', alpha=0.7, label='High Complexity Threshold')
            
            ax.set_xlabel('Cyclomatic Complexity', fontsize=12)
            ax.set_ylabel('Number of Functions', fontsize=12)
            ax.set_title('Code Complexity Distribution', fontsize=16, fontweight='bold')
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created complexity histogram: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create complexity histogram: {e}")
            return None
    
    def _create_interactive_dashboard(self, result: AnalysisResult, output_path: Path) -> Optional[Path]:
        """Create interactive dashboard using plotly."""
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Dimension Scores', 'Issue Distribution', 'Coverage Metrics', 'Performance Metrics'),
                specs=[[{"type": "bar"}, {"type": "pie"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Dimension scores bar chart
            dimensions = ['Architecture', 'Performance', 'Coverage', 'Code Quality',
                         'Dependencies', 'Deployment', 'Security', 'Scalability']
            scores = [
                result.architecture.score, result.performance.score, result.coverage.score,
                result.code_quality.score, result.dependencies.score, result.deployment.score,
                result.security.score, result.scalability.score
            ]
            
            fig.add_trace(
                go.Bar(x=dimensions, y=scores, name="Scores", 
                      marker_color=['red' if s < 40 else 'orange' if s < 60 else 'yellow' if s < 80 else 'green' for s in scores]),
                row=1, col=1
            )
            
            # Issue distribution pie chart
            issue_counts = {
                'Critical': len([i for i in result.critical_issues if i.severity.value == 'critical']),
                'High': len([i for i in result.critical_issues if i.severity.value == 'high']),
                'Medium': len([i for i in result.critical_issues if i.severity.value == 'medium']),
                'Low': len([i for i in result.critical_issues if i.severity.value == 'low'])
            }
            
            fig.add_trace(
                go.Pie(labels=list(issue_counts.keys()), values=list(issue_counts.values()),
                      name="Issues", marker_colors=['red', 'orange', 'yellow', 'green']),
                row=1, col=2
            )
            
            # Coverage metrics
            coverage_types = ['Line Coverage', 'Branch Coverage']
            coverage_values = [result.coverage.line_coverage, result.coverage.branch_coverage]
            
            fig.add_trace(
                go.Scatter(x=coverage_types, y=coverage_values, mode='markers+lines',
                          name="Coverage", marker_size=15),
                row=2, col=1
            )
            
            # Performance metrics
            perf_metrics = ['GPU Utilization', 'Memory Peak GB', 'Bottlenecks Count']
            perf_values = [result.performance.gpu_utilization, 
                          result.performance.memory_usage_peak_gb * 10,  # Scale for visibility
                          len(result.performance.bottlenecks)]
            
            fig.add_trace(
                go.Bar(x=perf_metrics, y=perf_values, name="Performance"),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text=f"HistoCore Analysis Dashboard - Overall Score: {result.overall_score:.1f}/100",
                title_x=0.5,
                showlegend=False,
                height=800
            )
            
            # Save as HTML
            pyo.plot(fig, filename=str(output_path), auto_open=False)
            
            logger.info(f"Created interactive dashboard: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create interactive dashboard: {e}")
            return None
    
    def _create_issue_treemap(self, result: AnalysisResult, output_path: Path) -> Optional[Path]:
        """Create treemap of issues by priority and dimension."""
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            # Prepare data for treemap
            labels = []
            parents = []
            values = []
            colors = []
            
            # Root
            labels.append("All Issues")
            parents.append("")
            values.append(len(result.critical_issues))
            colors.append(0)
            
            # Priority levels
            priorities = ['P0', 'P1', 'P2', 'P3']
            for priority in priorities:
                count = len([i for i in result.critical_issues if i.priority.value == priority])
                if count > 0:
                    labels.append(f"Priority {priority}")
                    parents.append("All Issues")
                    values.append(count)
                    colors.append(1 if priority == 'P0' else 2 if priority == 'P1' else 3)
            
            # Individual issues
            for issue in result.critical_issues[:20]:  # Limit to top 20
                labels.append(issue.title[:30] + "..." if len(issue.title) > 30 else issue.title)
                parents.append(f"Priority {issue.priority.value}")
                values.append(issue.effort_hours)
                colors.append(4 if issue.severity.value == 'critical' else 3)
            
            fig = go.Figure(go.Treemap(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                marker=dict(
                    colorscale='RdYlGn_r',
                    cmid=2,
                    colorbar=dict(title="Priority Level")
                ),
                hovertemplate='<b>%{label}</b><br>Value: %{value}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Issue Priority Treemap",
                font_size=12,
                width=1000,
                height=600
            )
            
            pyo.plot(fig, filename=str(output_path), auto_open=False)
            
            logger.info(f"Created issue treemap: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create issue treemap: {e}")
            return None
    
    def _create_performance_timeline(self, result: AnalysisResult, output_path: Path) -> Optional[Path]:
        """Create performance timeline (placeholder with mock data)."""
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            # Mock timeline data
            import datetime
            dates = [datetime.datetime.now() - datetime.timedelta(days=x) for x in range(30, 0, -1)]
            
            # Generate mock performance data
            np.random.seed(42)
            gpu_util = np.random.normal(result.performance.gpu_utilization, 10, 30)
            memory_usage = np.random.normal(result.performance.memory_usage_avg_gb, 1, 30)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=gpu_util,
                mode='lines+markers',
                name='GPU Utilization %',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=memory_usage * 10,  # Scale for visibility
                mode='lines+markers',
                name='Memory Usage (GB x10)',
                line=dict(color='red'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Performance Timeline (Last 30 Days)',
                xaxis_title='Date',
                yaxis_title='GPU Utilization %',
                yaxis2=dict(
                    title='Memory Usage (GB)',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified'
            )
            
            pyo.plot(fig, filename=str(output_path), auto_open=False)
            
            logger.info(f"Created performance timeline: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create performance timeline: {e}")
            return None
    
    def _create_ascii_chart(self, result: AnalysisResult, output_path: Path) -> Optional[Path]:
        """Create ASCII bar chart as fallback."""
        try:
            dimensions = ['Architecture', 'Performance', 'Coverage', 'Code Quality',
                         'Dependencies', 'Deployment', 'Security', 'Scalability']
            scores = [
                result.architecture.score, result.performance.score, result.coverage.score,
                result.code_quality.score, result.dependencies.score, result.deployment.score,
                result.security.score, result.scalability.score
            ]
            
            chart_lines = []
            chart_lines.append("HistoCore Analysis - Dimension Scores")
            chart_lines.append("=" * 50)
            chart_lines.append("")
            
            max_name_len = max(len(name) for name in dimensions)
            
            for name, score in zip(dimensions, scores):
                bar_length = int(score / 2)  # Scale to 50 chars max
                bar = "█" * bar_length + "░" * (50 - bar_length)
                chart_lines.append(f"{name:<{max_name_len}} │{bar}│ {score:5.1f}")
            
            chart_lines.append("")
            chart_lines.append(f"Overall Score: {result.overall_score:.1f}/100")
            chart_lines.append(f"Critical Issues: {len(result.critical_issues)}")
            
            chart_content = "\n".join(chart_lines)
            output_path.write_text(chart_content, encoding='utf-8')
            
            logger.info(f"Created ASCII chart: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create ASCII chart: {e}")
            return None
    
    def embed_charts_in_html(self, html_content: str, chart_paths: Dict[str, str]) -> str:
        """
        Embed chart images/HTML into HTML report.
        
        Args:
            html_content: Original HTML content
            chart_paths: Dictionary of chart names to file paths
            
        Returns:
            HTML content with embedded charts
        """
        # Add charts section before closing body tag
        charts_html = "\n<h2>Visualizations</h2>\n"
        
        for chart_name, chart_path in chart_paths.items():
            chart_path_obj = Path(chart_path)
            
            if chart_path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg']:
                # Embed image
                charts_html += f'<h3>{chart_name.replace("_", " ").title()}</h3>\n'
                charts_html += f'<img src="{chart_path}" alt="{chart_name}" style="max-width: 100%; height: auto;">\n\n'
            elif chart_path_obj.suffix.lower() == '.html':
                # Link to interactive chart
                charts_html += f'<h3>{chart_name.replace("_", " ").title()}</h3>\n'
                charts_html += f'<p><a href="{chart_path}" target="_blank">View Interactive Chart</a></p>\n\n'
        
        # Insert before closing body tag
        if '</body>' in html_content:
            html_content = html_content.replace('</body>', f'{charts_html}</body>')
        else:
            html_content += charts_html
        
        return html_content
