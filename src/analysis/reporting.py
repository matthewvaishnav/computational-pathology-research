"""
Report Generator for HistoCore Project Optimization Analysis System.

Generates comprehensive reports in Markdown, HTML, and PDF formats with
executive summaries, detailed analysis, and prioritized task lists.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import tempfile

from .models import AnalysisResult, Priority, Severity
from .aggregator import ResultAggregator

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates analysis reports in multiple formats."""
    
    def __init__(self):
        """Initialize report generator."""
        self.aggregator = ResultAggregator()
    
    def generate_markdown(self, result: AnalysisResult) -> str:
        """
        Generate comprehensive Markdown report.
        
        Args:
            result: Analysis result to report on
            
        Returns:
            Markdown report as string
        """
        logger.info("Generating Markdown report...")
        
        # Get dimension summaries
        summaries = self.aggregator.get_dimension_summary(result)
        top_issues = self.aggregator.get_top_issues_by_dimension(result)
        
        # Build report sections
        sections = []
        
        # Header
        sections.append(self._generate_header(result))
        
        # Executive Summary
        sections.append(self._generate_executive_summary(result, summaries))
        
        # Overall Metrics
        sections.append(self._generate_overall_metrics(result, summaries))
        
        # Critical Issues
        sections.append(self._generate_critical_issues(result))
        
        # Detailed Analysis by Dimension
        sections.append(self._generate_architecture_section(result.architecture, summaries['architecture']))
        sections.append(self._generate_performance_section(result.performance, summaries['performance']))
        sections.append(self._generate_coverage_section(result.coverage, summaries['coverage']))
        sections.append(self._generate_code_quality_section(result.code_quality, summaries['code_quality']))
        sections.append(self._generate_dependencies_section(result.dependencies, summaries['dependencies']))
        sections.append(self._generate_deployment_section(result.deployment, summaries['deployment']))
        sections.append(self._generate_security_section(result.security, summaries['security']))
        sections.append(self._generate_scalability_section(result.scalability, summaries['scalability']))
        
        # Prioritized Task List
        sections.append(self._generate_task_list(result, top_issues))
        
        # Recommendations
        sections.append(self._generate_recommendations(result, summaries))
        
        # Footer
        sections.append(self._generate_footer(result))
        
        return '\n\n'.join(sections)
    
    def _generate_header(self, result: AnalysisResult) -> str:
        """Generate report header."""
        try:
            timestamp = datetime.fromisoformat(result.timestamp.replace('Z', '+00:00'))
            formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
        except (ValueError, AttributeError):
            # Fallback for invalid timestamps
            formatted_time = result.timestamp
        
        return f"""# HistoCore Project Optimization Analysis Report

**Generated:** {formatted_time}  
**Project:** {result.project_path}  
**Git Commit:** `{result.git_commit}`  
**Overall Score:** {result.overall_score:.1f}/100 {self._get_score_emoji(result.overall_score)}

---"""
    
    def _generate_executive_summary(self, result: AnalysisResult, summaries: Dict[str, Dict[str, Any]]) -> str:
        """Generate executive summary."""
        # Count issues by severity
        critical_count = len([i for i in result.critical_issues if i.severity == Severity.CRITICAL])
        high_count = len([i for i in result.critical_issues if i.severity == Severity.HIGH])
        
        # Get worst performing dimensions
        worst_dims = sorted(summaries.items(), key=lambda x: x[1]['score'])[:3]
        
        summary = f"""## Executive Summary

This analysis evaluated the HistoCore computational pathology project across 8 dimensions of software quality. The project achieved an **overall score of {result.overall_score:.1f}/100**, indicating {"excellent" if result.overall_score >= 80 else "good" if result.overall_score >= 60 else "needs improvement" if result.overall_score >= 40 else "critical"} health.

### Key Findings

- **{len(result.critical_issues)} critical issues** identified requiring immediate attention
- **{critical_count} critical** and **{high_count} high-severity** issues found
- **Top concerns:** {', '.join([dim.replace('_', ' ').title() for dim, _ in worst_dims])}

### Immediate Actions Required

"""
        
        # Add top 3 critical issues
        for i, issue in enumerate(result.critical_issues[:3], 1):
            summary += f"{i}. **{issue.title}** ({issue.severity.value}) - {issue.file_path}\n"
        
        return summary
    
    def _generate_overall_metrics(self, result: AnalysisResult, summaries: Dict[str, Dict[str, Any]]) -> str:
        """Generate overall metrics table."""
        metrics = """## Overall Metrics

| Dimension | Score | Status | Key Metric |
|-----------|-------|--------|------------|"""
        
        for dim_name, summary in summaries.items():
            score = summary['score']
            status = summary['status'].replace('_', ' ').title()
            
            # Get key metric for each dimension
            key_metric = self._get_key_metric(dim_name, summary)
            
            metrics += f"\n| {dim_name.replace('_', ' ').title()} | {score:.1f}/100 | {status} | {key_metric} |"
        
        return metrics
    
    def _get_key_metric(self, dimension: str, summary: Dict[str, Any]) -> str:
        """Get the most important metric for each dimension."""
        if dimension == 'architecture':
            return f"{summary['large_files_count']} large files"
        elif dimension == 'performance':
            return f"{summary['gpu_utilization']:.1f}% GPU utilization"
        elif dimension == 'coverage':
            return f"{summary['line_coverage']:.1f}% line coverage"
        elif dimension == 'code_quality':
            return f"{summary['average_complexity']:.1f} avg complexity"
        elif dimension == 'dependencies':
            return f"{summary['vulnerabilities_count']} vulnerabilities"
        elif dimension == 'deployment':
            return f"{summary['ci_cd_completeness']:.1f}% CI/CD complete"
        elif dimension == 'security':
            return f"{summary['vulnerabilities_count']} security issues"
        elif dimension == 'scalability':
            return f"DDP: {'✓' if summary['ddp_correctness'] else '✗'}"
        return "N/A"
    
    def _generate_critical_issues(self, result: AnalysisResult) -> str:
        """Generate critical issues section."""
        if not result.critical_issues:
            return "## Critical Issues\n\n✅ No critical issues found!"
        
        section = f"## Critical Issues ({len(result.critical_issues)})\n\n"
        
        # Build issues efficiently
        issue_lines = []
        for i, issue in enumerate(result.critical_issues, 1):
            priority_emoji = "🔴" if issue.priority == Priority.P0 else "🟡"
            severity_emoji = "🚨" if issue.severity == Severity.CRITICAL else "⚠️"
            
            issue_lines.append(f"""### {i}. {issue.title} {priority_emoji} {severity_emoji}

**File:** `{issue.file_path}`{f" (Line {issue.line_number})" if issue.line_number else ""}  
**Category:** {issue.category}  
**Priority:** {issue.priority.value} | **Severity:** {issue.severity.value}  
**Effort:** {issue.effort_hours:.1f} hours | **Role:** {issue.role.value}

{issue.description}

**Recommendation:** {issue.recommendation}

""")
        
        section += ''.join(issue_lines)
        return section
    
    def _generate_architecture_section(self, arch, summary: Dict[str, Any]) -> str:
        """Generate architecture analysis section."""
        section = f"""## Architecture Analysis

**Score:** {arch.score:.1f}/100 ({summary['status'].replace('_', ' ').title()})

### Metrics
- **Total Files:** {arch.total_files:,}
- **Large Files (>500 lines):** {len(arch.large_files)}
- **Circular Dependencies:** {len(arch.circular_dependencies)}
- **SOLID Violations:** {len(arch.solid_violations)}

### Large Files Requiring Refactoring
"""
        
        if arch.large_files:
            for file_info in arch.large_files[:10]:  # Top 10
                lines = file_info.get('lines', 0)
                complexity = file_info.get('complexity', 0)
                section += f"- `{file_info.get('path', 'unknown')}` ({lines:,} lines, complexity: {complexity:.1f})\n"
        else:
            section += "✅ No large files detected\n"
        
        section += "\n### Circular Dependencies\n"
        if arch.circular_dependencies:
            for cycle in arch.circular_dependencies[:5]:  # Top 5
                section += f"- {' → '.join(cycle)}\n"
        else:
            section += "✅ No circular dependencies detected\n"
        
        return section
    
    def _generate_performance_section(self, perf, summary: Dict[str, Any]) -> str:
        """Generate performance analysis section."""
        section = f"""## Performance Analysis

**Score:** {perf.score:.1f}/100 ({summary['status'].replace('_', ' ').title()})

### Metrics
- **GPU Utilization:** {perf.gpu_utilization:.1f}%
- **Memory Peak:** {perf.memory_usage_peak_gb:.1f} GB
- **Memory Average:** {perf.memory_usage_avg_gb:.1f} GB
- **Bottlenecks Detected:** {len(perf.bottlenecks)}

### Performance Bottlenecks
"""
        
        if perf.bottlenecks:
            for bottleneck in perf.bottlenecks[:10]:
                func_name = bottleneck.get('function', 'unknown')
                time_ms = bottleneck.get('time_ms', 0)
                file_path = bottleneck.get('file', 'unknown')
                section += f"- `{func_name}` in `{file_path}` ({time_ms:.1f}ms)\n"
        else:
            section += "✅ No significant bottlenecks detected\n"
        
        if perf.flame_graph_path:
            section += f"\n**Flame Graph:** `{perf.flame_graph_path}`\n"
        
        return section
    
    def _generate_coverage_section(self, cov, summary: Dict[str, Any]) -> str:
        """Generate coverage analysis section."""
        section = f"""## Test Coverage Analysis

**Score:** {cov.score:.1f}/100 ({summary['status'].replace('_', ' ').title()})

### Metrics
- **Line Coverage:** {cov.line_coverage:.1f}%
- **Branch Coverage:** {cov.branch_coverage:.1f}%
- **Untested Critical Paths:** {len(cov.untested_critical_paths)}
- **Missing Property Tests:** {len(cov.missing_property_tests)}
- **Flaky Tests:** {len(cov.flaky_tests)}

### Coverage Gaps
"""
        
        if cov.untested_critical_paths:
            section += "**Untested Critical Paths:**\n"
            for path in cov.untested_critical_paths[:10]:
                section += f"- `{path}`\n"
        
        if cov.missing_property_tests:
            section += "\n**Functions Needing Property Tests:**\n"
            for func in cov.missing_property_tests[:10]:
                section += f"- `{func}`\n"
        
        if cov.flaky_tests:
            section += "\n**Flaky Tests:**\n"
            for test in cov.flaky_tests[:5]:
                section += f"- `{test}`\n"
        
        if not any([cov.untested_critical_paths, cov.missing_property_tests, cov.flaky_tests]):
            section += "✅ No significant coverage gaps detected\n"
        
        return section
    
    def _generate_code_quality_section(self, qual, summary: Dict[str, Any]) -> str:
        """Generate code quality section."""
        section = f"""## Code Quality Analysis

**Score:** {qual.score:.1f}/100 ({summary['status'].replace('_', ' ').title()})

### Metrics
- **Average Complexity:** {qual.average_complexity:.1f}
- **High Complexity Functions:** {len(qual.high_complexity_functions)}
- **Code Duplication:** {qual.duplication_percentage:.1f}%
- **Documentation Coverage:** {qual.documentation_coverage:.1f}%
- **PyLint Score:** {qual.pylint_score:.1f}/10

### High Complexity Functions
"""
        
        if qual.high_complexity_functions:
            for func in qual.high_complexity_functions[:10]:
                name = func.get('name', 'unknown')
                complexity = func.get('complexity', 0)
                file_path = func.get('file', 'unknown')
                section += f"- `{name}` in `{file_path}` (complexity: {complexity:.1f})\n"
        else:
            section += "✅ No high complexity functions detected\n"
        
        return section
    
    def _generate_dependencies_section(self, deps, summary: Dict[str, Any]) -> str:
        """Generate dependencies section."""
        section = f"""## Dependencies Analysis

**Score:** {deps.score:.1f}/100 ({summary['status'].replace('_', ' ').title()})

### Metrics
- **Total Dependencies:** {deps.total_dependencies}
- **Security Vulnerabilities:** {len(deps.vulnerabilities)}
- **Outdated Packages:** {len(deps.outdated_packages)}
- **License Issues:** {len(deps.license_issues)}

### Security Vulnerabilities
"""
        
        if deps.vulnerabilities:
            for vuln in deps.vulnerabilities[:10]:
                package = vuln.get('package', 'unknown')
                severity = vuln.get('severity', 'unknown')
                cve = vuln.get('cve', 'N/A')
                section += f"- `{package}` - {severity} ({cve})\n"
        else:
            section += "✅ No security vulnerabilities detected\n"
        
        if deps.outdated_packages:
            section += f"\n### Outdated Packages ({len(deps.outdated_packages)})\n"
            for pkg in deps.outdated_packages[:10]:
                section += f"- `{pkg}`\n"
        
        return section
    
    def _generate_deployment_section(self, deploy, summary: Dict[str, Any]) -> str:
        """Generate deployment section."""
        return f"""## Deployment Analysis

**Score:** {deploy.score:.1f}/100 ({summary['status'].replace('_', ' ').title()})

### Metrics
- **Dockerfile Score:** {deploy.dockerfile_score:.1f}/100
- **Kubernetes Readiness:** {deploy.k8s_readiness:.1f}/100
- **CI/CD Completeness:** {deploy.ci_cd_completeness:.1f}/100
- **Monitoring Score:** {deploy.monitoring_score:.1f}/100

### Recommendations
- Review Dockerfile best practices
- Implement comprehensive health checks
- Add monitoring and observability
- Enhance CI/CD pipeline security
"""
    
    def _generate_security_section(self, sec, summary: Dict[str, Any]) -> str:
        """Generate security section."""
        section = f"""## Security Analysis

**Score:** {sec.score:.1f}/100 ({summary['status'].replace('_', ' ').title()})

### Metrics
- **Security Vulnerabilities:** {len(sec.vulnerabilities)}
- **HIPAA Compliance Score:** {sec.hipaa_compliance_score:.1f}/100
- **Hardcoded Secrets:** {len(sec.hardcoded_secrets)}
- **Injection Risks:** {len(sec.injection_risks)}

### Security Issues
"""
        
        if sec.vulnerabilities:
            for vuln in sec.vulnerabilities[:10]:
                title = vuln.get('title', 'Security vulnerability')
                severity = vuln.get('severity', 'unknown')
                file_path = vuln.get('file', 'unknown')
                section += f"- `{title}` in `{file_path}` ({severity})\n"
        
        if sec.hardcoded_secrets:
            section += f"\n### Hardcoded Secrets ({len(sec.hardcoded_secrets)})\n"
            for secret in sec.hardcoded_secrets[:5]:
                section += f"- `{secret}`\n"
        
        if not sec.vulnerabilities and not sec.hardcoded_secrets:
            section += "✅ No major security issues detected\n"
        
        return section
    
    def _generate_scalability_section(self, scale, summary: Dict[str, Any]) -> str:
        """Generate scalability section."""
        return f"""## Scalability Analysis

**Score:** {scale.score:.1f}/100 ({summary['status'].replace('_', ' ').title()})

### Metrics
- **DDP Implementation:** {'✓ Correct' if scale.ddp_correctness else '✗ Issues detected'}
- **Scaling Efficiency:** {scale.scaling_efficiency}
- **Memory Bottlenecks:** {len(scale.memory_bottlenecks)}
- **Communication Overhead:** {scale.communication_overhead_ms:.1f}ms

### Scaling Recommendations
- {"Optimize DDP implementation" if not scale.ddp_correctness else "DDP implementation looks good"}
- {"Address memory bottlenecks" if scale.memory_bottlenecks else "Memory usage appears optimal"}
- {"Reduce communication overhead" if scale.communication_overhead_ms > 100 else "Communication overhead is acceptable"}
"""
    
    def _generate_task_list(self, result: AnalysisResult, top_issues: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate prioritized task list."""
        section = """## Prioritized Task List

### P0 - Critical (Immediate Action Required)
"""
        
        p0_tasks = [issue for issue in result.critical_issues if issue.priority == Priority.P0]
        if p0_tasks:
            task_lines = []
            for i, task in enumerate(p0_tasks, 1):
                task_lines.append(f"{i}. **{task.title}** ({task.effort_hours:.1f}h, {task.role.value})\n")
                task_lines.append(f"   - {task.recommendation}\n\n")
            section += ''.join(task_lines)
        else:
            section += "✅ No P0 tasks\n\n"
        
        section += "### P1 - High Priority (This Sprint)\n"
        p1_tasks = [issue for issue in result.critical_issues if issue.priority == Priority.P1]
        if p1_tasks:
            task_lines = []
            for i, task in enumerate(p1_tasks, 1):
                task_lines.append(f"{i}. **{task.title}** ({task.effort_hours:.1f}h, {task.role.value})\n")
                task_lines.append(f"   - {task.recommendation}\n\n")
            section += ''.join(task_lines)
        else:
            section += "✅ No P1 tasks\n\n"
        
        return section
    
    def _generate_recommendations(self, result: AnalysisResult, summaries: Dict[str, Dict[str, Any]]) -> str:
        """Generate overall recommendations."""
        recommendations = []
        
        # Score-based recommendations
        if result.overall_score < 40:
            recommendations.append("🚨 **Critical**: Immediate architectural review required")
        elif result.overall_score < 60:
            recommendations.append("⚠️ **High Priority**: Focus on security and testing improvements")
        elif result.overall_score < 80:
            recommendations.append("📈 **Medium Priority**: Optimize performance and code quality")
        else:
            recommendations.append("✅ **Good Health**: Focus on continuous improvement")
        
        # Dimension-specific recommendations
        if summaries['security']['score'] < 60:
            recommendations.append("🔒 Prioritize security vulnerabilities and HIPAA compliance")
        
        if summaries['coverage']['score'] < 70:
            recommendations.append("🧪 Increase test coverage, especially for critical paths")
        
        if summaries['performance']['score'] < 60:
            recommendations.append("⚡ Address performance bottlenecks and GPU utilization")
        
        section = "## Recommendations\n\n"
        section += '\n'.join(f"- {rec}" for rec in recommendations)
        section += '\n'
        
        return section
    
    def _generate_footer(self, result: AnalysisResult) -> str:
        """Generate report footer."""
        return f"""---

**Report generated by HistoCore Project Optimization Analysis System**  
**Timestamp:** {result.timestamp}  
**Analysis Duration:** {len(result.critical_issues)} issues analyzed across 8 dimensions

For questions or support, please refer to the project documentation.
"""
    
    def _get_score_emoji(self, score: float) -> str:
        """Get emoji for score."""
        if score >= 80:
            return "🟢"
        elif score >= 60:
            return "🟡"
        elif score >= 40:
            return "🟠"
        else:
            return "🔴"
    
    def generate_html(self, result: AnalysisResult, output_path: Optional[str] = None) -> str:
        """
        Generate HTML report from Markdown.
        
        Args:
            result: Analysis result
            output_path: Optional output file path
            
        Returns:
            HTML content or file path
        """
        logger.info("Generating HTML report...")
        
        # Generate Markdown first
        markdown_content = self.generate_markdown(result)
        
        # Try to convert using pandoc
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as md_file:
                md_file.write(markdown_content)
                md_file.flush()
                
                if output_path:
                    # Convert to HTML file
                    subprocess.run([
                        'pandoc', md_file.name, '-o', output_path,
                        '--standalone', '--css', 'style.css'
                    ], check=True)
                    logger.info(f"HTML report saved to {output_path}")
                    return output_path
                else:
                    # Convert to HTML string
                    result = subprocess.run([
                        'pandoc', md_file.name, '-t', 'html',
                        '--standalone'
                    ], capture_output=True, text=True, check=True)
                    return result.stdout
                    
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Pandoc conversion failed: {e}")
            # Fallback: basic HTML wrapper
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>HistoCore Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
<pre>{markdown_content}</pre>
</body>
</html>"""
            
            if output_path:
                Path(output_path).write_text(html_content, encoding='utf-8')
                logger.info(f"Basic HTML report saved to {output_path}")
                return output_path
            else:
                return html_content
    
    def generate_pdf(self, result: AnalysisResult, output_path: str) -> str:
        """
        Generate PDF report.
        
        Args:
            result: Analysis result
            output_path: Output PDF file path
            
        Returns:
            Output file path
        """
        logger.info("Generating PDF report...")
        
        try:
            # Generate HTML first
            html_content = self.generate_html(result)
            
            # Convert HTML to PDF using weasyprint or pandoc
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as html_file:
                html_file.write(html_content)
                html_file.flush()
                
                try:
                    # Try weasyprint first
                    import weasyprint
                    weasyprint.HTML(filename=html_file.name).write_pdf(output_path)
                    logger.info(f"PDF report saved to {output_path} (weasyprint)")
                    return output_path
                except ImportError:
                    # Fallback to pandoc
                    subprocess.run([
                        'pandoc', html_file.name, '-o', output_path,
                        '--pdf-engine=wkhtmltopdf'
                    ], check=True)
                    logger.info(f"PDF report saved to {output_path} (pandoc)")
                    return output_path
                    
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            # Fallback: save as text file
            txt_path = output_path.replace('.pdf', '.txt')
            markdown_content = self.generate_markdown(result)
            Path(txt_path).write_text(markdown_content, encoding='utf-8')
            logger.info(f"Saved as text file instead: {txt_path}")
            return txt_path
