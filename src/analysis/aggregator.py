"""
Result aggregator for HistoCore Project Optimization Analysis System.

Merges results from all 8 analyzers into unified AnalysisResult.
"""

from typing import List, Dict, Any
from datetime import datetime
import subprocess
from pathlib import Path

from .models import (
    AnalysisResult,
    Issue,
    ArchitectureAnalysis,
    PerformanceAnalysis,
    CoverageAnalysis,
    CodeQualityAnalysis,
    DependencyAnalysis,
    DeploymentAnalysis,
    SecurityAnalysis,
    ScalabilityAnalysis,
    Severity,
    Priority
)


class ResultAggregator:
    """Aggregates results from all analysis dimensions."""

    def __init__(self, project_path: str):
        """
        Initialize aggregator.

        Args:
            project_path: Path to project root directory
        """
        self.project_path = project_path

    def aggregate(
        self,
        architecture: ArchitectureAnalysis,
        performance: PerformanceAnalysis,
        coverage: CoverageAnalysis,
        code_quality: CodeQualityAnalysis,
        dependencies: DependencyAnalysis,
        deployment: DeploymentAnalysis,
        security: SecurityAnalysis,
        scalability: ScalabilityAnalysis
    ) -> AnalysisResult:
        """
        Merge results from all 8 analyzers into unified AnalysisResult.

        Args:
            architecture: Architecture analysis results
            performance: Performance profiling results
            coverage: Test coverage analysis results
            code_quality: Code quality metrics
            dependencies: Dependency security and health
            deployment: Deployment readiness assessment
            security: Security vulnerability assessment
            scalability: Scalability assessment

        Returns:
            Unified AnalysisResult with aggregated data
        """
        # Get git commit hash
        git_commit = self._get_git_commit()

        # Compute overall score (weighted average)
        overall_score = self._compute_overall_score(
            architecture, performance, coverage, code_quality,
            dependencies, deployment, security, scalability
        )

        # Extract critical issues from all dimensions
        critical_issues = self._extract_critical_issues(
            architecture, performance, coverage, code_quality,
            dependencies, deployment, security, scalability
        )

        # Deduplicate issues
        critical_issues = self._deduplicate_issues(critical_issues)

        # Sort by priority and severity
        critical_issues = self._sort_issues(critical_issues)

        # Create unified result
        result = AnalysisResult(
            timestamp=datetime.now().isoformat(),
            project_path=self.project_path,
            git_commit=git_commit,
            architecture=architecture,
            performance=performance,
            coverage=coverage,
            code_quality=code_quality,
            dependencies=dependencies,
            deployment=deployment,
            security=security,
            scalability=scalability,
            overall_score=overall_score,
            critical_issues=critical_issues
        )

        return result

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return "unknown"

    def _compute_overall_score(
        self,
        architecture: ArchitectureAnalysis,
        performance: PerformanceAnalysis,
        coverage: CoverageAnalysis,
        code_quality: CodeQualityAnalysis,
        dependencies: DependencyAnalysis,
        deployment: DeploymentAnalysis,
        security: SecurityAnalysis,
        scalability: ScalabilityAnalysis
    ) -> float:
        """
        Compute weighted overall health score (0-100).

        Weights:
        - Security: 25% (highest priority for production)
        - Coverage: 20% (reliability)
        - Code Quality: 15%
        - Dependencies: 15%
        - Architecture: 10%
        - Performance: 5%
        - Deployment: 5%
        - Scalability: 5%
        """
        weights = {
            'security': 0.25,
            'coverage': 0.20,
            'code_quality': 0.15,
            'dependencies': 0.15,
            'architecture': 0.10,
            'performance': 0.05,
            'deployment': 0.05,
            'scalability': 0.05
        }

        scores = {
            'security': security.score,
            'coverage': coverage.score,
            'code_quality': code_quality.score,
            'dependencies': dependencies.score,
            'architecture': architecture.score,
            'performance': performance.score,
            'deployment': deployment.score,
            'scalability': scalability.score
        }

        overall = sum(scores[dim] * weights[dim] for dim in weights)
        return round(overall, 2)

    def _extract_critical_issues(
        self,
        architecture: ArchitectureAnalysis,
        performance: PerformanceAnalysis,
        coverage: CoverageAnalysis,
        code_quality: CodeQualityAnalysis,
        dependencies: DependencyAnalysis,
        deployment: DeploymentAnalysis,
        security: SecurityAnalysis,
        scalability: ScalabilityAnalysis
    ) -> List[Issue]:
        """Extract critical and high-severity issues from all dimensions."""
        issues = []

        # Architecture issues
        issues.extend(architecture.solid_violations)

        # Security issues (convert vulnerabilities to Issue objects)
        for vuln in security.vulnerabilities:
            if vuln.get('severity') in ['critical', 'high']:
                issue = Issue(
                    id=f"security-{vuln.get('id', 'unknown')}",
                    dimension='security',
                    severity=Severity(vuln['severity']),
                    category='vulnerability',
                    title=vuln.get('title', 'Security vulnerability'),
                    description=vuln.get('description', ''),
                    file_path=vuln.get('file', ''),
                    line_number=vuln.get('line'),
                    recommendation=vuln.get('recommendation', ''),
                    effort_hours=vuln.get('effort_hours', 2.0),
                    priority=Priority.P0 if vuln['severity'] == 'critical' else Priority.P1,
                    role=vuln.get('role', 'security')
                )
                issues.append(issue)

        # Dependency vulnerabilities
        for vuln in dependencies.vulnerabilities:
            if vuln.get('severity') in ['critical', 'high']:
                issue = Issue(
                    id=f"dependency-{vuln.get('cve_id', 'unknown')}",
                    dimension='dependencies',
                    severity=Severity(vuln['severity']),
                    category='cve',
                    title=f"CVE in {vuln.get('package', 'unknown')}",
                    description=vuln.get('description', ''),
                    file_path='requirements.txt',
                    recommendation=f"Upgrade to {vuln.get('fix_version', 'latest')}",
                    effort_hours=1.0,
                    priority=Priority.P0 if vuln['severity'] == 'critical' else Priority.P1
                )
                issues.append(issue)

        # Coverage gaps (critical paths)
        for path in coverage.untested_critical_paths[:10]:  # Top 10
            issue = Issue(
                id=f"coverage-{hash(path) % 10000}",
                dimension='coverage',
                severity=Severity.HIGH,
                category='untested_critical_path',
                title=f"Untested critical path: {Path(path).name}",
                description=f"Critical code path lacks test coverage: {path}",
                file_path=path,
                recommendation="Add unit tests covering error handling and edge cases",
                effort_hours=4.0,
                priority=Priority.P1
            )
            issues.append(issue)

        # Code quality (high complexity)
        for func in code_quality.high_complexity_functions[:5]:  # Top 5
            if func.get('complexity', 0) > 15:  # Very high complexity
                issue = Issue(
                    id=f"complexity-{func.get('name', 'unknown')}",
                    dimension='code_quality',
                    severity=Severity.MEDIUM,
                    category='complexity',
                    title=f"High complexity: {func.get('name', 'unknown')}",
                    description=f"Cyclomatic complexity: {func.get('complexity', 0)}",
                    file_path=func.get('file', ''),
                    line_number=func.get('line'),
                    recommendation="Refactor into smaller functions",
                    effort_hours=8.0,
                    priority=Priority.P2
                )
                issues.append(issue)

        # Performance bottlenecks
        for bottleneck in performance.bottlenecks[:3]:  # Top 3
            if bottleneck.get('time_ms', 0) > 500:  # >500ms
                issue = Issue(
                    id=f"performance-{bottleneck.get('operation', 'unknown')}",
                    dimension='performance',
                    severity=Severity.MEDIUM,
                    category='bottleneck',
                    title=f"Performance bottleneck: {bottleneck.get('operation', 'unknown')}",
                    description=f"Operation takes {bottleneck.get('time_ms', 0):.1f}ms",
                    file_path='',
                    recommendation="Profile and optimize hot path",
                    effort_hours=6.0,
                    priority=Priority.P2
                )
                issues.append(issue)

        return issues

    def _deduplicate_issues(self, issues: List[Issue]) -> List[Issue]:
        """
        Remove duplicate issues based on file_path and title.

        Args:
            issues: List of issues potentially containing duplicates

        Returns:
            Deduplicated list of issues
        """
        seen = set()
        deduplicated = []

        for issue in issues:
            # Create unique key from file_path and title
            key = (issue.file_path, issue.title)
            if key not in seen:
                seen.add(key)
                deduplicated.append(issue)

        return deduplicated

    def _sort_issues(self, issues: List[Issue]) -> List[Issue]:
        """
        Sort issues by priority (P0 > P1 > P2 > P3) then severity.

        Args:
            issues: List of issues to sort

        Returns:
            Sorted list of issues
        """
        priority_order = {Priority.P0: 0, Priority.P1: 1, Priority.P2: 2, Priority.P3: 3}
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3
        }

        return sorted(
            issues,
            key=lambda x: (priority_order[x.priority], severity_order[x.severity])
        )
