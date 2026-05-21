"""
Regression Detector for HistoCore Project Optimization Analysis System.

Compares baseline vs current analysis results to detect regressions in:
- Test coverage (>2% decrease)
- Performance metrics (>10% slowdown)
- Security vulnerabilities (new CVEs)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Tuple

from .models import AnalysisResult


class RegressionType(str, Enum):
    """Types of regressions that can be detected."""

    COVERAGE = "coverage"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CODE_QUALITY = "code_quality"


class RegressionSeverity(str, Enum):
    """Severity levels for regressions."""

    CRITICAL = "critical"  # Blocks CI
    HIGH = "high"  # Warning
    MEDIUM = "medium"  # Info
    LOW = "low"  # Info


@dataclass
class Regression:
    """Individual regression finding."""

    type: RegressionType
    severity: RegressionSeverity
    metric: str
    baseline_value: float
    current_value: float
    change_percentage: float
    description: str
    root_cause: str = ""
    recommendation: str = ""


@dataclass
class RegressionReport:
    """Comprehensive regression analysis report."""

    has_regressions: bool
    critical_regressions: List[Regression] = field(default_factory=list)
    high_regressions: List[Regression] = field(default_factory=list)
    medium_regressions: List[Regression] = field(default_factory=list)
    low_regressions: List[Regression] = field(default_factory=list)
    improvements: List[Regression] = field(default_factory=list)
    summary: str = ""

    def should_fail_ci(self) -> bool:
        """Determine if CI build should fail based on critical regressions."""
        return len(self.critical_regressions) > 0

    def get_all_regressions(self) -> List[Regression]:
        """Get all regressions sorted by severity."""
        return (
            self.critical_regressions
            + self.high_regressions
            + self.medium_regressions
            + self.low_regressions
        )


class RegressionDetector:
    """
    Detects regressions by comparing baseline and current analysis results.

    Thresholds:
    - Coverage: >2% decrease is critical
    - Performance: >10% slowdown is critical
    - Security: Any new CVE is critical
    """

    def __init__(self, coverage_threshold: float = 2.0, performance_threshold: float = 10.0):
        """
        Initialize regression detector.

        Args:
            coverage_threshold: Coverage decrease % to flag as critical (default: 2.0)
            performance_threshold: Performance slowdown % to flag as critical (default: 10.0)
        """
        self.coverage_threshold = coverage_threshold
        self.performance_threshold = performance_threshold

    def detect_regressions(
        self, baseline: AnalysisResult, current: AnalysisResult
    ) -> RegressionReport:
        """
        Compare baseline vs current analysis results and detect regressions.

        Args:
            baseline: Baseline analysis results (e.g., from main branch)
            current: Current analysis results (e.g., from PR branch)

        Returns:
            RegressionReport with all detected regressions and improvements
        """
        regressions: List[Regression] = []
        improvements: List[Regression] = []

        # Detect coverage regressions
        coverage_results = self._detect_coverage_regressions(baseline, current)
        regressions.extend(coverage_results[0])
        improvements.extend(coverage_results[1])

        # Detect performance regressions
        perf_results = self._detect_performance_regressions(baseline, current)
        regressions.extend(perf_results[0])
        improvements.extend(perf_results[1])

        # Detect security regressions
        security_results = self._detect_security_regressions(baseline, current)
        regressions.extend(security_results[0])
        improvements.extend(security_results[1])

        # Detect code quality regressions
        quality_results = self._detect_code_quality_regressions(baseline, current)
        regressions.extend(quality_results[0])
        improvements.extend(quality_results[1])

        # Categorize by severity
        critical = [r for r in regressions if r.severity == RegressionSeverity.CRITICAL]
        high = [r for r in regressions if r.severity == RegressionSeverity.HIGH]
        medium = [r for r in regressions if r.severity == RegressionSeverity.MEDIUM]
        low = [r for r in regressions if r.severity == RegressionSeverity.LOW]

        # Generate summary
        summary = self._generate_summary(critical, high, medium, low, improvements)

        return RegressionReport(
            has_regressions=len(regressions) > 0,
            critical_regressions=critical,
            high_regressions=high,
            medium_regressions=medium,
            low_regressions=low,
            improvements=improvements,
            summary=summary,
        )

    def _detect_coverage_regressions(
        self, baseline: AnalysisResult, current: AnalysisResult
    ) -> Tuple[List[Regression], List[Regression]]:
        """Detect test coverage regressions."""
        regressions = []
        improvements = []

        # Line coverage
        baseline_line = baseline.coverage.line_coverage
        current_line = current.coverage.line_coverage
        line_change = current_line - baseline_line

        if line_change < 0:
            severity = (
                RegressionSeverity.CRITICAL
                if abs(line_change) >= self.coverage_threshold
                else RegressionSeverity.HIGH
            )
            regressions.append(
                Regression(
                    type=RegressionType.COVERAGE,
                    severity=severity,
                    metric="line_coverage",
                    baseline_value=baseline_line,
                    current_value=current_line,
                    change_percentage=line_change,
                    description=f"Line coverage decreased by {abs(line_change):.1f}%",
                    root_cause="New code added without tests or existing tests removed",
                    recommendation="Add tests for uncovered code paths",
                )
            )
        elif line_change > 0:
            improvements.append(
                Regression(
                    type=RegressionType.COVERAGE,
                    severity=RegressionSeverity.LOW,
                    metric="line_coverage",
                    baseline_value=baseline_line,
                    current_value=current_line,
                    change_percentage=line_change,
                    description=f"Line coverage improved by {line_change:.1f}%",
                )
            )

        # Branch coverage
        baseline_branch = baseline.coverage.branch_coverage
        current_branch = current.coverage.branch_coverage
        branch_change = current_branch - baseline_branch

        if branch_change < 0:
            severity = (
                RegressionSeverity.CRITICAL
                if abs(branch_change) >= self.coverage_threshold
                else RegressionSeverity.HIGH
            )
            regressions.append(
                Regression(
                    type=RegressionType.COVERAGE,
                    severity=severity,
                    metric="branch_coverage",
                    baseline_value=baseline_branch,
                    current_value=current_branch,
                    change_percentage=branch_change,
                    description=f"Branch coverage decreased by {abs(branch_change):.1f}%",
                    root_cause="New conditional logic added without tests",
                    recommendation="Add tests for all code branches",
                )
            )
        elif branch_change > 0:
            improvements.append(
                Regression(
                    type=RegressionType.COVERAGE,
                    severity=RegressionSeverity.LOW,
                    metric="branch_coverage",
                    baseline_value=baseline_branch,
                    current_value=current_branch,
                    change_percentage=branch_change,
                    description=f"Branch coverage improved by {branch_change:.1f}%",
                )
            )

        return regressions, improvements

    def _detect_performance_regressions(
        self, baseline: AnalysisResult, current: AnalysisResult
    ) -> Tuple[List[Regression], List[Regression]]:
        """Detect performance regressions."""
        regressions = []
        improvements = []

        # GPU utilization
        baseline_gpu = baseline.performance.gpu_utilization
        current_gpu = current.performance.gpu_utilization

        if baseline_gpu > 0:  # Only compare if baseline has GPU data
            gpu_change = ((current_gpu - baseline_gpu) / baseline_gpu) * 100

            if gpu_change < -self.performance_threshold:
                regressions.append(
                    Regression(
                        type=RegressionType.PERFORMANCE,
                        severity=RegressionSeverity.CRITICAL,
                        metric="gpu_utilization",
                        baseline_value=baseline_gpu,
                        current_value=current_gpu,
                        change_percentage=gpu_change,
                        description=f"GPU utilization decreased by {abs(gpu_change):.1f}%",
                        root_cause="Inefficient GPU operations or increased CPU bottlenecks",
                        recommendation="Profile GPU kernels and optimize data loading",
                    )
                )
            elif gpu_change > self.performance_threshold:
                improvements.append(
                    Regression(
                        type=RegressionType.PERFORMANCE,
                        severity=RegressionSeverity.LOW,
                        metric="gpu_utilization",
                        baseline_value=baseline_gpu,
                        current_value=current_gpu,
                        change_percentage=gpu_change,
                        description=f"GPU utilization improved by {gpu_change:.1f}%",
                    )
                )

        # Memory usage
        baseline_mem = baseline.performance.memory_usage_peak_gb
        current_mem = current.performance.memory_usage_peak_gb

        if baseline_mem > 0:
            mem_change = ((current_mem - baseline_mem) / baseline_mem) * 100

            if mem_change > self.performance_threshold:
                regressions.append(
                    Regression(
                        type=RegressionType.PERFORMANCE,
                        severity=RegressionSeverity.HIGH,
                        metric="memory_usage_peak_gb",
                        baseline_value=baseline_mem,
                        current_value=current_mem,
                        change_percentage=mem_change,
                        description=f"Peak memory usage increased by {mem_change:.1f}%",
                        root_cause="Memory leaks or inefficient data structures",
                        recommendation="Profile memory allocations and optimize data handling",
                    )
                )
            elif mem_change < -self.performance_threshold:
                improvements.append(
                    Regression(
                        type=RegressionType.PERFORMANCE,
                        severity=RegressionSeverity.LOW,
                        metric="memory_usage_peak_gb",
                        baseline_value=baseline_mem,
                        current_value=current_mem,
                        change_percentage=mem_change,
                        description=f"Peak memory usage decreased by {abs(mem_change):.1f}%",
                    )
                )

        # New bottlenecks
        baseline_bottlenecks = {b.get("operation", "") for b in baseline.performance.bottlenecks}
        current_bottlenecks = {b.get("operation", "") for b in current.performance.bottlenecks}
        new_bottlenecks = current_bottlenecks - baseline_bottlenecks

        if new_bottlenecks:
            regressions.append(
                Regression(
                    type=RegressionType.PERFORMANCE,
                    severity=RegressionSeverity.HIGH,
                    metric="bottlenecks",
                    baseline_value=len(baseline_bottlenecks),
                    current_value=len(current_bottlenecks),
                    change_percentage=0.0,
                    description=f"New performance bottlenecks detected: {', '.join(new_bottlenecks)}",
                    root_cause="Inefficient operations introduced in new code",
                    recommendation="Profile and optimize slow operations",
                )
            )

        return regressions, improvements

    def _detect_security_regressions(
        self, baseline: AnalysisResult, current: AnalysisResult
    ) -> Tuple[List[Regression], List[Regression]]:
        """Detect security regressions."""
        regressions = []
        improvements = []

        # Vulnerability count
        baseline_vulns = len(baseline.security.vulnerabilities)
        current_vulns = len(current.security.vulnerabilities)

        if current_vulns > baseline_vulns:
            new_vuln_count = current_vulns - baseline_vulns
            regressions.append(
                Regression(
                    type=RegressionType.SECURITY,
                    severity=RegressionSeverity.CRITICAL,
                    metric="vulnerabilities",
                    baseline_value=baseline_vulns,
                    current_value=current_vulns,
                    change_percentage=0.0,
                    description=f"{new_vuln_count} new security vulnerabilities detected",
                    root_cause="New vulnerable code or dependencies introduced",
                    recommendation="Review and fix security vulnerabilities immediately",
                )
            )
        elif current_vulns < baseline_vulns:
            fixed_count = baseline_vulns - current_vulns
            improvements.append(
                Regression(
                    type=RegressionType.SECURITY,
                    severity=RegressionSeverity.LOW,
                    metric="vulnerabilities",
                    baseline_value=baseline_vulns,
                    current_value=current_vulns,
                    change_percentage=0.0,
                    description=f"{fixed_count} security vulnerabilities fixed",
                )
            )

        # New CVEs in dependencies
        baseline_cves = {v.get("cve_id", "") for v in baseline.dependencies.vulnerabilities}
        current_cves = {v.get("cve_id", "") for v in current.dependencies.vulnerabilities}
        new_cves = current_cves - baseline_cves

        if new_cves:
            regressions.append(
                Regression(
                    type=RegressionType.SECURITY,
                    severity=RegressionSeverity.CRITICAL,
                    metric="dependency_cves",
                    baseline_value=len(baseline_cves),
                    current_value=len(current_cves),
                    change_percentage=0.0,
                    description=f"New CVEs in dependencies: {', '.join(new_cves)}",
                    root_cause="Vulnerable dependencies added or updated",
                    recommendation="Update dependencies to patched versions",
                )
            )

        # Hardcoded secrets
        baseline_secrets = len(baseline.security.hardcoded_secrets)
        current_secrets = len(current.security.hardcoded_secrets)

        if current_secrets > baseline_secrets:
            new_secret_count = current_secrets - baseline_secrets
            regressions.append(
                Regression(
                    type=RegressionType.SECURITY,
                    severity=RegressionSeverity.CRITICAL,
                    metric="hardcoded_secrets",
                    baseline_value=baseline_secrets,
                    current_value=current_secrets,
                    change_percentage=0.0,
                    description=f"{new_secret_count} new hardcoded secrets detected",
                    root_cause="Secrets committed to codebase",
                    recommendation="Remove secrets and use environment variables",
                )
            )

        return regressions, improvements

    def _detect_code_quality_regressions(
        self, baseline: AnalysisResult, current: AnalysisResult
    ) -> Tuple[List[Regression], List[Regression]]:
        """Detect code quality regressions."""
        regressions = []
        improvements = []

        # Average complexity
        baseline_complexity = baseline.code_quality.average_complexity
        current_complexity = current.code_quality.average_complexity

        if baseline_complexity > 0:
            complexity_change = (
                (current_complexity - baseline_complexity) / baseline_complexity
            ) * 100

            if complexity_change > 20:  # >20% increase in complexity
                regressions.append(
                    Regression(
                        type=RegressionType.CODE_QUALITY,
                        severity=RegressionSeverity.MEDIUM,
                        metric="average_complexity",
                        baseline_value=baseline_complexity,
                        current_value=current_complexity,
                        change_percentage=complexity_change,
                        description=f"Average complexity increased by {complexity_change:.1f}%",
                        root_cause="Complex logic added without refactoring",
                        recommendation="Refactor complex functions into smaller units",
                    )
                )
            elif complexity_change < -20:
                improvements.append(
                    Regression(
                        type=RegressionType.CODE_QUALITY,
                        severity=RegressionSeverity.LOW,
                        metric="average_complexity",
                        baseline_value=baseline_complexity,
                        current_value=current_complexity,
                        change_percentage=complexity_change,
                        description=f"Average complexity decreased by {abs(complexity_change):.1f}%",
                    )
                )

        # Duplication percentage
        baseline_dup = baseline.code_quality.duplication_percentage
        current_dup = current.code_quality.duplication_percentage
        dup_change = current_dup - baseline_dup

        if dup_change > 5:  # >5% increase in duplication
            regressions.append(
                Regression(
                    type=RegressionType.CODE_QUALITY,
                    severity=RegressionSeverity.MEDIUM,
                    metric="duplication_percentage",
                    baseline_value=baseline_dup,
                    current_value=current_dup,
                    change_percentage=dup_change,
                    description=f"Code duplication increased by {dup_change:.1f}%",
                    root_cause="Copy-paste code instead of refactoring",
                    recommendation="Extract duplicated code into reusable functions",
                )
            )
        elif dup_change < -5:
            improvements.append(
                Regression(
                    type=RegressionType.CODE_QUALITY,
                    severity=RegressionSeverity.LOW,
                    metric="duplication_percentage",
                    baseline_value=baseline_dup,
                    current_value=current_dup,
                    change_percentage=dup_change,
                    description=f"Code duplication decreased by {abs(dup_change):.1f}%",
                )
            )

        return regressions, improvements

    def _generate_summary(
        self,
        critical: List[Regression],
        high: List[Regression],
        medium: List[Regression],
        low: List[Regression],
        improvements: List[Regression],
    ) -> str:
        """Generate human-readable summary of regression analysis."""
        lines = []

        if critical:
            lines.append(f"🚨 {len(critical)} CRITICAL regressions detected:")
            for r in critical:
                lines.append(f"  - {r.description}")

        if high:
            lines.append(f"⚠️  {len(high)} HIGH severity regressions detected:")
            for r in high:
                lines.append(f"  - {r.description}")

        if medium:
            lines.append(f"ℹ️  {len(medium)} MEDIUM severity regressions detected:")
            for r in medium:
                lines.append(f"  - {r.description}")

        if improvements:
            lines.append(f"✅ {len(improvements)} improvements detected:")
            for imp in improvements[:5]:  # Show top 5 improvements
                lines.append(f"  - {imp.description}")

        if not (critical or high or medium or low):
            lines.append("✅ No regressions detected - all metrics stable or improved")

        return "\n".join(lines)

    def generate_diff_report(
        self, baseline: AnalysisResult, current: AnalysisResult, report: RegressionReport
    ) -> str:
        """
        Generate side-by-side comparison report (baseline vs current).

        Args:
            baseline: Baseline analysis results
            current: Current analysis results
            report: Regression report

        Returns:
            Formatted diff report in Markdown
        """
        lines = [
            "# Regression Analysis Report",
            "",
            f"**Baseline**: {baseline.git_commit[:8]} ({baseline.timestamp})",
            f"**Current**: {current.git_commit[:8]} ({current.timestamp})",
            "",
            "## Summary",
            "",
            report.summary,
            "",
            "## Detailed Comparison",
            "",
            "### Coverage Metrics",
            "",
            "| Metric | Baseline | Current | Change |",
            "|--------|----------|---------|--------|",
            f"| Line Coverage | {baseline.coverage.line_coverage:.1f}% | "
            f"{current.coverage.line_coverage:.1f}% | "
            f"{self._format_change(current.coverage.line_coverage - baseline.coverage.line_coverage)} |",
            f"| Branch Coverage | {baseline.coverage.branch_coverage:.1f}% | "
            f"{current.coverage.branch_coverage:.1f}% | "
            f"{self._format_change(current.coverage.branch_coverage - baseline.coverage.branch_coverage)} |",
            "",
            "### Performance Metrics",
            "",
            "| Metric | Baseline | Current | Change |",
            "|--------|----------|---------|--------|",
            f"| GPU Utilization | {baseline.performance.gpu_utilization:.1f}% | "
            f"{current.performance.gpu_utilization:.1f}% | "
            f"{self._format_change(current.performance.gpu_utilization - baseline.performance.gpu_utilization)} |",
            f"| Peak Memory (GB) | {baseline.performance.memory_usage_peak_gb:.2f} | "
            f"{current.performance.memory_usage_peak_gb:.2f} | "
            f"{self._format_change(current.performance.memory_usage_peak_gb - baseline.performance.memory_usage_peak_gb)} |",
            "",
            "### Security Metrics",
            "",
            "| Metric | Baseline | Current | Change |",
            "|--------|----------|---------|--------|",
            f"| Vulnerabilities | {len(baseline.security.vulnerabilities)} | "
            f"{len(current.security.vulnerabilities)} | "
            f"{self._format_change(len(current.security.vulnerabilities) - len(baseline.security.vulnerabilities))} |",
            f"| Hardcoded Secrets | {len(baseline.security.hardcoded_secrets)} | "
            f"{len(current.security.hardcoded_secrets)} | "
            f"{self._format_change(len(current.security.hardcoded_secrets) - len(baseline.security.hardcoded_secrets))} |",
            "",
            "### Code Quality Metrics",
            "",
            "| Metric | Baseline | Current | Change |",
            "|--------|----------|---------|--------|",
            f"| Avg Complexity | {baseline.code_quality.average_complexity:.2f} | "
            f"{current.code_quality.average_complexity:.2f} | "
            f"{self._format_change(current.code_quality.average_complexity - baseline.code_quality.average_complexity)} |",
            f"| Duplication % | {baseline.code_quality.duplication_percentage:.1f}% | "
            f"{current.code_quality.duplication_percentage:.1f}% | "
            f"{self._format_change(current.code_quality.duplication_percentage - baseline.code_quality.duplication_percentage)} |",
            "",
        ]

        # Add regression details
        if report.get_all_regressions():
            lines.extend(
                [
                    "## Regressions",
                    "",
                ]
            )

            for regression in report.get_all_regressions():
                lines.extend(
                    [
                        f"### {regression.severity.value.upper()}: {regression.description}",
                        "",
                        f"**Type**: {regression.type.value}",
                        f"**Metric**: {regression.metric}",
                        f"**Change**: {regression.change_percentage:.1f}%",
                        "",
                        f"**Root Cause**: {regression.root_cause}",
                        "",
                        f"**Recommendation**: {regression.recommendation}",
                        "",
                    ]
                )

        return "\n".join(lines)

    def _format_change(self, value: float) -> str:
        """Format change value with color indicators."""
        if value > 0:
            return f"🔴 +{value:.2f}"
        elif value < 0:
            return f"🟢 {value:.2f}"
        else:
            return "⚪ 0.00"

    def exit_code_for_ci(self, report: RegressionReport) -> int:
        """
        Return appropriate exit code for CI build.

        Args:
            report: Regression report

        Returns:
            0 if no critical regressions, 1 if critical regressions detected
        """
        return 1 if report.should_fail_ci() else 0

    @staticmethod
    def load_baseline(baseline_path: str) -> AnalysisResult:
        """
        Load baseline analysis results from JSON file.

        Args:
            baseline_path: Path to baseline JSON file

        Returns:
            AnalysisResult object

        Raises:
            FileNotFoundError: If baseline file doesn't exist
            ValueError: If JSON is invalid
        """
        path = Path(baseline_path)
        if not path.exists():
            raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

        with open(path, "r", encoding="utf-8") as f:
            json_str = f.read()

        return AnalysisResult.from_json(json_str)
