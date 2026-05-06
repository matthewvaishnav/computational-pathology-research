"""
Result Aggregator for HistoCore Project Optimization Analysis System.

Merges results from all 8 analyzers into unified AnalysisResult with
conflict resolution, deduplication, and overall scoring.
"""

import logging
from typing import List, Dict, Any, Set
from collections import defaultdict

from src.analysis.models import (
    AnalysisResult,
    Issue,
    Severity,
    Priority,
    Role,
    ArchitectureAnalysis,
    PerformanceAnalysis,
    CoverageAnalysis,
    CodeQualityAnalysis,
    DependencyAnalysis,
    DeploymentAnalysis,
    SecurityAnalysis,
    ScalabilityAnalysis,
)

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Aggregates and merges analysis results from multiple analyzers."""
    
    def __init__(self):
        """Initialize aggregator."""
        self.dimension_weights = {
            'security': 0.20,      # Highest priority
            'coverage': 0.15,
            'code_quality': 0.15,
            'architecture': 0.15,
            'performance': 0.10,
            'dependencies': 0.10,
            'deployment': 0.10,
            'scalability': 0.05,   # Lowest priority
        }
    
    def merge_results(
        self,
        architecture: ArchitectureAnalysis,
        performance: PerformanceAnalysis,
        coverage: CoverageAnalysis,
        code_quality: CodeQualityAnalysis,
        dependencies: DependencyAnalysis,
        deployment: DeploymentAnalysis,
        security: SecurityAnalysis,
        scalability: ScalabilityAnalysis,
        timestamp: str,
        project_path: str,
        git_commit: str
    ) -> AnalysisResult:
        """
        Merge results from all analyzers into unified AnalysisResult.
        
        Args:
            All analyzer results plus metadata
            
        Returns:
            Unified AnalysisResult with aggregated data
        """
        logger.info("Merging results from all analyzers...")
        
        # Collect all issues from analyzers
        all_issues = []
        
        # Extract issues from architecture analysis
        if hasattr(architecture, 'solid_violations'):
            all_issues.extend(architecture.solid_violations)
        
        # Extract issues from other analyzers (when they have issue lists)
        # Note: Current analyzers don't expose issues directly, but they could
        
        # Deduplicate issues
        deduplicated_issues = self._deduplicate_issues(all_issues)
        
        # Extract critical issues (P0 and P1)
        critical_issues = self._extract_critical_issues(deduplicated_issues)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score({
            'architecture': architecture,
            'performance': performance,
            'coverage': coverage,
            'code_quality': code_quality,
            'dependencies': dependencies,
            'deployment': deployment,
            'security': security,
            'scalability': scalability,
        })
        
        # Create unified result
        result = AnalysisResult(
            timestamp=timestamp,
            project_path=project_path,
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
            critical_issues=critical_issues[:10]  # Top 10 critical issues
        )
        
        logger.info(f"Merged results: {len(deduplicated_issues)} total issues, "
                   f"{len(critical_issues)} critical, overall score: {overall_score:.1f}")
        
        return result
    
    def _deduplicate_issues(self, issues: List[Issue]) -> List[Issue]:
        """
        Remove duplicate issues based on file path, line number, and category.
        
        Args:
            issues: List of issues to deduplicate
            
        Returns:
            Deduplicated list of issues
        """
        seen_signatures: Set[str] = set()
        deduplicated = []
        
        for issue in issues:
            # Create signature for deduplication
            signature = f"{issue.file_path}:{issue.line_number}:{issue.category}:{issue.title}"
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                deduplicated.append(issue)
            else:
                logger.debug(f"Deduplicated issue: {issue.title} in {issue.file_path}")
        
        logger.info(f"Deduplicated {len(issues)} -> {len(deduplicated)} issues")
        return deduplicated
    
    def _extract_critical_issues(self, issues: List[Issue]) -> List[Issue]:
        """
        Extract and prioritize critical issues (P0 and P1).
        
        Args:
            issues: All issues
            
        Returns:
            Sorted list of critical issues
        """
        critical = [
            issue for issue in issues
            if issue.priority in [Priority.P0, Priority.P1]
        ]
        
        # Sort by priority (P0 first) then by severity
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3
        }
        
        priority_order = {
            Priority.P0: 0,
            Priority.P1: 1,
            Priority.P2: 2,
            Priority.P3: 3
        }
        
        critical.sort(key=lambda x: (
            priority_order.get(x.priority, 99),
            severity_order.get(x.severity, 99),
            x.effort_hours  # Prefer lower effort for same priority/severity
        ))
        
        return critical
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate weighted overall score from all dimensions.
        
        Args:
            results: Dictionary of analyzer results
            
        Returns:
            Overall score (0-100)
        """
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, weight in self.dimension_weights.items():
            if dimension in results and hasattr(results[dimension], 'score'):
                score = results[dimension].score
                total_score += score * weight
                total_weight += weight
                logger.debug(f"{dimension}: {score:.1f} (weight: {weight:.2f})")
        
        # Normalize by actual weights used (in case some analyzers failed)
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
        
        return round(final_score, 2)
    
    def get_dimension_summary(self, result: AnalysisResult) -> Dict[str, Dict[str, Any]]:
        """
        Generate summary statistics for each dimension.
        
        Args:
            result: Analysis result
            
        Returns:
            Dictionary with dimension summaries
        """
        summary = {}
        
        # Architecture summary
        arch = result.architecture
        summary['architecture'] = {
            'score': arch.score,
            'total_files': arch.total_files,
            'large_files_count': len(arch.large_files),
            'circular_dependencies_count': len(arch.circular_dependencies),
            'solid_violations_count': len(arch.solid_violations),
            'status': self._get_status_from_score(arch.score)
        }
        
        # Performance summary
        perf = result.performance
        summary['performance'] = {
            'score': perf.score,
            'gpu_utilization': perf.gpu_utilization,
            'bottlenecks_count': len(perf.bottlenecks),
            'memory_peak_gb': perf.memory_usage_peak_gb,
            'status': self._get_status_from_score(perf.score)
        }
        
        # Coverage summary
        cov = result.coverage
        summary['coverage'] = {
            'score': cov.score,
            'line_coverage': cov.line_coverage,
            'branch_coverage': cov.branch_coverage,
            'untested_paths_count': len(cov.untested_critical_paths),
            'flaky_tests_count': len(cov.flaky_tests),
            'status': self._get_status_from_score(cov.score)
        }
        
        # Code Quality summary
        qual = result.code_quality
        summary['code_quality'] = {
            'score': qual.score,
            'average_complexity': qual.average_complexity,
            'high_complexity_count': len(qual.high_complexity_functions),
            'duplication_percentage': qual.duplication_percentage,
            'documentation_coverage': qual.documentation_coverage,
            'pylint_score': qual.pylint_score,
            'status': self._get_status_from_score(qual.score)
        }
        
        # Dependencies summary
        deps = result.dependencies
        summary['dependencies'] = {
            'score': deps.score,
            'total_dependencies': deps.total_dependencies,
            'vulnerabilities_count': len(deps.vulnerabilities),
            'outdated_count': len(deps.outdated_packages),
            'license_issues_count': len(deps.license_issues),
            'status': self._get_status_from_score(deps.score)
        }
        
        # Deployment summary
        deploy = result.deployment
        summary['deployment'] = {
            'score': deploy.score,
            'dockerfile_score': deploy.dockerfile_score,
            'k8s_readiness': deploy.k8s_readiness,
            'ci_cd_completeness': deploy.ci_cd_completeness,
            'monitoring_score': deploy.monitoring_score,
            'status': self._get_status_from_score(deploy.score)
        }
        
        # Security summary
        sec = result.security
        summary['security'] = {
            'score': sec.score,
            'vulnerabilities_count': len(sec.vulnerabilities),
            'hipaa_compliance_score': sec.hipaa_compliance_score,
            'hardcoded_secrets_count': len(sec.hardcoded_secrets),
            'injection_risks_count': len(sec.injection_risks),
            'status': self._get_status_from_score(sec.score)
        }
        
        # Scalability summary
        scale = result.scalability
        summary['scalability'] = {
            'score': scale.score,
            'ddp_correctness': scale.ddp_correctness,
            'scaling_efficiency': scale.scaling_efficiency,
            'memory_bottlenecks_count': len(scale.memory_bottlenecks),
            'communication_overhead_ms': scale.communication_overhead_ms,
            'status': self._get_status_from_score(scale.score)
        }
        
        return summary
    
    def _get_status_from_score(self, score: float) -> str:
        """Convert numeric score to status string."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "needs_improvement"
        else:
            return "critical"
    
    def get_top_issues_by_dimension(
        self,
        result: AnalysisResult,
        limit: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get top issues for each dimension.
        
        Args:
            result: Analysis result
            limit: Max issues per dimension
            
        Returns:
            Dictionary mapping dimension to top issues
        """
        top_issues = {}
        
        # Architecture issues (from SOLID violations)
        arch_issues = []
        for issue in result.architecture.solid_violations[:limit]:
            arch_issues.append({
                'title': issue.title,
                'severity': issue.severity.value,
                'file_path': issue.file_path,
                'recommendation': issue.recommendation
            })
        top_issues['architecture'] = arch_issues
        
        # Performance issues (from bottlenecks)
        perf_issues = []
        for bottleneck in result.performance.bottlenecks[:limit]:
            perf_issues.append({
                'title': f"Performance bottleneck: {bottleneck.get('function', 'unknown')}",
                'severity': 'high' if bottleneck.get('time_ms', 0) > 1000 else 'medium',
                'file_path': bottleneck.get('file', 'unknown'),
                'recommendation': f"Optimize function taking {bottleneck.get('time_ms', 0):.1f}ms"
            })
        top_issues['performance'] = perf_issues
        
        # Coverage issues (from untested paths)
        cov_issues = []
        for path in result.coverage.untested_critical_paths[:limit]:
            cov_issues.append({
                'title': f"Untested critical path: {path}",
                'severity': 'high',
                'file_path': path,
                'recommendation': 'Add test coverage for this critical path'
            })
        top_issues['coverage'] = cov_issues
        
        # Security issues (from vulnerabilities)
        sec_issues = []
        for vuln in result.security.vulnerabilities[:limit]:
            sec_issues.append({
                'title': vuln.get('title', 'Security vulnerability'),
                'severity': vuln.get('severity', 'medium'),
                'file_path': vuln.get('file', 'unknown'),
                'recommendation': vuln.get('recommendation', 'Review and fix security issue')
            })
        top_issues['security'] = sec_issues
        
        # Add other dimensions as needed
        top_issues['code_quality'] = []
        top_issues['dependencies'] = []
        top_issues['deployment'] = []
        top_issues['scalability'] = []
        
        return top_issues
