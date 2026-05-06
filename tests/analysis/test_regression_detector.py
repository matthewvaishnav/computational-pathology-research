"""
Unit tests for Regression Detector.

Tests coverage regression detection, performance regression detection,
security regression detection, and CI failure logic.
Requirements: 12.2, 12.3, 12.4, 12.6
"""

import pytest
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, mock_open

from src.analysis.regression_detector import RegressionDetector
from src.analysis.models import (
    AnalysisResult,
    ArchitectureAnalysis,
    PerformanceAnalysis,
    CoverageAnalysis,
    CodeQualityAnalysis,
    DependencyAnalysis,
    DeploymentAnalysis,
    SecurityAnalysis,
    ScalabilityAnalysis,
    Issue,
    Severity
)


class TestRegressionDetector:
    """Test suite for RegressionDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = RegressionDetector()
        self.baseline_result = self._create_baseline_result()
        self.current_result = self._create_current_result()
    
    def _create_baseline_result(self) -> AnalysisResult:
        """Create a baseline analysis result for testing."""
        return AnalysisResult(
            timestamp=datetime.now(),
            project_path="/test/project",
            git_commit="baseline123",
            architecture=ArchitectureAnalysis(
                total_files=100,
                large_files=[],
                circular_dependencies=[],
                coupling_metrics={},
                solid_violations=[],
                score=80.0
            ),
            performance=PerformanceAnalysis(
                gpu_utilization=75.0,
                memory_usage_peak_gb=12.0,
                bottlenecks=[
                    {"function": "baseline_func", "time_ms": 500, "percentage": 25.0}
                ],
                flame_graph_path="",
                score=75.0
            ),
            coverage=CoverageAnalysis(
                line_coverage=85.0,
                branch_coverage=80.0,
                untested_critical_paths=[],
                missing_property_tests=[],
                flaky_tests=[],
                score=82.0
            ),
            code_quality=CodeQualityAnalysis(
                average_complexity=8.0,
                high_complexity_functions=[],
                duplication_percentage=5.0,
                documentation_coverage=75.0,
                pylint_score=8.5,
                score=78.0
            ),
            dependencies=DependencyAnalysis(
                total_dependencies=30,
                outdated_packages=[],
                vulnerabilities=[],
                unused_dependencies=[],
                license_issues=[],
                score=85.0
            ),
            deployment=DeploymentAnalysis(
                dockerfile_score=80.0,
                k8s_readiness=75.0,
                ci_cd_completeness=85.0,
                monitoring_score=70.0,
                score=77.5
            ),
            security=SecurityAnalysis(
                vulnerabilities=[],
                hardcoded_secrets=[],
                tls_issues=[],
                hipaa_compliance_score=85.0,
                score=85.0
            ),
            scalability=ScalabilityAnalysis(
                ddp_correctness=True,
                memory_bottlenecks=[],
                communication_overhead_ms=20.0,
                scaling_efficiency="linear",
                recommendations={},
                score=80.0
            ),
            overall_score=80.0,
            critical_issues=[]
        )
    
    def _create_current_result(self) -> AnalysisResult:
        """Create a current analysis result for testing."""
        return AnalysisResult(
            timestamp=datetime.now(),
            project_path="/test/project",
            git_commit="current456",
            architecture=ArchitectureAnalysis(
                total_files=105,
                large_files=[
                    {"path": "src/new_large.py", "lines": 600, "complexity": 25}
                ],
                circular_dependencies=[],
                coupling_metrics={},
                solid_violations=[],
                score=78.0
            ),
            performance=PerformanceAnalysis(
                gpu_utilization=70.0,
                memory_usage_peak_gb=14.0,
                bottlenecks=[
                    {"function": "baseline_func", "time_ms": 600, "percentage": 30.0},
                    {"function": "new_bottleneck", "time_ms": 400, "percentage": 20.0}
                ],
                flame_graph_path="",
                score=70.0
            ),
            coverage=CoverageAnalysis(
                line_coverage=82.0,  # 3% decrease
                branch_coverage=77.0,  # 3% decrease
                untested_critical_paths=[
                    "src/new_module.py:critical_function"
                ],
                missing_property_tests=[],
                flaky_tests=[],
                score=79.0
            ),
            code_quality=CodeQualityAnalysis(
                average_complexity=9.5,
                high_complexity_functions=[
                    {"function": "complex_func", "complexity": 15, "file": "src/complex.py"}
                ],
                duplication_percentage=7.0,
                documentation_coverage=70.0,
                pylint_score=8.0,
                score=75.0
            ),
            dependencies=DependencyAnalysis(
                total_dependencies=32,
                outdated_packages=[
                    {"name": "requests", "current": "2.25.0", "latest": "2.28.0"}
                ],
                vulnerabilities=[
                    {"package": "urllib3", "severity": "medium", "cve": "CVE-2023-1234"}
                ],
                unused_dependencies=[],
                license_issues=[],
                score=80.0
            ),
            deployment=DeploymentAnalysis(
                dockerfile_score=78.0,
                k8s_readiness=73.0,
                ci_cd_completeness=83.0,
                monitoring_score=68.0,
                score=75.5
            ),
            security=SecurityAnalysis(
                vulnerabilities=[
                    Issue(
                        id="SEC-001",
                        dimension="security",
                        severity=Severity.HIGH,
                        category="Security",
                        title="New SQL Injection",
                        description="SQL injection vulnerability",
                        file_path="src/database.py",
                        line_number=45,
                        recommendation="Use parameterized queries"
                    )
                ],
                hardcoded_secrets=[
                    {"file": "config.py", "line": 10, "type": "API_KEY"}
                ],
                tls_issues=[],
                hipaa_compliance_score=80.0,
                score=75.0
            ),
            scalability=ScalabilityAnalysis(
                ddp_correctness=True,
                memory_bottlenecks=[
                    "New memory leak in data loader"
                ],
                communication_overhead_ms=25.0,
                scaling_efficiency="sub-linear",
                recommendations={},
                score=75.0
            ),
            overall_score=76.0,  # 4 point decrease
            critical_issues=[
                Issue(
                    id="SEC-001",
                    dimension="security",
                    severity=Severity.HIGH,
                    category="Security",
                    title="New SQL Injection",
                    description="SQL injection vulnerability",
                    file_path="src/database.py",
                    line_number=45,
                    recommendation="Use parameterized queries"
                )
            ]
        )
    
    def test_detect_regressions_with_baseline(self):
        """Test regression detection with valid baseline."""
        # Create temporary baseline file with all required fields
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            baseline_data = {
                "timestamp": self.baseline_result.timestamp.isoformat(),
                "project_path": self.baseline_result.project_path,
                "git_commit": self.baseline_result.git_commit,
                "overall_score": self.baseline_result.overall_score,
                "architecture": {
                    "total_files": self.baseline_result.architecture.total_files,
                    "large_files": self.baseline_result.architecture.large_files,
                    "circular_dependencies": self.baseline_result.architecture.circular_dependencies,
                    "coupling_metrics": self.baseline_result.architecture.coupling_metrics,
                    "solid_violations": [],
                    "score": self.baseline_result.architecture.score
                },
                "coverage": {
                    "line_coverage": self.baseline_result.coverage.line_coverage,
                    "branch_coverage": self.baseline_result.coverage.branch_coverage,
                    "untested_critical_paths": self.baseline_result.coverage.untested_critical_paths,
                    "missing_property_tests": self.baseline_result.coverage.missing_property_tests,
                    "flaky_tests": self.baseline_result.coverage.flaky_tests,
                    "slow_tests": [],
                    "score": self.baseline_result.coverage.score
                },
                "performance": {
                    "gpu_utilization": self.baseline_result.performance.gpu_utilization,
                    "memory_usage_peak_gb": self.baseline_result.performance.memory_usage_peak_gb,
                    "memory_usage_avg_gb": 0.0,
                    "bottlenecks": self.baseline_result.performance.bottlenecks,
                    "flame_graph_path": self.baseline_result.performance.flame_graph_path,
                    "score": self.baseline_result.performance.score
                },
                "code_quality": {
                    "average_complexity": self.baseline_result.code_quality.average_complexity,
                    "high_complexity_functions": self.baseline_result.code_quality.high_complexity_functions,
                    "duplication_percentage": self.baseline_result.code_quality.duplication_percentage,
                    "documentation_coverage": self.baseline_result.code_quality.documentation_coverage,
                    "pylint_score": self.baseline_result.code_quality.pylint_score,
                    "score": self.baseline_result.code_quality.score,
                    "fix_suggestions": []
                },
                "dependencies": {
                    "total_dependencies": self.baseline_result.dependencies.total_dependencies,
                    "vulnerabilities": self.baseline_result.dependencies.vulnerabilities,
                    "outdated_packages": self.baseline_result.dependencies.outdated_packages,
                    "license_issues": self.baseline_result.dependencies.license_issues,
                    "unused_dependencies": self.baseline_result.dependencies.unused_dependencies,
                    "redundant_dependencies": [],
                    "security_report": {},
                    "score": self.baseline_result.dependencies.score
                },
                "deployment": {
                    "dockerfile_score": self.baseline_result.deployment.dockerfile_score,
                    "k8s_readiness": self.baseline_result.deployment.k8s_readiness,
                    "ci_cd_completeness": self.baseline_result.deployment.ci_cd_completeness,
                    "monitoring_score": self.baseline_result.deployment.monitoring_score,
                    "score": self.baseline_result.deployment.score
                },
                "security": {
                    "vulnerabilities": [],
                    "hardcoded_secrets": [],
                    "tls_issues": [],
                    "injection_risks": [],
                    "hipaa_compliance_score": self.baseline_result.security.hipaa_compliance_score,
                    "score": self.baseline_result.security.score
                },
                "scalability": {
                    "ddp_correctness": self.baseline_result.scalability.ddp_correctness,
                    "scaling_efficiency": self.baseline_result.scalability.scaling_efficiency,
                    "memory_bottlenecks": self.baseline_result.scalability.memory_bottlenecks,
                    "communication_overhead_ms": self.baseline_result.scalability.communication_overhead_ms,
                    "score": self.baseline_result.scalability.score,
                    "recommendations": self.baseline_result.scalability.recommendations
                },
                "critical_issues": []
            }
            json.dump(baseline_data, f)
            baseline_path = f.name
        
        try:
            result = self.detector.detect_regressions(
                current=self.current_result,
                baseline_path=baseline_path
            )
            
            # Should detect regressions
            assert result is not None
            assert result["has_regressions"] is True
            assert "critical_regressions" in result
            assert "comparison_details" in result
            
            # Should have detected some regressions (coverage, performance, security)
            assert len(result["critical_regressions"]) > 0
            
        finally:
            os.unlink(baseline_path)
    
    def test_detect_regressions_no_baseline(self):
        """Test regression detection without baseline."""
        result = self.detector.detect_regressions(
            current=self.current_result,
            baseline_path="nonexistent.json"
        )
        
        # Should return no-baseline result
        assert result is not None
        assert result["has_regressions"] is False
        assert result["should_fail_ci"] is False
        assert result["baseline_timestamp"] is None
        assert "No baseline available" in result["summary"]["message"]
    
    def test_coverage_regression_detection(self):
        """Test coverage regression detection logic."""
        regressions = self.detector._check_coverage_regressions(
            current=self.current_result,
            baseline=self.baseline_result
        )
        
        # Should detect coverage regressions
        assert regressions["coverage_regression_critical"] is True
        
        # Should have critical regressions
        assert "critical_regressions" in regressions
        critical_regressions = regressions["critical_regressions"]
        
        # Check for line coverage regression
        line_regression = next((r for r in critical_regressions if r["metric"] == "line_coverage"), None)
        assert line_regression is not None
        assert line_regression["current"] == 82.0
        assert line_regression["baseline"] == 85.0
        assert line_regression["change"] == -3.0
        
        # Check for branch coverage regression
        branch_regression = next((r for r in critical_regressions if r["metric"] == "branch_coverage"), None)
        assert branch_regression is not None
        assert branch_regression["current"] == 77.0
        assert branch_regression["baseline"] == 80.0
        assert branch_regression["change"] == -3.0
    
    def test_coverage_no_regression(self):
        """Test coverage regression detection with no regressions."""
        # Create current result with improved coverage
        improved_result = AnalysisResult(
            timestamp=datetime.now(),
            project_path="/test/project",
            git_commit="improved123",
            architecture=self.baseline_result.architecture,
            performance=self.baseline_result.performance,
            coverage=CoverageAnalysis(
                line_coverage=87.0,  # Improvement
                branch_coverage=82.0,  # Improvement
                untested_critical_paths=[],
                missing_property_tests=[],
                flaky_tests=[],
                score=85.0
            ),
            code_quality=self.baseline_result.code_quality,
            dependencies=self.baseline_result.dependencies,
            deployment=self.baseline_result.deployment,
            security=self.baseline_result.security,
            scalability=self.baseline_result.scalability,
            overall_score=85.0,
            critical_issues=[]
        )
        
        regressions = self.detector._check_coverage_regressions(
            current=improved_result,
            baseline=self.baseline_result
        )
        
        # Should not detect regressions
        assert regressions["coverage_regression_critical"] is False
        assert "critical_regressions" not in regressions or len(regressions["critical_regressions"]) == 0
    
    def test_performance_regression_detection(self):
        """Test performance regression detection logic."""
        regressions = self.detector._check_performance_regressions(
            current=self.current_result,
            baseline=self.baseline_result
        )
        
        # Should detect performance regressions
        assert regressions["performance_regression_critical"] is True
        
        # Should have critical regressions
        assert "critical_regressions" in regressions
        critical_regressions = regressions["critical_regressions"]
        assert len(critical_regressions) > 0
        
        # Check for GPU or memory regression
        has_gpu_or_memory_regression = any(
            r['metric'] in ['gpu_utilization', 'memory_usage'] 
            for r in critical_regressions
        )
        assert has_gpu_or_memory_regression
        
        # Check for warnings about new bottlenecks
        if "warnings" in regressions:
            warnings = regressions["warnings"]
            new_bottleneck_warning = next((w for w in warnings if w['type'] == 'new_bottlenecks'), None)
            if new_bottleneck_warning:
                assert new_bottleneck_warning['count'] >= 1
    
    def test_performance_no_regression(self):
        """Test performance regression detection with no regressions."""
        # Create current result with improved performance
        improved_result = AnalysisResult(
            timestamp=datetime.now(),
            project_path="/test/project",
            git_commit="improved123",
            architecture=self.baseline_result.architecture,
            performance=PerformanceAnalysis(
                gpu_utilization=80.0,  # Improvement
                memory_usage_peak_gb=10.0,  # Improvement
                bottlenecks=[
                    {"function": "baseline_func", "time_ms": 400, "percentage": 20.0}  # Improved
                ],
                flame_graph_path="",
                score=80.0
            ),
            coverage=self.baseline_result.coverage,
            code_quality=self.baseline_result.code_quality,
            dependencies=self.baseline_result.dependencies,
            deployment=self.baseline_result.deployment,
            security=self.baseline_result.security,
            scalability=self.baseline_result.scalability,
            overall_score=85.0,
            critical_issues=[]
        )
        
        regressions = self.detector._check_performance_regressions(
            current=improved_result,
            baseline=self.baseline_result
        )
        
        # Should not detect regressions
        assert regressions["performance_regression_critical"] is False
        assert "critical_regressions" not in regressions or len(regressions.get("critical_regressions", [])) == 0
    
    def test_security_regression_detection(self):
        """Test security regression detection logic."""
        regressions = self.detector._check_security_regressions(
            current=self.current_result,
            baseline=self.baseline_result
        )
        
        # Should detect security regressions
        assert regressions["security_regression_critical"] is True
        
        # Should have critical regressions
        assert "critical_regressions" in regressions
        critical_regressions = regressions["critical_regressions"]
        assert len(critical_regressions) > 0
        
        # Check for vulnerability or secret regressions
        has_vuln_or_secret_regression = any(
            r['metric'] in ['vulnerabilities', 'hardcoded_secrets'] 
            for r in critical_regressions
        )
        assert has_vuln_or_secret_regression
    
    def test_security_no_regression(self):
        """Test security regression detection with no regressions."""
        # Create current result with no new security issues
        secure_result = AnalysisResult(
            timestamp=datetime.now(),
            project_path="/test/project",
            git_commit="secure123",
            architecture=self.baseline_result.architecture,
            performance=self.baseline_result.performance,
            coverage=self.baseline_result.coverage,
            code_quality=self.baseline_result.code_quality,
            dependencies=self.baseline_result.dependencies,
            deployment=self.baseline_result.deployment,
            security=SecurityAnalysis(
                vulnerabilities=[],
                hardcoded_secrets=[],
                tls_issues=[],
                hipaa_compliance_score=87.0,  # Improvement
                score=87.0
            ),
            scalability=self.baseline_result.scalability,
            overall_score=85.0,
            critical_issues=[]
        )
        
        regressions = self.detector._check_security_regressions(
            current=secure_result,
            baseline=self.baseline_result
        )
        
        # Should not detect regressions
        assert regressions["security_regression_critical"] is False
        assert "critical_regressions" not in regressions or len(regressions.get("critical_regressions", [])) == 0
    
    def test_overall_score_regression(self):
        """Test overall score regression detection."""
        regression = self.detector._check_overall_score_regression(
            current=self.current_result,
            baseline=self.baseline_result
        )
        
        # Should detect overall score regression (76 vs 80 = -4, threshold is -5, so no critical regression)
        # But should still have comparison details
        assert 'comparison_details' in regression
        assert 'overall' in regression['comparison_details']
        
        # Check comparison details
        overall_details = regression['comparison_details']['overall']['score']
        assert overall_details['baseline'] == 80.0
        assert overall_details['current'] == 76.0
        assert overall_details['change'] == -4.0
    
    def test_overall_score_no_regression(self):
        """Test overall score regression with improvement."""
        improved_result = AnalysisResult(
            timestamp=datetime.now(),
            project_path="/test/project",
            git_commit="improved123",
            architecture=self.baseline_result.architecture,
            performance=self.baseline_result.performance,
            coverage=self.baseline_result.coverage,
            code_quality=self.baseline_result.code_quality,
            dependencies=self.baseline_result.dependencies,
            deployment=self.baseline_result.deployment,
            security=self.baseline_result.security,
            scalability=self.baseline_result.scalability,
            overall_score=83.0,  # +3 points, above the >2.0 threshold
            critical_issues=[]
        )
        
        regression = self.detector._check_overall_score_regression(
            current=improved_result,
            baseline=self.baseline_result
        )
        
        # Should not detect regression, should have improvements
        assert 'critical_regressions' not in regression or len(regression.get('critical_regressions', [])) == 0
        # Should have improvements (+3 is > 2.0 threshold)
        assert 'improvements' in regression
        assert len(regression['improvements']) > 0
    
    def test_generate_diff_report_markdown(self):
        """Test Markdown diff report generation."""
        # Create temporary baseline file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            baseline_data = {
                "timestamp": self.baseline_result.timestamp.isoformat(),
                "project_path": self.baseline_result.project_path,
                "git_commit": self.baseline_result.git_commit,
                "overall_score": self.baseline_result.overall_score,
                "architecture": {
                    "total_files": self.baseline_result.architecture.total_files,
                    "large_files": self.baseline_result.architecture.large_files,
                    "circular_dependencies": self.baseline_result.architecture.circular_dependencies,
                    "coupling_metrics": self.baseline_result.architecture.coupling_metrics,
                    "solid_violations": [],
                    "score": self.baseline_result.architecture.score
                },
                "coverage": {
                    "line_coverage": self.baseline_result.coverage.line_coverage,
                    "branch_coverage": self.baseline_result.coverage.branch_coverage,
                    "untested_critical_paths": self.baseline_result.coverage.untested_critical_paths,
                    "missing_property_tests": self.baseline_result.coverage.missing_property_tests,
                    "flaky_tests": self.baseline_result.coverage.flaky_tests,
                    "slow_tests": [],
                    "score": self.baseline_result.coverage.score
                },
                "performance": {
                    "gpu_utilization": self.baseline_result.performance.gpu_utilization,
                    "memory_usage_peak_gb": self.baseline_result.performance.memory_usage_peak_gb,
                    "memory_usage_avg_gb": 0.0,
                    "bottlenecks": self.baseline_result.performance.bottlenecks,
                    "flame_graph_path": self.baseline_result.performance.flame_graph_path,
                    "score": self.baseline_result.performance.score
                },
                "code_quality": {
                    "average_complexity": self.baseline_result.code_quality.average_complexity,
                    "high_complexity_functions": self.baseline_result.code_quality.high_complexity_functions,
                    "duplication_percentage": self.baseline_result.code_quality.duplication_percentage,
                    "documentation_coverage": self.baseline_result.code_quality.documentation_coverage,
                    "pylint_score": self.baseline_result.code_quality.pylint_score,
                    "score": self.baseline_result.code_quality.score,
                    "fix_suggestions": []
                },
                "dependencies": {
                    "total_dependencies": self.baseline_result.dependencies.total_dependencies,
                    "vulnerabilities": self.baseline_result.dependencies.vulnerabilities,
                    "outdated_packages": self.baseline_result.dependencies.outdated_packages,
                    "license_issues": self.baseline_result.dependencies.license_issues,
                    "unused_dependencies": self.baseline_result.dependencies.unused_dependencies,
                    "redundant_dependencies": [],
                    "security_report": {},
                    "score": self.baseline_result.dependencies.score
                },
                "deployment": {
                    "dockerfile_score": self.baseline_result.deployment.dockerfile_score,
                    "k8s_readiness": self.baseline_result.deployment.k8s_readiness,
                    "ci_cd_completeness": self.baseline_result.deployment.ci_cd_completeness,
                    "monitoring_score": self.baseline_result.deployment.monitoring_score,
                    "score": self.baseline_result.deployment.score
                },
                "security": {
                    "vulnerabilities": [],
                    "hardcoded_secrets": [],
                    "tls_issues": [],
                    "injection_risks": [],
                    "hipaa_compliance_score": self.baseline_result.security.hipaa_compliance_score,
                    "score": self.baseline_result.security.score
                },
                "scalability": {
                    "ddp_correctness": self.baseline_result.scalability.ddp_correctness,
                    "scaling_efficiency": self.baseline_result.scalability.scaling_efficiency,
                    "memory_bottlenecks": self.baseline_result.scalability.memory_bottlenecks,
                    "communication_overhead_ms": self.baseline_result.scalability.communication_overhead_ms,
                    "score": self.baseline_result.scalability.score,
                    "recommendations": self.baseline_result.scalability.recommendations
                },
                "critical_issues": []
            }
            json.dump(baseline_data, f)
            baseline_path = f.name
        
        try:
            report = self.detector.generate_diff_report(
                current=self.current_result,
                baseline_path=baseline_path,
                format="markdown"
            )
            
            # Should generate Markdown report
            assert "# Regression Analysis Report" in report or "Regression" in report
            assert "Status:" in report or "status" in report.lower()
            
        finally:
            os.unlink(baseline_path)
    
    def test_generate_diff_report_text(self):
        """Test text diff report generation."""
        # Create temporary baseline file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            baseline_data = {
                "timestamp": self.baseline_result.timestamp.isoformat(),
                "project_path": self.baseline_result.project_path,
                "git_commit": self.baseline_result.git_commit,
                "overall_score": self.baseline_result.overall_score,
                "architecture": {
                    "total_files": self.baseline_result.architecture.total_files,
                    "large_files": self.baseline_result.architecture.large_files,
                    "circular_dependencies": self.baseline_result.architecture.circular_dependencies,
                    "coupling_metrics": self.baseline_result.architecture.coupling_metrics,
                    "solid_violations": [],
                    "score": self.baseline_result.architecture.score
                },
                "coverage": {
                    "line_coverage": self.baseline_result.coverage.line_coverage,
                    "branch_coverage": self.baseline_result.coverage.branch_coverage,
                    "untested_critical_paths": self.baseline_result.coverage.untested_critical_paths,
                    "missing_property_tests": self.baseline_result.coverage.missing_property_tests,
                    "flaky_tests": self.baseline_result.coverage.flaky_tests,
                    "slow_tests": [],
                    "score": self.baseline_result.coverage.score
                },
                "performance": {
                    "gpu_utilization": self.baseline_result.performance.gpu_utilization,
                    "memory_usage_peak_gb": self.baseline_result.performance.memory_usage_peak_gb,
                    "memory_usage_avg_gb": 0.0,
                    "bottlenecks": self.baseline_result.performance.bottlenecks,
                    "flame_graph_path": self.baseline_result.performance.flame_graph_path,
                    "score": self.baseline_result.performance.score
                },
                "code_quality": {
                    "average_complexity": self.baseline_result.code_quality.average_complexity,
                    "high_complexity_functions": self.baseline_result.code_quality.high_complexity_functions,
                    "duplication_percentage": self.baseline_result.code_quality.duplication_percentage,
                    "documentation_coverage": self.baseline_result.code_quality.documentation_coverage,
                    "pylint_score": self.baseline_result.code_quality.pylint_score,
                    "score": self.baseline_result.code_quality.score,
                    "fix_suggestions": []
                },
                "dependencies": {
                    "total_dependencies": self.baseline_result.dependencies.total_dependencies,
                    "vulnerabilities": self.baseline_result.dependencies.vulnerabilities,
                    "outdated_packages": self.baseline_result.dependencies.outdated_packages,
                    "license_issues": self.baseline_result.dependencies.license_issues,
                    "unused_dependencies": self.baseline_result.dependencies.unused_dependencies,
                    "redundant_dependencies": [],
                    "security_report": {},
                    "score": self.baseline_result.dependencies.score
                },
                "deployment": {
                    "dockerfile_score": self.baseline_result.deployment.dockerfile_score,
                    "k8s_readiness": self.baseline_result.deployment.k8s_readiness,
                    "ci_cd_completeness": self.baseline_result.deployment.ci_cd_completeness,
                    "monitoring_score": self.baseline_result.deployment.monitoring_score,
                    "score": self.baseline_result.deployment.score
                },
                "security": {
                    "vulnerabilities": [],
                    "hardcoded_secrets": [],
                    "tls_issues": [],
                    "injection_risks": [],
                    "hipaa_compliance_score": self.baseline_result.security.hipaa_compliance_score,
                    "score": self.baseline_result.security.score
                },
                "scalability": {
                    "ddp_correctness": self.baseline_result.scalability.ddp_correctness,
                    "scaling_efficiency": self.baseline_result.scalability.scaling_efficiency,
                    "memory_bottlenecks": self.baseline_result.scalability.memory_bottlenecks,
                    "communication_overhead_ms": self.baseline_result.scalability.communication_overhead_ms,
                    "score": self.baseline_result.scalability.score,
                    "recommendations": self.baseline_result.scalability.recommendations
                },
                "critical_issues": []
            }
            json.dump(baseline_data, f)
            baseline_path = f.name
        
        try:
            report = self.detector.generate_diff_report(
                current=self.current_result,
                baseline_path=baseline_path,
                format="text"
            )
            
            # Should generate text report
            assert "REGRESSION" in report.upper() or "Status:" in report
            
        finally:
            os.unlink(baseline_path)
    
    def test_should_fail_build_critical_regressions(self):
        """Test CI build failure logic with critical regressions."""
        # First detect regressions to get proper result structure
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            baseline_data = {
                "timestamp": self.baseline_result.timestamp.isoformat(),
                "project_path": self.baseline_result.project_path,
                "git_commit": self.baseline_result.git_commit,
                "overall_score": self.baseline_result.overall_score,
                "architecture": {
                    "total_files": self.baseline_result.architecture.total_files,
                    "large_files": self.baseline_result.architecture.large_files,
                    "circular_dependencies": self.baseline_result.architecture.circular_dependencies,
                    "coupling_metrics": self.baseline_result.architecture.coupling_metrics,
                    "solid_violations": [],
                    "score": self.baseline_result.architecture.score
                },
                "coverage": {
                    "line_coverage": self.baseline_result.coverage.line_coverage,
                    "branch_coverage": self.baseline_result.coverage.branch_coverage,
                    "untested_critical_paths": self.baseline_result.coverage.untested_critical_paths,
                    "missing_property_tests": self.baseline_result.coverage.missing_property_tests,
                    "flaky_tests": self.baseline_result.coverage.flaky_tests,
                    "slow_tests": [],
                    "score": self.baseline_result.coverage.score
                },
                "performance": {
                    "gpu_utilization": self.baseline_result.performance.gpu_utilization,
                    "memory_usage_peak_gb": self.baseline_result.performance.memory_usage_peak_gb,
                    "memory_usage_avg_gb": 0.0,
                    "bottlenecks": self.baseline_result.performance.bottlenecks,
                    "flame_graph_path": self.baseline_result.performance.flame_graph_path,
                    "score": self.baseline_result.performance.score
                },
                "code_quality": {
                    "average_complexity": self.baseline_result.code_quality.average_complexity,
                    "high_complexity_functions": self.baseline_result.code_quality.high_complexity_functions,
                    "duplication_percentage": self.baseline_result.code_quality.duplication_percentage,
                    "documentation_coverage": self.baseline_result.code_quality.documentation_coverage,
                    "pylint_score": self.baseline_result.code_quality.pylint_score,
                    "score": self.baseline_result.code_quality.score,
                    "fix_suggestions": []
                },
                "dependencies": {
                    "total_dependencies": self.baseline_result.dependencies.total_dependencies,
                    "vulnerabilities": self.baseline_result.dependencies.vulnerabilities,
                    "outdated_packages": self.baseline_result.dependencies.outdated_packages,
                    "license_issues": self.baseline_result.dependencies.license_issues,
                    "unused_dependencies": self.baseline_result.dependencies.unused_dependencies,
                    "redundant_dependencies": [],
                    "security_report": {},
                    "score": self.baseline_result.dependencies.score
                },
                "deployment": {
                    "dockerfile_score": self.baseline_result.deployment.dockerfile_score,
                    "k8s_readiness": self.baseline_result.deployment.k8s_readiness,
                    "ci_cd_completeness": self.baseline_result.deployment.ci_cd_completeness,
                    "monitoring_score": self.baseline_result.deployment.monitoring_score,
                    "score": self.baseline_result.deployment.score
                },
                "security": {
                    "vulnerabilities": [],
                    "hardcoded_secrets": [],
                    "tls_issues": [],
                    "injection_risks": [],
                    "hipaa_compliance_score": self.baseline_result.security.hipaa_compliance_score,
                    "score": self.baseline_result.security.score
                },
                "scalability": {
                    "ddp_correctness": self.baseline_result.scalability.ddp_correctness,
                    "scaling_efficiency": self.baseline_result.scalability.scaling_efficiency,
                    "memory_bottlenecks": self.baseline_result.scalability.memory_bottlenecks,
                    "communication_overhead_ms": self.baseline_result.scalability.communication_overhead_ms,
                    "score": self.baseline_result.scalability.score,
                    "recommendations": self.baseline_result.scalability.recommendations
                },
                "critical_issues": []
            }
            json.dump(baseline_data, f)
            baseline_path = f.name
        
        try:
            regression_result = self.detector.detect_regressions(
                current=self.current_result,
                baseline_path=baseline_path
            )
            
            should_fail, reason = self.detector.should_fail_build(regression_result)
            
            # Should fail build due to critical regressions
            assert should_fail is True
            assert "critical" in reason.lower() or "regression" in reason.lower()
            
        finally:
            os.unlink(baseline_path)
    
    def test_should_fail_build_no_critical_regressions(self):
        """Test CI build failure logic with no critical regressions."""
        # Create improved result with no critical regressions
        improved_result = AnalysisResult(
            timestamp=datetime.now(),
            project_path="/test/project",
            git_commit="improved123",
            architecture=self.baseline_result.architecture,
            performance=self.baseline_result.performance,
            coverage=CoverageAnalysis(
                line_coverage=84.0,  # Small decrease, not critical
                branch_coverage=79.0,  # Small decrease, not critical
                untested_critical_paths=[],
                missing_property_tests=[],
                flaky_tests=[],
                score=81.0
            ),
            code_quality=self.baseline_result.code_quality,
            dependencies=self.baseline_result.dependencies,
            deployment=self.baseline_result.deployment,
            security=self.baseline_result.security,
            scalability=self.baseline_result.scalability,
            overall_score=79.0,  # Small decrease, not critical
            critical_issues=[]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            baseline_data = {
                "timestamp": self.baseline_result.timestamp.isoformat(),
                "project_path": self.baseline_result.project_path,
                "git_commit": self.baseline_result.git_commit,
                "overall_score": self.baseline_result.overall_score,
                "architecture": {
                    "total_files": self.baseline_result.architecture.total_files,
                    "large_files": self.baseline_result.architecture.large_files,
                    "circular_dependencies": self.baseline_result.architecture.circular_dependencies,
                    "coupling_metrics": self.baseline_result.architecture.coupling_metrics,
                    "solid_violations": [],
                    "score": self.baseline_result.architecture.score
                },
                "coverage": {
                    "line_coverage": self.baseline_result.coverage.line_coverage,
                    "branch_coverage": self.baseline_result.coverage.branch_coverage,
                    "untested_critical_paths": self.baseline_result.coverage.untested_critical_paths,
                    "missing_property_tests": self.baseline_result.coverage.missing_property_tests,
                    "flaky_tests": self.baseline_result.coverage.flaky_tests,
                    "slow_tests": [],
                    "score": self.baseline_result.coverage.score
                },
                "performance": {
                    "gpu_utilization": self.baseline_result.performance.gpu_utilization,
                    "memory_usage_peak_gb": self.baseline_result.performance.memory_usage_peak_gb,
                    "memory_usage_avg_gb": 0.0,
                    "bottlenecks": self.baseline_result.performance.bottlenecks,
                    "flame_graph_path": self.baseline_result.performance.flame_graph_path,
                    "score": self.baseline_result.performance.score
                },
                "code_quality": {
                    "average_complexity": self.baseline_result.code_quality.average_complexity,
                    "high_complexity_functions": self.baseline_result.code_quality.high_complexity_functions,
                    "duplication_percentage": self.baseline_result.code_quality.duplication_percentage,
                    "documentation_coverage": self.baseline_result.code_quality.documentation_coverage,
                    "pylint_score": self.baseline_result.code_quality.pylint_score,
                    "score": self.baseline_result.code_quality.score,
                    "fix_suggestions": []
                },
                "dependencies": {
                    "total_dependencies": self.baseline_result.dependencies.total_dependencies,
                    "vulnerabilities": self.baseline_result.dependencies.vulnerabilities,
                    "outdated_packages": self.baseline_result.dependencies.outdated_packages,
                    "license_issues": self.baseline_result.dependencies.license_issues,
                    "unused_dependencies": self.baseline_result.dependencies.unused_dependencies,
                    "redundant_dependencies": [],
                    "security_report": {},
                    "score": self.baseline_result.dependencies.score
                },
                "deployment": {
                    "dockerfile_score": self.baseline_result.deployment.dockerfile_score,
                    "k8s_readiness": self.baseline_result.deployment.k8s_readiness,
                    "ci_cd_completeness": self.baseline_result.deployment.ci_cd_completeness,
                    "monitoring_score": self.baseline_result.deployment.monitoring_score,
                    "score": self.baseline_result.deployment.score
                },
                "security": {
                    "vulnerabilities": [],
                    "hardcoded_secrets": [],
                    "tls_issues": [],
                    "injection_risks": [],
                    "hipaa_compliance_score": self.baseline_result.security.hipaa_compliance_score,
                    "score": self.baseline_result.security.score
                },
                "scalability": {
                    "ddp_correctness": self.baseline_result.scalability.ddp_correctness,
                    "scaling_efficiency": self.baseline_result.scalability.scaling_efficiency,
                    "memory_bottlenecks": self.baseline_result.scalability.memory_bottlenecks,
                    "communication_overhead_ms": self.baseline_result.scalability.communication_overhead_ms,
                    "score": self.baseline_result.scalability.score,
                    "recommendations": self.baseline_result.scalability.recommendations
                },
                "critical_issues": []
            }
            json.dump(baseline_data, f)
            baseline_path = f.name
        
        try:
            regression_result = self.detector.detect_regressions(
                current=improved_result,
                baseline_path=baseline_path
            )
            
            should_fail, reason = self.detector.should_fail_build(regression_result)
            
            # Should not fail build
            assert should_fail is False
            assert "no critical" in reason.lower() or "no regressions" in reason.lower()
            
        finally:
            os.unlink(baseline_path)
    
    def test_should_fail_build_no_regressions(self):
        """Test CI build failure logic with no regressions."""
        # Create improved result
        improved_result = AnalysisResult(
            timestamp=datetime.now(),
            project_path="/test/project",
            git_commit="improved123",
            architecture=self.baseline_result.architecture,
            performance=self.baseline_result.performance,
            coverage=CoverageAnalysis(
                line_coverage=87.0,  # Improvement
                branch_coverage=82.0,  # Improvement
                untested_critical_paths=[],
                missing_property_tests=[],
                flaky_tests=[],
                score=85.0
            ),
            code_quality=self.baseline_result.code_quality,
            dependencies=self.baseline_result.dependencies,
            deployment=self.baseline_result.deployment,
            security=self.baseline_result.security,
            scalability=self.baseline_result.scalability,
            overall_score=85.0,
            critical_issues=[]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            baseline_data = {
                "timestamp": self.baseline_result.timestamp.isoformat(),
                "project_path": self.baseline_result.project_path,
                "git_commit": self.baseline_result.git_commit,
                "overall_score": self.baseline_result.overall_score,
                "architecture": {
                    "total_files": self.baseline_result.architecture.total_files,
                    "large_files": self.baseline_result.architecture.large_files,
                    "circular_dependencies": self.baseline_result.architecture.circular_dependencies,
                    "coupling_metrics": self.baseline_result.architecture.coupling_metrics,
                    "solid_violations": [],
                    "score": self.baseline_result.architecture.score
                },
                "coverage": {
                    "line_coverage": self.baseline_result.coverage.line_coverage,
                    "branch_coverage": self.baseline_result.coverage.branch_coverage,
                    "untested_critical_paths": self.baseline_result.coverage.untested_critical_paths,
                    "missing_property_tests": self.baseline_result.coverage.missing_property_tests,
                    "flaky_tests": self.baseline_result.coverage.flaky_tests,
                    "slow_tests": [],
                    "score": self.baseline_result.coverage.score
                },
                "performance": {
                    "gpu_utilization": self.baseline_result.performance.gpu_utilization,
                    "memory_usage_peak_gb": self.baseline_result.performance.memory_usage_peak_gb,
                    "memory_usage_avg_gb": 0.0,
                    "bottlenecks": self.baseline_result.performance.bottlenecks,
                    "flame_graph_path": self.baseline_result.performance.flame_graph_path,
                    "score": self.baseline_result.performance.score
                },
                "code_quality": {
                    "average_complexity": self.baseline_result.code_quality.average_complexity,
                    "high_complexity_functions": self.baseline_result.code_quality.high_complexity_functions,
                    "duplication_percentage": self.baseline_result.code_quality.duplication_percentage,
                    "documentation_coverage": self.baseline_result.code_quality.documentation_coverage,
                    "pylint_score": self.baseline_result.code_quality.pylint_score,
                    "score": self.baseline_result.code_quality.score,
                    "fix_suggestions": []
                },
                "dependencies": {
                    "total_dependencies": self.baseline_result.dependencies.total_dependencies,
                    "vulnerabilities": self.baseline_result.dependencies.vulnerabilities,
                    "outdated_packages": self.baseline_result.dependencies.outdated_packages,
                    "license_issues": self.baseline_result.dependencies.license_issues,
                    "unused_dependencies": self.baseline_result.dependencies.unused_dependencies,
                    "redundant_dependencies": [],
                    "security_report": {},
                    "score": self.baseline_result.dependencies.score
                },
                "deployment": {
                    "dockerfile_score": self.baseline_result.deployment.dockerfile_score,
                    "k8s_readiness": self.baseline_result.deployment.k8s_readiness,
                    "ci_cd_completeness": self.baseline_result.deployment.ci_cd_completeness,
                    "monitoring_score": self.baseline_result.deployment.monitoring_score,
                    "score": self.baseline_result.deployment.score
                },
                "security": {
                    "vulnerabilities": [],
                    "hardcoded_secrets": [],
                    "tls_issues": [],
                    "injection_risks": [],
                    "hipaa_compliance_score": self.baseline_result.security.hipaa_compliance_score,
                    "score": self.baseline_result.security.score
                },
                "scalability": {
                    "ddp_correctness": self.baseline_result.scalability.ddp_correctness,
                    "scaling_efficiency": self.baseline_result.scalability.scaling_efficiency,
                    "memory_bottlenecks": self.baseline_result.scalability.memory_bottlenecks,
                    "communication_overhead_ms": self.baseline_result.scalability.communication_overhead_ms,
                    "score": self.baseline_result.scalability.score,
                    "recommendations": self.baseline_result.scalability.recommendations
                },
                "critical_issues": []
            }
            json.dump(baseline_data, f)
            baseline_path = f.name
        
        try:
            regression_result = self.detector.detect_regressions(
                current=improved_result,
                baseline_path=baseline_path
            )
            
            should_fail, reason = self.detector.should_fail_build(regression_result)
            
            # Should not fail build
            assert should_fail is False
            assert "no" in reason.lower() and "regression" in reason.lower()
            
        finally:
            os.unlink(baseline_path)
    
    def test_load_baseline_valid_file(self):
        """Test loading valid baseline file."""
        baseline_data = {
            "timestamp": "2023-01-01T00:00:00",
            "project_path": "/test",
            "git_commit": "abc123",
            "overall_score": 85.0,
            "architecture": {
                "total_files": 100,
                "large_files": [],
                "circular_dependencies": [],
                "coupling_metrics": {},
                "solid_violations": [],
                "score": 85.0
            },
            "coverage": {
                "line_coverage": 90.0,
                "branch_coverage": 85.0,
                "untested_critical_paths": [],
                "missing_property_tests": [],
                "flaky_tests": [],
                "slow_tests": [],
                "score": 87.0
            },
            "performance": {
                "gpu_utilization": 80.0,
                "memory_usage_peak_gb": 10.0,
                "memory_usage_avg_gb": 8.0,
                "bottlenecks": [],
                "flame_graph_path": "",
                "score": 80.0
            },
            "code_quality": {
                "average_complexity": 5.0,
                "high_complexity_functions": [],
                "duplication_percentage": 3.0,
                "documentation_coverage": 80.0,
                "pylint_score": 9.0,
                "score": 85.0,
                "fix_suggestions": []
            },
            "dependencies": {
                "total_dependencies": 20,
                "vulnerabilities": [],
                "outdated_packages": [],
                "license_issues": [],
                "unused_dependencies": [],
                "redundant_dependencies": [],
                "security_report": {},
                "score": 90.0
            },
            "deployment": {
                "dockerfile_score": 85.0,
                "k8s_readiness": 80.0,
                "ci_cd_completeness": 90.0,
                "monitoring_score": 75.0,
                "score": 82.5
            },
            "security": {
                "vulnerabilities": [],
                "hardcoded_secrets": [],
                "tls_issues": [],
                "injection_risks": [],
                "hipaa_compliance_score": 90.0,
                "score": 90.0
            },
            "scalability": {
                "ddp_correctness": True,
                "scaling_efficiency": "linear",
                "memory_bottlenecks": [],
                "communication_overhead_ms": 10.0,
                "score": 85.0,
                "recommendations": {}
            },
            "critical_issues": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(baseline_data, f)
            baseline_path = f.name
        
        try:
            result = self.detector._load_baseline(baseline_path)
            
            # Should load successfully
            assert result is not None
            assert result.overall_score == 85.0
            assert result.coverage.line_coverage == 90.0
            assert result.performance.gpu_utilization == 80.0
            
        finally:
            os.unlink(baseline_path)
    
    def test_load_baseline_invalid_file(self):
        """Test loading invalid baseline file."""
        result = self.detector._load_baseline("nonexistent.json")
        
        # Should return None for invalid file
        assert result is None
    
    def test_load_baseline_malformed_json(self):
        """Test loading malformed JSON baseline file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            baseline_path = f.name
        
        try:
            result = self.detector._load_baseline(baseline_path)
            
            # Should return None for malformed JSON
            assert result is None
            
        finally:
            os.unlink(baseline_path)
    
    def test_regression_summary_generation(self):
        """Test regression summary generation."""
        # Create temporary baseline file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            baseline_data = {
                "timestamp": self.baseline_result.timestamp.isoformat(),
                "project_path": self.baseline_result.project_path,
                "git_commit": self.baseline_result.git_commit,
                "overall_score": self.baseline_result.overall_score,
                "architecture": {
                    "total_files": self.baseline_result.architecture.total_files,
                    "large_files": self.baseline_result.architecture.large_files,
                    "circular_dependencies": self.baseline_result.architecture.circular_dependencies,
                    "coupling_metrics": self.baseline_result.architecture.coupling_metrics,
                    "solid_violations": [],
                    "score": self.baseline_result.architecture.score
                },
                "coverage": {
                    "line_coverage": self.baseline_result.coverage.line_coverage,
                    "branch_coverage": self.baseline_result.coverage.branch_coverage,
                    "untested_critical_paths": self.baseline_result.coverage.untested_critical_paths,
                    "missing_property_tests": self.baseline_result.coverage.missing_property_tests,
                    "flaky_tests": self.baseline_result.coverage.flaky_tests,
                    "slow_tests": [],
                    "score": self.baseline_result.coverage.score
                },
                "performance": {
                    "gpu_utilization": self.baseline_result.performance.gpu_utilization,
                    "memory_usage_peak_gb": self.baseline_result.performance.memory_usage_peak_gb,
                    "memory_usage_avg_gb": 0.0,
                    "bottlenecks": self.baseline_result.performance.bottlenecks,
                    "flame_graph_path": self.baseline_result.performance.flame_graph_path,
                    "score": self.baseline_result.performance.score
                },
                "code_quality": {
                    "average_complexity": self.baseline_result.code_quality.average_complexity,
                    "high_complexity_functions": self.baseline_result.code_quality.high_complexity_functions,
                    "duplication_percentage": self.baseline_result.code_quality.duplication_percentage,
                    "documentation_coverage": self.baseline_result.code_quality.documentation_coverage,
                    "pylint_score": self.baseline_result.code_quality.pylint_score,
                    "score": self.baseline_result.code_quality.score,
                    "fix_suggestions": []
                },
                "dependencies": {
                    "total_dependencies": self.baseline_result.dependencies.total_dependencies,
                    "vulnerabilities": self.baseline_result.dependencies.vulnerabilities,
                    "outdated_packages": self.baseline_result.dependencies.outdated_packages,
                    "license_issues": self.baseline_result.dependencies.license_issues,
                    "unused_dependencies": self.baseline_result.dependencies.unused_dependencies,
                    "redundant_dependencies": [],
                    "security_report": {},
                    "score": self.baseline_result.dependencies.score
                },
                "deployment": {
                    "dockerfile_score": self.baseline_result.deployment.dockerfile_score,
                    "k8s_readiness": self.baseline_result.deployment.k8s_readiness,
                    "ci_cd_completeness": self.baseline_result.deployment.ci_cd_completeness,
                    "monitoring_score": self.baseline_result.deployment.monitoring_score,
                    "score": self.baseline_result.deployment.score
                },
                "security": {
                    "vulnerabilities": [],
                    "hardcoded_secrets": [],
                    "tls_issues": [],
                    "injection_risks": [],
                    "hipaa_compliance_score": self.baseline_result.security.hipaa_compliance_score,
                    "score": self.baseline_result.security.score
                },
                "scalability": {
                    "ddp_correctness": self.baseline_result.scalability.ddp_correctness,
                    "scaling_efficiency": self.baseline_result.scalability.scaling_efficiency,
                    "memory_bottlenecks": self.baseline_result.scalability.memory_bottlenecks,
                    "communication_overhead_ms": self.baseline_result.scalability.communication_overhead_ms,
                    "score": self.baseline_result.scalability.score,
                    "recommendations": self.baseline_result.scalability.recommendations
                },
                "critical_issues": []
            }
            json.dump(baseline_data, f)
            baseline_path = f.name
        
        try:
            regression_result = self.detector.detect_regressions(
                current=self.current_result,
                baseline_path=baseline_path
            )
            
            summary = regression_result['summary']
            
            # Should generate comprehensive summary
            assert 'status' in summary
            assert 'message' in summary
            assert 'critical_count' in summary
            assert summary['critical_count'] > 0  # Should have critical regressions
            
        finally:
            os.unlink(baseline_path)
    
    def test_regression_summary_no_regressions(self):
        """Test regression summary with no regressions."""
        # Create improved result
        improved_result = AnalysisResult(
            timestamp=datetime.now(),
            project_path="/test/project",
            git_commit="improved123",
            architecture=self.baseline_result.architecture,
            performance=self.baseline_result.performance,
            coverage=CoverageAnalysis(
                line_coverage=87.0,  # Improvement
                branch_coverage=82.0,  # Improvement
                untested_critical_paths=[],
                missing_property_tests=[],
                flaky_tests=[],
                score=85.0
            ),
            code_quality=self.baseline_result.code_quality,
            dependencies=self.baseline_result.dependencies,
            deployment=self.baseline_result.deployment,
            security=self.baseline_result.security,
            scalability=self.baseline_result.scalability,
            overall_score=85.0,
            critical_issues=[]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            baseline_data = {
                "timestamp": self.baseline_result.timestamp.isoformat(),
                "project_path": self.baseline_result.project_path,
                "git_commit": self.baseline_result.git_commit,
                "overall_score": self.baseline_result.overall_score,
                "architecture": {
                    "total_files": self.baseline_result.architecture.total_files,
                    "large_files": self.baseline_result.architecture.large_files,
                    "circular_dependencies": self.baseline_result.architecture.circular_dependencies,
                    "coupling_metrics": self.baseline_result.architecture.coupling_metrics,
                    "solid_violations": [],
                    "score": self.baseline_result.architecture.score
                },
                "coverage": {
                    "line_coverage": self.baseline_result.coverage.line_coverage,
                    "branch_coverage": self.baseline_result.coverage.branch_coverage,
                    "untested_critical_paths": self.baseline_result.coverage.untested_critical_paths,
                    "missing_property_tests": self.baseline_result.coverage.missing_property_tests,
                    "flaky_tests": self.baseline_result.coverage.flaky_tests,
                    "slow_tests": [],
                    "score": self.baseline_result.coverage.score
                },
                "performance": {
                    "gpu_utilization": self.baseline_result.performance.gpu_utilization,
                    "memory_usage_peak_gb": self.baseline_result.performance.memory_usage_peak_gb,
                    "memory_usage_avg_gb": 0.0,
                    "bottlenecks": self.baseline_result.performance.bottlenecks,
                    "flame_graph_path": self.baseline_result.performance.flame_graph_path,
                    "score": self.baseline_result.performance.score
                },
                "code_quality": {
                    "average_complexity": self.baseline_result.code_quality.average_complexity,
                    "high_complexity_functions": self.baseline_result.code_quality.high_complexity_functions,
                    "duplication_percentage": self.baseline_result.code_quality.duplication_percentage,
                    "documentation_coverage": self.baseline_result.code_quality.documentation_coverage,
                    "pylint_score": self.baseline_result.code_quality.pylint_score,
                    "score": self.baseline_result.code_quality.score,
                    "fix_suggestions": []
                },
                "dependencies": {
                    "total_dependencies": self.baseline_result.dependencies.total_dependencies,
                    "vulnerabilities": self.baseline_result.dependencies.vulnerabilities,
                    "outdated_packages": self.baseline_result.dependencies.outdated_packages,
                    "license_issues": self.baseline_result.dependencies.license_issues,
                    "unused_dependencies": self.baseline_result.dependencies.unused_dependencies,
                    "redundant_dependencies": [],
                    "security_report": {},
                    "score": self.baseline_result.dependencies.score
                },
                "deployment": {
                    "dockerfile_score": self.baseline_result.deployment.dockerfile_score,
                    "k8s_readiness": self.baseline_result.deployment.k8s_readiness,
                    "ci_cd_completeness": self.baseline_result.deployment.ci_cd_completeness,
                    "monitoring_score": self.baseline_result.deployment.monitoring_score,
                    "score": self.baseline_result.deployment.score
                },
                "security": {
                    "vulnerabilities": [],
                    "hardcoded_secrets": [],
                    "tls_issues": [],
                    "injection_risks": [],
                    "hipaa_compliance_score": self.baseline_result.security.hipaa_compliance_score,
                    "score": self.baseline_result.security.score
                },
                "scalability": {
                    "ddp_correctness": self.baseline_result.scalability.ddp_correctness,
                    "scaling_efficiency": self.baseline_result.scalability.scaling_efficiency,
                    "memory_bottlenecks": self.baseline_result.scalability.memory_bottlenecks,
                    "communication_overhead_ms": self.baseline_result.scalability.communication_overhead_ms,
                    "score": self.baseline_result.scalability.score,
                    "recommendations": self.baseline_result.scalability.recommendations
                },
                "critical_issues": []
            }
            json.dump(baseline_data, f)
            baseline_path = f.name
        
        try:
            regression_result = self.detector.detect_regressions(
                current=improved_result,
                baseline_path=baseline_path
            )
            
            summary = regression_result['summary']
            
            # Should generate positive summary
            assert 'status' in summary
            assert 'message' in summary
            assert summary['status'] == 'PASS'
            
        finally:
            os.unlink(baseline_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])