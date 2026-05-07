"""
Unit tests for Result Aggregator.

Tests merging of analyzer results, deduplication, and overall score calculation.
"""

import pytest
from src.analysis.aggregator import ResultAggregator
from src.analysis.models import (
    ArchitectureAnalysis,
    PerformanceAnalysis,
    CoverageAnalysis,
    CodeQualityAnalysis,
    DependencyAnalysis,
    DeploymentAnalysis,
    SecurityAnalysis,
    ScalabilityAnalysis,
    Issue,
    Severity,
    Priority,
    Role
)


@pytest.fixture
def sample_architecture():
    """Sample architecture analysis."""
    return ArchitectureAnalysis(
        total_files=100,
        large_files=[{'path': 'src/large.py', 'lines': 600}],
        circular_dependencies=[['module_a', 'module_b', 'module_a']],
        coupling_metrics={'avg_fanout': 5.2},
        solid_violations=[
            Issue(
                id='arch-1',
                dimension='architecture',
                severity=Severity.HIGH,
                category='srp_violation',
                title='Large class violation',
                description='Class exceeds 500 lines',
                file_path='src/large.py',
                line_number=1,
                priority=Priority.P1,
                role=Role.BACKEND
            )
        ],
        score=75.0
    )


@pytest.fixture
def sample_performance():
    """Sample performance analysis."""
    return PerformanceAnalysis(
        gpu_utilization=85.5,
        bottlenecks=[
            {'operation': 'data_loading', 'time_ms': 650.0, 'percentage': 25.0}
        ],
        flame_graph_path='profile.svg',
        memory_usage_peak_gb=12.5,
        memory_usage_avg_gb=8.2,
        score=80.0
    )


@pytest.fixture
def sample_coverage():
    """Sample coverage analysis."""
    return CoverageAnalysis(
        line_coverage=55.0,
        branch_coverage=48.0,
        untested_critical_paths=['src/error_handler.py', 'src/validator.py'],
        missing_property_tests=['src/transforms.py'],
        flaky_tests=['test_integration.py::test_flaky'],
        slow_tests=[{'name': 'test_slow', 'duration_s': 12.5}],
        score=60.0
    )


@pytest.fixture
def sample_code_quality():
    """Sample code quality analysis."""
    return CodeQualityAnalysis(
        average_complexity=6.5,
        high_complexity_functions=[
            {'name': 'complex_func', 'file': 'src/utils.py', 'line': 42, 'complexity': 18}
        ],
        duplication_percentage=5.2,
        documentation_coverage=72.0,
        pylint_score=8.5,
        score=78.0
    )


@pytest.fixture
def sample_dependencies():
    """Sample dependency analysis."""
    return DependencyAnalysis(
        total_dependencies=45,
        vulnerabilities=[
            {
                'cve_id': 'CVE-2024-1234',
                'package': 'cryptography',
                'severity': 'critical',
                'cvss_score': 9.8,
                'fix_version': '42.0.0',
                'description': 'Critical vulnerability'
            }
        ],
        outdated_packages=['numpy==1.20.0'],
        license_issues=[],
        unused_dependencies=['unused-pkg'],
        redundant_dependencies=[],
        score=65.0
    )


@pytest.fixture
def sample_deployment():
    """Sample deployment analysis."""
    return DeploymentAnalysis(
        dockerfile_score=85.0,
        k8s_readiness=70.0,
        ci_cd_completeness=90.0,
        monitoring_score=60.0,
        score=76.25
    )


@pytest.fixture
def sample_security():
    """Sample security analysis."""
    return SecurityAnalysis(
        vulnerabilities=[
            {
                'id': 'sec-1',
                'severity': 'high',
                'title': 'SQL injection risk',
                'description': 'Unsafe query construction',
                'file': 'src/db.py',
                'line': 123,
                'recommendation': 'Use parameterized queries',
                'effort_hours': 3.0,
                'role': 'security'
            }
        ],
        hipaa_compliance_score=82.0,
        hardcoded_secrets=[],
        injection_risks=[],
        tls_issues=[],
        score=70.0
    )


@pytest.fixture
def sample_scalability():
    """Sample scalability analysis."""
    return ScalabilityAnalysis(
        ddp_correctness=True,
        scaling_efficiency='linear',
        memory_bottlenecks=['large_batch_processing'],
        communication_overhead_ms=15.5,
        score=85.0,
        recommendations={'use_gradient_accumulation': True}
    )


def test_aggregate_creates_unified_result(
    sample_architecture,
    sample_performance,
    sample_coverage,
    sample_code_quality,
    sample_dependencies,
    sample_deployment,
    sample_security,
    sample_scalability
):
    """Test that aggregator creates unified AnalysisResult."""
    aggregator = ResultAggregator(project_path='.')

    result = aggregator.aggregate(
        architecture=sample_architecture,
        performance=sample_performance,
        coverage=sample_coverage,
        code_quality=sample_code_quality,
        dependencies=sample_dependencies,
        deployment=sample_deployment,
        security=sample_security,
        scalability=sample_scalability
    )

    # Verify result structure
    assert result.project_path == '.'
    assert result.git_commit is not None
    assert result.timestamp is not None

    # Verify all dimensions present
    assert result.architecture == sample_architecture
    assert result.performance == sample_performance
    assert result.coverage == sample_coverage
    assert result.code_quality == sample_code_quality
    assert result.dependencies == sample_dependencies
    assert result.deployment == sample_deployment
    assert result.security == sample_security
    assert result.scalability == sample_scalability


def test_compute_overall_score(
    sample_architecture,
    sample_performance,
    sample_coverage,
    sample_code_quality,
    sample_dependencies,
    sample_deployment,
    sample_security,
    sample_scalability
):
    """Test overall score calculation with weighted average."""
    aggregator = ResultAggregator(project_path='.')

    result = aggregator.aggregate(
        architecture=sample_architecture,
        performance=sample_performance,
        coverage=sample_coverage,
        code_quality=sample_code_quality,
        dependencies=sample_dependencies,
        deployment=sample_deployment,
        security=sample_security,
        scalability=sample_scalability
    )

    # Expected: weighted average
    # security(70)*0.25 + coverage(60)*0.20 + code_quality(78)*0.15 +
    # dependencies(65)*0.15 + architecture(75)*0.10 + performance(80)*0.05 +
    # deployment(76.25)*0.05 + scalability(85)*0.05
    expected = (70*0.25 + 60*0.20 + 78*0.15 + 65*0.15 + 75*0.10 +
                80*0.05 + 76.25*0.05 + 85*0.05)

    assert result.overall_score == pytest.approx(expected, rel=0.01)


def test_extract_critical_issues(
    sample_architecture,
    sample_performance,
    sample_coverage,
    sample_code_quality,
    sample_dependencies,
    sample_deployment,
    sample_security,
    sample_scalability
):
    """Test extraction of critical issues from all dimensions."""
    aggregator = ResultAggregator(project_path='.')

    result = aggregator.aggregate(
        architecture=sample_architecture,
        performance=sample_performance,
        coverage=sample_coverage,
        code_quality=sample_code_quality,
        dependencies=sample_dependencies,
        deployment=sample_deployment,
        security=sample_security,
        scalability=sample_scalability
    )

    # Should extract issues from multiple dimensions
    assert len(result.critical_issues) > 0

    # Check for architecture issue
    arch_issues = [i for i in result.critical_issues if i.dimension == 'architecture']
    assert len(arch_issues) == 1
    assert arch_issues[0].id == 'arch-1'

    # Check for dependency CVE
    dep_issues = [i for i in result.critical_issues if i.dimension == 'dependencies']
    assert len(dep_issues) == 1
    assert 'CVE-2024-1234' in dep_issues[0].id

    # Check for security issue
    sec_issues = [i for i in result.critical_issues if i.dimension == 'security']
    assert len(sec_issues) == 1
    assert sec_issues[0].severity == Severity.HIGH

    # Check for coverage gaps
    cov_issues = [i for i in result.critical_issues if i.dimension == 'coverage']
    assert len(cov_issues) == 2  # 2 untested critical paths


def test_deduplicate_issues():
    """Test deduplication of issues with same file_path and title."""
    aggregator = ResultAggregator(project_path='.')

    issues = [
        Issue(
            id='issue-1',
            dimension='security',
            severity=Severity.HIGH,
            category='vulnerability',
            title='SQL injection',
            description='Unsafe query',
            file_path='src/db.py',
            priority=Priority.P1,
            role=Role.SECURITY
        ),
        Issue(
            id='issue-2',
            dimension='security',
            severity=Severity.HIGH,
            category='vulnerability',
            title='SQL injection',  # Same title
            description='Different description',
            file_path='src/db.py',  # Same file
            priority=Priority.P1,
            role=Role.SECURITY
        ),
        Issue(
            id='issue-3',
            dimension='security',
            severity=Severity.MEDIUM,
            category='vulnerability',
            title='XSS vulnerability',  # Different title
            description='Unsafe output',
            file_path='src/db.py',  # Same file
            priority=Priority.P2,
            role=Role.SECURITY
        )
    ]

    deduplicated = aggregator._deduplicate_issues(issues)

    # Should keep only 2 issues (first SQL injection + XSS)
    assert len(deduplicated) == 2
    assert deduplicated[0].id == 'issue-1'
    assert deduplicated[1].id == 'issue-3'


def test_sort_issues_by_priority_and_severity():
    """Test sorting issues by priority then severity."""
    aggregator = ResultAggregator(project_path='.')

    issues = [
        Issue(
            id='p2-medium',
            dimension='code_quality',
            severity=Severity.MEDIUM,
            category='complexity',
            title='Medium P2',
            description='',
            file_path='',
            priority=Priority.P2,
            role=Role.BACKEND
        ),
        Issue(
            id='p0-critical',
            dimension='security',
            severity=Severity.CRITICAL,
            category='vulnerability',
            title='Critical P0',
            description='',
            file_path='',
            priority=Priority.P0,
            role=Role.SECURITY
        ),
        Issue(
            id='p1-high',
            dimension='dependencies',
            severity=Severity.HIGH,
            category='cve',
            title='High P1',
            description='',
            file_path='',
            priority=Priority.P1,
            role=Role.DEVOPS
        ),
        Issue(
            id='p0-high',
            dimension='security',
            severity=Severity.HIGH,
            category='vulnerability',
            title='High P0',
            description='',
            file_path='',
            priority=Priority.P0,
            role=Role.SECURITY
        )
    ]

    sorted_issues = aggregator._sort_issues(issues)

    # Expected order: P0 critical, P0 high, P1 high, P2 medium
    assert sorted_issues[0].id == 'p0-critical'
    assert sorted_issues[1].id == 'p0-high'
    assert sorted_issues[2].id == 'p1-high'
    assert sorted_issues[3].id == 'p2-medium'


def test_aggregate_with_empty_results():
    """Test aggregation with empty analyzer results."""
    aggregator = ResultAggregator(project_path='.')

    result = aggregator.aggregate(
        architecture=ArchitectureAnalysis(),
        performance=PerformanceAnalysis(),
        coverage=CoverageAnalysis(),
        code_quality=CodeQualityAnalysis(),
        dependencies=DependencyAnalysis(),
        deployment=DeploymentAnalysis(),
        security=SecurityAnalysis(),
        scalability=ScalabilityAnalysis()
    )

    # Should create valid result with zero scores
    assert result.overall_score == 0.0
    assert len(result.critical_issues) == 0
