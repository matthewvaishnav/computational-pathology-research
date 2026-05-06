"""
Property-based tests for JSON serialization round-trip consistency.

Tests that AnalysisResult objects can be serialized to JSON and deserialized
back without data loss.
"""

import pytest
from hypothesis import given, strategies as st
from datetime import datetime

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
    Severity,
    Priority,
    Role,
)


# Hypothesis strategies for generating test data
@st.composite
def issue_strategy(draw):
    """Generate random Issue objects."""
    return Issue(
        id=draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))),
        dimension=draw(st.sampled_from(['architecture', 'performance', 'coverage', 'code_quality',
                                        'dependencies', 'deployment', 'security', 'scalability'])),
        severity=draw(st.sampled_from(list(Severity))),
        category=draw(st.text(min_size=1, max_size=30)),
        title=draw(st.text(min_size=1, max_size=100)),
        description=draw(st.text(min_size=1, max_size=500)),
        file_path=draw(st.text(min_size=1, max_size=200)),
        line_number=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10000))),
        recommendation=draw(st.text(max_size=500)),
        effort_hours=draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
        priority=draw(st.sampled_from(list(Priority))),
        role=draw(st.sampled_from(list(Role))),
        references=draw(st.lists(st.text(max_size=200), max_size=10)),
    )


@st.composite
def architecture_analysis_strategy(draw):
    """Generate random ArchitectureAnalysis objects."""
    return ArchitectureAnalysis(
        total_files=draw(st.integers(min_value=0, max_value=100000)),
        large_files=draw(st.lists(st.dictionaries(
            st.sampled_from(['path', 'lines', 'complexity']),
            st.one_of(st.text(max_size=200), st.integers(min_value=0, max_value=10000), st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)),
            min_size=1, max_size=3
        ), max_size=10)),
        circular_dependencies=draw(st.lists(st.lists(st.text(max_size=100), min_size=2, max_size=5), max_size=10)),
        coupling_metrics=draw(st.dictionaries(st.text(max_size=50), st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False), max_size=10)),
        solid_violations=draw(st.lists(issue_strategy(), max_size=10)),
        score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def performance_analysis_strategy(draw):
    """Generate random PerformanceAnalysis objects."""
    return PerformanceAnalysis(
        gpu_utilization=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        bottlenecks=draw(st.lists(st.dictionaries(
            st.sampled_from(['operation', 'time_ms', 'percentage']),
            st.one_of(st.text(max_size=100), st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False)),
            min_size=1, max_size=3
        ), max_size=10)),
        flame_graph_path=draw(st.text(max_size=200)),
        memory_usage_peak_gb=draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
        memory_usage_avg_gb=draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
        score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def coverage_analysis_strategy(draw):
    """Generate random CoverageAnalysis objects."""
    return CoverageAnalysis(
        line_coverage=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        branch_coverage=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        untested_critical_paths=draw(st.lists(st.text(max_size=200), max_size=20)),
        missing_property_tests=draw(st.lists(st.text(max_size=200), max_size=20)),
        flaky_tests=draw(st.lists(st.text(max_size=200), max_size=20)),
        slow_tests=draw(st.lists(st.dictionaries(
            st.sampled_from(['name', 'duration_ms']),
            st.one_of(st.text(max_size=100), st.floats(min_value=0, max_value=100000, allow_nan=False, allow_infinity=False)),
            min_size=1, max_size=2
        ), max_size=10)),
        score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def code_quality_analysis_strategy(draw):
    """Generate random CodeQualityAnalysis objects."""
    return CodeQualityAnalysis(
        average_complexity=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        high_complexity_functions=draw(st.lists(st.dictionaries(
            st.sampled_from(['name', 'file', 'line', 'complexity']),
            st.one_of(st.text(max_size=100), st.integers(min_value=1, max_value=10000), st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)),
            min_size=1, max_size=4
        ), max_size=10)),
        duplication_percentage=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        documentation_coverage=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        pylint_score=draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
        score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def dependency_analysis_strategy(draw):
    """Generate random DependencyAnalysis objects."""
    return DependencyAnalysis(
        total_dependencies=draw(st.integers(min_value=0, max_value=1000)),
        vulnerabilities=draw(st.lists(st.dictionaries(
            st.sampled_from(['cve_id', 'package', 'severity', 'cvss_score', 'fix_version']),
            st.one_of(st.text(max_size=50), st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False)),
            min_size=1, max_size=5
        ), max_size=10)),
        outdated_packages=draw(st.lists(st.text(max_size=100), max_size=20)),
        license_issues=draw(st.lists(st.text(max_size=200), max_size=10)),
        score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def deployment_analysis_strategy(draw):
    """Generate random DeploymentAnalysis objects."""
    return DeploymentAnalysis(
        dockerfile_score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        k8s_readiness=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        ci_cd_completeness=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        monitoring_score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def security_analysis_strategy(draw):
    """Generate random SecurityAnalysis objects."""
    return SecurityAnalysis(
        vulnerabilities=draw(st.lists(st.dictionaries(
            st.sampled_from(['type', 'severity', 'file', 'line', 'description']),
            st.one_of(st.text(max_size=100), st.integers(min_value=1, max_value=10000)),
            min_size=1, max_size=5
        ), max_size=10)),
        hipaa_compliance_score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        hardcoded_secrets=draw(st.lists(st.text(max_size=200), max_size=10)),
        injection_risks=draw(st.lists(st.dictionaries(
            st.sampled_from(['type', 'file', 'line']),
            st.one_of(st.text(max_size=100), st.integers(min_value=1, max_value=10000)),
            min_size=1, max_size=3
        ), max_size=10)),
        score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def scalability_analysis_strategy(draw):
    """Generate random ScalabilityAnalysis objects."""
    return ScalabilityAnalysis(
        ddp_correctness=draw(st.booleans()),
        scaling_efficiency=draw(st.sampled_from(['linear', 'sub-linear', 'super-linear', 'unknown'])),
        memory_bottlenecks=draw(st.lists(st.text(max_size=200), max_size=10)),
        communication_overhead_ms=draw(st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)),
        score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def analysis_result_strategy(draw):
    """Generate random AnalysisResult objects."""
    return AnalysisResult(
        timestamp=datetime.now().isoformat(),
        project_path=draw(st.text(min_size=1, max_size=200)),
        git_commit=draw(st.text(min_size=40, max_size=40, alphabet='0123456789abcdef')),
        architecture=draw(architecture_analysis_strategy()),
        performance=draw(performance_analysis_strategy()),
        coverage=draw(coverage_analysis_strategy()),
        code_quality=draw(code_quality_analysis_strategy()),
        dependencies=draw(dependency_analysis_strategy()),
        deployment=draw(deployment_analysis_strategy()),
        security=draw(security_analysis_strategy()),
        scalability=draw(scalability_analysis_strategy()),
        overall_score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        critical_issues=draw(st.lists(issue_strategy(), max_size=10)),
    )


# Property-based test
@given(analysis_result_strategy())
def test_round_trip_serialization(result: AnalysisResult):
    """
    Property 1: Round-trip consistency.
    
    FOR ALL valid AnalysisResult objects,
    parse(serialize(obj)) == obj
    
    Validates: Requirements 11.4 - Round-trip serialization preserves all data
    """
    # Serialize to JSON
    json_str = result.to_json(validate_schema=True)
    
    # Deserialize back
    deserialized = AnalysisResult.from_json(json_str, validate_schema=True)
    
    # Verify all fields match
    assert deserialized.timestamp == result.timestamp
    assert deserialized.project_path == result.project_path
    assert deserialized.git_commit == result.git_commit
    assert deserialized.overall_score == result.overall_score
    
    # Verify nested structures
    assert deserialized.architecture.total_files == result.architecture.total_files
    assert deserialized.architecture.score == result.architecture.score
    assert deserialized.performance.gpu_utilization == result.performance.gpu_utilization
    assert deserialized.performance.score == result.performance.score
    assert deserialized.coverage.line_coverage == result.coverage.line_coverage
    assert deserialized.coverage.branch_coverage == result.coverage.branch_coverage
    assert deserialized.code_quality.average_complexity == result.code_quality.average_complexity
    assert deserialized.dependencies.total_dependencies == result.dependencies.total_dependencies
    assert deserialized.deployment.dockerfile_score == result.deployment.dockerfile_score
    assert deserialized.security.hipaa_compliance_score == result.security.hipaa_compliance_score
    assert deserialized.scalability.ddp_correctness == result.scalability.ddp_correctness
    assert deserialized.scalability.scaling_efficiency == result.scalability.scaling_efficiency
    
    # Verify critical issues
    assert len(deserialized.critical_issues) == len(result.critical_issues)
    for orig_issue, deser_issue in zip(result.critical_issues, deserialized.critical_issues):
        assert deser_issue.id == orig_issue.id
        assert deser_issue.dimension == orig_issue.dimension
        assert deser_issue.severity == orig_issue.severity
        assert deser_issue.title == orig_issue.title
        assert deser_issue.file_path == orig_issue.file_path
        assert deser_issue.priority == orig_issue.priority
        assert deser_issue.role == orig_issue.role


# Unit tests for error handling
def test_invalid_json_raises_error():
    """Test that invalid JSON raises ValueError with descriptive message."""
    invalid_json = "{ invalid json }"
    
    with pytest.raises(ValueError, match="Invalid JSON"):
        AnalysisResult.from_json(invalid_json)


def test_missing_required_field_raises_error():
    """Test that missing required fields raise ValueError."""
    incomplete_json = '{"timestamp": "2024-01-01T00:00:00", "project_path": "/test"}'
    
    with pytest.raises(ValueError, match="Schema validation failed"):
        AnalysisResult.from_json(incomplete_json, validate_schema=True)


def test_invalid_score_range_raises_error():
    """Test that scores outside 0-100 range raise ValueError."""
    result = AnalysisResult(
        timestamp=datetime.now().isoformat(),
        project_path="/test",
        git_commit="a" * 40,
        architecture=ArchitectureAnalysis(),
        performance=PerformanceAnalysis(),
        coverage=CoverageAnalysis(),
        code_quality=CodeQualityAnalysis(),
        dependencies=DependencyAnalysis(),
        deployment=DeploymentAnalysis(),
        security=SecurityAnalysis(),
        scalability=ScalabilityAnalysis(),
        overall_score=150.0,  # Invalid: >100
    )
    
    with pytest.raises(ValueError, match="Schema validation failed"):
        result.to_json(validate_schema=True)


def test_schema_validation_can_be_disabled():
    """Test that schema validation can be disabled for performance."""
    result = AnalysisResult(
        timestamp=datetime.now().isoformat(),
        project_path="/test",
        git_commit="a" * 40,
        architecture=ArchitectureAnalysis(),
        performance=PerformanceAnalysis(),
        coverage=CoverageAnalysis(),
        code_quality=CodeQualityAnalysis(),
        dependencies=DependencyAnalysis(),
        deployment=DeploymentAnalysis(),
        security=SecurityAnalysis(),
        scalability=ScalabilityAnalysis(),
        overall_score=150.0,  # Invalid but validation disabled
    )
    
    # Should not raise error when validation disabled
    json_str = result.to_json(validate_schema=False)
    assert json_str is not None
    
    # Deserialization should also work without validation
    deserialized = AnalysisResult.from_json(json_str, validate_schema=False)
    assert deserialized.overall_score == 150.0
