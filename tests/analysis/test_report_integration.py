"""
Integration tests for end-to-end report generation.

Tests JSON → Markdown → HTML conversion pipeline and visualization embedding.
"""

import pytest
import json
import tempfile
from pathlib import Path
from src.analysis.reporting import ReportGenerator
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
)


@pytest.fixture
def generator():
    """Create report generator instance."""
    return ReportGenerator()


@pytest.fixture
def complete_result():
    """Create complete analysis result for integration testing."""
    return AnalysisResult(
        timestamp="2024-01-01T00:00:00Z",
        project_path="/path/to/project",
        git_commit="abc123def456",
        architecture=ArchitectureAnalysis(
            total_files=250,
            large_files=[
                {"path": "src/models/large_model.py", "lines": 850, "complexity": 25.0},
                {"path": "src/utils/helpers.py", "lines": 650, "complexity": 18.0},
            ],
            circular_dependencies=[
                ["module_a", "module_b", "module_c", "module_a"],
                ["service_x", "service_y", "service_x"],
            ],
            coupling_metrics={"high_coupling": ["src/core.py", "src/main.py"]},
            solid_violations=[],
            score=72.5,
        ),
        performance=PerformanceAnalysis(
            gpu_utilization=65.0,
            bottlenecks=[
                {
                    "function": "data_preprocessing",
                    "time_ms": 2500,
                    "file": "src/data/preprocess.py",
                },
                {"function": "model_inference", "time_ms": 1800, "file": "src/models/inference.py"},
            ],
            flame_graph_path="/tmp/flame_graph.svg",
            memory_usage_peak_gb=12.5,
            memory_usage_avg_gb=9.8,
            score=68.0,
        ),
        coverage=CoverageAnalysis(
            line_coverage=78.5,
            branch_coverage=72.0,
            untested_critical_paths=["src/error_handler.py", "src/security/auth.py"],
            missing_property_tests=["src/transforms.py", "src/validators.py"],
            flaky_tests=["test_integration_flaky", "test_network_timeout"],
            slow_tests=[
                {"test": "test_full_pipeline", "duration_ms": 8000},
                {"test": "test_large_dataset", "duration_ms": 6500},
            ],
            score=75.0,
        ),
        code_quality=CodeQualityAnalysis(
            average_complexity=6.8,
            high_complexity_functions=[
                {"name": "process_batch", "complexity": 18, "file": "src/batch.py"},
                {"name": "validate_input", "complexity": 15, "file": "src/validation.py"},
            ],
            duplication_percentage=12.5,
            documentation_coverage=65.0,
            pylint_score=8.2,
            score=70.0,
            fix_suggestions=[
                {"type": "unused_import", "file": "src/main.py", "line": 5},
                {"type": "naming_convention", "file": "src/utils.py", "line": 42},
            ],
        ),
        dependencies=DependencyAnalysis(
            total_dependencies=68,
            vulnerabilities=[
                {"package": "requests", "cve": "CVE-2023-1234", "severity": "high"},
                {"package": "pillow", "cve": "CVE-2023-5678", "severity": "medium"},
            ],
            outdated_packages=["numpy==1.20.0", "pandas==1.3.0", "torch==1.12.0"],
            license_issues=["gpl-licensed-package"],
            unused_dependencies=["unused-lib", "deprecated-tool"],
            redundant_dependencies=["duplicate-util"],
            security_report={"critical": 0, "high": 1, "medium": 1, "low": 3},
            score=72.0,
        ),
        deployment=DeploymentAnalysis(
            dockerfile_score=85.0,
            k8s_readiness=75.0,
            ci_cd_completeness=90.0,
            monitoring_score=70.0,
            score=80.0,
        ),
        security=SecurityAnalysis(
            vulnerabilities=[
                {
                    "title": "SQL injection risk",
                    "severity": "critical",
                    "file": "src/db/queries.py",
                },
                {"title": "XSS vulnerability", "severity": "high", "file": "src/web/views.py"},
            ],
            hipaa_compliance_score=68.0,
            hardcoded_secrets=["src/config.py:API_KEY", "src/settings.py:SECRET_TOKEN"],
            injection_risks=[
                {"type": "sql", "file": "src/db/queries.py", "line": 42},
                {"type": "command", "file": "src/utils/shell.py", "line": 18},
            ],
            tls_issues=[
                {"issue": "weak_cipher", "location": "src/network/ssl.py"},
                {"issue": "missing_cert_validation", "location": "src/api/client.py"},
            ],
            score=62.0,
        ),
        scalability=ScalabilityAnalysis(
            ddp_correctness=True,
            scaling_efficiency="sub-linear",
            memory_bottlenecks=["large_tensor_allocation", "inefficient_caching"],
            communication_overhead_ms=180.0,
            score=74.0,
            recommendations={
                "gpu_count": 4,
                "expected_speedup": 3.2,
                "optimization_strategies": ["gradient_accumulation", "mixed_precision"],
            },
        ),
        overall_score=71.5,
        critical_issues=[],
    )


class TestReportIntegration:
    """Integration tests for report generation pipeline."""

    def test_json_to_markdown_pipeline(self, generator, complete_result):
        """Test complete JSON to Markdown conversion."""
        # Generate Markdown report
        markdown = generator.generate_markdown(complete_result)

        # Verify report structure
        assert isinstance(markdown, str)
        assert len(markdown) > 1000  # Should be substantial

        # Verify all major sections present
        required_sections = [
            "# HistoCore Project Optimization Analysis Report",
            "## Executive Summary",
            "## Overall Metrics",
            "## Architecture Analysis",
            "## Performance Analysis",
            "## Test Coverage Analysis",
            "## Code Quality Analysis",
            "## Dependencies Analysis",
            "## Deployment Analysis",
            "## Security Analysis",
            "## Scalability Analysis",
            "## Prioritized Task List",
            "## Recommendations",
        ]

        for section in required_sections:
            assert section in markdown, f"Missing section: {section}"

        # Verify data integrity
        assert "71.5/100" in markdown  # Overall score
        assert "abc123def456" in markdown  # Git commit
        assert "/path/to/project" in markdown  # Project path

    def test_markdown_to_html_conversion(self, generator, complete_result):
        """Test Markdown to HTML conversion."""
        # Generate HTML (without pandoc, uses fallback)
        html = generator.generate_html(complete_result)

        # Verify HTML structure
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html or "<html>" in html
        assert "</html>" in html

        # Verify content preservation
        assert "HistoCore" in html
        assert "71.5" in html or "71.5/100" in html

    def test_html_file_generation(self, generator, complete_result):
        """Test HTML file generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"

            # Generate HTML file
            result_path = generator.generate_html(complete_result, str(output_path))

            # Verify file created
            assert Path(result_path).exists()

            # Verify file content
            content = Path(result_path).read_text(encoding="utf-8")
            assert len(content) > 0
            assert "HistoCore" in content

    def test_round_trip_json_serialization(self, complete_result):
        """Test round-trip JSON serialization."""
        # Serialize to JSON
        json_str = complete_result.to_json()

        # Verify valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

        # Deserialize back
        restored = AnalysisResult.from_json(json_str)

        # Verify data integrity
        assert restored.timestamp == complete_result.timestamp
        assert restored.project_path == complete_result.project_path
        assert restored.git_commit == complete_result.git_commit
        assert restored.overall_score == complete_result.overall_score

        # Verify dimension scores
        assert restored.architecture.score == complete_result.architecture.score
        assert restored.performance.score == complete_result.performance.score
        assert restored.coverage.score == complete_result.coverage.score
        assert restored.code_quality.score == complete_result.code_quality.score
        assert restored.dependencies.score == complete_result.dependencies.score
        assert restored.deployment.score == complete_result.deployment.score
        assert restored.security.score == complete_result.security.score
        assert restored.scalability.score == complete_result.scalability.score

    def test_json_file_persistence(self, complete_result):
        """Test JSON file save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "analysis_result.json"

            # Save to file
            json_path.write_text(complete_result.to_json(), encoding="utf-8")

            # Load from file
            loaded_json = json_path.read_text(encoding="utf-8")
            restored = AnalysisResult.from_json(loaded_json)

            # Verify restoration
            assert restored.overall_score == complete_result.overall_score
            assert restored.git_commit == complete_result.git_commit

    def test_markdown_table_formatting(self, generator, complete_result):
        """Test Markdown table generation."""
        markdown = generator.generate_markdown(complete_result)

        # Verify table structure
        assert "| Dimension | Score | Status | Key Metric |" in markdown
        assert "|-----------|-------|--------|------------|" in markdown

        # Verify all dimensions in table
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
        for dim in dimensions:
            assert dim in markdown

    def test_empty_sections_handling(self, generator):
        """Test handling of empty analysis sections."""
        minimal_result = AnalysisResult(
            timestamp="2024-01-01T00:00:00Z",
            project_path="/minimal/project",
            git_commit="minimal123",
            architecture=ArchitectureAnalysis(),
            performance=PerformanceAnalysis(),
            coverage=CoverageAnalysis(),
            code_quality=CodeQualityAnalysis(),
            dependencies=DependencyAnalysis(),
            deployment=DeploymentAnalysis(),
            security=SecurityAnalysis(),
            scalability=ScalabilityAnalysis(),
            overall_score=0.0,
            critical_issues=[],
        )

        # Should not crash with empty data
        markdown = generator.generate_markdown(minimal_result)

        assert isinstance(markdown, str)
        assert len(markdown) > 0
        assert "✅" in markdown  # Should have checkmarks for empty sections

    def test_large_dataset_handling(self, generator):
        """Test handling of large analysis results."""
        # Create result with many items
        large_result = AnalysisResult(
            timestamp="2024-01-01T00:00:00Z",
            project_path="/large/project",
            git_commit="large123",
            architecture=ArchitectureAnalysis(
                total_files=1000,
                large_files=[
                    {"path": f"src/file_{i}.py", "lines": 600 + i, "complexity": 10.0 + i}
                    for i in range(50)
                ],
                circular_dependencies=[[f"mod_{i}", f"mod_{i+1}", f"mod_{i}"] for i in range(20)],
                score=60.0,
            ),
            performance=PerformanceAnalysis(
                bottlenecks=[
                    {"function": f"func_{i}", "time_ms": 100 + i, "file": f"src/perf_{i}.py"}
                    for i in range(30)
                ],
                score=55.0,
            ),
            coverage=CoverageAnalysis(
                untested_critical_paths=[f"src/critical_{i}.py" for i in range(40)],
                missing_property_tests=[f"src/test_{i}.py" for i in range(35)],
                flaky_tests=[f"test_flaky_{i}" for i in range(25)],
                score=65.0,
            ),
            code_quality=CodeQualityAnalysis(
                high_complexity_functions=[
                    {"name": f"complex_{i}", "complexity": 15 + i, "file": f"src/complex_{i}.py"}
                    for i in range(45)
                ],
                score=62.0,
            ),
            dependencies=DependencyAnalysis(
                vulnerabilities=[
                    {"package": f"pkg_{i}", "cve": f"CVE-2023-{1000+i}", "severity": "high"}
                    for i in range(15)
                ],
                outdated_packages=[f"package_{i}==1.0.0" for i in range(30)],
                score=70.0,
            ),
            deployment=DeploymentAnalysis(score=75.0),
            security=SecurityAnalysis(
                vulnerabilities=[
                    {"title": f"vuln_{i}", "severity": "high", "file": f"src/sec_{i}.py"}
                    for i in range(20)
                ],
                hardcoded_secrets=[f"src/secret_{i}.py:KEY" for i in range(10)],
                score=58.0,
            ),
            scalability=ScalabilityAnalysis(
                memory_bottlenecks=[f"bottleneck_{i}" for i in range(12)], score=68.0
            ),
            overall_score=64.5,
            critical_issues=[],
        )

        # Should handle large dataset without issues
        markdown = generator.generate_markdown(large_result)

        assert isinstance(markdown, str)
        assert len(markdown) > 5000  # Should be very large

        # Verify truncation (should show top 10 for most lists)
        assert "src/file_0.py" in markdown
        assert "src/file_9.py" in markdown

    def test_special_characters_in_paths(self, generator):
        """Test handling of special characters in file paths."""
        result = AnalysisResult(
            timestamp="2024-01-01T00:00:00Z",
            project_path="/path/with spaces/and-dashes",
            git_commit="special123",
            architecture=ArchitectureAnalysis(
                large_files=[
                    {"path": "src/file with spaces.py", "lines": 600, "complexity": 10.0},
                    {"path": "src/file-with-dashes.py", "lines": 550, "complexity": 8.0},
                    {"path": "src/file_with_underscores.py", "lines": 520, "complexity": 7.0},
                ],
                score=70.0,
            ),
            performance=PerformanceAnalysis(score=65.0),
            coverage=CoverageAnalysis(score=72.0),
            code_quality=CodeQualityAnalysis(score=68.0),
            dependencies=DependencyAnalysis(score=75.0),
            deployment=DeploymentAnalysis(score=80.0),
            security=SecurityAnalysis(score=70.0),
            scalability=ScalabilityAnalysis(score=73.0),
            overall_score=71.5,
            critical_issues=[],
        )

        markdown = generator.generate_markdown(result)

        # Should preserve special characters
        assert "file with spaces.py" in markdown
        assert "file-with-dashes.py" in markdown
        assert "file_with_underscores.py" in markdown

    def test_unicode_content_handling(self, generator):
        """Test handling of Unicode characters."""
        result = AnalysisResult(
            timestamp="2024-01-01T00:00:00Z",
            project_path="/path/to/项目",  # Chinese characters
            git_commit="unicode123",
            architecture=ArchitectureAnalysis(
                large_files=[{"path": "src/файл.py", "lines": 600, "complexity": 10.0}],  # Cyrillic
                score=70.0,
            ),
            performance=PerformanceAnalysis(score=65.0),
            coverage=CoverageAnalysis(score=72.0),
            code_quality=CodeQualityAnalysis(score=68.0),
            dependencies=DependencyAnalysis(score=75.0),
            deployment=DeploymentAnalysis(score=80.0),
            security=SecurityAnalysis(score=70.0),
            scalability=ScalabilityAnalysis(score=73.0),
            overall_score=71.5,
            critical_issues=[],
        )

        markdown = generator.generate_markdown(result)

        # Should handle Unicode
        assert isinstance(markdown, str)
        assert len(markdown) > 0

    def test_report_consistency_across_formats(self, generator, complete_result):
        """Test that key information is preserved across formats."""
        # Generate both formats
        markdown = generator.generate_markdown(complete_result)
        html = generator.generate_html(complete_result)

        # Key information should be in both
        key_info = [
            "71.5",  # Overall score
            "abc123def456",  # Git commit
            "Architecture",
            "Performance",
            "Security",
        ]

        for info in key_info:
            assert info in markdown, f"Missing {info} in Markdown"
            assert info in html, f"Missing {info} in HTML"

    def test_pdf_generation_fallback(self, generator, complete_result):
        """Test PDF generation with fallback to text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "report.pdf"

            # Try to generate PDF (will likely fall back to text)
            result_path = generator.generate_pdf(complete_result, str(pdf_path))

            # Should create some output file
            assert Path(result_path).exists()

            # Verify file has content
            content = Path(result_path).read_text(encoding="utf-8")
            assert len(content) > 0

    def test_concurrent_report_generation(self, generator, complete_result):
        """Test thread-safety of report generation."""
        import concurrent.futures

        def generate_report():
            return generator.generate_markdown(complete_result)

        # Generate reports concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(generate_report) for _ in range(10)]
            results = [f.result() for f in futures]

        # All results should be identical
        assert len(set(results)) == 1  # All the same
        assert len(results[0]) > 1000

    def test_report_generation_performance(self, generator, complete_result):
        """Test report generation performance."""
        import time

        start = time.time()
        markdown = generator.generate_markdown(complete_result)
        duration = time.time() - start

        # Should complete quickly (< 1 second for typical report)
        assert duration < 1.0
        assert len(markdown) > 0

    def test_error_recovery_invalid_data(self, generator):
        """Test error handling with invalid data."""
        # Create result with some invalid/missing data
        result = AnalysisResult(
            timestamp="invalid-timestamp",  # Invalid format
            project_path="",  # Empty path
            git_commit="",  # Empty commit
            architecture=ArchitectureAnalysis(),
            performance=PerformanceAnalysis(),
            coverage=CoverageAnalysis(),
            code_quality=CodeQualityAnalysis(),
            dependencies=DependencyAnalysis(),
            deployment=DeploymentAnalysis(),
            security=SecurityAnalysis(),
            scalability=ScalabilityAnalysis(),
            overall_score=-1.0,  # Invalid score
            critical_issues=[],
        )

        # Should not crash, handle gracefully
        markdown = generator.generate_markdown(result)

        assert isinstance(markdown, str)
        assert len(markdown) > 0
