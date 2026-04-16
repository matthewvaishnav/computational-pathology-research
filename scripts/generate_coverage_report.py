#!/usr/bin/env python3
"""
Generate comprehensive test coverage report for dataset testing suite.
Includes detailed coverage metrics, HTML reports, and gap analysis.
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET


class CoverageReporter:
    """Generate comprehensive coverage reports for dataset testing."""

    def __init__(self, output_dir: str = "coverage_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_coverage_tests(self, test_paths: List[str], source_paths: List[str]) -> bool:
        """Run tests with coverage collection."""
        print("Running tests with coverage...")

        cmd = [
            "python",
            "-m",
            "pytest",
            "--cov=" + ":".join(source_paths),
            "--cov-report=html:" + str(self.output_dir / "html"),
            "--cov-report=xml:" + str(self.output_dir / "coverage.xml"),
            "--cov-report=json:" + str(self.output_dir / "coverage.json"),
            "--cov-report=term-missing",
            "-v",
        ] + test_paths

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Tests completed: {result.returncode}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Test execution failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return False

    def parse_coverage_json(self) -> Dict:
        """Parse coverage JSON report."""
        json_path = self.output_dir / "coverage.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Coverage JSON not found: {json_path}")

        with open(json_path) as f:
            return json.load(f)

    def generate_summary_report(self, coverage_data: Dict) -> str:
        """Generate summary coverage report."""
        summary = coverage_data.get("totals", {})

        report = f"""# Dataset Testing Coverage Report

## Overall Coverage
- **Total Coverage**: {summary.get('percent_covered', 0):.1f}%
- **Lines Covered**: {summary.get('covered_lines', 0)} / {summary.get('num_statements', 0)}
- **Missing Lines**: {summary.get('missing_lines', 0)}
- **Excluded Lines**: {summary.get('excluded_lines', 0)}

## File Coverage Details

| File | Coverage | Lines | Missing | Excluded |
|------|----------|-------|---------|----------|
"""

        files = coverage_data.get("files", {})
        for filepath, file_data in sorted(files.items()):
            summary_data = file_data.get("summary", {})
            coverage_pct = summary_data.get("percent_covered", 0)
            covered = summary_data.get("covered_lines", 0)
            total = summary_data.get("num_statements", 0)
            missing = summary_data.get("missing_lines", 0)
            excluded = summary_data.get("excluded_lines", 0)

            # Shorten filepath for display
            short_path = filepath.replace("src/", "").replace("tests/", "")

            report += f"| {short_path} | {coverage_pct:.1f}% | {covered}/{total} | {missing} | {excluded} |\n"

        return report

    def identify_coverage_gaps(self, coverage_data: Dict) -> List[Dict]:
        """Identify files with low coverage and suggest improvements."""
        gaps = []
        files = coverage_data.get("files", {})

        for filepath, file_data in files.items():
            summary_data = file_data.get("summary", {})
            coverage_pct = summary_data.get("percent_covered", 0)
            missing_lines = file_data.get("missing_lines", [])

            if coverage_pct < 80:  # Low coverage threshold
                gaps.append(
                    {
                        "file": filepath,
                        "coverage": coverage_pct,
                        "missing_lines": missing_lines,
                        "suggestions": self._generate_suggestions(filepath, missing_lines),
                    }
                )

        return gaps

    def _generate_suggestions(self, filepath: str, missing_lines: List[int]) -> List[str]:
        """Generate test suggestions for missing coverage."""
        suggestions = []

        # Basic suggestions based on file type
        if "preprocessing" in filepath:
            suggestions.extend(
                [
                    "Add edge case tests for empty inputs",
                    "Test error handling for invalid parameters",
                    "Add property-based tests for transform consistency",
                ]
            )
        elif "openslide" in filepath:
            suggestions.extend(
                [
                    "Test error handling for corrupted files",
                    "Add tests for different WSI formats",
                    "Test memory constraints with large files",
                ]
            )
        elif "dataset" in filepath:
            suggestions.extend(
                [
                    "Add tests for dataset corruption scenarios",
                    "Test batch loading edge cases",
                    "Add property tests for data integrity",
                ]
            )

        # Line-specific suggestions
        if len(missing_lines) > 10:
            suggestions.append(
                f"High missing line count ({len(missing_lines)}). Consider refactoring complex functions."
            )

        return suggestions

    def generate_gap_analysis(self, gaps: List[Dict]) -> str:
        """Generate coverage gap analysis report."""
        if not gaps:
            return "\n## Coverage Gap Analysis\n\n✅ **Excellent coverage!** All files above 80% threshold.\n"

        report = "\n## Coverage Gap Analysis\n\n"

        for gap in sorted(gaps, key=lambda x: x["coverage"]):
            filepath = gap["file"].replace("src/", "")
            coverage = gap["coverage"]
            missing_count = len(gap["missing_lines"])

            report += f"### {filepath} ({coverage:.1f}% coverage)\n\n"
            report += f"- **Missing lines**: {missing_count}\n"
            report += (
                f"- **Lines**: {gap['missing_lines'][:10]}{'...' if missing_count > 10 else ''}\n\n"
            )

            report += "**Suggested improvements**:\n"
            for suggestion in gap["suggestions"]:
                report += f"- {suggestion}\n"
            report += "\n"

        return report

    def count_test_files(self, test_dir: str) -> Dict[str, int]:
        """Count test files and tests by category."""
        test_path = Path(test_dir)
        counts = {
            "total_files": 0,
            "unit_tests": 0,
            "integration_tests": 0,
            "performance_tests": 0,
            "property_tests": 0,
        }

        for test_file in test_path.rglob("test_*.py"):
            counts["total_files"] += 1

            # Count by directory
            if "unit" in str(test_file):
                counts["unit_tests"] += 1
            elif "integration" in str(test_file):
                counts["integration_tests"] += 1
            elif "performance" in str(test_file):
                counts["performance_tests"] += 1
            elif "property" in str(test_file):
                counts["property_tests"] += 1

        return counts

    def generate_full_report(self, test_dir: str = "tests/dataset_testing") -> str:
        """Generate complete coverage report."""
        print("Parsing coverage data...")
        coverage_data = self.parse_coverage_json()

        print("Generating summary...")
        summary_report = self.generate_summary_report(coverage_data)

        print("Analyzing coverage gaps...")
        gaps = self.identify_coverage_gaps(coverage_data)
        gap_analysis = self.generate_gap_analysis(gaps)

        print("Counting test files...")
        test_counts = self.count_test_files(test_dir)

        # Test statistics
        test_stats = f"""
## Test Suite Statistics

- **Total test files**: {test_counts['total_files']}
- **Unit test files**: {test_counts['unit_tests']}
- **Integration test files**: {test_counts['integration_tests']}
- **Performance test files**: {test_counts['performance_tests']}
- **Property test files**: {test_counts['property_tests']}

## Coverage Reports

- **HTML Report**: `{self.output_dir}/html/index.html`
- **XML Report**: `{self.output_dir}/coverage.xml`
- **JSON Report**: `{self.output_dir}/coverage.json`
"""

        full_report = summary_report + test_stats + gap_analysis

        # Save report
        report_path = self.output_dir / "coverage_report.md"
        with open(report_path, "w") as f:
            f.write(full_report)

        print(f"Coverage report saved: {report_path}")
        return full_report


def main():
    parser = argparse.ArgumentParser(description="Generate dataset testing coverage report")
    parser.add_argument("--test-dir", default="tests/dataset_testing", help="Test directory path")
    parser.add_argument(
        "--source-paths",
        nargs="+",
        default=["src/data", "src/models", "src/utils"],
        help="Source code paths to analyze",
    )
    parser.add_argument(
        "--output-dir", default="coverage_reports", help="Output directory for reports"
    )
    parser.add_argument(
        "--skip-tests", action="store_true", help="Skip running tests, use existing coverage data"
    )

    args = parser.parse_args()

    reporter = CoverageReporter(args.output_dir)

    if not args.skip_tests:
        # Run tests with coverage
        test_paths = [args.test_dir]
        success = reporter.run_coverage_tests(test_paths, args.source_paths)
        if not success:
            print("Test execution failed. Exiting.")
            sys.exit(1)

    try:
        # Generate report
        report = reporter.generate_full_report(args.test_dir)
        print("\n" + "=" * 60)
        print("COVERAGE REPORT GENERATED")
        print("=" * 60)
        print(report[:500] + "..." if len(report) > 500 else report)

    except Exception as e:
        print(f"Report generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
