#!/usr/bin/env python3
"""
Comprehensive dataset test runner.
Combines coverage reporting and execution logging.
"""

import argparse
import sys
from pathlib import Path
import subprocess
import json
from datetime import datetime

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from generate_coverage_report import CoverageReporter
from test_execution_logger import TestExecutionLogger


class DatasetTestRunner:
    """Run dataset tests with comprehensive reporting."""

    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.coverage_dir = self.output_dir / "coverage"
        self.logs_dir = self.output_dir / "logs"

        self.coverage_reporter = CoverageReporter(str(self.coverage_dir))
        self.execution_logger = TestExecutionLogger(str(self.logs_dir))

    def run_test_suite(
        self, test_categories: list = None, source_paths: list = None, pytest_args: list = None
    ) -> dict:
        """Run complete test suite with reporting."""

        if test_categories is None:
            test_categories = ["unit", "integration", "performance"]

        if source_paths is None:
            source_paths = ["src/data", "src/models", "src/utils"]

        if pytest_args is None:
            pytest_args = ["-v", "--tb=short"]
        elif len(pytest_args) == 0:
            pytest_args = ["-v", "--tb=short"]

        results = {
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "overall_success": True,
            "total_tests": 0,
            "total_passed": 0,
            "total_failed": 0,
        }

        print("=" * 60)
        print("DATASET TEST SUITE EXECUTION")
        print("=" * 60)

        # Run tests by category
        for category in test_categories:
            print(f"\n🧪 Running {category} tests...")

            test_path = f"tests/dataset_testing/{category}"
            if not Path(test_path).exists():
                print(f"⚠️  Skipping {category} - directory not found: {test_path}")
                continue

            # Run with logging only (no coverage for speed)
            category_args = pytest_args.copy()

            success, execution_info = self.execution_logger.run_tests_with_logging(
                [test_path], category_args
            )

            # Parse results
            test_results = self.execution_logger.parse_pytest_output(
                execution_info["stdout"], execution_info["stderr"]
            )

            results["categories"][category] = {
                "success": success,
                "execution_time": execution_info["execution_time"],
                "tests": test_results["total_tests"],
                "passed": test_results["passed"],
                "failed": test_results["failed"],
                "skipped": test_results["skipped"],
            }

            # Update totals
            results["total_tests"] += test_results["total_tests"]
            results["total_passed"] += test_results["passed"]
            results["total_failed"] += test_results["failed"]

            if not success:
                results["overall_success"] = False

            print(f"✅ {category}: {test_results['passed']}/{test_results['total_tests']} passed")

        return results

    def run_full_coverage_analysis(self, source_paths: list = None) -> str:
        """Run complete coverage analysis across all tests."""
        print("\n📊 Generating coverage analysis...")

        if source_paths is None:
            source_paths = ["src/data", "src/models", "src/utils"]

        # Run all tests with coverage
        test_paths = ["tests/dataset_testing"]
        success = self.coverage_reporter.run_coverage_tests(test_paths, source_paths)

        if success:
            report = self.coverage_reporter.generate_full_report()
            return report
        else:
            return "Coverage analysis failed"

    def generate_summary_report(self, results: dict, coverage_report: str = None) -> str:
        """Generate comprehensive summary report."""

        report = f"""# Dataset Testing Suite - Execution Summary

**Generated**: {results["timestamp"]}
**Overall Status**: {"✅ PASSED" if results["overall_success"] else "❌ FAILED"}

## Test Results by Category

| Category | Tests | Passed | Failed | Skipped | Success Rate | Duration |
|----------|-------|--------|--------|---------|--------------|----------|
"""

        for category, data in results["categories"].items():
            success_rate = (data["passed"] / max(data["tests"], 1)) * 100
            status = "✅" if data["success"] else "❌"

            report += f"| {category} | {data['tests']} | {data['passed']} | {data['failed']} | {data['skipped']} | {success_rate:.1f}% | {data['execution_time']:.1f}s |\n"

        # Overall summary
        overall_rate = (results["total_passed"] / max(results["total_tests"], 1)) * 100

        report += f"""
## Overall Summary

- **Total Tests**: {results["total_tests"]}
- **Passed**: {results["total_passed"]} ✅
- **Failed**: {results["total_failed"]} ❌
- **Success Rate**: {overall_rate:.1f}%

## Test Categories

"""

        for category in results["categories"]:
            report += f"- **{category.title()}**: `tests/dataset_testing/{category}/`\n"

        # Add coverage info if available
        if coverage_report:
            report += "\n## Coverage Analysis\n\n"
            # Extract key coverage metrics
            lines = coverage_report.split("\n")
            for line in lines:
                if "Total Coverage" in line:
                    report += f"- {line.strip()}\n"
                elif "Lines Covered" in line:
                    report += f"- {line.strip()}\n"

        report += f"""
## Generated Reports

- **Coverage HTML**: `{self.coverage_dir}/html/index.html`
- **Test Logs**: `{self.logs_dir}/`
- **Full Results**: `{self.output_dir}/`

## Quick Commands

```bash
# Run specific category
python -m pytest tests/dataset_testing/unit/ -v

# Run with coverage
python -m pytest tests/dataset_testing/ --cov=src --cov-report=html

# View coverage report
open {self.coverage_dir}/html/index.html
```
"""

        return report

    def save_results(self, results: dict, summary_report: str) -> str:
        """Save all results and reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        results_file = self.output_dir / f"results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save summary report
        summary_file = self.output_dir / f"summary_{timestamp}.md"
        with open(summary_file, "w") as f:
            f.write(summary_report)

        # Create latest symlinks
        latest_results = self.output_dir / "latest_results.json"
        latest_summary = self.output_dir / "latest_summary.md"

        if latest_results.exists():
            latest_results.unlink()
        if latest_summary.exists():
            latest_summary.unlink()

        # Create symlinks (or copy on Windows)
        try:
            latest_results.symlink_to(results_file.name)
            latest_summary.symlink_to(summary_file.name)
        except OSError:
            # Fallback for Windows
            import shutil

            shutil.copy2(results_file, latest_results)
            shutil.copy2(summary_file, latest_summary)

        return str(summary_file)


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive dataset test suite")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["unit", "integration", "performance"],
        help="Test categories to run",
    )
    parser.add_argument(
        "--source-paths",
        nargs="+",
        default=["src/data", "src/models", "src/utils"],
        help="Source paths for coverage analysis",
    )
    parser.add_argument(
        "--output-dir", default="test_results", help="Output directory for all reports"
    )
    parser.add_argument("--skip-coverage", action="store_true", help="Skip full coverage analysis")
    parser.add_argument("--pytest-args", nargs="*", default=[], help="Additional pytest arguments")

    args = parser.parse_args()

    runner = DatasetTestRunner(args.output_dir)

    try:
        # Run test suite
        results = runner.run_test_suite(args.categories, args.source_paths, args.pytest_args)

        # Run coverage analysis
        coverage_report = None
        if not args.skip_coverage:
            coverage_report = runner.run_full_coverage_analysis(args.source_paths)

        # Generate summary
        summary_report = runner.generate_summary_report(results, coverage_report)

        # Save results
        summary_file = runner.save_results(results, summary_report)

        # Print final summary
        print("\n" + "=" * 60)
        print("DATASET TEST SUITE COMPLETE")
        print("=" * 60)
        print(f"Status: {'PASSED' if results['overall_success'] else 'FAILED'}")
        print(f"Tests: {results['total_passed']}/{results['total_tests']} passed")
        print(f"Summary: {summary_file}")
        print(f"Coverage: {runner.coverage_dir}/html/index.html")

        # Exit with appropriate code
        sys.exit(0 if results["overall_success"] else 1)

    except Exception as e:
        print(f"Test suite execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
