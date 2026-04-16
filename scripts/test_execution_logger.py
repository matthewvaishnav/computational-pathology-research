#!/usr/bin/env python3
"""
Test execution logging and failure reporting system.
Captures detailed test execution info, failure analysis, and reproduction steps.
"""

import argparse
import subprocess
import sys
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import platform
import psutil


class TestExecutionLogger:
    """Log test execution with detailed failure reporting."""

    def __init__(self, output_dir: str = "test_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.start_time = None
        self.end_time = None

    def capture_environment_info(self) -> Dict:
        """Capture system and environment information."""
        return {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
            },
            "system": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": psutil.disk_usage(".").free,
            },
            "environment": {
                "pwd": os.getcwd(),
                "path": os.environ.get("PATH", ""),
                "pythonpath": os.environ.get("PYTHONPATH", ""),
            },
        }

    def run_tests_with_logging(
        self, test_paths: List[str], pytest_args: List[str] = None
    ) -> Tuple[bool, Dict]:
        """Run tests with comprehensive logging."""
        if pytest_args is None:
            pytest_args = ["-v", "--tb=short"]

        # Capture environment
        env_info = self.capture_environment_info()

        # Build command
        cmd = ["python", "-m", "pytest"] + pytest_args + test_paths

        print(f"Running: {' '.join(cmd)}")
        self.start_time = time.time()

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600  # 1 hour timeout
            )
            self.end_time = time.time()

            execution_info = {
                "command": cmd,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": self.end_time - self.start_time,
                "environment": env_info,
                "success": result.returncode == 0,
            }

            return result.returncode == 0, execution_info

        except subprocess.TimeoutExpired as e:
            self.end_time = time.time()
            execution_info = {
                "command": cmd,
                "return_code": -1,
                "stdout": e.stdout or "",
                "stderr": e.stderr or "",
                "execution_time": self.end_time - self.start_time,
                "environment": env_info,
                "success": False,
                "timeout": True,
                "timeout_duration": 3600,
            }
            return False, execution_info

    def parse_pytest_output(self, stdout: str, stderr: str) -> Dict:
        """Parse pytest output for test results and failures."""
        lines = stdout.split("\n")

        results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "failures": [],
            "test_summary": {},
        }

        # Parse test results
        in_failure = False
        current_failure = {}

        for line in lines:
            # Test count summary
            if "failed" in line and "passed" in line:
                # Extract numbers from summary line
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "failed,":
                        results["failed"] = int(parts[i - 1])
                    elif part == "passed":
                        results["passed"] = int(parts[i - 1])
                    elif part == "skipped":
                        results["skipped"] = int(parts[i - 1])

            # Failure details
            if line.startswith("FAILED "):
                test_name = line.split()[1]
                current_failure = {"test_name": test_name, "failure_line": line, "traceback": []}
                in_failure = True
            elif line.startswith("=") and "FAILURES" in line:
                in_failure = True
            elif line.startswith("=") and in_failure:
                if current_failure:
                    results["failures"].append(current_failure)
                    current_failure = {}
                in_failure = False
            elif in_failure and current_failure:
                current_failure["traceback"].append(line)

        results["total_tests"] = results["passed"] + results["failed"] + results["skipped"]
        return results

    def analyze_failures(self, failures: List[Dict]) -> List[Dict]:
        """Analyze test failures and generate reproduction steps."""
        analyzed_failures = []

        for failure in failures:
            test_name = failure.get("test_name", "")
            traceback = failure.get("traceback", [])

            # Extract error type and message
            error_type = "Unknown"
            error_message = ""
            file_location = ""

            for line in traceback:
                if "AssertionError" in line:
                    error_type = "AssertionError"
                    error_message = line.strip()
                elif "FileNotFoundError" in line:
                    error_type = "FileNotFoundError"
                    error_message = line.strip()
                elif "ImportError" in line:
                    error_type = "ImportError"
                    error_message = line.strip()
                elif ".py:" in line and "in " in line:
                    file_location = line.strip()

            # Generate reproduction steps
            repro_steps = self._generate_reproduction_steps(
                test_name, error_type, error_message, file_location
            )

            analyzed_failures.append(
                {
                    "test_name": test_name,
                    "error_type": error_type,
                    "error_message": error_message,
                    "file_location": file_location,
                    "full_traceback": traceback,
                    "reproduction_steps": repro_steps,
                    "suggested_fixes": self._suggest_fixes(error_type, error_message),
                }
            )

        return analyzed_failures

    def _generate_reproduction_steps(
        self, test_name: str, error_type: str, error_message: str, file_location: str
    ) -> List[str]:
        """Generate steps to reproduce the failure."""
        steps = [
            "# Reproduction Steps",
            "",
            "1. **Environment Setup**:",
            "   ```bash",
            "   cd /path/to/project",
            "   source venv/bin/activate  # or venv\\Scripts\\activate on Windows",
            "   pip install -r requirements.txt",
            "   ```",
            "",
            "2. **Run Specific Test**:",
            f"   ```bash",
            f"   python -m pytest {test_name} -v -s",
            f"   ```",
            "",
            "3. **Expected Error**:",
            f"   - **Type**: {error_type}",
            f"   - **Message**: {error_message}",
            f"   - **Location**: {file_location}",
            "",
        ]

        # Add specific debugging steps based on error type
        if error_type == "FileNotFoundError":
            steps.extend(
                [
                    "4. **Debug File Issues**:",
                    "   ```bash",
                    "   ls -la data/  # Check if data files exist",
                    "   find . -name '*.h5' -o -name '*.hdf5'  # Find HDF5 files",
                    "   ```",
                ]
            )
        elif error_type == "ImportError":
            steps.extend(
                [
                    "4. **Debug Import Issues**:",
                    "   ```bash",
                    "   python -c 'import sys; print(sys.path)'",
                    "   pip list | grep -E '(torch|numpy|h5py)'",
                    "   ```",
                ]
            )
        elif error_type == "AssertionError":
            steps.extend(
                [
                    "4. **Debug Assertion**:",
                    "   - Add print statements before assertion",
                    "   - Check input data shapes/types",
                    "   - Verify expected vs actual values",
                ]
            )

        return steps

    def _suggest_fixes(self, error_type: str, error_message: str) -> List[str]:
        """Suggest potential fixes based on error type."""
        fixes = []

        if error_type == "FileNotFoundError":
            fixes.extend(
                [
                    "Create missing test data files",
                    "Update file paths in test configuration",
                    "Add file existence checks in test setup",
                ]
            )
        elif error_type == "ImportError":
            fixes.extend(
                [
                    "Install missing dependencies: pip install -r requirements.txt",
                    "Check Python path configuration",
                    "Verify package installation: pip list",
                ]
            )
        elif error_type == "AssertionError":
            if "shape" in error_message.lower():
                fixes.append("Fix tensor/array shape mismatches")
            if "type" in error_message.lower():
                fixes.append("Fix data type conversions")
            fixes.extend(
                ["Update expected values in assertions", "Add input validation in test setup"]
            )
        elif "timeout" in error_message.lower():
            fixes.extend(
                [
                    "Increase test timeout values",
                    "Optimize test data size",
                    "Use faster test fixtures",
                ]
            )

        return fixes

    def generate_execution_report(
        self, execution_info: Dict, test_results: Dict, analyzed_failures: List[Dict]
    ) -> str:
        """Generate comprehensive test execution report."""
        env = execution_info["environment"]

        # Header
        report = f"""# Test Execution Report

**Generated**: {env["timestamp"]}
**Duration**: {execution_info["execution_time"]:.2f} seconds
**Status**: {"✅ PASSED" if execution_info["success"] else "❌ FAILED"}

## Environment Information

### System
- **OS**: {env["platform"]["system"]} {env["platform"]["release"]}
- **Architecture**: {env["platform"]["machine"]}
- **CPU Cores**: {env["system"]["cpu_count"]}
- **Memory**: {env["system"]["memory_total"] / (1024**3):.1f} GB total, {env["system"]["memory_available"] / (1024**3):.1f} GB available
- **Disk Space**: {env["system"]["disk_usage"] / (1024**3):.1f} GB free

### Python Environment
- **Version**: {env["python"]["version"]} ({env["python"]["implementation"]})
- **Executable**: {env["python"]["executable"]}
- **Working Directory**: {env["environment"]["pwd"]}

## Test Results Summary

- **Total Tests**: {test_results["total_tests"]}
- **Passed**: {test_results["passed"]} ✅
- **Failed**: {test_results["failed"]} ❌
- **Skipped**: {test_results["skipped"]} ⏭️
- **Success Rate**: {(test_results["passed"] / max(test_results["total_tests"], 1)) * 100:.1f}%

## Command Executed

```bash
{' '.join(execution_info["command"])}
```

"""

        # Add failure details if any
        if analyzed_failures:
            report += "## Failure Analysis\n\n"

            for i, failure in enumerate(analyzed_failures, 1):
                report += f"### Failure {i}: {failure['test_name']}\n\n"
                report += f"**Error Type**: {failure['error_type']}\n"
                report += f"**Error Message**: {failure['error_message']}\n"
                report += f"**Location**: {failure['file_location']}\n\n"

                # Reproduction steps
                report += "#### Reproduction Steps\n\n"
                for step in failure["reproduction_steps"]:
                    report += step + "\n"
                report += "\n"

                # Suggested fixes
                if failure["suggested_fixes"]:
                    report += "#### Suggested Fixes\n\n"
                    for fix in failure["suggested_fixes"]:
                        report += f"- {fix}\n"
                    report += "\n"

        # Add stdout/stderr if there were issues
        if not execution_info["success"]:
            report += "## Raw Output\n\n"
            report += "### STDOUT\n```\n"
            report += execution_info["stdout"][-2000:]  # Last 2000 chars
            report += "\n```\n\n"

            if execution_info["stderr"]:
                report += "### STDERR\n```\n"
                report += execution_info["stderr"][-1000:]  # Last 1000 chars
                report += "\n```\n"

        return report

    def save_execution_log(
        self, execution_info: Dict, test_results: Dict, analyzed_failures: List[Dict]
    ) -> str:
        """Save complete execution log and report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw execution data
        log_file = self.output_dir / f"execution_{timestamp}.json"
        with open(log_file, "w") as f:
            json.dump(
                {
                    "execution_info": execution_info,
                    "test_results": test_results,
                    "analyzed_failures": analyzed_failures,
                },
                f,
                indent=2,
            )

        # Generate and save report
        report = self.generate_execution_report(execution_info, test_results, analyzed_failures)
        report_file = self.output_dir / f"report_{timestamp}.md"
        with open(report_file, "w") as f:
            f.write(report)

        print(f"Execution log saved: {log_file}")
        print(f"Execution report saved: {report_file}")

        return str(report_file)


def main():
    parser = argparse.ArgumentParser(description="Run tests with detailed logging")
    parser.add_argument("test_paths", nargs="+", help="Test paths to run")
    parser.add_argument("--output-dir", default="test_logs", help="Output directory for logs")
    parser.add_argument(
        "--pytest-args", nargs="*", default=["-v", "--tb=short"], help="Additional pytest arguments"
    )

    args = parser.parse_args()

    logger = TestExecutionLogger(args.output_dir)

    # Run tests
    success, execution_info = logger.run_tests_with_logging(args.test_paths, args.pytest_args)

    # Parse results
    test_results = logger.parse_pytest_output(execution_info["stdout"], execution_info["stderr"])

    # Analyze failures
    analyzed_failures = logger.analyze_failures(test_results.get("failures", []))

    # Save logs and report
    report_file = logger.save_execution_log(execution_info, test_results, analyzed_failures)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Status: {'PASSED' if success else 'FAILED'}")
    print(f"Tests: {test_results['passed']}/{test_results['total_tests']} passed")
    print(f"Duration: {execution_info['execution_time']:.2f}s")
    print(f"Report: {report_file}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
