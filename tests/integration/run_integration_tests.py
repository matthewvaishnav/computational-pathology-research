#!/usr/bin/env python3
"""
Integration Test Runner

Comprehensive test runner that orchestrates all integration tests for the Medical AI platform.
Includes test discovery, execution, reporting, and CI/CD integration.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from test_api_endpoints import APIEndpointTests
from test_data_fixtures import TestDataFixtures
from test_full_workflow import IntegrationTestSuite
from test_performance_regression import PerformanceRegressionTests


class IntegrationTestRunner:
    """Comprehensive integration test runner."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize test runner with configuration."""

        self.config = config or self._load_default_config()
        self.results = {}
        self.start_time = None
        self.end_time = None

        # Test suites
        self.test_suites = {}

        print("🚀 Integration Test Runner initialized")
        print(f"⚙️ Configuration: {self.config}")

    def _load_default_config(self) -> Dict:
        """Load default test configuration."""

        return {
            "base_url": "http://localhost:8000",
            "timeout": 300,  # 5 minutes
            "retry_attempts": 3,
            "parallel_execution": False,
            "generate_fixtures": True,
            "cleanup_fixtures": True,
            "save_results": True,
            "results_dir": "tests/integration/results",
            "test_suites": {
                "workflow": True,
                "api_endpoints": True,
                "performance": True,
                "fixtures": True,
            },
            "performance_thresholds": {
                "api_response_time": 2.0,
                "inference_time": 30.0,
                "throughput_rps": 5.0,
            },
            "docker": {
                "auto_start": False,
                "compose_file": "docker-compose.yml",
                "services": ["api", "postgres", "redis"],
            },
        }

    def setup_test_environment(self) -> bool:
        """Setup test environment including Docker services if needed."""

        print("\n🔧 Setting up test environment...")

        # Create results directory
        results_dir = Path(self.config["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate test fixtures if enabled
        if self.config["generate_fixtures"]:
            print("🧪 Generating test fixtures...")
            try:
                fixtures = TestDataFixtures()
                fixtures.save_fixtures_to_files()
                fixtures.create_sample_images_and_dicoms(10)
                print("✅ Test fixtures generated successfully")
            except Exception as e:
                print(f"❌ Failed to generate fixtures: {e}")
                return False

        # Start Docker services if enabled
        if self.config["docker"]["auto_start"]:
            print("🐳 Starting Docker services...")
            try:
                compose_file = self.config["docker"]["compose_file"]
                services = " ".join(self.config["docker"]["services"])

                cmd = ["docker-compose", "-f", str(compose_file), "up", "-d"] + services.split()
                result = subprocess.run(cmd, shell=False, capture_output=True, text=True)

                if result.returncode == 0:
                    print("✅ Docker services started successfully")

                    # Wait for services to be ready
                    print("⏳ Waiting for services to be ready...")
                    time.sleep(30)  # Give services time to start

                else:
                    print(f"❌ Failed to start Docker services: {result.stderr}")
                    return False

            except Exception as e:
                print(f"❌ Docker startup error: {e}")
                return False

        # Verify API is accessible
        print("🔍 Verifying API accessibility...")
        try:
            import requests

            response = requests.get(f"{self.config['base_url']}/health", timeout=10)

            if response.status_code == 200:
                print("✅ API is accessible")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ API accessibility check failed: {e}")
            return False

    def initialize_test_suites(self):
        """Initialize all test suites."""

        base_url = self.config["base_url"]

        if self.config["test_suites"]["workflow"]:
            self.test_suites["workflow"] = IntegrationTestSuite(base_url=base_url)

        if self.config["test_suites"]["api_endpoints"]:
            self.test_suites["api_endpoints"] = APIEndpointTests(base_url=base_url)

        if self.config["test_suites"]["performance"]:
            self.test_suites["performance"] = PerformanceRegressionTests(base_url=base_url)
            # Update performance thresholds
            self.test_suites["performance"].thresholds.update(self.config["performance_thresholds"])

        print(f"🧪 Initialized {len(self.test_suites)} test suites")

    def run_workflow_tests(self) -> Dict[str, bool]:
        """Run full workflow integration tests."""

        print("\n" + "=" * 60)
        print("🔄 RUNNING WORKFLOW INTEGRATION TESTS")
        print("=" * 60)

        if "workflow" not in self.test_suites:
            return {"workflow": False}

        try:
            results = self.test_suites["workflow"].run_all_tests()
            return results

        except Exception as e:
            print(f"💥 Workflow tests error: {e}")
            return {"workflow_error": False}

    def run_api_endpoint_tests(self) -> Dict[str, bool]:
        """Run API endpoint tests."""

        print("\n" + "=" * 60)
        print("📡 RUNNING API ENDPOINT TESTS")
        print("=" * 60)

        if "api_endpoints" not in self.test_suites:
            return {"api_endpoints": False}

        try:
            results = self.test_suites["api_endpoints"].run_all_endpoint_tests()
            return results

        except Exception as e:
            print(f"💥 API endpoint tests error: {e}")
            return {"api_endpoints_error": False}

    def run_performance_tests(self) -> Dict[str, Dict]:
        """Run performance regression tests."""

        print("\n" + "=" * 60)
        print("⚡ RUNNING PERFORMANCE REGRESSION TESTS")
        print("=" * 60)

        if "performance" not in self.test_suites:
            return {"performance": {}}

        try:
            results = self.test_suites["performance"].run_all_performance_tests()
            return results

        except Exception as e:
            print(f"💥 Performance tests error: {e}")
            return {"performance_error": {}}

    def run_all_tests(self) -> Dict:
        """Run all integration tests."""

        self.start_time = datetime.now()

        print("🚀 STARTING COMPREHENSIVE INTEGRATION TESTS")
        print("=" * 80)
        print(f"⏰ Start time: {self.start_time}")
        print(f"🎯 Target URL: {self.config['base_url']}")
        print(f"⚙️ Test suites: {list(self.config['test_suites'].keys())}")

        all_results = {}

        # Run workflow tests
        if self.config["test_suites"]["workflow"]:
            workflow_results = self.run_workflow_tests()
            all_results["workflow"] = workflow_results

        # Run API endpoint tests
        if self.config["test_suites"]["api_endpoints"]:
            api_results = self.run_api_endpoint_tests()
            all_results["api_endpoints"] = api_results

        # Run performance tests
        if self.config["test_suites"]["performance"]:
            performance_results = self.run_performance_tests()
            all_results["performance"] = performance_results

        self.end_time = datetime.now()
        self.results = all_results

        return all_results

    def generate_test_report(self) -> Dict:
        """Generate comprehensive test report."""

        if not self.results:
            return {}

        # Calculate overall statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        for suite_name, suite_results in self.results.items():
            if isinstance(suite_results, dict):
                for test_name, test_result in suite_results.items():
                    total_tests += 1
                    if test_result is True:
                        passed_tests += 1
                    else:
                        failed_tests += 1

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # Test duration
        duration = (
            (self.end_time - self.start_time).total_seconds()
            if self.start_time and self.end_time
            else 0
        )

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "duration_seconds": duration,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
            },
            "configuration": self.config,
            "results": self.results,
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "base_url": self.config["base_url"],
            },
        }

        return report

    def save_test_results(self, report: Dict) -> Path:
        """Save test results to file."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = (
            Path(self.config["results_dir"]) / f"integration_test_results_{timestamp}.json"
        )

        with open(results_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return results_file

    def print_test_summary(self, report: Dict):
        """Print comprehensive test summary."""

        print("\n" + "=" * 80)
        print("🎯 INTEGRATION TEST SUMMARY")
        print("=" * 80)

        summary = report["summary"]

        print(f"⏰ Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")

        # Detailed results by suite
        print("\n📋 DETAILED RESULTS BY SUITE:")

        for suite_name, suite_results in report["results"].items():
            if isinstance(suite_results, dict):
                suite_passed = sum(1 for result in suite_results.values() if result is True)
                suite_total = len(suite_results)
                suite_rate = (suite_passed / suite_total * 100) if suite_total > 0 else 0

                print(f"\n🧪 {suite_name.upper()}:")
                print(f"   Tests: {suite_passed}/{suite_total} passed ({suite_rate:.1f}%)")

                for test_name, test_result in suite_results.items():
                    status = "✅" if test_result is True else "❌"
                    print(f"   {status} {test_name}")

        # Overall status
        print("\n" + "=" * 80)
        if summary["success_rate"] == 100:
            print("🎉 ALL TESTS PASSED! Platform is ready for deployment.")
        elif summary["success_rate"] >= 90:
            print("✅ TESTS MOSTLY PASSED. Minor issues detected.")
        elif summary["success_rate"] >= 70:
            print("⚠️ TESTS PARTIALLY PASSED. Significant issues detected.")
        else:
            print("❌ TESTS FAILED. Major issues detected. Platform not ready.")

        print("=" * 80)

    def cleanup_test_environment(self):
        """Clean up test environment."""

        print("\n🧹 Cleaning up test environment...")

        # Clean up fixtures if enabled
        if self.config["cleanup_fixtures"]:
            try:
                fixtures = TestDataFixtures()
                fixtures.cleanup_fixtures()
                print("✅ Test fixtures cleaned up")
            except Exception as e:
                print(f"⚠️ Fixture cleanup warning: {e}")

        # Stop Docker services if they were auto-started
        if self.config["docker"]["auto_start"]:
            try:
                compose_file = self.config["docker"]["compose_file"]
                cmd = f"docker-compose -f {compose_file} down"
                subprocess.run(cmd, shell=True, capture_output=True)
                print("✅ Docker services stopped")
            except Exception as e:
                print(f"⚠️ Docker cleanup warning: {e}")

    def run(self) -> int:
        """Run complete integration test suite and return exit code."""

        try:
            # Setup environment
            if not self.setup_test_environment():
                print("❌ Test environment setup failed")
                return 1

            # Initialize test suites
            self.initialize_test_suites()

            # Run all tests
            results = self.run_all_tests()

            # Generate report
            report = self.generate_test_report()

            # Save results
            if self.config["save_results"]:
                results_file = self.save_test_results(report)
                print(f"💾 Results saved to: {results_file}")

            # Print summary
            self.print_test_summary(report)

            # Determine exit code
            success_rate = report["summary"]["success_rate"]
            exit_code = 0 if success_rate >= 90 else 1

            return exit_code

        except KeyboardInterrupt:
            print("\n⚠️ Tests interrupted by user")
            return 130

        except Exception as e:
            print(f"\n💥 Test runner error: {e}")
            return 1

        finally:
            # Always cleanup
            self.cleanup_test_environment()


def main():
    """Main entry point for integration test runner."""

    parser = argparse.ArgumentParser(description="Medical AI Integration Test Runner")

    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for API server (default: http://localhost:8000)",
    )

    parser.add_argument("--config", type=str, help="Path to configuration file (JSON)")

    parser.add_argument(
        "--docker", action="store_true", help="Auto-start Docker services before testing"
    )

    parser.add_argument("--no-fixtures", action="store_true", help="Skip test fixture generation")

    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup after tests")

    parser.add_argument(
        "--suite",
        choices=["workflow", "api", "performance", "all"],
        default="all",
        help="Test suite to run (default: all)",
    )

    parser.add_argument(
        "--timeout", type=int, default=300, help="Test timeout in seconds (default: 300)"
    )

    args = parser.parse_args()

    # Load configuration
    config = None
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

    # Override config with command line arguments
    if not config:
        config = {}

    # Ensure all required keys exist
    if "test_suites" not in config:
        config["test_suites"] = {}
    if "docker" not in config:
        config["docker"] = {}

    config["base_url"] = args.base_url
    config["timeout"] = args.timeout
    config["results_dir"] = config.get("results_dir", "tests/integration/results")
    config["save_results"] = config.get("save_results", True)
    config["docker"]["auto_start"] = args.docker
    config["generate_fixtures"] = not args.no_fixtures
    config["cleanup_fixtures"] = not args.no_cleanup

    # Configure test suites
    if args.suite != "all":
        config["test_suites"] = {
            "workflow": args.suite == "workflow",
            "api_endpoints": args.suite == "api",
            "performance": args.suite == "performance",
        }

    # Run tests
    runner = IntegrationTestRunner(config=config)
    exit_code = runner.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
