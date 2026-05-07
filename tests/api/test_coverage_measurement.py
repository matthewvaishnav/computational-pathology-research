#!/usr/bin/env python3
"""
Test Coverage Measurement for API Routes Refactoring

Measures and validates test coverage across the refactored API modules
to ensure >80% coverage is maintained.

**Validates: Requirements 12.1-12.4**
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest


class TestCoverageMeasurement:
    """Test coverage measurement and validation."""

    def setup_method(self):
        """Set up coverage measurement."""
        self.coverage_threshold = 80.0  # 80% minimum coverage
        self.api_modules = [
            "src/api/main.py",
            "src/api/routers/auth.py",
            "src/api/routers/analysis.py", 
            "src/api/routers/admin.py",
            "src/api/routers/mobile.py",
            "src/api/routers/monitoring.py",
            "src/api/dependencies.py",
            "src/api/validators.py",
            "src/api/errors.py",
            "src/api/security.py",
            "src/api/middleware.py",
        ]

    def _run_coverage_analysis(self) -> Tuple[Dict, float]:
        """Run coverage analysis on API modules."""
        try:
            # Run pytest with coverage for API modules
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/api/",
                "--cov=src/api",
                "--cov-report=json:tests/api/coverage.json",
                "--cov-report=term-missing",
                "-v"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            print("Coverage Analysis Output:")
            print(result.stdout)
            
            if result.stderr:
                print("Coverage Analysis Errors:")
                print(result.stderr)
            
            # Read coverage report
            coverage_file = Path("tests/api/coverage.json")
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                return coverage_data, total_coverage
            else:
                print("Coverage report file not found")
                return {}, 0.0
                
        except Exception as e:
            print(f"Error running coverage analysis: {e}")
            return {}, 0.0

    def _analyze_module_coverage(self, coverage_data: Dict) -> Dict[str, float]:
        """Analyze coverage for individual modules."""
        module_coverage = {}
        
        files = coverage_data.get("files", {})
        
        for module_path in self.api_modules:
            # Normalize path for comparison
            normalized_path = module_path.replace("/", "\\") if sys.platform == "win32" else module_path
            
            # Find matching file in coverage data
            matching_file = None
            for file_path in files.keys():
                if module_path in file_path or normalized_path in file_path:
                    matching_file = file_path
                    break
            
            if matching_file:
                file_data = files[matching_file]
                coverage_percent = file_data.get("summary", {}).get("percent_covered", 0)
                module_coverage[module_path] = coverage_percent
            else:
                print(f"Warning: No coverage data found for {module_path}")
                module_coverage[module_path] = 0.0
        
        return module_coverage

    def _identify_uncovered_lines(self, coverage_data: Dict) -> Dict[str, List[int]]:
        """Identify uncovered lines in each module."""
        uncovered_lines = {}
        
        files = coverage_data.get("files", {})
        
        for module_path in self.api_modules:
            # Find matching file in coverage data
            matching_file = None
            for file_path in files.keys():
                if module_path in file_path:
                    matching_file = file_path
                    break
            
            if matching_file:
                file_data = files[matching_file]
                missing_lines = file_data.get("missing_lines", [])
                uncovered_lines[module_path] = missing_lines
            else:
                uncovered_lines[module_path] = []
        
        return uncovered_lines

    def test_overall_coverage_threshold(self):
        """
        Test that overall API coverage is above 80%.
        
        **Validates: Requirements 12.1-12.4**
        """
        print("\n📊 Testing Overall Coverage Threshold...")
        
        coverage_data, total_coverage = self._run_coverage_analysis()
        
        print(f"Overall API Coverage: {total_coverage:.2f}%")
        print(f"Required Threshold: {self.coverage_threshold}%")
        
        if total_coverage >= self.coverage_threshold:
            print("✅ Coverage threshold met")
        else:
            print("❌ Coverage threshold not met")
        
        assert total_coverage >= self.coverage_threshold, \
            f"Overall coverage {total_coverage:.2f}% is below threshold {self.coverage_threshold}%"

    def test_individual_module_coverage(self):
        """
        Test coverage for individual API modules.
        
        **Validates: Requirements 12.1-12.4**
        """
        print("\n📋 Testing Individual Module Coverage...")
        
        coverage_data, _ = self._run_coverage_analysis()
        module_coverage = self._analyze_module_coverage(coverage_data)
        
        low_coverage_modules = []
        module_threshold = 70.0  # 70% threshold for individual modules
        
        print(f"\nModule Coverage Report (Threshold: {module_threshold}%):")
        print("-" * 60)
        
        for module_path, coverage_percent in module_coverage.items():
            status = "✅" if coverage_percent >= module_threshold else "❌"
            print(f"{status} {module_path}: {coverage_percent:.2f}%")
            
            if coverage_percent < module_threshold:
                low_coverage_modules.append((module_path, coverage_percent))
        
        if low_coverage_modules:
            print(f"\n⚠️ Modules with low coverage:")
            for module_path, coverage_percent in low_coverage_modules:
                print(f"  - {module_path}: {coverage_percent:.2f}%")
        
        # Allow some modules to have lower coverage (e.g., main.py might be mostly imports)
        critical_modules = [
            "src/api/routers/auth.py",
            "src/api/routers/analysis.py",
            "src/api/routers/admin.py",
            "src/api/dependencies.py",
        ]
        
        critical_low_coverage = [
            (module, coverage) for module, coverage in low_coverage_modules
            if module in critical_modules
        ]
        
        assert len(critical_low_coverage) == 0, \
            f"Critical modules have low coverage: {critical_low_coverage}"

    def test_uncovered_code_analysis(self):
        """
        Analyze uncovered code to identify testing gaps.
        
        **Validates: Requirements 12.1-12.4**
        """
        print("\n🔍 Analyzing Uncovered Code...")
        
        coverage_data, _ = self._run_coverage_analysis()
        uncovered_lines = self._identify_uncovered_lines(coverage_data)
        
        total_uncovered = 0
        modules_with_gaps = []
        
        print("\nUncovered Lines Report:")
        print("-" * 40)
        
        for module_path, missing_lines in uncovered_lines.items():
            if missing_lines:
                total_uncovered += len(missing_lines)
                modules_with_gaps.append(module_path)
                
                print(f"\n{module_path}:")
                print(f"  Uncovered lines: {len(missing_lines)}")
                
                # Show first few uncovered lines
                if len(missing_lines) <= 10:
                    print(f"  Lines: {missing_lines}")
                else:
                    print(f"  Lines: {missing_lines[:5]} ... {missing_lines[-5:]} (showing first/last 5)")
        
        print(f"\nSummary:")
        print(f"  Total uncovered lines: {total_uncovered}")
        print(f"  Modules with gaps: {len(modules_with_gaps)}")
        
        if modules_with_gaps:
            print(f"  Modules needing attention: {modules_with_gaps}")

    def test_test_file_coverage(self):
        """
        Verify that all API modules have corresponding test files.
        
        **Validates: Requirements 12.1-12.4**
        """
        print("\n📝 Testing Test File Coverage...")
        
        # Map of modules to their expected test files
        expected_test_files = {
            "src/api/routers/auth.py": ["tests/api/test_auth.py", "tests/api/test_auth_unit.py"],
            "src/api/routers/analysis.py": ["tests/api/test_analysis.py", "tests/api/test_analysis_unit.py"],
            "src/api/routers/admin.py": ["tests/api/test_admin.py", "tests/api/test_admin_unit.py"],
            "src/api/routers/mobile.py": ["tests/api/test_mobile.py", "tests/api/test_mobile_unit.py"],
            "src/api/routers/monitoring.py": ["tests/api/test_monitoring.py", "tests/api/test_monitoring_unit.py"],
            "src/api/validators.py": ["tests/api/test_validators.py"],
            "src/api/errors.py": ["tests/api/test_errors.py"],
        }
        
        missing_test_files = []
        
        print("Test File Coverage Report:")
        print("-" * 40)
        
        for module_path, test_files in expected_test_files.items():
            module_has_tests = False
            
            for test_file in test_files:
                if Path(test_file).exists():
                    module_has_tests = True
                    print(f"✅ {module_path} -> {test_file}")
                    break
            
            if not module_has_tests:
                missing_test_files.append(module_path)
                print(f"❌ {module_path} -> No test file found")
        
        if missing_test_files:
            print(f"\n⚠️ Modules without test files: {missing_test_files}")
        
        # Allow some modules to not have dedicated test files (e.g., main.py)
        critical_modules_without_tests = [
            module for module in missing_test_files
            if "routers/" in module or "validators.py" in module or "errors.py" in module
        ]
        
        assert len(critical_modules_without_tests) == 0, \
            f"Critical modules missing test files: {critical_modules_without_tests}"

    def test_integration_test_coverage(self):
        """
        Verify integration tests cover end-to-end workflows.
        
        **Validates: Requirements 12.1-12.4**
        """
        print("\n🔗 Testing Integration Test Coverage...")
        
        # Check for integration test files
        integration_test_files = [
            "tests/api/test_integration_flows.py",
            "tests/api/test_security.py",
            "tests/api/test_performance.py",
        ]
        
        existing_integration_tests = []
        missing_integration_tests = []
        
        for test_file in integration_test_files:
            if Path(test_file).exists():
                existing_integration_tests.append(test_file)
                print(f"✅ {test_file}")
            else:
                missing_integration_tests.append(test_file)
                print(f"❌ {test_file}")
        
        print(f"\nIntegration Test Summary:")
        print(f"  Existing: {len(existing_integration_tests)}")
        print(f"  Missing: {len(missing_integration_tests)}")
        
        # At least basic integration tests should exist
        assert len(existing_integration_tests) >= 2, \
            f"Insufficient integration test coverage: {existing_integration_tests}"

    def test_property_based_test_coverage(self):
        """
        Verify property-based tests exist for API equivalence.
        
        **Validates: Requirements 12.1-12.4**
        """
        print("\n🎲 Testing Property-Based Test Coverage...")
        
        # Check for property-based test files
        pbt_test_files = [
            "tests/api/test_api_refactor_equivalence.py",
        ]
        
        existing_pbt_tests = []
        
        for test_file in pbt_test_files:
            if Path(test_file).exists():
                existing_pbt_tests.append(test_file)
                print(f"✅ {test_file}")
            else:
                print(f"❌ {test_file}")
        
        print(f"\nProperty-Based Test Summary:")
        print(f"  Existing: {len(existing_pbt_tests)}")
        
        # Property-based tests are important for refactoring validation
        assert len(existing_pbt_tests) >= 1, \
            "Property-based tests missing for API equivalence validation"

    def test_coverage_report_generation(self):
        """
        Generate comprehensive coverage report.
        
        **Validates: Requirements 12.1-12.4**
        """
        print("\n📄 Generating Coverage Report...")
        
        coverage_data, total_coverage = self._run_coverage_analysis()
        module_coverage = self._analyze_module_coverage(coverage_data)
        uncovered_lines = self._identify_uncovered_lines(coverage_data)
        
        # Generate comprehensive report
        report = {
            "timestamp": pytest.current_timestamp if hasattr(pytest, 'current_timestamp') else "unknown",
            "overall_coverage": total_coverage,
            "coverage_threshold": self.coverage_threshold,
            "threshold_met": total_coverage >= self.coverage_threshold,
            "module_coverage": module_coverage,
            "uncovered_lines": uncovered_lines,
            "summary": {
                "total_modules": len(self.api_modules),
                "modules_above_threshold": len([
                    m for m, c in module_coverage.items() 
                    if c >= 70.0
                ]),
                "total_uncovered_lines": sum(len(lines) for lines in uncovered_lines.values()),
                "modules_with_gaps": len([
                    m for m, lines in uncovered_lines.items() 
                    if lines
                ])
            },
            "recommendations": []
        }
        
        # Add recommendations based on analysis
        if total_coverage < self.coverage_threshold:
            report["recommendations"].append(
                f"Increase overall coverage from {total_coverage:.2f}% to {self.coverage_threshold}%"
            )
        
        low_coverage_modules = [
            m for m, c in module_coverage.items() 
            if c < 70.0 and c > 0
        ]
        
        if low_coverage_modules:
            report["recommendations"].append(
                f"Improve coverage for modules: {low_coverage_modules}"
            )
        
        modules_with_many_gaps = [
            m for m, lines in uncovered_lines.items()
            if len(lines) > 20
        ]
        
        if modules_with_many_gaps:
            report["recommendations"].append(
                f"Focus testing on modules with many uncovered lines: {modules_with_many_gaps}"
            )
        
        # Save report
        report_file = Path("tests/api/coverage_report.json")
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Coverage report saved to: {report_file}")
        except Exception as e:
            print(f"Could not save coverage report: {e}")
        
        # Print summary
        print(f"\nCoverage Report Summary:")
        print(f"  Overall Coverage: {total_coverage:.2f}%")
        print(f"  Threshold Met: {'✅ Yes' if report['threshold_met'] else '❌ No'}")
        print(f"  Modules Above 70%: {report['summary']['modules_above_threshold']}/{report['summary']['total_modules']}")
        print(f"  Total Uncovered Lines: {report['summary']['total_uncovered_lines']}")
        
        if report["recommendations"]:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"  {i}. {rec}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])