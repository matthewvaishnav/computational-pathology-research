#!/usr/bin/env python3
"""
Master test runner for complete dataset testing suite.
Integrates all test categories with comprehensive reporting.
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import json
from datetime import datetime
import time


class MasterTestRunner:
    """Run complete dataset testing suite with CI integration."""
    
    def __init__(self, output_dir: str = "dataset_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.start_time = time.time()
        
    def check_dependencies(self) -> bool:
        """Check required dependencies are installed."""
        required_packages = [
            "pytest", "pytest-cov", "hypothesis", "h5py", 
            "numpy", "torch", "psutil"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"❌ Missing dependencies: {', '.join(missing)}")
            print("Install with: pip install " + " ".join(missing))
            return False
        
        print("✅ All dependencies available")
        return True
    
    def run_test_category(self, category: str, fast_mode: bool = False) -> dict:
        """Run specific test category."""
        test_path = f"tests/dataset_testing/{category}"
        
        if not Path(test_path).exists():
            return {
                "category": category,
                "status": "skipped",
                "reason": f"Directory not found: {test_path}",
                "tests": 0,
                "passed": 0,
                "failed": 0,
                "duration": 0
            }
        
        print(f"\n🧪 Running {category} tests...")
        
        # Build pytest command
        cmd = ["python", "-m", "pytest", test_path, "-v"]
        
        if fast_mode:
            cmd.extend(["-x", "--tb=line"])  # Stop on first failure, short traceback
        else:
            cmd.extend(["--tb=short"])
        
        # Add coverage for unit tests
        if category == "unit":
            cmd.extend([
                "--cov=src/data/preprocessing",
                "--cov=src/data/openslide_utils", 
                f"--cov-report=html:{self.output_dir}/coverage_{category}",
                f"--cov-report=json:{self.output_dir}/coverage_{category}.json"
            ])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            duration = time.time() - start_time
            
            # Parse output for test counts
            stdout_lines = result.stdout.split('\n')
            tests, passed, failed = 0, 0, 0
            
            for line in stdout_lines:
                if " passed" in line or " failed" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            passed = int(parts[i-1])
                        elif part == "failed,":
                            failed = int(parts[i-1])
                        elif part == "passed,":
                            passed = int(parts[i-1])
            
            tests = passed + failed
            
            return {
                "category": category,
                "status": "passed" if result.returncode == 0 else "failed",
                "tests": tests,
                "passed": passed,
                "failed": failed,
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout[-1000:],  # Last 1000 chars
                "stderr": result.stderr[-500:] if result.stderr else ""
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "category": category,
                "status": "timeout",
                "tests": 0,
                "passed": 0,
                "failed": 0,
                "duration": duration,
                "timeout": True
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "category": category,
                "status": "error",
                "tests": 0,
                "passed": 0,
                "failed": 0,
                "duration": duration,
                "error": str(e)
            }
    
    def run_all_categories(self, categories: list = None, fast_mode: bool = False) -> dict:
        """Run all test categories."""
        if categories is None:
            categories = ["unit", "integration", "performance"]
        
        print("="*60)
        print("COMPREHENSIVE DATASET TESTING SUITE")
        print("="*60)
        print(f"Categories: {', '.join(categories)}")
        print(f"Fast mode: {fast_mode}")
        print(f"Output: {self.output_dir}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "summary": {
                "total_categories": len(categories),
                "passed_categories": 0,
                "failed_categories": 0,
                "total_tests": 0,
                "total_passed": 0,
                "total_failed": 0,
                "total_duration": 0
            }
        }
        
        for category in categories:
            category_result = self.run_test_category(category, fast_mode)
            results["categories"][category] = category_result
            
            # Update summary
            if category_result["status"] == "passed":
                results["summary"]["passed_categories"] += 1
            elif category_result["status"] in ["failed", "error", "timeout"]:
                results["summary"]["failed_categories"] += 1
            
            results["summary"]["total_tests"] += category_result["tests"]
            results["summary"]["total_passed"] += category_result["passed"]
            results["summary"]["total_failed"] += category_result["failed"]
            results["summary"]["total_duration"] += category_result["duration"]
            
            # Print category result
            status_emoji = {
                "passed": "PASS",
                "failed": "FAIL", 
                "error": "ERROR",
                "timeout": "TIMEOUT",
                "skipped": "SKIP"
            }
            
            emoji = status_emoji.get(category_result["status"], "?")
            print(f"[{emoji}] {category}: {category_result['passed']}/{category_result['tests']} passed ({category_result['duration']:.1f}s)")
        
        results["summary"]["success_rate"] = (
            results["summary"]["total_passed"] / max(results["summary"]["total_tests"], 1)
        ) * 100
        
        results["summary"]["overall_success"] = (
            results["summary"]["failed_categories"] == 0 and 
            results["summary"]["total_tests"] > 0
        )
        
        return results
    
    def generate_ci_summary(self, results: dict) -> str:
        """Generate CI-friendly summary."""
        summary = results["summary"]
        
        status = "PASSED" if summary["overall_success"] else "FAILED"
        
        ci_summary = f"""
## Dataset Testing Suite Results

**Status**: {status}
**Categories**: {summary["passed_categories"]}/{summary["total_categories"]} passed
**Tests**: {summary["total_passed"]}/{summary["total_tests"]} passed ({summary["success_rate"]:.1f}%)
**Duration**: {summary["total_duration"]:.1f}s

### Category Results

| Category | Status | Tests | Passed | Failed | Duration |
|----------|--------|-------|--------|--------|----------|
"""
        
        for category, data in results["categories"].items():
            status_icon = "[PASS]" if data["status"] == "passed" else "[FAIL]"
            ci_summary += f"| {category} | {status_icon} {data['status']} | {data['tests']} | {data['passed']} | {data['failed']} | {data['duration']:.1f}s |\n"
        
        return ci_summary
    
    def save_results(self, results: dict) -> tuple:
        """Save results and generate reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = self.output_dir / f"results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate CI summary
        ci_summary = self.generate_ci_summary(results)
        summary_file = self.output_dir / f"ci_summary_{timestamp}.md"
        with open(summary_file, "w") as f:
            f.write(ci_summary)
        
        # Create latest links
        latest_results = self.output_dir / "latest_results.json"
        latest_summary = self.output_dir / "latest_summary.md"
        
        # Remove existing
        for f in [latest_results, latest_summary]:
            if f.exists():
                f.unlink()
        
        # Copy to latest (Windows compatible)
        import shutil
        shutil.copy2(results_file, latest_results)
        shutil.copy2(summary_file, latest_summary)
        
        return str(results_file), str(summary_file)
    
    def print_final_summary(self, results: dict):
        """Print final execution summary."""
        summary = results["summary"]
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("DATASET TESTING SUITE COMPLETE")
        print("="*60)
        
        status = "PASSED" if summary["overall_success"] else "FAILED"
        print(f"Overall Status: {status}")
        print(f"Categories: {summary['passed_categories']}/{summary['total_categories']} passed")
        print(f"Tests: {summary['total_passed']}/{summary['total_tests']} passed ({summary['success_rate']:.1f}%)")
        print(f"Total Duration: {total_time:.1f}s")
        
        if not summary["overall_success"]:
            print("\n[FAIL] Failed Categories:")
            for category, data in results["categories"].items():
                if data["status"] != "passed":
                    print(f"  - {category}: {data['status']}")
                    if data.get("error"):
                        print(f"    Error: {data['error']}")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive dataset testing suite")
    parser.add_argument("--categories", nargs="+",
                       default=["unit", "integration", "performance"],
                       help="Test categories to run")
    parser.add_argument("--output-dir", default="dataset_test_results",
                       help="Output directory")
    parser.add_argument("--fast", action="store_true",
                       help="Fast mode: stop on first failure, minimal output")
    parser.add_argument("--check-deps", action="store_true",
                       help="Only check dependencies, don't run tests")
    parser.add_argument("--ci", action="store_true",
                       help="CI mode: optimized for continuous integration")
    
    args = parser.parse_args()
    
    runner = MasterTestRunner(args.output_dir)
    
    # Check dependencies
    if not runner.check_dependencies():
        sys.exit(1)
    
    if args.check_deps:
        print("✅ Dependency check passed")
        sys.exit(0)
    
    # Run tests
    try:
        fast_mode = args.fast or args.ci
        results = runner.run_all_categories(args.categories, fast_mode)
        
        # Save results
        results_file, summary_file = runner.save_results(results)
        
        # Print summary
        runner.print_final_summary(results)
        
        print(f"\nResults saved:")
        print(f"  - Raw data: {results_file}")
        print(f"  - CI summary: {summary_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results["summary"]["overall_success"] else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()