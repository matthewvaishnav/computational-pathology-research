"""
Setup Validation Script for the Competitor Benchmark System.

This standalone script validates the system setup before running benchmarks.
It provides clear feedback about any issues that would prevent benchmarks from
running successfully.

Requirements: 3.1, 9.2, 9.3, 9.4

Usage:
    python experiments/benchmark_system/validate_setup.py
    python experiments/benchmark_system/validate_setup.py --output validation_report.json
    python experiments/benchmark_system/validate_setup.py --verbose
"""

import argparse
import json
import logging
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Use ASCII-compatible symbols for cross-platform compatibility
CHECK_MARK = "[OK]"
CROSS_MARK = "[FAIL]"
WARNING_MARK = "[WARN]"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    check_name: str
    passed: bool
    message: str
    details: Dict[str, any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SetupValidationReport:
    """Complete setup validation report."""
    
    timestamp: str
    overall_status: str  # "PASS", "FAIL", "WARNING"
    checks: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "checks": [
                {
                    "check_name": check.check_name,
                    "passed": check.passed,
                    "message": check.message,
                    "details": check.details,
                    "warnings": check.warnings,
                }
                for check in self.checks
            ],
            "summary": self.summary,
        }


class SetupValidator:
    """Validates system setup for benchmark execution."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize Setup Validator.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        
    def run_all_checks(self) -> SetupValidationReport:
        """
        Run all validation checks.
        
        Returns:
            SetupValidationReport with all check results
        """
        print("=" * 80)
        print("Competitor Benchmark System - Setup Validation")
        print("=" * 80)
        print()
        
        # Run all validation checks
        self._check_gpu_availability()
        self._check_cuda_cudnn_versions()
        self._check_disk_space()
        self._check_python_compatibility()
        self._check_framework_imports()
        
        # Generate report
        report = self._generate_report()
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _check_gpu_availability(self) -> None:
        """
        Verify GPU availability and specifications.
        
        Requirements: 3.1, 9.3
        """
        print("1. Checking GPU availability and specifications...")
        print("-" * 80)
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                result = ValidationResult(
                    check_name="GPU Availability",
                    passed=False,
                    message="CUDA is not available",
                    details={
                        "cuda_available": False,
                        "error": "PyTorch cannot detect CUDA",
                    },
                )
                self.results.append(result)
                print(f"   {CROSS_MARK} CUDA is not available")
                print()
                return
            
            # Get GPU information
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_properties = torch.cuda.get_device_properties(0)
            gpu_memory_mb = gpu_properties.total_memory / (1024 ** 2)
            
            # Get driver version from nvidia-smi
            driver_version = "unknown"
            try:
                result_smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                driver_version = result_smi.stdout.strip()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Check if GPU matches expected (RTX 4070)
            expected_gpu = "RTX 4070"
            gpu_matches = expected_gpu.lower() in gpu_name.lower()
            
            warnings = []
            if not gpu_matches:
                warnings.append(
                    f"Expected {expected_gpu}, found {gpu_name}. "
                    f"Benchmarks may not be comparable to reference results."
                )
            
            if gpu_memory_mb < 10000:  # Less than 10GB
                warnings.append(
                    f"GPU has {gpu_memory_mb:.0f}MB memory. "
                    f"Some benchmarks may fail due to insufficient memory."
                )
            
            result = ValidationResult(
                check_name="GPU Availability",
                passed=True,
                message=f"GPU detected: {gpu_name}",
                details={
                    "gpu_count": gpu_count,
                    "gpu_name": gpu_name,
                    "gpu_memory_mb": gpu_memory_mb,
                    "driver_version": driver_version,
                    "compute_capability": f"{gpu_properties.major}.{gpu_properties.minor}",
                    "matches_expected": gpu_matches,
                },
                warnings=warnings,
            )
            self.results.append(result)
            
            print(f"   {CHECK_MARK} GPU detected: {gpu_name}")
            print(f"   {CHECK_MARK} GPU count: {gpu_count}")
            print(f"   {CHECK_MARK} GPU memory: {gpu_memory_mb:.0f} MB")
            print(f"   {CHECK_MARK} Driver version: {driver_version}")
            print(f"   {CHECK_MARK} Compute capability: {gpu_properties.major}.{gpu_properties.minor}")
            
            if warnings:
                for warning in warnings:
                    print(f"   {WARNING_MARK} {warning}")
            
        except ImportError:
            result = ValidationResult(
                check_name="GPU Availability",
                passed=False,
                message="PyTorch not installed",
                details={"error": "Cannot import torch"},
            )
            self.results.append(result)
            print(f"   {CROSS_MARK} PyTorch not installed")
        except Exception as e:
            result = ValidationResult(
                check_name="GPU Availability",
                passed=False,
                message=f"Error checking GPU: {str(e)}",
                details={"error": str(e)},
            )
            self.results.append(result)
            print(f"   {CROSS_MARK} Error: {e}")
        
        print()
    
    def _check_cuda_cudnn_versions(self) -> None:
        """
        Verify CUDA and cuDNN versions.
        
        Requirements: 9.2
        """
        print("2. Checking CUDA and cuDNN versions...")
        print("-" * 80)
        
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else "not_available"
            cudnn_available = torch.backends.cudnn.is_available() if cuda_available else False
            cudnn_version = str(torch.backends.cudnn.version()) if cudnn_available else "not_available"
            
            warnings = []
            
            # Check CUDA version
            if cuda_available:
                cuda_major = int(cuda_version.split(".")[0]) if cuda_version != "not_available" else 0
                if cuda_major < 11:
                    warnings.append(
                        f"CUDA {cuda_version} is older than recommended (11.8+). "
                        f"Some features may not work correctly."
                    )
            else:
                warnings.append("CUDA is not available. GPU acceleration will not work.")
            
            # Check cuDNN
            if not cudnn_available:
                warnings.append("cuDNN is not available. Training may be slower.")
            
            result = ValidationResult(
                check_name="CUDA and cuDNN Versions",
                passed=cuda_available,
                message=f"CUDA {cuda_version}, cuDNN {cudnn_version}",
                details={
                    "cuda_available": cuda_available,
                    "cuda_version": cuda_version,
                    "cudnn_available": cudnn_available,
                    "cudnn_version": cudnn_version,
                },
                warnings=warnings,
            )
            self.results.append(result)
            
            if cuda_available:
                print(f"   {CHECK_MARK} CUDA version: {cuda_version}")
            else:
                print(f"   {CROSS_MARK} CUDA not available")
            
            if cudnn_available:
                print(f"   {CHECK_MARK} cuDNN version: {cudnn_version}")
            else:
                print(f"   {CROSS_MARK} cuDNN not available")
            
            if warnings:
                for warning in warnings:
                    print(f"   {WARNING_MARK} {warning}")
            
        except ImportError:
            result = ValidationResult(
                check_name="CUDA and cuDNN Versions",
                passed=False,
                message="PyTorch not installed",
                details={"error": "Cannot import torch"},
            )
            self.results.append(result)
            print(f"   {CROSS_MARK} PyTorch not installed")
        except Exception as e:
            result = ValidationResult(
                check_name="CUDA and cuDNN Versions",
                passed=False,
                message=f"Error checking CUDA/cuDNN: {str(e)}",
                details={"error": str(e)},
            )
            self.results.append(result)
            print(f"   {CROSS_MARK} Error: {e}")
        
        print()
    
    def _check_disk_space(self) -> None:
        """
        Verify disk space availability.
        
        Requirements: 9.3 (indirectly - system resources)
        """
        print("3. Checking disk space availability...")
        print("-" * 80)
        
        try:
            stat = shutil.disk_usage(".")
            total_gb = stat.total / (1024 ** 3)
            used_gb = stat.used / (1024 ** 3)
            free_gb = stat.free / (1024 ** 3)
            percent_used = (stat.used / stat.total) * 100
            
            warnings = []
            passed = True
            
            # Check minimum requirements
            if free_gb < 100:
                warnings.append(
                    f"Only {free_gb:.1f} GB free. Recommended: 100+ GB for "
                    f"environments, checkpoints, and results."
                )
                if free_gb < 50:
                    passed = False
                    warnings.append("Insufficient disk space. Benchmarks will likely fail.")
            
            result = ValidationResult(
                check_name="Disk Space",
                passed=passed,
                message=f"{free_gb:.1f} GB free ({percent_used:.1f}% used)",
                details={
                    "total_gb": total_gb,
                    "used_gb": used_gb,
                    "free_gb": free_gb,
                    "percent_used": percent_used,
                },
                warnings=warnings,
            )
            self.results.append(result)
            
            if passed:
                print(f"   {CHECK_MARK} Total disk space: {total_gb:.1f} GB")
                print(f"   {CHECK_MARK} Free disk space: {free_gb:.1f} GB ({100 - percent_used:.1f}% free)")
            else:
                print(f"   {CROSS_MARK} Insufficient disk space: {free_gb:.1f} GB free")
            
            if warnings:
                for warning in warnings:
                    print(f"   {WARNING_MARK} {warning}")
            
        except Exception as e:
            result = ValidationResult(
                check_name="Disk Space",
                passed=False,
                message=f"Error checking disk space: {str(e)}",
                details={"error": str(e)},
            )
            self.results.append(result)
            print(f"   {CROSS_MARK} Error: {e}")
        
        print()
    
    def _check_python_compatibility(self) -> None:
        """
        Verify Python version compatibility.
        
        Requirements: 9.4
        """
        print("4. Checking Python version compatibility...")
        print("-" * 80)
        
        try:
            python_version = platform.python_version()
            python_implementation = platform.python_implementation()
            major, minor, micro = sys.version_info[:3]
            
            warnings = []
            passed = True
            
            # Check Python version
            if major < 3 or (major == 3 and minor < 9):
                passed = False
                warnings.append(
                    f"Python {python_version} is too old. Required: Python 3.9+"
                )
            elif major == 3 and minor >= 14:
                warnings.append(
                    f"Python {python_version} may have compatibility issues with "
                    f"some frameworks (e.g., PathML numpy/pandas). Patches will be applied."
                )
            
            # Check implementation
            if python_implementation != "CPython":
                warnings.append(
                    f"Python implementation is {python_implementation}. "
                    f"CPython is recommended for best compatibility."
                )
            
            result = ValidationResult(
                check_name="Python Compatibility",
                passed=passed,
                message=f"Python {python_version} ({python_implementation})",
                details={
                    "python_version": python_version,
                    "python_implementation": python_implementation,
                    "major": major,
                    "minor": minor,
                    "micro": micro,
                    "os_name": platform.system(),
                    "os_version": platform.version(),
                    "os_release": platform.release(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                },
                warnings=warnings,
            )
            self.results.append(result)
            
            if passed:
                print(f"   {CHECK_MARK} Python version: {python_version}")
                print(f"   {CHECK_MARK} Implementation: {python_implementation}")
                print(f"   {CHECK_MARK} OS: {platform.system()} {platform.release()}")
                print(f"   {CHECK_MARK} Architecture: {platform.machine()}")
            else:
                print(f"   {CROSS_MARK} Python version incompatible: {python_version}")
            
            if warnings:
                for warning in warnings:
                    print(f"   {WARNING_MARK} {warning}")
            
        except Exception as e:
            result = ValidationResult(
                check_name="Python Compatibility",
                passed=False,
                message=f"Error checking Python: {str(e)}",
                details={"error": str(e)},
            )
            self.results.append(result)
            print(f"   {CROSS_MARK} Error: {e}")
        
        print()
    
    def _check_framework_imports(self) -> None:
        """
        Run smoke tests for framework imports.
        
        Requirements: 3.1 (indirectly - validates PyTorch setup)
        """
        print("5. Running smoke tests for framework imports...")
        print("-" * 80)
        
        frameworks_to_test = [
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("scipy", "SciPy"),
            ("sklearn", "scikit-learn"),
            ("matplotlib", "Matplotlib"),
            ("PIL", "Pillow"),
            ("yaml", "PyYAML"),
            ("psutil", "psutil"),
        ]
        
        import_results = {}
        all_passed = True
        warnings = []
        
        for module_name, display_name in frameworks_to_test:
            try:
                module = __import__(module_name)
                version = getattr(module, "__version__", "unknown")
                import_results[display_name] = {
                    "success": True,
                    "version": version,
                }
                print(f"   {CHECK_MARK} {display_name}: {version}")
            except ImportError as e:
                import_results[display_name] = {
                    "success": False,
                    "error": str(e),
                }
                print(f"   {CROSS_MARK} {display_name}: Not installed")
                warnings.append(f"{display_name} not installed: {e}")
                
                # Only fail for critical packages
                if module_name in ["torch", "numpy"]:
                    all_passed = False
            except Exception as e:
                import_results[display_name] = {
                    "success": False,
                    "error": str(e),
                }
                print(f"   {WARNING_MARK} {display_name}: Error - {e}")
                warnings.append(f"{display_name} error: {e}")
        
        result = ValidationResult(
            check_name="Framework Imports",
            passed=all_passed,
            message=f"{sum(1 for r in import_results.values() if r['success'])}/{len(frameworks_to_test)} frameworks imported successfully",
            details=import_results,
            warnings=warnings,
        )
        self.results.append(result)
        
        print()
    
    def _generate_report(self) -> SetupValidationReport:
        """
        Generate setup validation report.
        
        Returns:
            SetupValidationReport with all results
        """
        passed_count = sum(1 for r in self.results if r.passed)
        failed_count = len(self.results) - passed_count
        warning_count = sum(len(r.warnings) for r in self.results)
        
        # Determine overall status
        if failed_count > 0:
            overall_status = "FAIL"
        elif warning_count > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASS"
        
        report = SetupValidationReport(
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            checks=self.results,
            summary={
                "total_checks": len(self.results),
                "passed": passed_count,
                "failed": failed_count,
                "warnings": warning_count,
            },
        )
        
        return report
    
    def _print_summary(self, report: SetupValidationReport) -> None:
        """
        Print validation summary.
        
        Args:
            report: SetupValidationReport to summarize
        """
        print("=" * 80)
        print("Validation Summary")
        print("=" * 80)
        print()
        print(f"Total checks: {report.summary['total_checks']}")
        print(f"Passed: {report.summary['passed']}")
        print(f"Failed: {report.summary['failed']}")
        print(f"Warnings: {report.summary['warnings']}")
        print()
        
        if report.overall_status == "PASS":
            print(f"{CHECK_MARK} Overall Status: PASS")
            print()
            print("System is ready for benchmarking!")
        elif report.overall_status == "WARNING":
            print(f"{WARNING_MARK} Overall Status: WARNING")
            print()
            print("System can run benchmarks, but there are warnings to address.")
            print("Review the warnings above for potential issues.")
        else:
            print(f"{CROSS_MARK} Overall Status: FAIL")
            print()
            print("System is NOT ready for benchmarking.")
            print("Fix the failed checks above before running benchmarks.")
        
        print("=" * 80)


def main():
    """Main entry point for setup validation."""
    parser = argparse.ArgumentParser(
        description="Validate setup for Competitor Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run validation with default settings
  python experiments/benchmark_system/validate_setup.py
  
  # Run validation with verbose output
  python experiments/benchmark_system/validate_setup.py --verbose
  
  # Save validation report to JSON file
  python experiments/benchmark_system/validate_setup.py --output validation_report.json
        """,
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save validation report (JSON format)",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = SetupValidator(verbose=args.verbose)
    report = validator.run_all_checks()
    
    # Save report if requested
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print()
            print(f"Validation report saved to: {args.output}")
        except Exception as e:
            print()
            print(f"Error saving report: {e}", file=sys.stderr)
            return 1
    
    # Return exit code based on overall status
    if report.overall_status == "FAIL":
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
