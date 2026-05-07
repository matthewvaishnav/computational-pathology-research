#!/usr/bin/env python3
"""
Integration test script for WSI Processing Pipeline.

This script runs comprehensive tests to validate the WSI processing pipeline
implementation, including performance benchmarks and end-to-end validation.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.wsi_pipeline.config import ProcessingConfig
from data.wsi_pipeline.validation import run_comprehensive_validation
from data.wsi_pipeline.benchmarks import run_performance_benchmarks


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def test_basic_functionality() -> bool:
    """Test basic pipeline functionality."""
    print("\n" + "="*50)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*50)
    
    try:
        # Test configuration
        config = ProcessingConfig(
            patch_size=256,
            encoder_name="resnet50",
            batch_size=8,
        )
        print(f"✅ Configuration created: patch_size={config.patch_size}")
        
        # Test component imports
        from data.wsi_pipeline.batch_processor import BatchProcessor
        from data.wsi_pipeline.cache import FeatureCache
        from data.wsi_pipeline.extractor import PatchExtractor
        from data.wsi_pipeline.feature_generator import FeatureGenerator
        from data.wsi_pipeline.tissue_detector import TissueDetector
        from data.wsi_pipeline.quality_control import QualityControl
        print("✅ All components imported successfully")
        
        # Test component initialization
        extractor = PatchExtractor(patch_size=256)
        detector = TissueDetector()
        generator = FeatureGenerator(encoder_name="resnet50", device="cpu")
        print("✅ Core components initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_performance_benchmarks(quick_mode: bool = True) -> bool:
    """Test performance benchmarks."""
    print("\n" + "="*50)
    print("TESTING PERFORMANCE BENCHMARKS")
    print("="*50)
    
    try:
        config = ProcessingConfig(
            patch_size=256,
            encoder_name="resnet50",
            batch_size=16,
        )
        
        results = run_performance_benchmarks(
            config=config,
            quick_mode=quick_mode,
        )
        
        # Check if any benchmarks passed
        passed_benchmarks = 0
        total_benchmarks = 0
        
        if "summary" in results:
            summary = results["summary"]
            passed_benchmarks = summary.get("passed_tests", 0)
            total_benchmarks = summary.get("total_tests", 0)
        
        print(f"\n📊 Performance benchmark results: {passed_benchmarks}/{total_benchmarks} tests passed")
        
        if passed_benchmarks > 0:
            print("✅ Performance benchmarks completed successfully")
            return True
        else:
            print("⚠️ Performance benchmarks completed but no tests passed")
            return False
            
    except Exception as e:
        print(f"❌ Performance benchmark test failed: {e}")
        return False


def test_comprehensive_validation() -> bool:
    """Test comprehensive validation."""
    print("\n" + "="*50)
    print("TESTING COMPREHENSIVE VALIDATION")
    print("="*50)
    
    try:
        config = ProcessingConfig(
            patch_size=256,
            encoder_name="resnet50",
            batch_size=8,
        )
        
        results = run_comprehensive_validation(config=config)
        
        # Check overall validation status
        if "overall_summary" in results:
            summary = results["overall_summary"]
            validation_passed = summary.get("validation_passed", False)
            passed_tests = summary.get("passed_tests", 0)
            total_tests = summary.get("total_tests", 0)
            
            print(f"\n🔍 Validation results: {passed_tests}/{total_tests} tests passed")
            
            if validation_passed:
                print("✅ Comprehensive validation passed")
                return True
            else:
                print("⚠️ Comprehensive validation completed with some failures")
                return False
        else:
            print("⚠️ Validation completed but no summary available")
            return False
            
    except Exception as e:
        print(f"❌ Comprehensive validation test failed: {e}")
        return False


def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(description="Test WSI Processing Pipeline")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick tests only")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip performance benchmarks")
    parser.add_argument("--skip-validation", action="store_true", help="Skip comprehensive validation")
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    print("WSI PROCESSING PIPELINE INTEGRATION TEST")
    print("="*60)
    print(f"Quick mode: {args.quick}")
    print(f"Skip benchmarks: {args.skip_benchmarks}")
    print(f"Skip validation: {args.skip_validation}")
    
    # Track test results
    test_results = []
    
    # Test 1: Basic functionality
    test_results.append(("Basic Functionality", test_basic_functionality()))
    
    # Test 2: Performance benchmarks (optional)
    if not args.skip_benchmarks:
        test_results.append(("Performance Benchmarks", test_performance_benchmarks(args.quick)))
    
    # Test 3: Comprehensive validation (optional)
    if not args.skip_validation:
        test_results.append(("Comprehensive Validation", test_comprehensive_validation()))
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results:
        status = "PASS" if passed else "FAIL"
        icon = "✅" if passed else "❌"
        print(f"{icon} {test_name}: {status}")
        if passed:
            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! WSI Processing Pipeline is ready for use.")
        return 0
    else:
        print("⚠️ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())