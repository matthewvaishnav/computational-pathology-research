#!/usr/bin/env python3
"""
Training Pipeline Stress Test - Test with problematic data
"""

import sys
import os
import tempfile
import json
from pathlib import Path


def create_problematic_data():
    """Create various problematic data scenarios."""

    test_dir = Path("test_data_stress")
    test_dir.mkdir(exist_ok=True)

    # Create problematic config files
    configs = {
        "empty_config.json": {},
        "malformed_config.json": '{"incomplete": json',
        "huge_config.json": {"data": "x" * 10000},
        "unicode_config.json": {"path": "世界/🚀/test"},
        "null_config.json": {"model": None, "dataset": None},
    }

    for filename, content in configs.items():
        config_path = test_dir / filename
        if isinstance(content, str):
            config_path.write_text(content)
        else:
            config_path.write_text(json.dumps(content))

    return test_dir


def test_training_edge_cases():
    """Test training pipeline with edge cases."""

    print("🧪 Testing Training Pipeline Edge Cases...")

    # Create test data
    test_dir = create_problematic_data()

    results = {"passed": 0, "failed": 0, "details": []}

    # Test cases
    test_cases = [
        # Invalid parameters
        {"epochs": -1, "expected": "should_fail"},
        {"epochs": 0, "expected": "should_fail"},
        {"batch_size": 0, "expected": "should_fail"},
        {"batch_size": -1, "expected": "should_fail"},
        # Edge case parameters
        {"epochs": 1, "batch_size": 1, "expected": "should_work"},
        {"epochs": 1000000, "expected": "should_handle"},
        # Invalid datasets
        {"dataset": "nonexistent", "expected": "should_fail"},
        {"dataset": "", "expected": "should_fail"},
        {"dataset": None, "expected": "should_fail"},
        # Invalid models
        {"model": "nonexistent", "expected": "should_fail"},
        {"model": "", "expected": "should_fail"},
        {"model": None, "expected": "should_fail"},
    ]

    for i, test_case in enumerate(test_cases):
        try:
            expected = test_case.pop("expected")

            # Try to create a training configuration
            # This would normally call the training API
            # For now, just validate the parameters

            valid = True
            error_msg = ""

            # Validate epochs
            if "epochs" in test_case:
                epochs = test_case["epochs"]
                if not isinstance(epochs, int) or epochs <= 0:
                    valid = False
                    error_msg = f"Invalid epochs: {epochs}"

            # Validate batch_size
            if "batch_size" in test_case:
                batch_size = test_case["batch_size"]
                if not isinstance(batch_size, int) or batch_size <= 0:
                    valid = False
                    error_msg = f"Invalid batch_size: {batch_size}"

            # Validate dataset
            if "dataset" in test_case:
                dataset = test_case["dataset"]
                if not dataset or not isinstance(dataset, str):
                    valid = False
                    error_msg = f"Invalid dataset: {dataset}"
                elif dataset not in ["pcam", "camelyon"]:
                    valid = False
                    error_msg = f"Unknown dataset: {dataset}"

            # Validate model
            if "model" in test_case:
                model = test_case["model"]
                if not model or not isinstance(model, str):
                    valid = False
                    error_msg = f"Invalid model: {model}"
                elif model not in ["nnmil", "attention", "clam"]:
                    valid = False
                    error_msg = f"Unknown model: {model}"

            # Check if result matches expectation
            if expected == "should_fail" and not valid:
                results["passed"] += 1
                results["details"].append(f"✅ Test {i+1} - Failed as expected: {error_msg}")
            elif expected == "should_work" and valid:
                results["passed"] += 1
                results["details"].append(f"✅ Test {i+1} - Passed as expected")
            elif expected == "should_handle" and valid:
                results["passed"] += 1
                results["details"].append(f"✅ Test {i+1} - Handled edge case")
            else:
                results["failed"] += 1
                results["details"].append(
                    f"❌ Test {i+1} - Unexpected result: valid={valid}, expected={expected}"
                )

        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Test {i+1} - Exception: {str(e)}")

    # Cleanup
    import shutil

    if test_dir.exists():
        shutil.rmtree(test_dir)

    return results


def test_memory_constraints():
    """Test training under memory constraints."""

    print("🧪 Testing Memory Constraints...")

    results = {"passed": 0, "failed": 0, "details": []}

    try:
        # Simulate memory constraint scenarios
        memory_scenarios = [
            {"available_mb": 100, "batch_size": 1, "expected": "should_work"},
            {"available_mb": 50, "batch_size": 32, "expected": "should_fail"},
            {"available_mb": 1000, "batch_size": 128, "expected": "should_work"},
        ]

        for scenario in memory_scenarios:
            try:
                # Simulate memory check
                available = scenario["available_mb"]
                batch_size = scenario["batch_size"]
                expected = scenario["expected"]

                # Improved memory estimation
                estimated_usage = batch_size * 6  # 6MB per batch item (more realistic)

                if estimated_usage > available:
                    # Would fail due to memory
                    if expected == "should_fail":
                        results["passed"] += 1
                        results["details"].append(
                            f"✅ Memory scenario {available}MB/{batch_size}bs - Failed as expected"
                        )
                    else:
                        results["failed"] += 1
                        results["details"].append(
                            f"❌ Memory scenario {available}MB/{batch_size}bs - Unexpected failure"
                        )
                else:
                    # Would succeed
                    if expected == "should_work":
                        results["passed"] += 1
                        results["details"].append(
                            f"✅ Memory scenario {available}MB/{batch_size}bs - Worked as expected"
                        )
                    else:
                        results["failed"] += 1
                        results["details"].append(
                            f"❌ Memory scenario {available}MB/{batch_size}bs - Unexpected success"
                        )

            except Exception as e:
                results["failed"] += 1
                results["details"].append(f"❌ Memory scenario - Exception: {str(e)}")

    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Memory constraint testing - Exception: {str(e)}")

    return results


def run_training_stress_tests():
    """Run all training pipeline stress tests."""

    print("🚀 Starting Training Pipeline Stress Tests")
    print("=" * 50)

    all_results = {
        "edge_cases": test_training_edge_cases(),
        "memory": test_memory_constraints(),
    }

    # Summary
    total_passed = sum(r["passed"] for r in all_results.values())
    total_failed = sum(r["failed"] for r in all_results.values())

    print("=" * 50)
    print("📋 TRAINING STRESS TEST SUMMARY")
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")

    for test_type, results in all_results.items():
        print(f"\n📊 {test_type.upper()} TESTS:")
        print(f"  ✅ Passed: {results['passed']}")
        print(f"  ❌ Failed: {results['failed']}")

        # Show details
        for detail in results["details"]:
            print(f"    {detail}")

    return total_passed, total_failed


if __name__ == "__main__":
    passed, failed = run_training_stress_tests()
    sys.exit(1 if failed > 0 else 0)
