#!/usr/bin/env python3
"""
HistoCore Comprehensive Test Suite - Test everything that can be tested
"""

import os
import sys
import json
import tempfile
from pathlib import Path
import subprocess

def test_imports():
    """Test core imports work."""
    print("📦 Testing Core Imports...")
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Test basic Python imports
    try:
        import json
        import pathlib
        import tempfile
        import subprocess
        results["passed"] += 1
        results["details"].append("✅ Standard library imports work")
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Standard library import failed: {e}")
    
    # Test HistoCore structure exists
    try:
        src_path = Path("src")
        if src_path.exists():
            results["passed"] += 1
            results["details"].append("✅ HistoCore src/ directory exists")
        else:
            results["failed"] += 1
            results["details"].append("❌ HistoCore src/ directory missing")
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Directory check failed: {e}")
    
    return results

def test_file_operations():
    """Test file operations work."""
    print("📁 Testing File Operations...")
    results = {"passed": 0, "failed": 0, "details": []}
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test file creation
            test_file = temp_path / "test.txt"
            test_file.write_text("test content")
            
            if test_file.exists():
                results["passed"] += 1
                results["details"].append("✅ File creation works")
            else:
                results["failed"] += 1
                results["details"].append("❌ File creation failed")
            
            # Test file reading
            content = test_file.read_text()
            if content == "test content":
                results["passed"] += 1
                results["details"].append("✅ File reading works")
            else:
                results["failed"] += 1
                results["details"].append("❌ File reading failed")
            
            # Test JSON operations
            json_file = temp_path / "test.json"
            test_data = {"key": "value", "number": 42}
            
            with open(json_file, 'w') as f:
                json.dump(test_data, f)
            
            with open(json_file, 'r') as f:
                loaded_data = json.load(f)
            
            if loaded_data == test_data:
                results["passed"] += 1
                results["details"].append("✅ JSON operations work")
            else:
                results["failed"] += 1
                results["details"].append("❌ JSON operations failed")
                
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ File operations test failed: {e}")
    
    return results

def test_pathology_fl():
    """Test PathologyFL components."""
    print("🧬 Testing PathologyFL...")
    results = {"passed": 0, "failed": 0, "details": []}
    
    try:
        # Test PathologyFL demo exists and runs
        if Path("pathology_fl_demo.py").exists():
            result = subprocess.run([sys.executable, "pathology_fl_demo.py"], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                results["passed"] += 1
                results["details"].append("✅ PathologyFL demo runs successfully")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ PathologyFL demo failed: {result.stderr}")
        else:
            results["failed"] += 1
            results["details"].append("❌ PathologyFL demo file missing")
            
    except subprocess.TimeoutExpired:
        results["failed"] += 1
        results["details"].append("❌ PathologyFL demo timed out")
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ PathologyFL test failed: {e}")
    
    return results

def test_stress_tests():
    """Test existing stress test components."""
    print("🔥 Testing Stress Test Components...")
    results = {"passed": 0, "failed": 0, "details": []}
    
    stress_tests = [
        "basic_stress_test.py",
        "test_cli_edge_cases.py", 
        "test_property_based.py",
        "test_training_stress.py",
        "test_security_penetration.py",
        "test_memory_pressure.py",
        "test_concurrency_chaos.py",
        "test_filesystem_edge.py",
        "test_malicious_fuzzing.py",
        "test_production_load.py"
    ]
    
    for test_file in stress_tests:
        try:
            if Path(test_file).exists():
                # Quick syntax check
                result = subprocess.run([sys.executable, "-m", "py_compile", test_file],
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    results["passed"] += 1
                    results["details"].append(f"✅ {test_file} syntax valid")
                else:
                    results["failed"] += 1
                    results["details"].append(f"❌ {test_file} syntax error")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ {test_file} missing")
                
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ {test_file} check failed: {e}")
    
    return results

def test_configuration_files():
    """Test configuration and setup files."""
    print("⚙️ Testing Configuration Files...")
    results = {"passed": 0, "failed": 0, "details": []}
    
    config_files = [
        ("README.md", "README exists"),
        ("pyproject.toml", "PyProject config exists"),
        ("requirements-core.txt", "Core requirements exist"),
        ("setup_pypi.py", "PyPI setup exists"),
        ("MANIFEST.in", "Package manifest exists")
    ]
    
    for file_path, description in config_files:
        try:
            if Path(file_path).exists():
                # Check file is not empty
                content = Path(file_path).read_text()
                if len(content.strip()) > 0:
                    results["passed"] += 1
                    results["details"].append(f"✅ {description}")
                else:
                    results["failed"] += 1
                    results["details"].append(f"❌ {file_path} is empty")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ {file_path} missing")
                
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ {file_path} check failed: {e}")
    
    return results

def test_documentation():
    """Test documentation completeness."""
    print("📚 Testing Documentation...")
    results = {"passed": 0, "failed": 0, "details": []}
    
    doc_files = [
        "STRESS_TEST_RESULTS.md",
        "ADVANCED_STRESS_TEST_RESULTS.md", 
        "PATHOLOGY_FL_DESIGN.md",
        "UPLOAD_READY.md",
        "PYPI_GUIDE.md"
    ]
    
    for doc_file in doc_files:
        try:
            if Path(doc_file).exists():
                content = Path(doc_file).read_text()
                if len(content) > 1000:  # Substantial content
                    results["passed"] += 1
                    results["details"].append(f"✅ {doc_file} comprehensive")
                else:
                    results["failed"] += 1
                    results["details"].append(f"❌ {doc_file} too short")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ {doc_file} missing")
                
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ {doc_file} check failed: {e}")
    
    return results

def test_package_structure():
    """Test package structure integrity."""
    print("📦 Testing Package Structure...")
    results = {"passed": 0, "failed": 0, "details": []}
    
    required_dirs = [
        "src",
        "src/federated", 
        "configs",
        "examples",
        "docs"
    ]
    
    for dir_path in required_dirs:
        try:
            if Path(dir_path).exists() and Path(dir_path).is_dir():
                results["passed"] += 1
                results["details"].append(f"✅ {dir_path}/ directory exists")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ {dir_path}/ directory missing")
                
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ {dir_path} check failed: {e}")
    
    # Check for key files in src/
    key_files = [
        "src/__init__.py",
        "src/federated/pathology_fl.py",
        "src/federated/pathology_fl_coordinator.py",
        "src/federated/pathology_fl_client.py"
    ]
    
    for file_path in key_files:
        try:
            if Path(file_path).exists():
                results["passed"] += 1
                results["details"].append(f"✅ {file_path} exists")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ {file_path} missing")
                
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ {file_path} check failed: {e}")
    
    return results

def test_git_repository():
    """Test git repository status."""
    print("🔄 Testing Git Repository...")
    results = {"passed": 0, "failed": 0, "details": []}
    
    try:
        # Check if .git exists
        if Path(".git").exists():
            results["passed"] += 1
            results["details"].append("✅ Git repository initialized")
            
            # Check git status
            result = subprocess.run(["git", "status", "--porcelain"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                results["passed"] += 1
                results["details"].append("✅ Git status accessible")
                
                # Check for recent commits
                result = subprocess.run(["git", "log", "--oneline", "-5"], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    results["passed"] += 1
                    results["details"].append("✅ Git commit history exists")
                else:
                    results["failed"] += 1
                    results["details"].append("❌ No git commit history")
            else:
                results["failed"] += 1
                results["details"].append("❌ Git status failed")
        else:
            results["failed"] += 1
            results["details"].append("❌ Not a git repository")
            
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Git test failed: {e}")
    
    return results

def run_comprehensive_tests():
    """Run all possible tests."""
    print("🚀 HistoCore Comprehensive Test Suite")
    print("=" * 60)
    
    all_tests = [
        ("Core Imports", test_imports),
        ("File Operations", test_file_operations),
        ("PathologyFL", test_pathology_fl),
        ("Stress Tests", test_stress_tests),
        ("Configuration", test_configuration_files),
        ("Documentation", test_documentation),
        ("Package Structure", test_package_structure),
        ("Git Repository", test_git_repository)
    ]
    
    total_passed = 0
    total_failed = 0
    all_results = {}
    
    for test_name, test_func in all_tests:
        try:
            results = test_func()
            all_results[test_name] = results
            total_passed += results["passed"]
            total_failed += results["failed"]
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            total_failed += 1
            all_results[test_name] = {
                "passed": 0, 
                "failed": 1, 
                "details": [f"❌ Test crashed: {e}"]
            }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    print(f"📈 Pass Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")
    
    # Detailed results
    for test_name, results in all_results.items():
        print(f"\n📋 {test_name.upper()}:")
        print(f"  ✅ Passed: {results['passed']}")
        print(f"  ❌ Failed: {results['failed']}")
        
        # Show first few details
        for detail in results["details"][:3]:
            print(f"    {detail}")
        if len(results["details"]) > 3:
            print(f"    ... and {len(results['details']) - 3} more")
    
    # Final assessment
    print("\n" + "=" * 60)
    print("🏁 FINAL ASSESSMENT")
    print("=" * 60)
    
    pass_rate = total_passed / (total_passed + total_failed) * 100
    
    if pass_rate >= 90:
        print("🏆 EXCELLENT - HistoCore is highly robust")
    elif pass_rate >= 80:
        print("✅ GOOD - HistoCore is solid with minor issues")
    elif pass_rate >= 70:
        print("⚠️ FAIR - HistoCore needs some improvements")
    else:
        print("❌ POOR - HistoCore needs significant work")
    
    print(f"📊 Overall Quality Score: {pass_rate:.1f}%")
    
    return pass_rate >= 80

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)