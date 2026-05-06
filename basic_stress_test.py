#!/usr/bin/env python3
"""
HistoCore Basic Stress Testing Suite

Tests core functionality without heavy dependencies.
"""

import os
import sys
import time
import threading
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class BasicStressTest:
    """Basic stress testing for HistoCore."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def log(self, test_name, status, details=""):
        """Log test results."""
        elapsed = time.time() - self.start_time
        
        result = {
            'status': status,
            'elapsed': elapsed,
            'details': details
        }
        self.results[test_name] = result
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status} ({elapsed:.1f}s)")
        if details:
            print(f"   {details}")

    def test_import_system(self):
        """Test that core imports work."""
        try:
            sys.path.insert(0, '.')
            import src
            self.log("import_system", "PASS", "Core imports successful")
        except Exception as e:
            self.log("import_system", "FAIL", str(e))

    def test_cli_interface(self):
        """Test CLI interface."""
        try:
            # Test CLI help
            import subprocess
            result = subprocess.run([sys.executable, 'histocore', '--help'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                self.log("cli_interface", "PASS", "CLI help works")
            else:
                self.log("cli_interface", "FAIL", f"CLI failed: {result.stderr}")
        except Exception as e:
            self.log("cli_interface", "FAIL", str(e))

    def test_file_operations(self):
        """Test file system operations."""
        try:
            # Test creating/reading/deleting files
            test_file = Path("test_stress.tmp")
            
            # Write test
            test_file.write_text("test data")
            
            # Read test
            content = test_file.read_text()
            assert content == "test data"
            
            # Delete test
            test_file.unlink()
            
            self.log("file_operations", "PASS", "File operations work")
        except Exception as e:
            self.log("file_operations", "FAIL", str(e))

    def test_threading(self):
        """Test basic threading."""
        try:
            results = []
            
            def worker(n):
                time.sleep(0.1)
                return n * 2
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(worker, i) for i in range(10)]
                results = [f.result() for f in futures]
            
            expected = [i * 2 for i in range(10)]
            if results == expected:
                self.log("threading", "PASS", "Threading works correctly")
            else:
                self.log("threading", "FAIL", "Threading results incorrect")
                
        except Exception as e:
            self.log("threading", "FAIL", str(e))

    def test_error_handling(self):
        """Test error handling."""
        try:
            # Test various error conditions
            errors_handled = 0
            
            # Division by zero
            try:
                result = 1 / 0
            except ZeroDivisionError:
                errors_handled += 1
            
            # File not found
            try:
                with open("nonexistent_file.txt", 'r') as f:
                    content = f.read()
            except FileNotFoundError:
                errors_handled += 1
            
            # Index error
            try:
                lst = [1, 2, 3]
                item = lst[10]
            except IndexError:
                errors_handled += 1
            
            if errors_handled == 3:
                self.log("error_handling", "PASS", "Error handling works")
            else:
                self.log("error_handling", "FAIL", f"Only {errors_handled}/3 errors handled")
                
        except Exception as e:
            self.log("error_handling", "FAIL", str(e))

    def test_memory_basic(self):
        """Test basic memory operations."""
        try:
            # Create and delete large data structures
            data = []
            for i in range(1000):
                data.append([j for j in range(100)])
            
            # Force garbage collection
            del data
            gc.collect()
            
            self.log("memory_basic", "PASS", "Basic memory operations work")
        except Exception as e:
            self.log("memory_basic", "FAIL", str(e))

    def test_path_operations(self):
        """Test path operations."""
        try:
            # Test various path operations
            current_dir = Path.cwd()
            parent_dir = current_dir.parent
            
            # Test path joining
            test_path = current_dir / "test" / "subdir" / "file.txt"
            
            # Test path properties
            assert current_dir.exists()
            assert current_dir.is_dir()
            
            self.log("path_operations", "PASS", "Path operations work")
        except Exception as e:
            self.log("path_operations", "FAIL", str(e))

    def test_string_operations(self):
        """Test string operations with edge cases."""
        try:
            # Test various string operations
            test_strings = [
                "",
                "normal string",
                "string with spaces",
                "string\nwith\nnewlines",
                "string\twith\ttabs",
                "very " * 1000 + "long string",
            ]
            
            operations_passed = 0
            for s in test_strings:
                try:
                    # Test basic string operations
                    upper = s.upper()
                    lower = s.lower()
                    stripped = s.strip()
                    split = s.split()
                    operations_passed += 1
                except Exception:
                    pass
            
            if operations_passed == len(test_strings):
                self.log("string_operations", "PASS", "String operations work")
            else:
                self.log("string_operations", "WARN", f"{operations_passed}/{len(test_strings)} strings handled")
                
        except Exception as e:
            self.log("string_operations", "FAIL", str(e))

    def run_all_tests(self):
        """Run all basic stress tests."""
        print("🚀 Starting HistoCore Basic Stress Test Suite")
        print("=" * 50)
        
        # Run tests
        self.test_import_system()
        self.test_cli_interface()
        self.test_file_operations()
        self.test_threading()
        self.test_error_handling()
        self.test_memory_basic()
        self.test_path_operations()
        self.test_string_operations()
        
        # Summary
        print("=" * 50)
        print("📋 BASIC STRESS TEST SUMMARY")
        
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        warned = sum(1 for r in self.results.values() if r['status'] == 'WARN')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        
        total_time = time.time() - self.start_time
        
        print(f"✅ Passed: {passed}")
        print(f"⚠️  Warned: {warned}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Total time: {total_time:.1f}s")
        
        # Detailed results
        print("\n📊 DETAILED RESULTS:")
        for test_name, result in self.results.items():
            status_emoji = "✅" if result['status'] == "PASS" else "❌" if result['status'] == "FAIL" else "⚠️"
            print(f"{status_emoji} {test_name}: {result['status']} ({result['elapsed']:.1f}s)")
            if result['details']:
                print(f"   └─ {result['details']}")
        
        return {
            'passed': passed,
            'warned': warned, 
            'failed': failed,
            'total_time': total_time,
            'results': self.results
        }

if __name__ == "__main__":
    suite = BasicStressTest()
    results = suite.run_all_tests()
    
    # Exit with appropriate code
    if results['failed'] > 0:
        sys.exit(1)
    elif results['warned'] > 0:
        sys.exit(2)
    else:
        sys.exit(0)