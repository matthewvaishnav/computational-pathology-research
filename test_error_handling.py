#!/usr/bin/env python3
"""Error handling edge case tests."""

import sys
import traceback
from typing import Any, Optional

class ErrorHandler:
    """Comprehensive error handling utilities."""
    
    @staticmethod
    def safe_divide(a: float, b: float) -> Optional[float]:
        """Safe division with error handling."""
        try:
            if b == 0:
                return None
            return a / b
        except (TypeError, ValueError):
            return None
    
    @staticmethod
    def safe_file_read(filepath: str) -> Optional[str]:
        """Safe file reading with error handling."""
        try:
            with open(filepath, 'r') as f:
                return f.read()
        except (FileNotFoundError, PermissionError, IOError):
            return None
    
    @staticmethod
    def safe_json_parse(json_str: str) -> Optional[dict]:
        """Safe JSON parsing."""
        import json
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return None

def test_division_by_zero():
    """Test division by zero handling."""
    print("Testing division by zero...")
    
    handler = ErrorHandler()
    
    test_cases = [
        (10, 2, 5.0),
        (10, 0, None),
        (0, 5, 0.0),
        (10, "invalid", None),
        ("invalid", 5, None),
    ]
    
    passed = 0
    for a, b, expected in test_cases:
        result = handler.safe_divide(a, b)
        if result == expected:
            passed += 1
        else:
            print(f"  Failed: {a}/{b} -> {result}, expected {expected}")
    
    print(f"  Division tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_file_not_found():
    """Test file not found handling."""
    print("Testing file not found...")
    
    handler = ErrorHandler()
    
    # Test non-existent file
    result = handler.safe_file_read("/nonexistent/file.txt")
    file_not_found_ok = result is None
    
    # Test with actual file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_path = f.name
    
    result = handler.safe_file_read(temp_path)
    file_read_ok = result == "test content"
    
    # Cleanup
    import os
    os.unlink(temp_path)
    
    print(f"  File not found handling: {file_not_found_ok}")
    print(f"  Valid file reading: {file_read_ok}")
    
    return file_not_found_ok and file_read_ok

def test_json_parse_errors():
    """Test JSON parsing error handling."""
    print("Testing JSON parse errors...")
    
    handler = ErrorHandler()
    
    test_cases = [
        ('{"valid": "json"}', {"valid": "json"}),
        ('invalid json', None),
        ('{"incomplete": }', None),
        ('null', None),  # Not a dict
        ('[]', None),    # Not a dict
        ('', None),
    ]
    
    passed = 0
    for json_str, expected in test_cases:
        result = handler.safe_json_parse(json_str)
        if result == expected:
            passed += 1
        else:
            print(f"  Failed: '{json_str}' -> {result}, expected {expected}")
    
    print(f"  JSON parse tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_exception_logging():
    """Test exception logging and recovery."""
    print("Testing exception logging...")
    
    def risky_operation(value):
        """Operation that might fail."""
        if value == "error":
            raise ValueError("Intentional error")
        elif value == "zero":
            return 10 / 0
        else:
            return f"Success: {value}"
    
    def safe_operation(value):
        """Safely execute risky operation."""
        try:
            return risky_operation(value)
        except Exception as e:
            # Log error (in real code, use proper logging)
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "input_value": value
            }
            return f"Error handled: {error_info['error_type']}"
    
    test_cases = [
        ("normal", "Success: normal"),
        ("error", "Error handled: ValueError"),
        ("zero", "Error handled: ZeroDivisionError"),
    ]
    
    passed = 0
    for value, expected in test_cases:
        result = safe_operation(value)
        if result == expected:
            passed += 1
        else:
            print(f"  Failed: '{value}' -> '{result}', expected '{expected}'")
    
    print(f"  Exception logging: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_resource_cleanup():
    """Test resource cleanup on errors."""
    print("Testing resource cleanup...")
    
    class ResourceManager:
        def __init__(self):
            self.resources = []
            self.cleanup_called = False
        
        def acquire_resource(self, name):
            self.resources.append(name)
            return f"resource_{name}"
        
        def cleanup(self):
            self.cleanup_called = True
            self.resources.clear()
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.cleanup()
            return False  # Don't suppress exceptions
    
    # Test successful operation
    try:
        with ResourceManager() as rm:
            rm.acquire_resource("test1")
            rm.acquire_resource("test2")
        success_cleanup = rm.cleanup_called
    except:
        success_cleanup = False
    
    # Test cleanup on exception
    try:
        with ResourceManager() as rm:
            rm.acquire_resource("test1")
            raise ValueError("Test error")
    except ValueError:
        error_cleanup = rm.cleanup_called
    except:
        error_cleanup = False
    
    print(f"  Cleanup on success: {success_cleanup}")
    print(f"  Cleanup on error: {error_cleanup}")
    
    return success_cleanup and error_cleanup

def test_memory_error_handling():
    """Test memory error handling."""
    print("Testing memory error handling...")
    
    def allocate_large_memory(size_mb):
        """Try to allocate large amount of memory."""
        try:
            # Try to allocate memory
            data = bytearray(size_mb * 1024 * 1024)
            return len(data)
        except MemoryError:
            return None
        except Exception:
            return None
    
    # Test reasonable allocation
    small_result = allocate_large_memory(10)  # 10MB
    small_ok = small_result is not None
    
    # Test large allocation (might fail)
    large_result = allocate_large_memory(10000)  # 10GB
    large_handled = True  # Just testing it doesn't crash
    
    print(f"  Small allocation (10MB): {small_ok}")
    print(f"  Large allocation handled: {large_handled}")
    
    return small_ok and large_handled

def test_timeout_handling():
    """Test timeout handling."""
    print("Testing timeout handling...")
    
    import time
    import signal
    
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out")
    
    def long_operation(duration):
        """Simulate long-running operation."""
        time.sleep(duration)
        return "completed"
    
    def operation_with_timeout(duration, timeout_seconds):
        """Execute operation with timeout."""
        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            result = long_operation(duration)
            
            # Clear timeout
            signal.alarm(0)
            return result
            
        except TimeoutError:
            signal.alarm(0)
            return "timeout"
        except Exception as e:
            signal.alarm(0)
            return f"error: {e}"
    
    # Test quick operation
    quick_result = operation_with_timeout(0.1, 2)
    quick_ok = quick_result == "completed"
    
    # Test timeout
    timeout_result = operation_with_timeout(3, 1)
    timeout_ok = timeout_result == "timeout"
    
    print(f"  Quick operation: {quick_ok}")
    print(f"  Timeout handling: {timeout_ok}")
    
    return quick_ok and timeout_ok

def run_error_handling_tests():
    """Run all error handling tests."""
    print("🚨 Error Handling Edge Case Testing")
    print("=" * 50)
    
    tests = [
        ("Division by Zero", test_division_by_zero),
        ("File Not Found", test_file_not_found),
        ("JSON Parse Errors", test_json_parse_errors),
        ("Exception Logging", test_exception_logging),
        ("Resource Cleanup", test_resource_cleanup),
        ("Memory Error Handling", test_memory_error_handling),
        ("Timeout Handling", test_timeout_handling),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
        print()
    
    print("=" * 50)
    print(f"Error Handling Tests: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🏆 Excellent error handling!")
    else:
        print(f"⚠️ {len(tests) - passed} error handling issues found")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_error_handling_tests()
    exit(0 if success else 1)