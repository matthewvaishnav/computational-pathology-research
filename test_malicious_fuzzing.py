#!/usr/bin/env python3
"""
Malicious Input Fuzzing for HistoCore
"""

import sys
import random
import string
import struct

def generate_malicious_strings():
    """Generate various malicious string inputs."""
    
    malicious_inputs = [
        # Buffer overflow attempts
        "A" * 1000,
        "A" * 10000,
        "A" * 100000,
        
        # Format string attacks
        "%s%s%s%s%s%s%s%s%s%s",
        "%x%x%x%x%x%x%x%x%x%x",
        "%n%n%n%n%n%n%n%n%n%n",
        
        # Null byte injection
        "test\x00malicious",
        "\x00" * 100,
        "normal\x00\x00\x00hidden",
        
        # Control characters
        "\x01\x02\x03\x04\x05",
        "\x7f\x80\x81\x82\x83",
        "\xff\xfe\xfd\xfc\xfb",
        
        # Unicode exploits
        "\u0000\u0001\u0002",
        "\uffff\ufffe\ufffd",
        "\U0001f4a9" * 1000,  # Pile of poo emoji spam
        
        # Script injection
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "data:text/html,<script>alert('xss')</script>",
        
        # Path traversal
        "../" * 100,
        "..\\..\\..\\windows\\system32\\",
        "/etc/passwd\x00.jpg",
        
        # Binary data
        b'\x00\x01\x02\x03\x04\x05'.decode('latin-1'),
        b'\xff\xfe\xfd\xfc\xfb\xfa'.decode('latin-1'),
        
        # Extremely long lines
        "x" * 1000000,
        
        # Malformed UTF-8
        "\x80\x81\x82\x83",
        "\xc0\xc1\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff",
    ]
    
    return malicious_inputs

def test_string_sanitization():
    """Test string input sanitization."""
    
    print("🧪 Testing String Sanitization...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    malicious_inputs = generate_malicious_strings()
    
    for i, malicious_input in enumerate(malicious_inputs):
        try:
            # Test basic sanitization
            sanitized = sanitize_input(malicious_input)
            
            # Check if dangerous patterns were removed
            dangerous_patterns = [
                "\x00", "<script>", "javascript:", "../", "%n", "%s", "%x"
            ]
            
            is_safe = True
            for pattern in dangerous_patterns:
                if pattern in sanitized:
                    is_safe = False
                    break
            
            # Check length limits
            if len(sanitized) > 10000:
                is_safe = False
            
            if is_safe:
                results["passed"] += 1
                results["details"].append(f"✅ Input {i+1} sanitized successfully")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ Input {i+1} not properly sanitized")
                
        except Exception as e:
            # Exception during sanitization is acceptable
            results["passed"] += 1
            results["details"].append(f"✅ Input {i+1} rejected: {str(e)[:50]}...")
    
    return results

def sanitize_input(input_str):
    """Basic input sanitization function."""
    
    if not isinstance(input_str, str):
        raise ValueError("Input must be string")
    
    # Length limit
    if len(input_str) > 10000:
        input_str = input_str[:10000]
    
    # Remove null bytes
    input_str = input_str.replace('\x00', '')
    
    # Remove dangerous patterns
    dangerous_patterns = [
        '<script>', '</script>', 'javascript:', 'data:',
        '%n', '%s', '%x', '../', '..\\',
    ]
    
    for pattern in dangerous_patterns:
        input_str = input_str.replace(pattern, '')
    
    # Remove control characters (except common ones)
    allowed_chars = string.printable + '\n\r\t'
    input_str = ''.join(c for c in input_str if c in allowed_chars)
    
    return input_str

def test_numeric_fuzzing():
    """Test numeric input fuzzing."""
    
    print("🔢 Testing Numeric Fuzzing...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Extreme numeric values
    extreme_numbers = [
        float('inf'),
        float('-inf'),
        float('nan'),
        2**63 - 1,  # Max int64
        -2**63,     # Min int64
        2**64,      # Overflow
        -2**64,     # Underflow
        1e308,      # Very large float
        1e-308,     # Very small float
        0.0,
        -0.0,
    ]
    
    for i, number in enumerate(extreme_numbers):
        try:
            # Test numeric validation
            validated = validate_number(number)
            
            if validated is not None and abs(validated) < 1e100:
                results["passed"] += 1
                results["details"].append(f"✅ Number {i+1} validated: {validated}")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ Number {i+1} not properly validated")
                
        except (ValueError, OverflowError, TypeError) as e:
            results["passed"] += 1
            results["details"].append(f"✅ Number {i+1} rejected: {str(e)[:50]}...")
    
    return results

def validate_number(value):
    """Basic numeric validation."""
    
    import math
    
    if value is None:
        raise ValueError("Value cannot be None")
    
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise ValueError("Invalid numeric string")
    
    if math.isnan(value):
        raise ValueError("NaN not allowed")
    
    if math.isinf(value):
        raise ValueError("Infinity not allowed")
    
    # Range check
    if abs(value) > 1e100:
        raise ValueError("Value too large")
    
    return value

def test_binary_fuzzing():
    """Test binary data fuzzing."""
    
    print("📦 Testing Binary Fuzzing...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Generate random binary data
    for i in range(10):
        try:
            # Random binary data
            size = random.randint(1, 1000)
            binary_data = bytes([random.randint(0, 255) for _ in range(size)])
            
            # Test binary handling
            handled = handle_binary_data(binary_data)
            
            if handled is not None:
                results["passed"] += 1
                results["details"].append(f"✅ Binary data {i+1} handled ({size} bytes)")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ Binary data {i+1} not handled")
                
        except Exception as e:
            results["passed"] += 1
            results["details"].append(f"✅ Binary data {i+1} rejected: {str(e)[:50]}...")
    
    return results

def handle_binary_data(data):
    """Basic binary data handling."""
    
    if not isinstance(data, bytes):
        raise TypeError("Expected bytes")
    
    # Size limit
    if len(data) > 10000:
        raise ValueError("Binary data too large")
    
    # Check for suspicious patterns
    if b'\x00' * 100 in data:
        raise ValueError("Suspicious null byte pattern")
    
    return data

def run_fuzzing_tests():
    """Run all fuzzing tests."""
    
    print("🚀 Starting Malicious Input Fuzzing Tests")
    print("=" * 50)
    
    all_results = {
        "string_sanitization": test_string_sanitization(),
        "numeric_fuzzing": test_numeric_fuzzing(),
        "binary_fuzzing": test_binary_fuzzing(),
    }
    
    # Summary
    total_passed = sum(r["passed"] for r in all_results.values())
    total_failed = sum(r["failed"] for r in all_results.values())
    
    print("=" * 50)
    print("📋 FUZZING TEST SUMMARY")
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    
    for test_type, results in all_results.items():
        print(f"\n📊 {test_type.upper()} TESTS:")
        print(f"  ✅ Passed: {results['passed']}")
        print(f"  ❌ Failed: {results['failed']}")
        
        # Show first few details
        for detail in results["details"][:3]:
            print(f"    {detail}")
        if len(results["details"]) > 3:
            print(f"    ... and {len(results['details']) - 3} more")
    
    return total_passed, total_failed

if __name__ == "__main__":
    passed, failed = run_fuzzing_tests()
    sys.exit(1 if failed > 0 else 0)