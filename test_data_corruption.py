#!/usr/bin/env python3
"""Data corruption detection and recovery tests."""

import hashlib
import tempfile
import random
from pathlib import Path

class DataIntegrityChecker:
    """Data integrity checking utilities."""
    
    @staticmethod
    def calculate_checksum(data: bytes) -> str:
        """Calculate SHA-256 checksum."""
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def verify_checksum(data: bytes, expected_checksum: str) -> bool:
        """Verify data integrity using checksum."""
        actual_checksum = DataIntegrityChecker.calculate_checksum(data)
        return actual_checksum == expected_checksum
    
    @staticmethod
    def detect_bit_flip(original: bytes, corrupted: bytes) -> list:
        """Detect bit flips between original and corrupted data."""
        if len(original) != len(corrupted):
            return ["Length mismatch"]
        
        differences = []
        for i, (orig_byte, corr_byte) in enumerate(zip(original, corrupted)):
            if orig_byte != corr_byte:
                differences.append(f"Byte {i}: {orig_byte:02x} -> {corr_byte:02x}")
        
        return differences

def test_checksum_validation():
    """Test checksum-based data validation."""
    print("Testing checksum validation...")
    
    checker = DataIntegrityChecker()
    
    # Test data
    test_data = b"This is test data for integrity checking."
    checksum = checker.calculate_checksum(test_data)
    
    # Test valid data
    valid_check = checker.verify_checksum(test_data, checksum)
    
    # Test corrupted data
    corrupted_data = test_data[:-1] + b"X"  # Change last byte
    invalid_check = checker.verify_checksum(corrupted_data, checksum)
    
    print(f"  Valid data verification: {valid_check}")
    print(f"  Corrupted data detection: {not invalid_check}")
    
    return valid_check and not invalid_check

def test_file_corruption_detection():
    """Test file corruption detection."""
    print("Testing file corruption detection...")
    
    checker = DataIntegrityChecker()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test file
        test_file = Path(temp_dir) / "test_data.bin"
        original_data = b"Binary data for corruption testing" * 100
        
        test_file.write_bytes(original_data)
        original_checksum = checker.calculate_checksum(original_data)
        
        # Verify original file
        read_data = test_file.read_bytes()
        original_valid = checker.verify_checksum(read_data, original_checksum)
        
        # Simulate corruption by modifying file
        with open(test_file, "r+b") as f:
            f.seek(50)  # Go to middle of file
            f.write(b"CORRUPTED")
        
        # Check corrupted file
        corrupted_data = test_file.read_bytes()
        corrupted_detected = not checker.verify_checksum(corrupted_data, original_checksum)
        
        print(f"  Original file valid: {original_valid}")
        print(f"  Corruption detected: {corrupted_detected}")
        
        return original_valid and corrupted_detected

def test_bit_flip_detection():
    """Test single bit flip detection."""
    print("Testing bit flip detection...")
    
    checker = DataIntegrityChecker()
    
    original = b"Hello, World!"
    
    # Create single bit flip
    corrupted = bytearray(original)
    corrupted[0] ^= 0x01  # Flip least significant bit of first byte
    corrupted = bytes(corrupted)
    
    differences = checker.detect_bit_flip(original, corrupted)
    
    bit_flip_detected = len(differences) == 1
    correct_location = "Byte 0:" in differences[0] if differences else False
    
    print(f"  Bit flip detected: {bit_flip_detected}")
    print(f"  Correct location: {correct_location}")
    if differences:
        print(f"  Difference: {differences[0]}")
    
    return bit_flip_detected and correct_location

def test_data_recovery():
    """Test data recovery from corruption."""
    print("Testing data recovery...")
    
    def create_redundant_data(data: bytes, redundancy: int = 3):
        """Create redundant copies of data."""
        return [data] * redundancy
    
    def recover_data(redundant_copies: list) -> bytes:
        """Recover data using majority voting."""
        if not redundant_copies:
            return b""
        
        # Simple majority voting - return most common copy
        from collections import Counter
        counter = Counter(redundant_copies)
        return counter.most_common(1)[0][0]
    
    original_data = b"Important data that needs protection"
    
    # Create redundant copies
    copies = create_redundant_data(original_data, 5)
    
    # Corrupt some copies
    corrupted_copies = copies.copy()
    corrupted_copies[1] = b"Corrupted data 1"
    corrupted_copies[3] = b"Corrupted data 2"
    
    # Recover data
    recovered_data = recover_data(corrupted_copies)
    
    recovery_successful = recovered_data == original_data
    
    print(f"  Original copies: {len(copies)}")
    print(f"  Corrupted copies: 2")
    print(f"  Recovery successful: {recovery_successful}")
    
    return recovery_successful

def test_progressive_corruption():
    """Test detection of progressive data corruption."""
    print("Testing progressive corruption...")
    
    checker = DataIntegrityChecker()
    
    # Simulate data that gets progressively corrupted
    original_data = b"Data that will be progressively corrupted over time"
    
    corruption_stages = []
    current_data = original_data
    
    # Stage 1: Original
    corruption_stages.append((current_data, checker.calculate_checksum(current_data)))
    
    # Stage 2: Minor corruption
    current_data = current_data.replace(b"Data", b"Dxta")
    corruption_stages.append((current_data, checker.calculate_checksum(current_data)))
    
    # Stage 3: More corruption
    current_data = current_data.replace(b"will", b"wxll")
    corruption_stages.append((current_data, checker.calculate_checksum(current_data)))
    
    # Stage 4: Severe corruption
    current_data = current_data.replace(b"progressively", b"xxxxxxxxxxxx")
    corruption_stages.append((current_data, checker.calculate_checksum(current_data)))
    
    # Detect corruption progression
    original_checksum = corruption_stages[0][1]
    corruption_detected = []
    
    for i, (data, checksum) in enumerate(corruption_stages):
        is_corrupted = checksum != original_checksum
        corruption_detected.append(is_corrupted)
        print(f"  Stage {i}: {'Corrupted' if is_corrupted else 'Clean'}")
    
    # Should detect corruption in stages 1, 2, 3 but not 0
    expected_pattern = [False, True, True, True]
    pattern_correct = corruption_detected == expected_pattern
    
    print(f"  Corruption pattern correct: {pattern_correct}")
    
    return pattern_correct

def test_large_file_integrity():
    """Test integrity checking of large files."""
    print("Testing large file integrity...")
    
    checker = DataIntegrityChecker()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create large test file (1MB)
        large_file = Path(temp_dir) / "large_test.bin"
        
        # Generate pseudo-random data
        random.seed(42)  # For reproducibility
        large_data = bytes([random.randint(0, 255) for _ in range(1024 * 1024)])
        
        large_file.write_bytes(large_data)
        original_checksum = checker.calculate_checksum(large_data)
        
        # Verify large file
        read_data = large_file.read_bytes()
        large_file_valid = checker.verify_checksum(read_data, original_checksum)
        
        # Corrupt single byte in large file
        with open(large_file, "r+b") as f:
            f.seek(500000)  # Middle of file
            f.write(b"\xFF")
        
        # Check if corruption is detected
        corrupted_data = large_file.read_bytes()
        corruption_detected = not checker.verify_checksum(corrupted_data, original_checksum)
        
        print(f"  Large file size: {len(large_data)} bytes")
        print(f"  Original file valid: {large_file_valid}")
        print(f"  Single byte corruption detected: {corruption_detected}")
        
        return large_file_valid and corruption_detected

def test_network_data_integrity():
    """Test network data transmission integrity."""
    print("Testing network data integrity...")
    
    checker = DataIntegrityChecker()
    
    def simulate_network_transmission(data: bytes, error_rate: float = 0.0):
        """Simulate network transmission with potential errors."""
        if error_rate == 0.0:
            return data
        
        transmitted = bytearray(data)
        num_errors = int(len(data) * error_rate)
        
        for _ in range(num_errors):
            pos = random.randint(0, len(transmitted) - 1)
            transmitted[pos] ^= random.randint(1, 255)
        
        return bytes(transmitted)
    
    original_message = b"Network message with integrity checking"
    original_checksum = checker.calculate_checksum(original_message)
    
    # Test clean transmission
    clean_transmission = simulate_network_transmission(original_message, 0.0)
    clean_valid = checker.verify_checksum(clean_transmission, original_checksum)
    
    # Test transmission with errors
    random.seed(42)
    noisy_transmission = simulate_network_transmission(original_message, 0.05)  # 5% error rate
    noisy_detected = not checker.verify_checksum(noisy_transmission, original_checksum)
    
    print(f"  Clean transmission valid: {clean_valid}")
    print(f"  Noisy transmission detected: {noisy_detected}")
    
    return clean_valid and noisy_detected

def run_data_corruption_tests():
    """Run all data corruption tests."""
    print("🔍 Data Corruption Detection Testing")
    print("=" * 50)
    
    tests = [
        ("Checksum Validation", test_checksum_validation),
        ("File Corruption Detection", test_file_corruption_detection),
        ("Bit Flip Detection", test_bit_flip_detection),
        ("Data Recovery", test_data_recovery),
        ("Progressive Corruption", test_progressive_corruption),
        ("Large File Integrity", test_large_file_integrity),
        ("Network Data Integrity", test_network_data_integrity),
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
    print(f"Data Corruption Tests: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🏆 Excellent data corruption detection!")
    else:
        print(f"⚠️ {len(tests) - passed} corruption detection issues found")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_data_corruption_tests()
    exit(0 if success else 1)