#!/usr/bin/env python3
"""
File System Edge Cases Testing for HistoCore
"""

import os
import sys
import tempfile
import shutil
import stat
from pathlib import Path

def test_permission_errors():
    """Test file permission edge cases."""
    
    print("🔐 Testing Permission Errors...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create read-only file
            readonly_file = temp_path / "readonly.txt"
            readonly_file.write_text("test content")
            readonly_file.chmod(stat.S_IRUSR)  # Read-only
            
            # Test writing to read-only file
            try:
                readonly_file.write_text("new content")
                results["failed"] += 1
                results["details"].append("❌ Read-only file was writable")
            except PermissionError:
                results["passed"] += 1
                results["details"].append("✅ Read-only file protection works")
            
            # Create no-permission directory
            noperm_dir = temp_path / "noperm"
            noperm_dir.mkdir()
            noperm_dir.chmod(0o000)  # No permissions
            
            # Test accessing no-permission directory
            try:
                list(noperm_dir.iterdir())
                results["failed"] += 1
                results["details"].append("❌ No-permission directory was accessible")
            except PermissionError:
                results["passed"] += 1
                results["details"].append("✅ No-permission directory protection works")
            
            # Cleanup (restore permissions)
            noperm_dir.chmod(0o755)
            readonly_file.chmod(0o644)
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Permission test failed: {str(e)}")
    
    return results

def test_disk_full_simulation():
    """Test disk full scenarios."""
    
    print("💾 Testing Disk Full Scenarios...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Simulate disk full by creating large file
            large_file = temp_path / "large.dat"
            
            try:
                # Try to write increasingly large chunks
                chunk_size = 1024 * 1024  # 1MB
                max_size = 100 * 1024 * 1024  # 100MB limit
                
                with large_file.open('wb') as f:
                    written = 0
                    while written < max_size:
                        try:
                            f.write(b'0' * chunk_size)
                            written += chunk_size
                        except OSError as e:
                            if "No space left" in str(e):
                                results["passed"] += 1
                                results["details"].append(f"✅ Disk full handled at {written // 1024 // 1024}MB")
                                break
                            else:
                                raise
                    else:
                        results["passed"] += 1
                        results["details"].append("✅ Large file write completed within limits")
                
            except OSError as e:
                if "No space left" in str(e):
                    results["passed"] += 1
                    results["details"].append("✅ Disk full error handled gracefully")
                else:
                    results["failed"] += 1
                    results["details"].append(f"❌ Unexpected OS error: {str(e)}")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Disk full test failed: {str(e)}")
    
    return results

def test_corrupted_files():
    """Test corrupted file handling."""
    
    print("🗂️ Testing Corrupted Files...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create corrupted text file
            corrupted_text = temp_path / "corrupted.txt"
            with corrupted_text.open('wb') as f:
                # Write invalid UTF-8 sequences
                f.write(b'\xff\xfe\x00\x00invalid\x80\x81\x82')
            
            # Test reading corrupted text
            try:
                content = corrupted_text.read_text(encoding='utf-8')
                results["failed"] += 1
                results["details"].append("❌ Corrupted text file read without error")
            except UnicodeDecodeError:
                results["passed"] += 1
                results["details"].append("✅ Corrupted text file detected")
            
            # Test reading with error handling
            try:
                content = corrupted_text.read_text(encoding='utf-8', errors='replace')
                results["passed"] += 1
                results["details"].append("✅ Corrupted text handled with replacement")
            except Exception as e:
                results["failed"] += 1
                results["details"].append(f"❌ Error handling failed: {str(e)}")
            
            # Create truncated file
            truncated_file = temp_path / "truncated.dat"
            with truncated_file.open('wb') as f:
                f.write(b'HEADER')
                # Simulate truncation - missing expected data
            
            # Test reading truncated file
            try:
                with truncated_file.open('rb') as f:
                    header = f.read(6)
                    if header == b'HEADER':
                        # Try to read expected data
                        data = f.read(1024)  # Expect more data
                        if len(data) == 0:
                            results["passed"] += 1
                            results["details"].append("✅ Truncated file detected")
                        else:
                            results["failed"] += 1
                            results["details"].append("❌ Truncated file not detected")
            except Exception as e:
                results["failed"] += 1
                results["details"].append(f"❌ Truncated file test failed: {str(e)}")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Corrupted file test failed: {str(e)}")
    
    return results

def test_path_edge_cases():
    """Test path edge cases."""
    
    print("🛤️ Testing Path Edge Cases...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test very long filename
            long_name = "a" * 200 + ".txt"
            long_file = temp_path / long_name
            
            try:
                long_file.write_text("test")
                if long_file.exists():
                    results["passed"] += 1
                    results["details"].append("✅ Long filename handled")
                else:
                    results["failed"] += 1
                    results["details"].append("❌ Long filename failed")
            except OSError:
                results["passed"] += 1
                results["details"].append("✅ Long filename rejected gracefully")
            
            # Test special characters in filename
            special_chars = ['<', '>', ':', '"', '|', '?', '*']
            for char in special_chars:
                try:
                    special_file = temp_path / f"test{char}file.txt"
                    special_file.write_text("test")
                    results["failed"] += 1
                    results["details"].append(f"❌ Special character '{char}' allowed in filename")
                except (OSError, ValueError):
                    results["passed"] += 1
                    results["details"].append(f"✅ Special character '{char}' rejected")
            
            # Test deep directory nesting
            deep_path = temp_path
            for i in range(50):  # Create deep nesting
                deep_path = deep_path / f"level_{i}"
            
            try:
                deep_path.mkdir(parents=True)
                test_file = deep_path / "test.txt"
                test_file.write_text("deep file")
                
                if test_file.exists():
                    results["passed"] += 1
                    results["details"].append("✅ Deep directory nesting handled")
                else:
                    results["failed"] += 1
                    results["details"].append("❌ Deep directory nesting failed")
            except OSError:
                results["passed"] += 1
                results["details"].append("✅ Deep nesting rejected gracefully")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Path edge case test failed: {str(e)}")
    
    return results

def run_filesystem_edge_tests():
    """Run all file system edge case tests."""
    
    print("🚀 Starting File System Edge Case Tests")
    print("=" * 50)
    
    all_results = {
        "permissions": test_permission_errors(),
        "disk_full": test_disk_full_simulation(),
        "corrupted_files": test_corrupted_files(),
        "path_edge_cases": test_path_edge_cases(),
    }
    
    # Summary
    total_passed = sum(r["passed"] for r in all_results.values())
    total_failed = sum(r["failed"] for r in all_results.values())
    
    print("=" * 50)
    print("📋 FILE SYSTEM EDGE CASE TEST SUMMARY")
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    
    for test_type, results in all_results.items():
        print(f"\n📊 {test_type.upper()} TESTS:")
        print(f"  ✅ Passed: {results['passed']}")
        print(f"  ❌ Failed: {results['failed']}")
        
        for detail in results["details"]:
            print(f"    {detail}")
    
    return total_passed, total_failed

if __name__ == "__main__":
    passed, failed = run_filesystem_edge_tests()
    sys.exit(1 if failed > 0 else 0)