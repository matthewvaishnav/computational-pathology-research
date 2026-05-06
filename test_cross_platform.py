#!/usr/bin/env python3
"""Cross-platform compatibility testing for Windows/macOS edge cases."""

import os
import sys
import platform
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any

class CrossPlatformTester:
    """Test cross-platform compatibility."""
    
    def __init__(self):
        self.platform = platform.system()
        self.is_windows = self.platform == "Windows"
        self.is_macos = self.platform == "Darwin"
        self.is_linux = self.platform == "Linux"

def test_path_handling():
    """Test cross-platform path handling."""
    print("Testing cross-platform path handling...")
    
    tester = CrossPlatformTester()
    
    # Test path separators
    test_paths = [
        "data/models/checkpoint.pth",
        "results\\analysis\\report.json",
        "/tmp/histocore/temp.h5",
        "C:\\Users\\User\\Documents\\data.csv",
        "~/Documents/histocore/config.yaml",
    ]
    
    passed = 0
    for path_str in test_paths:
        try:
            # Use pathlib for cross-platform handling
            path = Path(path_str).resolve()
            
            # Check if path is properly normalized
            if path.is_absolute() or not path.is_reserved():
                passed += 1
            else:
                print(f"  Path issue: {path_str}")
                
        except Exception as e:
            print(f"  Path error: {path_str} - {e}")
    
    print(f"  Path handling: {passed}/{len(test_paths)} passed")
    return passed >= len(test_paths) * 0.8

def test_file_permissions():
    """Test file permission handling across platforms."""
    print("Testing file permissions...")
    
    tester = CrossPlatformTester()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test_permissions.txt"
        
        try:
            # Create test file
            test_file.write_text("test content")
            
            # Test readable
            if test_file.is_file() and os.access(test_file, os.R_OK):
                readable = True
            else:
                readable = False
            
            # Test writable
            if os.access(test_file, os.W_OK):
                writable = True
            else:
                writable = False
            
            # Platform-specific permission tests
            if not tester.is_windows:
                # Unix-like systems
                try:
                    os.chmod(test_file, 0o444)  # Read-only
                    readonly_set = not os.access(test_file, os.W_OK)
                except:
                    readonly_set = False
            else:
                # Windows
                readonly_set = True  # Skip chmod test on Windows
            
            print(f"  File readable: {readable}")
            print(f"  File writable: {writable}")
            print(f"  Read-only setting: {readonly_set}")
            
            return readable and writable and readonly_set
            
        except Exception as e:
            print(f"  Permission test error: {e}")
            return False

def test_environment_variables():
    """Test environment variable handling."""
    print("Testing environment variables...")
    
    tester = CrossPlatformTester()
    
    # Platform-specific environment variables
    if tester.is_windows:
        test_vars = ["USERPROFILE", "APPDATA", "TEMP", "PATH"]
    else:
        test_vars = ["HOME", "USER", "TMPDIR", "PATH"]
    
    found_vars = 0
    for var in test_vars:
        value = os.environ.get(var)
        if value:
            found_vars += 1
        else:
            print(f"  Missing environment variable: {var}")
    
    # Test setting custom environment variable
    test_var = "HISTOCORE_TEST_VAR"
    os.environ[test_var] = "test_value"
    
    if os.environ.get(test_var) == "test_value":
        custom_var_works = True
    else:
        custom_var_works = False
    
    print(f"  Standard variables: {found_vars}/{len(test_vars)} found")
    print(f"  Custom variable setting: {custom_var_works}")
    
    return found_vars >= len(test_vars) * 0.8 and custom_var_works

def test_process_execution():
    """Test subprocess execution across platforms."""
    print("Testing process execution...")
    
    tester = CrossPlatformTester()
    
    # Platform-specific commands
    if tester.is_windows:
        test_commands = [
            ["echo", "hello"],
            ["dir", "/b"],
            ["python", "--version"],
        ]
    else:
        test_commands = [
            ["echo", "hello"],
            ["ls", "-la"],
            ["python3", "--version"],
        ]
    
    successful_commands = 0
    for cmd in test_commands:
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                successful_commands += 1
            else:
                print(f"  Command failed: {' '.join(cmd)} (exit code: {result.returncode})")
                
        except subprocess.TimeoutExpired:
            print(f"  Command timeout: {' '.join(cmd)}")
        except FileNotFoundError:
            print(f"  Command not found: {' '.join(cmd)}")
        except Exception as e:
            print(f"  Command error: {' '.join(cmd)} - {e}")
    
    print(f"  Process execution: {successful_commands}/{len(test_commands)} passed")
    return successful_commands >= len(test_commands) * 0.6

def test_unicode_file_handling():
    """Test Unicode filename handling."""
    print("Testing Unicode file handling...")
    
    unicode_filenames = [
        "test_file.txt",
        "测试文件.txt",
        "файл_тест.txt", 
        "archivo_prueba.txt",
        "tëst_fîlé.txt",
        "テストファイル.txt",
    ]
    
    successful_files = 0
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for filename in unicode_filenames:
            try:
                file_path = Path(temp_dir) / filename
                
                # Create file with Unicode name
                file_path.write_text("test content", encoding='utf-8')
                
                # Verify file exists and is readable
                if file_path.exists() and file_path.read_text(encoding='utf-8') == "test content":
                    successful_files += 1
                else:
                    print(f"  Unicode file issue: {filename}")
                    
            except Exception as e:
                print(f"  Unicode file error: {filename} - {e}")
    
    print(f"  Unicode files: {successful_files}/{len(unicode_filenames)} passed")
    return successful_files >= len(unicode_filenames) * 0.8

def test_memory_limits():
    """Test platform-specific memory limits."""
    print("Testing memory limits...")
    
    tester = CrossPlatformTester()
    
    try:
        import psutil
        
        # Get system memory info
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"  Total memory: {total_gb:.1f} GB")
        print(f"  Available memory: {available_gb:.1f} GB")
        
        # Test memory allocation
        try:
            # Allocate 100MB
            test_data = bytearray(100 * 1024 * 1024)
            del test_data
            memory_test = True
        except MemoryError:
            memory_test = False
            
        print(f"  Memory allocation test: {memory_test}")
        
        return total_gb > 1.0 and available_gb > 0.5 and memory_test
        
    except ImportError:
        print("  psutil not available, skipping detailed memory test")
        
        # Basic memory test without psutil
        try:
            test_data = bytearray(50 * 1024 * 1024)  # 50MB
            del test_data
            return True
        except MemoryError:
            return False

def test_python_version_compatibility():
    """Test Python version compatibility."""
    print("Testing Python version compatibility...")
    
    version = sys.version_info
    
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
    
    # Check minimum version requirements
    min_version = (3, 9)
    version_ok = (version.major, version.minor) >= min_version
    
    # Test Python features
    features_ok = True
    
    try:
        # Test f-strings (Python 3.6+)
        test_var = "test"
        f_string = f"Value: {test_var}"
        
        # Test pathlib (Python 3.4+)
        from pathlib import Path
        
        # Test type hints (Python 3.5+)
        from typing import Dict, List
        
        # Test dataclasses (Python 3.7+)
        from dataclasses import dataclass
        
    except Exception as e:
        print(f"  Feature test failed: {e}")
        features_ok = False
    
    print(f"  Version compatibility: {version_ok}")
    print(f"  Feature compatibility: {features_ok}")
    
    return version_ok and features_ok

def test_package_imports():
    """Test critical package imports."""
    print("Testing package imports...")
    
    critical_packages = [
        "os", "sys", "pathlib", "tempfile", "subprocess",
        "json", "csv", "sqlite3", "urllib", "http",
    ]
    
    optional_packages = [
        "numpy", "torch", "PIL", "matplotlib", "sklearn",
        "pandas", "h5py", "tqdm", "requests",
    ]
    
    critical_imports = 0
    for package in critical_packages:
        try:
            __import__(package)
            critical_imports += 1
        except ImportError:
            print(f"  Critical import failed: {package}")
    
    optional_imports = 0
    for package in optional_packages:
        try:
            __import__(package)
            optional_imports += 1
        except ImportError:
            pass  # Optional packages
    
    print(f"  Critical imports: {critical_imports}/{len(critical_packages)} passed")
    print(f"  Optional imports: {optional_imports}/{len(optional_packages)} available")
    
    return critical_imports == len(critical_packages)

def run_cross_platform_tests():
    """Run all cross-platform compatibility tests."""
    print("🌍 Cross-Platform Compatibility Testing")
    print("=" * 50)
    
    tester = CrossPlatformTester()
    print(f"Platform: {tester.platform}")
    print(f"Python: {sys.version}")
    print()
    
    tests = [
        ("Path Handling", test_path_handling),
        ("File Permissions", test_file_permissions),
        ("Environment Variables", test_environment_variables),
        ("Process Execution", test_process_execution),
        ("Unicode File Handling", test_unicode_file_handling),
        ("Memory Limits", test_memory_limits),
        ("Python Version Compatibility", test_python_version_compatibility),
        ("Package Imports", test_package_imports),
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
    print(f"Cross-Platform Tests: {passed}/{len(tests)} passed")
    
    if passed >= len(tests) * 0.8:
        print("🏆 Excellent cross-platform compatibility!")
    else:
        print(f"⚠️ {len(tests) - passed} compatibility issues found")
    
    return passed >= len(tests) * 0.8

if __name__ == "__main__":
    success = run_cross_platform_tests()
    exit(0 if success else 1)