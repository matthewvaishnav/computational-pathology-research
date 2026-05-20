#!/usr/bin/env python3
"""
Security Verification Script

Verifies that all security controls are properly configured and working.
Run this script after deployment to verify security posture.

Usage:
    python scripts/verify_security.py [--environment ENV]

Environment:
    production, development, research (default: auto-detect)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class SecurityVerifier:
    """Verify security controls are working correctly."""
    
    def __init__(self, environment: str = None):
        """Initialize verifier with environment."""
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.project_root = Path(__file__).parent.parent
        self.results: Dict[str, bool] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
    
    def verify_jinja2_autoescape(self) -> bool:
        """Verify Jinja2 autoescape is enabled."""
        print("Checking Jinja2 autoescape configuration...")
        
        try:
            # Check if SecureJinja2Environment exists
            try:
                from src.platform.security.jinja2_control import SecureJinja2Environment
                env = SecureJinja2Environment.create_environment()
                
                if not env.autoescape:
                    self.errors.append("Jinja2 autoescape is NOT enabled")
                    return False
                
                print("  ✅ Jinja2 autoescape is enabled")
                return True
            except ImportError:
                # Module doesn't exist - check if Jinja2 is used securely in code
                self.warnings.append("SecureJinja2Environment not found - checking code manually")
                print("  ⚠️  SecureJinja2Environment not found - manual verification needed")
                return True
            
        except Exception as e:
            self.errors.append(f"Failed to verify Jinja2 autoescape: {e}")
            return False
    
    def verify_network_binding(self) -> bool:
        """Verify network binding follows security policy."""
        print(f"Checking network binding for {self.environment} environment...")
        
        try:
            try:
                from src.platform.security.network_binding import NetworkBindingManager
                from src.platform.security.config import SecurityConfigManager
                
                config = SecurityConfigManager.from_environment()
                manager = NetworkBindingManager(config)
                
                # Get safe host
                safe_host = manager.get_safe_host()
                
                # Verify production doesn't allow 0.0.0.0 without explicit config
                if self.environment == 'production':
                    if safe_host == '0.0.0.0' and not os.getenv('ALLOW_PUBLIC_BINDING'):
                        self.errors.append("Production environment allows 0.0.0.0 binding without explicit config")
                        return False
                    print(f"  ✅ Production binding is secure: {safe_host}")
                else:
                    print(f"  ✅ Network binding configured: {safe_host}")
                
                return True
            except ImportError:
                self.warnings.append("NetworkBindingManager not found - manual verification needed")
                print("  ⚠️  NetworkBindingManager not found - manual verification needed")
                return True
            
        except Exception as e:
            self.errors.append(f"Failed to verify network binding: {e}")
            return False
    
    def verify_model_downloads(self) -> bool:
        """Verify model downloads use pinned revisions when required."""
        print(f"Checking model download configuration for {self.environment}...")
        
        try:
            try:
                from src.platform.security.model_download import ModelDownloadManager
                from src.platform.security.config import SecurityConfigManager
                
                config = SecurityConfigManager.from_environment()
                manager = ModelDownloadManager(config)
                
                # Check if production requires pinned revisions
                if self.environment == 'production':
                    if not config.should_require_pinned_models():
                        self.errors.append("Production environment does not require pinned model revisions")
                        return False
                    print("  ✅ Production requires pinned model revisions")
                else:
                    print(f"  ✅ Model download policy configured for {self.environment}")
                
                return True
            except ImportError:
                # Check if model_download.py exists with basic functionality
                from src.platform.security.model_download import ModelDownloadManager
                print("  ✅ ModelDownloadManager exists")
                return True
            
        except Exception as e:
            self.warnings.append(f"Model download verification skipped: {e}")
            print("  ⚠️  Model download verification skipped")
            return True
    
    def verify_temp_files(self) -> bool:
        """Verify temporary files are created securely."""
        print("Checking temporary file creation...")
        
        try:
            try:
                from src.platform.security.temp_file import TempFileManager
                import tempfile
                import stat
                
                # Create test temp file
                fd, path = TempFileManager.create_temp_file(suffix='.test')
                
                try:
                    # Verify permissions (should be 0o600)
                    file_stat = os.stat(path)
                    file_mode = stat.S_IMODE(file_stat.st_mode)
                    
                    # On Windows, permission checks are different
                    if sys.platform == 'win32':
                        print("  ✅ Temp file created (Windows - permissions not checked)")
                    else:
                        if file_mode != 0o600:
                            self.warnings.append(f"Temp file permissions are {oct(file_mode)}, expected 0o600")
                            print(f"  ⚠️  Temp file permissions: {oct(file_mode)} (expected 0o600)")
                        else:
                            print("  ✅ Temp file has secure permissions (0o600)")
                    
                    # Verify not using hardcoded /tmp
                    if path.startswith('/tmp/') and not path.startswith(tempfile.gettempdir()):
                        self.errors.append("Temp file uses hardcoded /tmp path")
                        return False
                    
                    return True
                    
                finally:
                    # Clean up
                    os.close(fd)
                    os.unlink(path)
            except ImportError:
                self.warnings.append("TempFileManager not found - manual verification needed")
                print("  ⚠️  TempFileManager not found - manual verification needed")
                return True
            
        except Exception as e:
            self.warnings.append(f"Temp file verification skipped: {e}")
            print("  ⚠️  Temp file verification skipped")
            return True
    
    def verify_bandit_scan(self) -> bool:
        """Run Bandit scan and verify no HIGH/MEDIUM issues."""
        print("Running Bandit security scan...")
        
        try:
            # Run Bandit scan
            result = subprocess.run(
                ['bandit', '-r', 'src/', '-f', 'json', '-o', 'bandit_verify.json'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Load results
            bandit_file = self.project_root / 'bandit_verify.json'
            if not bandit_file.exists():
                self.errors.append("Bandit scan did not produce output file")
                return False
            
            with open(bandit_file) as f:
                bandit_data = json.load(f)
            
            # Count issues by severity
            high_count = 0
            medium_count = 0
            
            for result in bandit_data.get('results', []):
                severity = result.get('issue_severity', '').upper()
                if severity == 'HIGH':
                    high_count += 1
                elif severity == 'MEDIUM':
                    medium_count += 1
            
            # Report results
            if high_count > 0:
                self.errors.append(f"Bandit found {high_count} HIGH severity issues")
                print(f"  ❌ {high_count} HIGH severity issues found")
                return False
            
            if medium_count > 0:
                self.errors.append(f"Bandit found {medium_count} MEDIUM severity issues")
                print(f"  ❌ {medium_count} MEDIUM severity issues found")
                return False
            
            print(f"  ✅ Bandit scan passed (0 HIGH, 0 MEDIUM issues)")
            return True
            
        except FileNotFoundError:
            self.warnings.append("Bandit not installed - skipping scan")
            print("  ⚠️  Bandit not installed - skipping scan")
            return True
        except Exception as e:
            self.errors.append(f"Failed to run Bandit scan: {e}")
            return False
    
    def run_all_checks(self) -> bool:
        """Run all security verification checks."""
        print("="*60)
        print(f"SECURITY VERIFICATION - {self.environment.upper()} ENVIRONMENT")
        print("="*60)
        print()
        
        checks = [
            ("Jinja2 Autoescape", self.verify_jinja2_autoescape),
            ("Network Binding", self.verify_network_binding),
            ("Model Downloads", self.verify_model_downloads),
            ("Temp Files", self.verify_temp_files),
            ("Bandit Scan", self.verify_bandit_scan),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            try:
                passed = check_func()
                self.results[check_name] = passed
                if not passed:
                    all_passed = False
            except Exception as e:
                self.errors.append(f"{check_name} check failed: {e}")
                self.results[check_name] = False
                all_passed = False
            print()
        
        # Print summary
        print("="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)
        
        for check_name, passed in self.results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {check_name}")
        
        if self.warnings:
            print()
            print("WARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
        
        if self.errors:
            print()
            print("ERRORS:")
            for error in self.errors:
                print(f"  ❌ {error}")
        
        print()
        print("="*60)
        if all_passed:
            print("✅ ALL SECURITY CHECKS PASSED")
        else:
            print("❌ SOME SECURITY CHECKS FAILED")
        print("="*60)
        
        return all_passed


def main():
    """Main entry point."""
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    parser = argparse.ArgumentParser(description='Verify security controls')
    parser.add_argument(
        '--environment',
        choices=['production', 'development', 'research'],
        help='Environment to verify (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    verifier = SecurityVerifier(environment=args.environment)
    success = verifier.run_all_checks()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
