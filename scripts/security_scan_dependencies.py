#!/usr/bin/env python3
"""
Dependency Security Scanner

Scans Python dependencies for known security vulnerabilities.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def run_safety_check() -> Dict:
    """Run safety check on dependencies.
    
    Returns:
        Dictionary with vulnerability results
    """
    try:
        result = subprocess.run(
            ["safety", "check", "--json"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not run safety check: {e}")
        return {}


def run_pip_audit() -> List[Dict]:
    """Run pip-audit on dependencies.
    
    Returns:
        List of vulnerability dictionaries
    """
    try:
        result = subprocess.run(
            ["pip-audit", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.stdout:
            data = json.loads(result.stdout)
            return data.get("vulnerabilities", [])
        return []
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not run pip-audit: {e}")
        return []


def check_outdated_packages() -> List[Dict]:
    """Check for outdated packages.
    
    Returns:
        List of outdated package dictionaries
    """
    try:
        result = subprocess.run(
            ["pip", "list", "--outdated", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.stdout:
            return json.loads(result.stdout)
        return []
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not check outdated packages: {e}")
        return []


def generate_report(safety_results: Dict, audit_results: List[Dict], outdated: List[Dict]) -> None:
    """Generate security report.
    
    Args:
        safety_results: Results from safety check
        audit_results: Results from pip-audit
        outdated: Outdated packages
    """
    print("\n" + "="*80)
    print("DEPENDENCY SECURITY SCAN REPORT")
    print("="*80 + "\n")
    
    # Safety check results
    if safety_results:
        vulnerabilities = safety_results.get("vulnerabilities", [])
        if vulnerabilities:
            print(f"⚠️  CRITICAL: {len(vulnerabilities)} vulnerabilities found by safety\n")
            for vuln in vulnerabilities:
                print(f"  Package: {vuln.get('package')}")
                print(f"  Installed: {vuln.get('installed_version')}")
                print(f"  Affected: {vuln.get('affected_versions')}")
                print(f"  CVE: {vuln.get('cve', 'N/A')}")
                print(f"  Description: {vuln.get('advisory')}")
                print()
        else:
            print("✅ No vulnerabilities found by safety\n")
    
    # Pip-audit results
    if audit_results:
        print(f"⚠️  CRITICAL: {len(audit_results)} vulnerabilities found by pip-audit\n")
        for vuln in audit_results:
            print(f"  Package: {vuln.get('name')}")
            print(f"  Version: {vuln.get('version')}")
            print(f"  ID: {vuln.get('id')}")
            print(f"  Fix: {vuln.get('fix_versions', 'N/A')}")
            print()
    else:
        print("✅ No vulnerabilities found by pip-audit\n")
    
    # Outdated packages
    if outdated:
        print(f"ℹ️  {len(outdated)} outdated packages found\n")
        for pkg in outdated[:10]:  # Show first 10
            print(f"  {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
        if len(outdated) > 10:
            print(f"  ... and {len(outdated) - 10} more")
        print()
    else:
        print("✅ All packages are up to date\n")
    
    print("="*80)
    
    # Exit with error if vulnerabilities found
    if (safety_results and safety_results.get("vulnerabilities")) or audit_results:
        sys.exit(1)


def main():
    """Run dependency security scan."""
    print("Running dependency security scan...")
    print("This may take a few minutes...\n")
    
    # Check if tools are installed
    tools_missing = []
    for tool in ["safety", "pip-audit"]:
        try:
            subprocess.run([tool, "--version"], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            tools_missing.append(tool)
    
    if tools_missing:
        print(f"⚠️  Warning: {', '.join(tools_missing)} not installed")
        print("Install with: pip install safety pip-audit\n")
    
    # Run scans
    safety_results = run_safety_check()
    audit_results = run_pip_audit()
    outdated = check_outdated_packages()
    
    # Generate report
    generate_report(safety_results, audit_results, outdated)


if __name__ == "__main__":
    main()
