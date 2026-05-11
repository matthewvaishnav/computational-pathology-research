#!/usr/bin/env python3
"""
Bandit Results Checker

Parses Bandit JSON output and fails CI if HIGH or MEDIUM severity issues are found.
Used in CI/CD pipeline to enforce security standards.

Usage:
    python scripts/check_bandit_results.py <bandit-report.json>

Exit codes:
    0: No HIGH/MEDIUM issues found
    1: HIGH or MEDIUM severity issues found
    2: Error parsing report
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


def parse_bandit_report(report_path: Path) -> Dict:
    """Parse Bandit JSON report."""
    try:
        with open(report_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Report file not found: {report_path}")
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in report: {e}")
        sys.exit(2)


def count_by_severity(results: List[Dict]) -> Dict[str, int]:
    """Count issues by severity level."""
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for issue in results:
        severity = issue.get("issue_severity", "UNKNOWN")
        if severity in counts:
            counts[severity] += 1
    return counts


def print_summary(counts: Dict[str, int], total: int):
    """Print issue summary."""
    print("\n" + "=" * 60)
    print("BANDIT SECURITY SCAN RESULTS")
    print("=" * 60)
    print(f"\nTotal issues found: {total}")
    print(f"  HIGH severity:   {counts['HIGH']}")
    print(f"  MEDIUM severity: {counts['MEDIUM']}")
    print(f"  LOW severity:    {counts['LOW']}")
    print()


def print_issues(results: List[Dict], severity_filter: str = None):
    """Print detailed issue information."""
    filtered = [r for r in results if not severity_filter or r.get("issue_severity") == severity_filter]
    
    if not filtered:
        return
    
    print(f"\n{severity_filter or 'ALL'} SEVERITY ISSUES:")
    print("-" * 60)
    
    for issue in filtered:
        print(f"\nFile: {issue.get('filename', 'unknown')}")
        print(f"Line: {issue.get('line_number', 'unknown')}")
        print(f"Severity: {issue.get('issue_severity', 'unknown')}")
        print(f"Confidence: {issue.get('issue_confidence', 'unknown')}")
        print(f"Issue: {issue.get('issue_text', 'unknown')}")
        print(f"Test ID: {issue.get('test_id', 'unknown')}")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_bandit_results.py <bandit-report.json>")
        sys.exit(2)
    
    report_path = Path(sys.argv[1])
    report = parse_bandit_report(report_path)
    
    results = report.get("results", [])
    counts = count_by_severity(results)
    total = len(results)
    
    print_summary(counts, total)
    
    # Print HIGH and MEDIUM issues
    if counts["HIGH"] > 0:
        print_issues(results, "HIGH")
    if counts["MEDIUM"] > 0:
        print_issues(results, "MEDIUM")
    
    # Fail if HIGH or MEDIUM issues found
    if counts["HIGH"] > 0 or counts["MEDIUM"] > 0:
        print("\n" + "=" * 60)
        print("❌ SECURITY SCAN FAILED")
        print("=" * 60)
        print(f"\nFound {counts['HIGH']} HIGH and {counts['MEDIUM']} MEDIUM severity issues.")
        print("Please fix these security issues before merging.")
        print("\nFor false positives, add # nosec comments with justification.")
        print("=" * 60 + "\n")
        sys.exit(1)
    
    # Success
    print("=" * 60)
    print("✅ SECURITY SCAN PASSED")
    print("=" * 60)
    print("\nNo HIGH or MEDIUM severity issues found.")
    print("=" * 60 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
