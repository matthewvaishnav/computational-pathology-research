#!/usr/bin/env python3
"""
Security Posture Report Generator

Generates comprehensive security posture reports including:
- Issue counts by severity
- Test coverage metrics
- Audit trail completeness
- Trends over time
- Remediation guidance
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def load_bandit_report(report_path: str) -> Dict[str, Any]:
    """Load Bandit JSON report."""
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Bandit report not found: {report_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in Bandit report: {report_path}")
        return {}


def get_issue_counts(bandit_report: Dict[str, Any]) -> Dict[str, int]:
    """Extract issue counts by severity from Bandit report."""
    if not bandit_report or 'metrics' not in bandit_report:
        return {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    
    totals = bandit_report['metrics'].get('_totals', {})
    return {
        'HIGH': totals.get('SEVERITY.HIGH', 0),
        'MEDIUM': totals.get('SEVERITY.MEDIUM', 0),
        'LOW': totals.get('SEVERITY.LOW', 0),
    }


def get_test_coverage() -> Dict[str, float]:
    """Get test coverage metrics from coverage report."""
    coverage_file = Path('.coverage')
    if not coverage_file.exists():
        return {'overall': 0.0, 'security': 0.0}
    
    # Try to read coverage data
    try:
        import coverage
        cov = coverage.Coverage()
        cov.load()
        
        # Get overall coverage
        total = cov.report(show_missing=False, file=open(os.devnull, 'w'))
        
        # Get security module coverage
        security_files = [
            'src/security/jinja2_security.py',
            'src/security/network_binding.py',
            'src/security/model_download.py',
            'src/security/temp_file.py',
            'src/security/pickle_security.py',
            'src/security/url_fetcher.py',
        ]
        
        security_coverage = 0.0
        security_count = 0
        for file in security_files:
            if Path(file).exists():
                try:
                    analysis = cov.analysis(file)
                    if analysis:
                        executed = len(analysis[1])
                        missing = len(analysis[2])
                        total_lines = executed + missing
                        if total_lines > 0:
                            security_coverage += (executed / total_lines) * 100
                            security_count += 1
                except:
                    pass
        
        if security_count > 0:
            security_coverage /= security_count
        
        return {
            'overall': total,
            'security': security_coverage,
        }
    except:
        return {'overall': 0.0, 'security': 0.0}


def check_audit_trail() -> Dict[str, Any]:
    """Check audit trail completeness."""
    audit_log = Path('logs/security_audit.log')
    
    if not audit_log.exists():
        return {
            'exists': False,
            'entries': 0,
            'recent_entries': 0,
        }
    
    try:
        with open(audit_log, 'r') as f:
            lines = f.readlines()
        
        # Count entries
        total_entries = len(lines)
        
        # Count recent entries (last 24 hours)
        recent_entries = 0
        now = datetime.now()
        for line in lines[-1000:]:  # Check last 1000 entries
            try:
                # Parse timestamp from log entry
                if '"timestamp"' in line:
                    entry = json.loads(line)
                    timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                    if (now - timestamp).total_seconds() < 86400:
                        recent_entries += 1
            except:
                pass
        
        return {
            'exists': True,
            'entries': total_entries,
            'recent_entries': recent_entries,
        }
    except:
        return {
            'exists': True,
            'entries': 0,
            'recent_entries': 0,
        }


def get_remediation_guidance(issue_counts: Dict[str, int]) -> List[str]:
    """Generate remediation guidance based on issue counts."""
    guidance = []
    
    if issue_counts['HIGH'] > 0:
        guidance.append(f"🔴 CRITICAL: {issue_counts['HIGH']} HIGH severity issues require immediate attention")
        guidance.append("   - Review Bandit report for details")
        guidance.append("   - Fix HIGH severity issues before deployment")
        guidance.append("   - Run: python scripts/check_bandit_results.py bandit-report.json")
    
    if issue_counts['MEDIUM'] > 0:
        guidance.append(f"🟡 WARNING: {issue_counts['MEDIUM']} MEDIUM severity issues should be addressed")
        guidance.append("   - Review and prioritize MEDIUM severity issues")
        guidance.append("   - Add to sprint backlog for resolution")
        guidance.append("   - Consider adding # nosec comments for false positives")
    
    if issue_counts['LOW'] > 0:
        guidance.append(f"ℹ️  INFO: {issue_counts['LOW']} LOW severity issues detected")
        guidance.append("   - Review LOW severity issues for best practices")
        guidance.append("   - Address during regular maintenance")
    
    if issue_counts['HIGH'] == 0 and issue_counts['MEDIUM'] == 0:
        guidance.append("✅ EXCELLENT: No HIGH or MEDIUM severity issues detected")
        guidance.append("   - Security posture is good")
        guidance.append("   - Continue regular security audits")
    
    return guidance


def generate_report(bandit_report_path: str, output_path: str = None) -> None:
    """Generate comprehensive security posture report."""
    print("Generating security posture report...")
    
    # Load data
    bandit_report = load_bandit_report(bandit_report_path)
    issue_counts = get_issue_counts(bandit_report)
    test_coverage = get_test_coverage()
    audit_trail = check_audit_trail()
    remediation = get_remediation_guidance(issue_counts)
    
    # Generate report
    report = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'high_severity_issues': issue_counts['HIGH'],
            'medium_severity_issues': issue_counts['MEDIUM'],
            'low_severity_issues': issue_counts['LOW'],
            'total_issues': sum(issue_counts.values()),
        },
        'test_coverage': test_coverage,
        'audit_trail': audit_trail,
        'remediation_guidance': remediation,
    }
    
    # Print report
    print("\n" + "="*80)
    print("SECURITY POSTURE REPORT")
    print("="*80)
    print(f"\nGenerated: {report['generated_at']}")
    
    print("\n📊 ISSUE SUMMARY")
    print("-" * 80)
    print(f"  HIGH severity:   {issue_counts['HIGH']:3d}")
    print(f"  MEDIUM severity: {issue_counts['MEDIUM']:3d}")
    print(f"  LOW severity:    {issue_counts['LOW']:3d}")
    print(f"  Total issues:    {report['summary']['total_issues']:3d}")
    
    print("\n🧪 TEST COVERAGE")
    print("-" * 80)
    print(f"  Overall coverage:  {test_coverage['overall']:.1f}%")
    print(f"  Security coverage: {test_coverage['security']:.1f}%")
    
    print("\n📝 AUDIT TRAIL")
    print("-" * 80)
    print(f"  Audit log exists:  {'Yes' if audit_trail['exists'] else 'No'}")
    print(f"  Total entries:     {audit_trail['entries']}")
    print(f"  Recent entries:    {audit_trail['recent_entries']} (last 24h)")
    
    print("\n💡 REMEDIATION GUIDANCE")
    print("-" * 80)
    for item in remediation:
        print(f"  {item}")
    
    print("\n" + "="*80)
    
    # Save to file if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Report saved to: {output_path}")
    
    # Exit with error code if HIGH or MEDIUM issues found
    if issue_counts['HIGH'] > 0 or issue_counts['MEDIUM'] > 0:
        print("\n❌ Security posture check FAILED")
        sys.exit(1)
    else:
        print("\n✅ Security posture check PASSED")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Generate security posture report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report from Bandit scan
  python scripts/security_posture_report.py bandit-report.json
  
  # Generate report and save to file
  python scripts/security_posture_report.py bandit-report.json -o security-posture.json
  
  # Run full security audit
  python -m bandit -r src/ -f json -o bandit-report.json
  python scripts/security_posture_report.py bandit-report.json
        """
    )
    
    parser.add_argument(
        'bandit_report',
        help='Path to Bandit JSON report'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output path for JSON report (optional)',
        default=None
    )
    
    args = parser.parse_args()
    
    generate_report(args.bandit_report, args.output)


if __name__ == '__main__':
    main()
