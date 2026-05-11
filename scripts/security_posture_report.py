#!/usr/bin/env python3
"""
Security Posture Report Generator

Generates comprehensive security posture reports from multiple security scanning tools.
Includes metrics, trends, and remediation guidance.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class SecurityPostureReporter:
    """Generate security posture reports from scan results."""

    def __init__(self):
        self.scan_date = datetime.now().isoformat()
        self.results = {
            "scan_date": self.scan_date,
            "bandit": {},
            "safety": {},
            "pip_audit": {},
            "summary": {},
            "recommendations": []
        }

    def load_bandit_report(self, bandit_path: str) -> Dict[str, Any]:
        """Load and parse Bandit JSON report."""
        try:
            with open(bandit_path, 'r') as f:
                data = json.load(f)
            
            # Count issues by severity
            high_count = len([r for r in data.get('results', []) if r.get('issue_severity') == 'HIGH'])
            medium_count = len([r for r in data.get('results', []) if r.get('issue_severity') == 'MEDIUM'])
            low_count = len([r for r in data.get('results', []) if r.get('issue_severity') == 'LOW'])
            
            self.results["bandit"] = {
                "high_count": high_count,
                "medium_count": medium_count,
                "low_count": low_count,
                "total_count": len(data.get('results', [])),
                "files_scanned": len(data.get('metrics', {}).get('_totals', {}).get('loc', 0))
            }
            
            return self.results["bandit"]
        except Exception as e:
            print(f"Error loading Bandit report: {e}", file=sys.stderr)
            return {}

    def load_safety_report(self, safety_path: str) -> Dict[str, Any]:
        """Load and parse Safety JSON report."""
        try:
            with open(safety_path, 'r') as f:
                data = json.load(f)
            
            vulnerable_count = len(data) if isinstance(data, list) else 0
            
            self.results["safety"] = {
                "vulnerable_count": vulnerable_count,
                "vulnerabilities": data if isinstance(data, list) else []
            }
            
            return self.results["safety"]
        except Exception as e:
            print(f"Error loading Safety report: {e}", file=sys.stderr)
            return {}

    def load_pip_audit_report(self, pip_audit_path: str) -> Dict[str, Any]:
        """Load and parse pip-audit JSON report."""
        try:
            with open(pip_audit_path, 'r') as f:
                data = json.load(f)
            
            vulnerable_count = len(data.get('dependencies', [])) if isinstance(data, dict) else 0
            
            self.results["pip_audit"] = {
                "vulnerable_count": vulnerable_count,
                "vulnerabilities": data.get('dependencies', []) if isinstance(data, dict) else []
            }
            
            return self.results["pip_audit"]
        except Exception as e:
            print(f"Error loading pip-audit report: {e}", file=sys.stderr)
            return {}

    def generate_summary(self) -> Dict[str, Any]:
        """Generate overall security posture summary."""
        bandit = self.results.get("bandit", {})
        safety = self.results.get("safety", {})
        pip_audit = self.results.get("pip_audit", {})
        
        # Calculate overall risk score (0-100, lower is better)
        risk_score = (
            bandit.get("high_count", 0) * 10 +
            bandit.get("medium_count", 0) * 5 +
            bandit.get("low_count", 0) * 1 +
            safety.get("vulnerable_count", 0) * 8 +
            pip_audit.get("vulnerable_count", 0) * 8
        )
        
        # Determine security posture
        if risk_score == 0:
            posture = "EXCELLENT"
            status = "✅ PASSED"
        elif risk_score < 10:
            posture = "GOOD"
            status = "✅ PASSED"
        elif risk_score < 50:
            posture = "FAIR"
            status = "⚠️ WARNING"
        else:
            posture = "POOR"
            status = "❌ FAILED"
        
        self.results["summary"] = {
            "risk_score": risk_score,
            "posture": posture,
            "status": status,
            "critical_issues": bandit.get("high_count", 0),
            "total_issues": (
                bandit.get("total_count", 0) +
                safety.get("vulnerable_count", 0) +
                pip_audit.get("vulnerable_count", 0)
            )
        }
        
        return self.results["summary"]

    def generate_recommendations(self) -> List[str]:
        """Generate remediation recommendations."""
        recommendations = []
        
        bandit = self.results.get("bandit", {})
        safety = self.results.get("safety", {})
        pip_audit = self.results.get("pip_audit", {})
        
        # Bandit recommendations
        if bandit.get("high_count", 0) > 0:
            recommendations.append(
                f"🚨 CRITICAL: Address {bandit['high_count']} HIGH severity code security issues immediately"
            )
        
        if bandit.get("medium_count", 0) > 0:
            recommendations.append(
                f"⚠️ Address {bandit['medium_count']} MEDIUM severity code security issues"
            )
        
        if bandit.get("low_count", 0) > 10:
            recommendations.append(
                f"ℹ️ Consider addressing {bandit['low_count']} LOW severity issues for improved security posture"
            )
        
        # Dependency recommendations
        if safety.get("vulnerable_count", 0) > 0:
            recommendations.append(
                f"📦 Update {safety['vulnerable_count']} vulnerable dependencies identified by Safety"
            )
        
        if pip_audit.get("vulnerable_count", 0) > 0:
            recommendations.append(
                f"📦 Update {pip_audit['vulnerable_count']} vulnerable dependencies identified by pip-audit"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ No immediate security actions required - maintain current security practices")
        else:
            recommendations.append("📚 Review security documentation in docs/SECURITY.md")
            recommendations.append("🔍 Run security verification: python scripts/verify_security.py")
        
        self.results["recommendations"] = recommendations
        return recommendations

    def save_json_report(self, output_path: str):
        """Save report as JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"JSON report saved to: {output_path}")

    def save_html_report(self, output_path: str):
        """Save report as HTML."""
        summary = self.results.get("summary", {})
        bandit = self.results.get("bandit", {})
        safety = self.results.get("safety", {})
        pip_audit = self.results.get("pip_audit", {})
        recommendations = self.results.get("recommendations", [])
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Security Posture Report - {self.scan_date}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .status {{ font-size: 24px; font-weight: bold; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .status.passed {{ background: #d4edda; color: #155724; }}
        .status.warning {{ background: #fff3cd; color: #856404; }}
        .status.failed {{ background: #f8d7da; color: #721c24; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; min-width: 150px; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #333; }}
        .high {{ color: #dc3545; }}
        .medium {{ color: #ffc107; }}
        .low {{ color: #17a2b8; }}
        .recommendations {{ background: #e7f3ff; padding: 20px; border-radius: 5px; border-left: 4px solid #2196F3; }}
        .recommendations li {{ margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; font-weight: bold; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 Security Posture Report</h1>
        <p><strong>Scan Date:</strong> {self.scan_date}</p>
        
        <div class="status {summary.get('status', '').lower().split()[1] if ' ' in summary.get('status', '') else 'passed'}">
            {summary.get('status', 'UNKNOWN')} - Security Posture: {summary.get('posture', 'UNKNOWN')}
        </div>
        
        <h2>📊 Metrics Overview</h2>
        <div>
            <div class="metric">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">{summary.get('risk_score', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Critical Issues</div>
                <div class="metric-value high">{summary.get('critical_issues', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Issues</div>
                <div class="metric-value">{summary.get('total_issues', 0)}</div>
            </div>
        </div>
        
        <h2>🔍 Bandit Code Security Scan</h2>
        <table>
            <tr>
                <th>Severity</th>
                <th>Count</th>
            </tr>
            <tr>
                <td class="high">HIGH</td>
                <td class="high">{bandit.get('high_count', 0)}</td>
            </tr>
            <tr>
                <td class="medium">MEDIUM</td>
                <td class="medium">{bandit.get('medium_count', 0)}</td>
            </tr>
            <tr>
                <td class="low">LOW</td>
                <td class="low">{bandit.get('low_count', 0)}</td>
            </tr>
            <tr>
                <td><strong>TOTAL</strong></td>
                <td><strong>{bandit.get('total_count', 0)}</strong></td>
            </tr>
        </table>
        
        <h2>📦 Dependency Vulnerabilities</h2>
        <table>
            <tr>
                <th>Tool</th>
                <th>Vulnerable Packages</th>
            </tr>
            <tr>
                <td>Safety</td>
                <td>{safety.get('vulnerable_count', 0)}</td>
            </tr>
            <tr>
                <td>pip-audit</td>
                <td>{pip_audit.get('vulnerable_count', 0)}</td>
            </tr>
        </table>
        
        <h2>💡 Recommendations</h2>
        <div class="recommendations">
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in recommendations)}
            </ul>
        </div>
        
        <div class="footer">
            <p>Generated by HistoCore Security Posture Reporter</p>
            <p>For more information, see docs/SECURITY.md</p>
        </div>
    </div>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"HTML report saved to: {output_path}")

    def print_summary(self):
        """Print summary to console."""
        summary = self.results.get("summary", {})
        recommendations = self.results.get("recommendations", [])
        
        print("\n" + "="*60)
        print("SECURITY POSTURE REPORT")
        print("="*60)
        print(f"Scan Date: {self.scan_date}")
        print(f"Status: {summary.get('status', 'UNKNOWN')}")
        print(f"Posture: {summary.get('posture', 'UNKNOWN')}")
        print(f"Risk Score: {summary.get('risk_score', 0)}")
        print(f"Critical Issues: {summary.get('critical_issues', 0)}")
        print(f"Total Issues: {summary.get('total_issues', 0)}")
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  {rec}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate security posture report")
    parser.add_argument("--bandit", required=True, help="Path to Bandit JSON report")
    parser.add_argument("--safety", help="Path to Safety JSON report")
    parser.add_argument("--pip-audit", help="Path to pip-audit JSON report")
    parser.add_argument("--output", default="security-posture.json", help="Output JSON path")
    parser.add_argument("--html", help="Output HTML path")
    
    args = parser.parse_args()
    
    reporter = SecurityPostureReporter()
    
    # Load reports
    reporter.load_bandit_report(args.bandit)
    
    if args.safety:
        reporter.load_safety_report(args.safety)
    
    if args.pip_audit:
        reporter.load_pip_audit_report(args.pip_audit)
    
    # Generate analysis
    reporter.generate_summary()
    reporter.generate_recommendations()
    
    # Save reports
    reporter.save_json_report(args.output)
    
    if args.html:
        reporter.save_html_report(args.html)
    
    # Print summary
    reporter.print_summary()
    
    # Exit with error if critical issues found
    if reporter.results["summary"].get("critical_issues", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
