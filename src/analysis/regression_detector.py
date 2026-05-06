"""
Regression Detector for HistoCore Project Optimization Analysis System.

Compares current analysis results against baseline to detect regressions in
coverage, performance, security, and other quality metrics for CI/CD integration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

from src.analysis.models import AnalysisResult

logger = logging.getLogger(__name__)


class RegressionDetector:
    """Detects regressions by comparing current results against baseline."""
    
    def __init__(self):
        """Initialize regression detector."""
        self.thresholds = {
            'coverage_line_decrease': 2.0,      # % decrease that triggers failure
            'coverage_branch_decrease': 2.0,
            'performance_slowdown': 10.0,       # % slowdown that triggers failure
            'security_new_vulnerabilities': 0,   # Any new vulnerability fails
            'overall_score_decrease': 5.0,      # Overall score decrease threshold
            'memory_increase': 15.0,            # % memory increase threshold
        }
    
    def detect_regressions(
        self,
        current: AnalysisResult,
        baseline_path: str
    ) -> Dict[str, Any]:
        """
        Detect regressions by comparing current results against baseline.
        
        Args:
            current: Current analysis results
            baseline_path: Path to baseline analysis JSON file
            
        Returns:
            Dictionary with regression analysis results
        """
        logger.info(f"Detecting regressions against baseline: {baseline_path}")
        
        # Load baseline results
        baseline = self._load_baseline(baseline_path)
        if not baseline:
            return self._create_no_baseline_result(current)
        
        # Perform regression analysis
        regressions = {
            'has_regressions': False,
            'critical_regressions': [],
            'warnings': [],
            'improvements': [],
            'summary': {},
            'should_fail_ci': False,
            'baseline_timestamp': baseline.timestamp,
            'current_timestamp': current.timestamp,
            'comparison_details': {}
        }
        
        # Check each dimension for regressions
        regressions.update(self._check_coverage_regressions(current, baseline))
        regressions.update(self._check_performance_regressions(current, baseline))
        regressions.update(self._check_security_regressions(current, baseline))
        regressions.update(self._check_quality_regressions(current, baseline))
        regressions.update(self._check_overall_score_regression(current, baseline))
        
        # Determine if CI should fail
        regressions['should_fail_ci'] = (
            len(regressions['critical_regressions']) > 0 or
            regressions.get('coverage_regression_critical', False) or
            regressions.get('security_regression_critical', False) or
            regressions.get('performance_regression_critical', False)
        )
        
        regressions['has_regressions'] = (
            len(regressions['critical_regressions']) > 0 or
            len(regressions['warnings']) > 0
        )
        
        # Generate summary
        regressions['summary'] = self._generate_regression_summary(regressions, current, baseline)
        
        logger.info(f"Regression analysis complete. "
                   f"Critical: {len(regressions['critical_regressions'])}, "
                   f"Warnings: {len(regressions['warnings'])}, "
                   f"Should fail CI: {regressions['should_fail_ci']}")
        
        return regressions
    
    def _load_baseline(self, baseline_path: str) -> Optional[AnalysisResult]:
        """Load baseline analysis results from JSON file."""
        try:
            baseline_file = Path(baseline_path)
            if not baseline_file.exists():
                logger.warning(f"Baseline file not found: {baseline_path}")
                return None
            
            json_content = baseline_file.read_text(encoding='utf-8')
            baseline = AnalysisResult.from_json(json_content, validate_schema=False)
            logger.info(f"Loaded baseline from {baseline_path} (timestamp: {baseline.timestamp})")
            return baseline
            
        except Exception as e:
            logger.error(f"Failed to load baseline from {baseline_path}: {e}")
            return None
    
    def _create_no_baseline_result(self, current: AnalysisResult) -> Dict[str, Any]:
        """Create result when no baseline is available."""
        return {
            'has_regressions': False,
            'critical_regressions': [],
            'warnings': ['No baseline available for comparison'],
            'improvements': [],
            'summary': {
                'message': 'No baseline available - this will become the new baseline',
                'current_score': current.overall_score
            },
            'should_fail_ci': False,
            'baseline_timestamp': None,
            'current_timestamp': current.timestamp,
            'comparison_details': {}
        }
    
    def _check_coverage_regressions(
        self,
        current: AnalysisResult,
        baseline: AnalysisResult
    ) -> Dict[str, Any]:
        """Check for test coverage regressions."""
        result = {
            'coverage_regression_critical': False,
            'comparison_details': {}
        }
        
        # Line coverage regression
        line_diff = current.coverage.line_coverage - baseline.coverage.line_coverage
        if line_diff < -self.thresholds['coverage_line_decrease']:
            result['critical_regressions'] = result.get('critical_regressions', [])
            result['critical_regressions'].append({
                'type': 'coverage_regression',
                'metric': 'line_coverage',
                'current': current.coverage.line_coverage,
                'baseline': baseline.coverage.line_coverage,
                'change': line_diff,
                'threshold': -self.thresholds['coverage_line_decrease'],
                'message': f"Line coverage decreased by {abs(line_diff):.1f}% "
                          f"({baseline.coverage.line_coverage:.1f}% → {current.coverage.line_coverage:.1f}%)"
            })
            result['coverage_regression_critical'] = True
        elif line_diff > 1.0:  # Improvement
            result['improvements'] = result.get('improvements', [])
            result['improvements'].append({
                'type': 'coverage_improvement',
                'metric': 'line_coverage',
                'change': line_diff,
                'message': f"Line coverage improved by {line_diff:.1f}%"
            })
        
        # Branch coverage regression
        branch_diff = current.coverage.branch_coverage - baseline.coverage.branch_coverage
        if branch_diff < -self.thresholds['coverage_branch_decrease']:
            result['critical_regressions'] = result.get('critical_regressions', [])
            result['critical_regressions'].append({
                'type': 'coverage_regression',
                'metric': 'branch_coverage',
                'current': current.coverage.branch_coverage,
                'baseline': baseline.coverage.branch_coverage,
                'change': branch_diff,
                'threshold': -self.thresholds['coverage_branch_decrease'],
                'message': f"Branch coverage decreased by {abs(branch_diff):.1f}% "
                          f"({baseline.coverage.branch_coverage:.1f}% → {current.coverage.branch_coverage:.1f}%)"
            })
            result['coverage_regression_critical'] = True
        elif branch_diff > 1.0:  # Improvement
            result['improvements'] = result.get('improvements', [])
            result['improvements'].append({
                'type': 'coverage_improvement',
                'metric': 'branch_coverage',
                'change': branch_diff,
                'message': f"Branch coverage improved by {branch_diff:.1f}%"
            })
        
        # New untested critical paths
        new_untested = set(current.coverage.untested_critical_paths) - set(baseline.coverage.untested_critical_paths)
        if new_untested:
            result['warnings'] = result.get('warnings', [])
            result['warnings'].append({
                'type': 'new_untested_paths',
                'count': len(new_untested),
                'paths': list(new_untested)[:5],  # Show first 5
                'message': f"{len(new_untested)} new untested critical paths detected"
            })
        
        # Store comparison details
        result['comparison_details']['coverage'] = {
            'line_coverage': {
                'current': current.coverage.line_coverage,
                'baseline': baseline.coverage.line_coverage,
                'change': line_diff
            },
            'branch_coverage': {
                'current': current.coverage.branch_coverage,
                'baseline': baseline.coverage.branch_coverage,
                'change': branch_diff
            },
            'untested_paths': {
                'current': len(current.coverage.untested_critical_paths),
                'baseline': len(baseline.coverage.untested_critical_paths),
                'new_paths': len(new_untested)
            }
        }
        
        return result
    
    def _check_performance_regressions(
        self,
        current: AnalysisResult,
        baseline: AnalysisResult
    ) -> Dict[str, Any]:
        """Check for performance regressions."""
        result = {
            'performance_regression_critical': False,
            'comparison_details': {}
        }
        
        # GPU utilization regression
        gpu_diff = current.performance.gpu_utilization - baseline.performance.gpu_utilization
        if gpu_diff < -self.thresholds['performance_slowdown']:
            result['critical_regressions'] = result.get('critical_regressions', [])
            result['critical_regressions'].append({
                'type': 'performance_regression',
                'metric': 'gpu_utilization',
                'current': current.performance.gpu_utilization,
                'baseline': baseline.performance.gpu_utilization,
                'change': gpu_diff,
                'message': f"GPU utilization decreased by {abs(gpu_diff):.1f}% "
                          f"({baseline.performance.gpu_utilization:.1f}% → {current.performance.gpu_utilization:.1f}%)"
            })
            result['performance_regression_critical'] = True
        
        # Memory usage regression
        if baseline.performance.memory_usage_peak_gb > 0:
            memory_change_pct = ((current.performance.memory_usage_peak_gb - baseline.performance.memory_usage_peak_gb) 
                               / baseline.performance.memory_usage_peak_gb * 100)
            if memory_change_pct > self.thresholds['memory_increase']:
                result['critical_regressions'] = result.get('critical_regressions', [])
                result['critical_regressions'].append({
                    'type': 'performance_regression',
                    'metric': 'memory_usage',
                    'current': current.performance.memory_usage_peak_gb,
                    'baseline': baseline.performance.memory_usage_peak_gb,
                    'change_pct': memory_change_pct,
                    'message': f"Peak memory usage increased by {memory_change_pct:.1f}% "
                              f"({baseline.performance.memory_usage_peak_gb:.1f}GB → {current.performance.memory_usage_peak_gb:.1f}GB)"
                })
                result['performance_regression_critical'] = True
        
        # New bottlenecks
        baseline_bottleneck_funcs = {b.get('function', '') for b in baseline.performance.bottlenecks}
        current_bottleneck_funcs = {b.get('function', '') for b in current.performance.bottlenecks}
        new_bottlenecks = current_bottleneck_funcs - baseline_bottleneck_funcs
        
        if new_bottlenecks:
            result['warnings'] = result.get('warnings', [])
            result['warnings'].append({
                'type': 'new_bottlenecks',
                'count': len(new_bottlenecks),
                'functions': list(new_bottlenecks)[:5],
                'message': f"{len(new_bottlenecks)} new performance bottlenecks detected"
            })
        
        # Store comparison details
        result['comparison_details']['performance'] = {
            'gpu_utilization': {
                'current': current.performance.gpu_utilization,
                'baseline': baseline.performance.gpu_utilization,
                'change': gpu_diff
            },
            'memory_peak_gb': {
                'current': current.performance.memory_usage_peak_gb,
                'baseline': baseline.performance.memory_usage_peak_gb,
                'change': current.performance.memory_usage_peak_gb - baseline.performance.memory_usage_peak_gb
            },
            'bottlenecks': {
                'current': len(current.performance.bottlenecks),
                'baseline': len(baseline.performance.bottlenecks),
                'new_bottlenecks': len(new_bottlenecks)
            }
        }
        
        return result
    
    def _check_security_regressions(
        self,
        current: AnalysisResult,
        baseline: AnalysisResult
    ) -> Dict[str, Any]:
        """Check for security regressions."""
        result = {
            'security_regression_critical': False,
            'comparison_details': {}
        }
        
        # New vulnerabilities
        vuln_diff = len(current.security.vulnerabilities) - len(baseline.security.vulnerabilities)
        if vuln_diff > self.thresholds['security_new_vulnerabilities']:
            result['critical_regressions'] = result.get('critical_regressions', [])
            result['critical_regressions'].append({
                'type': 'security_regression',
                'metric': 'vulnerabilities',
                'current': len(current.security.vulnerabilities),
                'baseline': len(baseline.security.vulnerabilities),
                'change': vuln_diff,
                'message': f"{vuln_diff} new security vulnerabilities detected"
            })
            result['security_regression_critical'] = True
        
        # New hardcoded secrets
        # hardcoded_secrets is a list of dicts, so we need to compare by converting to tuples
        baseline_secret_keys = {(s.get('file', ''), s.get('line', 0)) for s in baseline.security.hardcoded_secrets}
        current_secret_keys = {(s.get('file', ''), s.get('line', 0)) for s in current.security.hardcoded_secrets}
        new_secret_keys = current_secret_keys - baseline_secret_keys
        
        if new_secret_keys:
            result['critical_regressions'] = result.get('critical_regressions', [])
            result['critical_regressions'].append({
                'type': 'security_regression',
                'metric': 'hardcoded_secrets',
                'current': len(current.security.hardcoded_secrets),
                'baseline': len(baseline.security.hardcoded_secrets),
                'new_secrets': [f"{file}:{line}" for file, line in list(new_secret_keys)[:3]],  # Show first 3
                'message': f"{len(new_secret_keys)} new hardcoded secrets detected"
            })
            result['security_regression_critical'] = True
        
        # HIPAA compliance regression
        hipaa_diff = current.security.hipaa_compliance_score - baseline.security.hipaa_compliance_score
        if hipaa_diff < -5.0:  # 5% decrease threshold
            result['warnings'] = result.get('warnings', [])
            result['warnings'].append({
                'type': 'hipaa_regression',
                'current': current.security.hipaa_compliance_score,
                'baseline': baseline.security.hipaa_compliance_score,
                'change': hipaa_diff,
                'message': f"HIPAA compliance score decreased by {abs(hipaa_diff):.1f}%"
            })
        
        # Store comparison details
        result['comparison_details']['security'] = {
            'vulnerabilities': {
                'current': len(current.security.vulnerabilities),
                'baseline': len(baseline.security.vulnerabilities),
                'change': vuln_diff
            },
            'hardcoded_secrets': {
                'current': len(current.security.hardcoded_secrets),
                'baseline': len(baseline.security.hardcoded_secrets),
                'new_secrets': len(new_secret_keys)
            },
            'hipaa_compliance': {
                'current': current.security.hipaa_compliance_score,
                'baseline': baseline.security.hipaa_compliance_score,
                'change': hipaa_diff
            }
        }
        
        return result
    
    def _check_quality_regressions(
        self,
        current: AnalysisResult,
        baseline: AnalysisResult
    ) -> Dict[str, Any]:
        """Check for code quality regressions."""
        result = {'comparison_details': {}}
        
        # Complexity regression
        complexity_diff = current.code_quality.average_complexity - baseline.code_quality.average_complexity
        if complexity_diff > 2.0:  # Significant complexity increase
            result['warnings'] = result.get('warnings', [])
            result['warnings'].append({
                'type': 'complexity_regression',
                'current': current.code_quality.average_complexity,
                'baseline': baseline.code_quality.average_complexity,
                'change': complexity_diff,
                'message': f"Average complexity increased by {complexity_diff:.1f}"
            })
        
        # Documentation coverage regression
        doc_diff = current.code_quality.documentation_coverage - baseline.code_quality.documentation_coverage
        if doc_diff < -5.0:  # 5% decrease threshold
            result['warnings'] = result.get('warnings', [])
            result['warnings'].append({
                'type': 'documentation_regression',
                'current': current.code_quality.documentation_coverage,
                'baseline': baseline.code_quality.documentation_coverage,
                'change': doc_diff,
                'message': f"Documentation coverage decreased by {abs(doc_diff):.1f}%"
            })
        
        # Store comparison details
        result['comparison_details']['code_quality'] = {
            'average_complexity': {
                'current': current.code_quality.average_complexity,
                'baseline': baseline.code_quality.average_complexity,
                'change': complexity_diff
            },
            'documentation_coverage': {
                'current': current.code_quality.documentation_coverage,
                'baseline': baseline.code_quality.documentation_coverage,
                'change': doc_diff
            }
        }
        
        return result
    
    def _check_overall_score_regression(
        self,
        current: AnalysisResult,
        baseline: AnalysisResult
    ) -> Dict[str, Any]:
        """Check for overall score regression."""
        result = {'comparison_details': {}}
        
        score_diff = current.overall_score - baseline.overall_score
        if score_diff < -self.thresholds['overall_score_decrease']:
            result['critical_regressions'] = result.get('critical_regressions', [])
            result['critical_regressions'].append({
                'type': 'overall_score_regression',
                'current': current.overall_score,
                'baseline': baseline.overall_score,
                'change': score_diff,
                'threshold': -self.thresholds['overall_score_decrease'],
                'message': f"Overall score decreased by {abs(score_diff):.1f} points "
                          f"({baseline.overall_score:.1f} → {current.overall_score:.1f})"
            })
        elif score_diff > 2.0:  # Significant improvement
            result['improvements'] = result.get('improvements', [])
            result['improvements'].append({
                'type': 'overall_score_improvement',
                'change': score_diff,
                'message': f"Overall score improved by {score_diff:.1f} points"
            })
        
        # Store comparison details
        result['comparison_details']['overall'] = {
            'score': {
                'current': current.overall_score,
                'baseline': baseline.overall_score,
                'change': score_diff
            }
        }
        
        return result
    
    def _generate_regression_summary(
        self,
        regressions: Dict[str, Any],
        current: AnalysisResult,
        baseline: AnalysisResult
    ) -> Dict[str, Any]:
        """Generate human-readable regression summary."""
        summary = {
            'status': 'PASS' if not regressions['should_fail_ci'] else 'FAIL',
            'message': '',
            'critical_count': len(regressions['critical_regressions']),
            'warning_count': len(regressions['warnings']),
            'improvement_count': len(regressions['improvements']),
            'score_change': current.overall_score - baseline.overall_score,
            'baseline_date': baseline.timestamp,
            'analysis_date': current.timestamp
        }
        
        # Generate status message
        if regressions['should_fail_ci']:
            summary['message'] = f"❌ REGRESSION DETECTED: {summary['critical_count']} critical issues found"
        elif regressions['warnings']:
            summary['message'] = f"⚠️ WARNINGS: {summary['warning_count']} potential issues detected"
        elif regressions['improvements']:
            summary['message'] = f"✅ IMPROVEMENTS: {summary['improvement_count']} metrics improved"
        else:
            summary['message'] = "✅ NO REGRESSIONS: All metrics stable or improved"
        
        return summary
    
    def generate_diff_report(
        self,
        current: AnalysisResult,
        baseline_path: str,
        format: str = 'markdown'
    ) -> str:
        """
        Generate detailed diff report comparing current vs baseline.
        
        Args:
            current: Current analysis results
            baseline_path: Path to baseline analysis
            format: Output format ('markdown', 'json', 'text')
            
        Returns:
            Formatted diff report
        """
        regressions = self.detect_regressions(current, baseline_path)
        
        if format == 'json':
            return json.dumps(regressions, indent=2, default=str)
        elif format == 'markdown':
            return self._generate_markdown_diff_report(regressions, current)
        else:  # text
            return self._generate_text_diff_report(regressions, current)
    
    def _generate_markdown_diff_report(
        self,
        regressions: Dict[str, Any],
        current: AnalysisResult
    ) -> str:
        """Generate Markdown diff report."""
        lines = []
        
        # Header
        lines.append("# Regression Analysis Report")
        lines.append("")
        lines.append(f"**Status:** {regressions['summary']['status']}")
        lines.append(f"**Message:** {regressions['summary']['message']}")
        lines.append(f"**Analysis Date:** {current.timestamp}")
        if regressions['baseline_timestamp']:
            lines.append(f"**Baseline Date:** {regressions['baseline_timestamp']}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Critical Regressions:** {regressions['summary']['critical_count']}")
        lines.append(f"- **Warnings:** {regressions['summary']['warning_count']}")
        lines.append(f"- **Improvements:** {regressions['summary']['improvement_count']}")
        lines.append(f"- **Score Change:** {regressions['summary']['score_change']:+.1f}")
        lines.append("")
        
        # Critical regressions
        if regressions['critical_regressions']:
            lines.append("## 🚨 Critical Regressions")
            lines.append("")
            for reg in regressions['critical_regressions']:
                lines.append(f"### {reg['type'].replace('_', ' ').title()}")
                lines.append(f"**Message:** {reg['message']}")
                if 'current' in reg and 'baseline' in reg:
                    lines.append(f"**Values:** {reg['baseline']} → {reg['current']}")
                lines.append("")
        
        # Warnings
        if regressions['warnings']:
            lines.append("## ⚠️ Warnings")
            lines.append("")
            for warning in regressions['warnings']:
                lines.append(f"- {warning.get('message', 'Warning detected')}")
            lines.append("")
        
        # Improvements
        if regressions['improvements']:
            lines.append("## ✅ Improvements")
            lines.append("")
            for improvement in regressions['improvements']:
                lines.append(f"- {improvement.get('message', 'Improvement detected')}")
            lines.append("")
        
        # Detailed comparison
        if regressions['comparison_details']:
            lines.append("## Detailed Comparison")
            lines.append("")
            for dimension, details in regressions['comparison_details'].items():
                lines.append(f"### {dimension.title()}")
                lines.append("")
                for metric, values in details.items():
                    if isinstance(values, dict) and 'current' in values:
                        change = values.get('change', 0)
                        change_str = f"{change:+.1f}" if isinstance(change, (int, float)) else str(change)
                        lines.append(f"- **{metric}:** {values['baseline']} → {values['current']} ({change_str})")
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_text_diff_report(
        self,
        regressions: Dict[str, Any],
        current: AnalysisResult
    ) -> str:
        """Generate plain text diff report."""
        lines = []
        
        lines.append("REGRESSION ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"Status: {regressions['summary']['status']}")
        lines.append(f"Message: {regressions['summary']['message']}")
        lines.append("")
        
        if regressions['critical_regressions']:
            lines.append("CRITICAL REGRESSIONS:")
            for reg in regressions['critical_regressions']:
                lines.append(f"  - {reg['message']}")
            lines.append("")
        
        if regressions['warnings']:
            lines.append("WARNINGS:")
            for warning in regressions['warnings']:
                lines.append(f"  - {warning.get('message', 'Warning detected')}")
            lines.append("")
        
        if regressions['improvements']:
            lines.append("IMPROVEMENTS:")
            for improvement in regressions['improvements']:
                lines.append(f"  - {improvement.get('message', 'Improvement detected')}")
            lines.append("")
        
        return "\n".join(lines)
    
    def should_fail_build(self, regression_result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if build should fail based on regression analysis.
        
        Args:
            regression_result: Result from detect_regressions()
            
        Returns:
            Tuple of (should_fail, reason)
        """
        if regression_result['should_fail_ci']:
            reasons = []
            if regression_result.get('coverage_regression_critical'):
                reasons.append("critical coverage regression")
            if regression_result.get('security_regression_critical'):
                reasons.append("critical security regression")
            if regression_result.get('performance_regression_critical'):
                reasons.append("critical performance regression")
            
            critical_count = len(regression_result['critical_regressions'])
            if critical_count > 0:
                reasons.append(f"{critical_count} critical regressions")
            
            reason = "Build failed due to: " + ", ".join(reasons)
            return True, reason
        
        return False, "No critical regressions detected"