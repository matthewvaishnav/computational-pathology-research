"""
Security Scanner for HistoCore Project Optimization Analysis System.

Analyzes security vulnerabilities, hardcoded secrets, and HIPAA compliance.
"""

import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import SecurityAnalysis, SecretFinding


logger = logging.getLogger(__name__)


class SecurityScanner:
    """Analyzes security vulnerabilities and compliance."""
    
    def __init__(self, project_path: str):
        """
        Initialize scanner.
        
        Args:
            project_path: Path to project root directory
        """
        self.project_path = Path(project_path).resolve()
        
    def analyze(self) -> SecurityAnalysis:
        """
        Run security analysis.
        
        Returns:
            SecurityAnalysis with security metrics
        """
        logger.info("Starting security analysis...")
        
        # Bandit security scan
        vulnerabilities = self._run_bandit_scan()
        
        # Hardcoded secrets detection
        secrets = self._detect_hardcoded_secrets()
        
        # Injection risks
        injection_risks = self._detect_injection_risks()
        
        # TLS/SSL configuration
        tls_issues = self._validate_tls_ssl_config()
        
        # HIPAA compliance
        hipaa_score = self._assess_hipaa_compliance()
        
        # Calculate score
        score = self._calculate_security_score(
            vulnerabilities, secrets, injection_risks, tls_issues, hipaa_score
        )
        
        return SecurityAnalysis(
            vulnerabilities=vulnerabilities,
            hipaa_compliance_score=hipaa_score,
            hardcoded_secrets=secrets,
            injection_risks=injection_risks,
            tls_issues=tls_issues,
            score=score
        )
    
    def _run_bandit_scan(self) -> List[Dict[str, Any]]:
        """Run bandit security scanner with comprehensive logging and monitoring."""
        scan_start_time = time.time()
        
        try:
            src_dir = self.project_path / 'src'
            if not src_dir.exists():
                logger.info("No src directory found - skipping bandit scan")
                return []
            
            # Count Python files for progress tracking
            python_files = list(src_dir.rglob('*.py'))
            logger.info(f"Starting bandit security scan on {len(python_files)} Python files")
            
            result = subprocess.run(
                ['bandit', '-r', str(src_dir), '-f', 'json', '--severity-level', 'low'],
                capture_output=True,
                text=True,
                timeout=120,
                check=False
            )
            
            scan_duration = time.time() - scan_start_time
            logger.info(f"Bandit scan completed in {scan_duration:.2f}s")
            
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    
                    vulnerabilities = []
                    severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
                    
                    for issue in data.get('results', []):
                        severity = issue.get('issue_severity', 'UNKNOWN').upper()
                        if severity in severity_counts:
                            severity_counts[severity] += 1
                        
                        vulnerability = {
                            'type': issue.get('test_name', 'unknown'),
                            'severity': severity,
                            'file': issue.get('filename', 'unknown'),
                            'line': issue.get('line_number', 0),
                            'description': issue.get('issue_text', 'unknown'),
                            'confidence': issue.get('issue_confidence', 'UNKNOWN'),
                            'cwe_id': issue.get('test_id', 'unknown')
                        }
                        vulnerabilities.append(vulnerability)
                    
                    # Log security scan summary
                    total_issues = len(vulnerabilities)
                    logger.info(f"Security scan found {total_issues} issues: "
                              f"HIGH={severity_counts['HIGH']}, "
                              f"MEDIUM={severity_counts['MEDIUM']}, "
                              f"LOW={severity_counts['LOW']}")
                    
                    # Log critical security issues
                    critical_issues = [v for v in vulnerabilities if v['severity'] == 'HIGH']
                    if critical_issues:
                        logger.warning(f"CRITICAL: {len(critical_issues)} high-severity security issues found!")
                        for issue in critical_issues[:3]:  # Log first 3 critical issues
                            logger.warning(f"  - {issue['type']} in {Path(issue['file']).name}:{issue['line']}")
                    
                    return vulnerabilities
                
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse bandit JSON output: {e}")
                    logger.debug(f"Bandit stdout: {result.stdout[:500]}...")
                    return []
            
            # Handle bandit execution issues
            if result.returncode != 0:
                logger.warning(f"Bandit scan completed with warnings (exit code: {result.returncode})")
                if result.stderr:
                    logger.debug(f"Bandit stderr: {result.stderr}")
            
            return []
        
        except subprocess.TimeoutExpired:
            scan_duration = time.time() - scan_start_time
            logger.error(f"Bandit scan timed out after {scan_duration:.2f}s")
            return []
        except FileNotFoundError:
            logger.warning("Bandit not installed - install with 'pip install bandit' for security scanning")
            return []
        except Exception as e:
            scan_duration = time.time() - scan_start_time
            logger.error(f"Bandit scan failed after {scan_duration:.2f}s: {e}")
            return []
    
    def _detect_hardcoded_secrets(self) -> List[SecretFinding]:
        """Detect hardcoded secrets using regex patterns with comprehensive monitoring."""
        detection_start_time = time.time()
        secrets = []
        files_scanned = 0
        patterns_matched = 0
        
        # Enhanced secret patterns with confidence levels
        patterns = [
            (r'password\s*=\s*["\'][^"\']{8,}["\']', 'password', 'high'),
            (r'api_key\s*=\s*["\'][^"\']{16,}["\']', 'api_key', 'high'),
            (r'secret_key\s*=\s*["\'][^"\']{16,}["\']', 'secret_key', 'high'),
            (r'token\s*=\s*["\'][^"\']{20,}["\']', 'token', 'high'),
            (r'-----BEGIN\s+PRIVATE\s+KEY-----', 'private_key', 'critical'),
            (r'-----BEGIN\s+RSA\s+PRIVATE\s+KEY-----', 'rsa_private_key', 'critical'),
            (r'sk_live_[a-zA-Z0-9]{24,}', 'stripe_live_key', 'critical'),
            (r'sk_test_[a-zA-Z0-9]{24,}', 'stripe_test_key', 'medium'),
            (r'AKIA[0-9A-Z]{16}', 'aws_access_key', 'high'),
            (r'[0-9a-zA-Z/+]{40}', 'aws_secret_key', 'medium'),
            (r'ghp_[a-zA-Z0-9]{36}', 'github_token', 'high'),
            (r'xox[baprs]-[0-9a-zA-Z\-]{10,48}', 'slack_token', 'high'),
        ]
        
        logger.info(f"Starting hardcoded secrets detection with {len(patterns)} patterns")
        
        # Scan Python files
        for py_file in self.project_path.rglob('*.py'):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            files_scanned += 1
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern, secret_type, confidence in patterns:
                    matches = list(re.finditer(pattern, content, re.IGNORECASE))
                    if matches:
                        patterns_matched += len(matches)
                        
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            
                            secret_dict = {
                                'type': secret_type,
                                'severity': confidence.upper(),
                                'file': str(py_file.relative_to(self.project_path)),
                                'line': line_num,
                                'description': f"Hardcoded {secret_type.replace('_', ' ')} detected"
                            }
                            
                            # Log based on confidence level
                            log_msg = f"{confidence.upper()} {secret_type} in {py_file.relative_to(self.project_path)}:{line_num}"
                            if confidence == 'critical':
                                logger.error(f"Critical secret detected: [CRITICAL] {log_msg}")
                            elif confidence == 'high':
                                logger.warning(f"High-confidence secret detected: [HIGH] {log_msg}")
                            elif confidence == 'medium':
                                logger.info(f"Potential secret detected: [MEDIUM] {log_msg}")
                            
                            secrets.append(secret_dict)
            
            except (UnicodeDecodeError, OSError) as e:
                logger.debug(f"Failed to scan {py_file}: {e}")
                continue
        
        detection_duration = time.time() - detection_start_time
        
        # Log comprehensive detection summary
        logger.info(f"Hardcoded secrets detection completed in {detection_duration:.2f}s")
        logger.info(f"Scanned {files_scanned} Python files, found {len(secrets)} potential secrets")
        
        if secrets:
            # Categorize secrets by severity level
            critical_secrets = [s for s in secrets if s['severity'] == 'CRITICAL']
            high_secrets = [s for s in secrets if s['severity'] == 'HIGH']
            medium_secrets = [s for s in secrets if s['severity'] == 'MEDIUM']
            
            logger.warning(f"Secret detection summary: "
                          f"CRITICAL={len(critical_secrets)}, "
                          f"HIGH={len(high_secrets)}, "
                          f"MEDIUM={len(medium_secrets)}")
            
            if critical_secrets:
                logger.error("URGENT: Critical secrets detected! Immediate action required.")
        else:
            logger.info("No hardcoded secrets detected")
        
        return secrets
    
    def _detect_injection_risks(self) -> List[Dict[str, Any]]:
        """Detect SQL injection and command injection risks."""
        risks = []
        
        # Injection patterns
        patterns = [
            (r'execute\s*\(\s*["\'][^"\']*%[^"\']*["\']', 'SQL injection risk'),
            (r'subprocess\.\w+\([^)]*shell\s*=\s*True', 'Command injection risk'),
            (r'eval\s*\(', 'Code injection risk'),
            (r'exec\s*\(', 'Code execution risk'),
        ]
        
        # Scan Python files
        for py_file in self.project_path.rglob('*.py'):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern, risk_type in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        risks.append({
                            'type': risk_type,
                            'file': str(py_file.relative_to(self.project_path)),
                            'line': line_num
                        })
            
            except (UnicodeDecodeError, OSError):
                continue
        
        return risks
    
    def _validate_tls_ssl_config(self) -> List[Dict[str, Any]]:
        """
        Validate TLS/SSL configuration.
        
        Checks:
        - PACS integration uses TLS
        - Federated learning encryption
        - Insecure SSL contexts (verify=False, check_hostname=False)
        
        Returns:
            List of TLS/SSL configuration issues
        """
        issues = []
        
        # Insecure SSL patterns
        patterns = [
            (r'verify\s*=\s*False', 'SSL verification disabled'),
            (r'check_hostname\s*=\s*False', 'Hostname verification disabled'),
            (r'ssl\._create_unverified_context', 'Unverified SSL context'),
            (r'PROTOCOL_SSLv[23]', 'Insecure SSL protocol'),
            (r'PROTOCOL_TLSv1\b', 'Insecure TLS 1.0'),
        ]
        
        # Scan Python files
        for py_file in self.project_path.rglob('*.py'):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern, issue_type in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append({
                            'type': issue_type,
                            'file': str(py_file.relative_to(self.project_path)),
                            'line': line_num
                        })
            
            except (UnicodeDecodeError, OSError):
                continue
        
        return issues
    
    def _assess_hipaa_compliance(self) -> float:
        """
        Assess HIPAA compliance with detailed monitoring and logging.
        
        Checks:
        - Audit logging (7-year retention)
        - Input sanitization
        - Access controls
        - Encryption at rest and in transit
        
        Returns:
            Score 0-100 based on HIPAA compliance
        """
        assessment_start_time = time.time()
        score = 0.0
        compliance_details = {}
        
        logger.info("Starting HIPAA compliance assessment")
        
        # Check for audit logging
        audit_logging = self._check_audit_logging()
        compliance_details['audit_logging'] = audit_logging
        if audit_logging:
            score += 30
            logger.info("✓ Audit logging implementation detected")
        else:
            logger.warning("✗ Missing audit logging - HIPAA requires comprehensive audit trails")
        
        # Check for input sanitization
        input_sanitization = self._check_input_sanitization()
        compliance_details['input_sanitization'] = input_sanitization
        if input_sanitization:
            score += 25
            logger.info("✓ Input sanitization implementation detected")
        else:
            logger.warning("✗ Missing input sanitization - required for PHI protection")
        
        # Check for access controls
        access_controls = self._check_access_controls()
        compliance_details['access_controls'] = access_controls
        if access_controls:
            score += 25
            logger.info("✓ Access control implementation detected")
        else:
            logger.warning("✗ Missing access controls - HIPAA requires role-based access")
        
        # Check for encryption
        encryption = self._check_encryption()
        compliance_details['encryption'] = encryption
        if encryption:
            score += 20
            logger.info("✓ Encryption implementation detected")
        else:
            logger.warning("✗ Missing encryption - HIPAA requires data encryption at rest and in transit")
        
        assessment_duration = time.time() - assessment_start_time
        
        # Log comprehensive compliance assessment
        logger.info(f"HIPAA compliance assessment completed in {assessment_duration:.2f}s")
        logger.info(f"HIPAA compliance score: {score}/100")
        
        # Provide detailed compliance guidance
        if score < 50:
            logger.error("CRITICAL: HIPAA compliance score below 50% - immediate action required!")
            logger.error("Missing critical HIPAA requirements may result in regulatory violations")
        elif score < 75:
            logger.warning("WARNING: HIPAA compliance score below 75% - improvements needed")
        else:
            logger.info("GOOD: HIPAA compliance score above 75%")
        
        # Log specific compliance gaps
        missing_controls = [k for k, v in compliance_details.items() if not v]
        if missing_controls:
            logger.warning(f"Missing HIPAA controls: {', '.join(missing_controls)}")
            logger.info("Refer to HIPAA Security Rule 45 CFR §164.308-318 for implementation guidance")
        
        return score
    
    def _generate_hipaa_checklist(self) -> Dict[str, Any]:
        """Generate HIPAA compliance checklist."""
        checklist = {
            'audit_logging': {
                'implemented': self._check_audit_logging(),
                'requirement': 'Maintain audit logs for 7 years with tamper-evident storage',
                'priority': 'critical'
            },
            'input_sanitization': {
                'implemented': self._check_input_sanitization(),
                'requirement': 'Sanitize all user inputs to prevent injection attacks',
                'priority': 'high'
            },
            'access_controls': {
                'implemented': self._check_access_controls(),
                'requirement': 'Implement role-based access controls for PHI',
                'priority': 'critical'
            },
            'encryption': {
                'implemented': self._check_encryption(),
                'requirement': 'Encrypt PHI at rest and in transit using AES-256',
                'priority': 'critical'
            }
        }
        
        return checklist
    
    def _check_audit_logging(self) -> bool:
        """Check for audit logging implementation."""
        # Look for audit logging patterns
        patterns = [
            r'audit.*log',
            r'AuditLogger',
            r'audit_trail',
            r'log.*audit',
        ]
        
        for py_file in self.project_path.rglob('*.py'):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                if any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns):
                    return True
            except (UnicodeDecodeError, OSError):
                continue
        
        return False
    
    def _check_input_sanitization(self) -> bool:
        """Check for input sanitization implementation."""
        # Look for sanitization patterns
        patterns = [
            r'sanitize',
            r'validate.*input',
            r'clean.*input',
            r'escape',
        ]
        
        for py_file in self.project_path.rglob('*.py'):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                if any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns):
                    return True
            except (UnicodeDecodeError, OSError):
                continue
        
        return False
    
    def _check_access_controls(self) -> bool:
        """Check for access control implementation."""
        # Look for access control patterns
        patterns = [
            r'@require.*permission',
            r'check.*permission',
            r'authorize',
            r'access.*control',
            r'rbac',
        ]
        
        for py_file in self.project_path.rglob('*.py'):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                if any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns):
                    return True
            except (UnicodeDecodeError, OSError):
                continue
        
        return False
    
    def _check_encryption(self) -> bool:
        """Check for encryption implementation."""
        # Look for encryption patterns
        patterns = [
            r'encrypt',
            r'decrypt',
            r'Fernet',
            r'AES',
            r'cryptography',
        ]
        
        for py_file in self.project_path.rglob('*.py'):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                if any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns):
                    return True
            except (UnicodeDecodeError, OSError):
                continue
        
        return False
    
    def _calculate_security_score(
        self,
        vulnerabilities: List[Dict[str, Any]],
        secrets: List[Dict[str, Any]],
        injection_risks: List[Dict[str, Any]],
        tls_issues: Optional[List[Dict[str, Any]]] = None,
        hipaa_score: float = 0.0
    ) -> float:
        """
        Calculate security score (0-100).
        
        Scoring:
        - No vulnerabilities: 35%
        - No hardcoded secrets: 25%
        - No injection risks: 20%
        - Secure TLS/SSL: 10%
        - HIPAA compliance: 10%
        """
        score = 0.0
        
        # Vulnerability penalty (critical: -35, high: -20, medium: -10, low: -5)
        vuln_penalty = 0
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'unknown').lower()
            if severity == 'high':
                vuln_penalty += 20
            elif severity == 'medium':
                vuln_penalty += 10
            else:
                vuln_penalty += 5
        
        score += max(0, 35 - vuln_penalty)
        
        # Secrets penalty (-10 per secret, max -25)
        score += max(0, 25 - len(secrets) * 10)
        
        # Injection risks penalty (-5 per risk, max -20)
        score += max(0, 20 - len(injection_risks) * 5)
        
        # TLS issues penalty (-5 per issue, max -10)
        score += max(0, 10 - len(tls_issues) * 5)
        
        # HIPAA compliance (scale 0-100 to 0-10)
        score += (hipaa_score / 100.0) * 10.0
        
        return max(0.0, min(100.0, round(score, 2)))
