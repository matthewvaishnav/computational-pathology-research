"""
Security Scanner for HistoCore Project Optimization Analysis System.

Analyzes security vulnerabilities, hardcoded secrets, and HIPAA compliance.
"""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from src.analysis.models import SecurityAnalysis


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
        
        # HIPAA compliance
        hipaa_score = self._assess_hipaa_compliance()
        
        # Calculate score
        score = self._calculate_security_score(vulnerabilities, secrets, injection_risks, hipaa_score)
        
        return SecurityAnalysis(
            vulnerabilities=vulnerabilities,
            hipaa_compliance_score=hipaa_score,
            hardcoded_secrets=secrets,
            injection_risks=injection_risks,
            score=score
        )
    
    def _run_bandit_scan(self) -> List[Dict[str, Any]]:
        """Run bandit security scanner."""
        try:
            src_dir = self.project_path / 'src'
            if not src_dir.exists():
                return []
            
            result = subprocess.run(
                ['bandit', '-r', str(src_dir), '-f', 'json'],
                capture_output=True,
                text=True,
                timeout=120,
                check=False
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                
                vulnerabilities = []
                for issue in data.get('results', []):
                    vulnerabilities.append({
                        'type': issue.get('test_name', 'unknown'),
                        'severity': issue.get('issue_severity', 'unknown'),
                        'file': issue.get('filename', 'unknown'),
                        'line': issue.get('line_number', 0),
                        'description': issue.get('issue_text', 'unknown')
                    })
                
                return vulnerabilities
            
            return []
        
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to run bandit scan: {e}")
            return []
    
    def _detect_hardcoded_secrets(self) -> List[str]:
        """Detect hardcoded secrets using regex patterns."""
        secrets = []
        
        # Common secret patterns
        patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'hardcoded password'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'hardcoded API key'),
            (r'secret_key\s*=\s*["\'][^"\']+["\']', 'hardcoded secret key'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'hardcoded token'),
            (r'-----BEGIN\s+PRIVATE\s+KEY-----', 'private key'),
        ]
        
        # Scan Python files
        for py_file in self.project_path.rglob('*.py'):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern, description in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        secrets.append(f"{description} in {py_file.relative_to(self.project_path)}:{line_num}")
            
            except (UnicodeDecodeError, OSError):
                continue
        
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
    
    def _assess_hipaa_compliance(self) -> float:
        """Assess HIPAA compliance (placeholder)."""
        # TODO: Implement HIPAA compliance checklist
        logger.info("HIPAA compliance assessment not yet implemented")
        return 50.0  # Default neutral score
    
    def _calculate_security_score(
        self,
        vulnerabilities: List[Dict[str, Any]],
        secrets: List[str],
        injection_risks: List[Dict[str, Any]],
        hipaa_score: float
    ) -> float:
        """
        Calculate security score (0-100).
        
        Scoring:
        - No vulnerabilities: 40%
        - No hardcoded secrets: 30%
        - No injection risks: 20%
        - HIPAA compliance: 10%
        """
        score = 0.0
        
        # Vulnerability penalty (critical: -40, high: -25, medium: -10, low: -5)
        vuln_penalty = 0
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'unknown').lower()
            if severity == 'high':
                vuln_penalty += 25
            elif severity == 'medium':
                vuln_penalty += 10
            else:
                vuln_penalty += 5
        
        score += max(0, 40 - vuln_penalty)
        
        # Secrets penalty (-10 per secret, max -30)
        score += max(10, 30 - len(secrets) * 10)
        
        # Injection risks penalty (-5 per risk, max -20)
        score += max(0, 20 - len(injection_risks) * 5)
        
        # HIPAA compliance (scale 0-100 to 0-10)
        score += (hipaa_score / 100.0) * 10.0
        
        return max(0.0, min(100.0, round(score, 2)))