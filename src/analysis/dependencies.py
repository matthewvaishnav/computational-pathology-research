"""
Dependency Auditor for HistoCore Project Optimization Analysis System.

Analyzes dependency security, outdated packages, and license compatibility.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from src.analysis.models import DependencyAnalysis


logger = logging.getLogger(__name__)


class DependencyAuditor:
    """Analyzes dependency security and health."""
    
    def __init__(self, project_path: str):
        """
        Initialize auditor.
        
        Args:
            project_path: Path to project root directory
        """
        self.project_path = Path(project_path).resolve()
        
    def analyze(self) -> DependencyAnalysis:
        """
        Run dependency analysis.
        
        Returns:
            DependencyAnalysis with security metrics
        """
        logger.info("Starting dependency analysis...")
        
        # Count total dependencies
        total_deps = self._count_dependencies()
        
        # Security vulnerabilities
        vulnerabilities = self._scan_vulnerabilities()
        
        # Outdated packages
        outdated = self._detect_outdated_packages()
        
        # License issues
        license_issues = self._check_license_compatibility()
        
        # Calculate score
        score = self._calculate_dependency_score(vulnerabilities, outdated, license_issues)
        
        return DependencyAnalysis(
            total_dependencies=total_deps,
            vulnerabilities=vulnerabilities,
            outdated_packages=outdated,
            license_issues=license_issues,
            score=score
        )
    
    def _count_dependencies(self) -> int:
        """Count total dependencies from requirements files."""
        count = 0
        
        # Check requirements.txt
        req_file = self.project_path / 'requirements.txt'
        if req_file.exists():
            lines = req_file.read_text().splitlines()
            count += len([line for line in lines if line.strip() and not line.startswith('#')])
        
        # Check pyproject.toml
        pyproject = self.project_path / 'pyproject.toml'
        if pyproject.exists():
            # TODO: Parse TOML dependencies
            pass
        
        return count
    
    def _scan_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Scan for security vulnerabilities using safety."""
        try:
            result = subprocess.run(
                ['safety', 'check', '--json'],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=60,
                check=False
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                
                vulnerabilities = []
                for vuln in data:
                    vulnerabilities.append({
                        'cve_id': vuln.get('id', 'unknown'),
                        'package': vuln.get('package_name', 'unknown'),
                        'severity': vuln.get('severity', 'unknown'),
                        'cvss_score': vuln.get('cvss', 0.0),
                        'fix_version': vuln.get('fixed_in', 'unknown')
                    })
                
                return vulnerabilities
            
            return []
        
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to scan vulnerabilities: {e}")
            return []
    
    def _detect_outdated_packages(self) -> List[str]:
        """Detect outdated packages using pip."""
        try:
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True,
                text=True,
                timeout=60,
                check=False
            )
            
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                return [pkg['name'] for pkg in data]
            
            return []
        
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to detect outdated packages: {e}")
            return []
    
    def _check_license_compatibility(self) -> List[str]:
        """Check license compatibility (placeholder)."""
        # TODO: Use pip-licenses to check compatibility
        logger.info("License compatibility check not yet implemented")
        return []
    
    def _calculate_dependency_score(
        self,
        vulnerabilities: List[Dict[str, Any]],
        outdated: List[str],
        license_issues: List[str]
    ) -> float:
        """
        Calculate dependency score (0-100).
        
        Scoring:
        - No vulnerabilities: 50%
        - Up-to-date packages: 30%
        - License compliance: 20%
        """
        score = 0.0
        
        # Vulnerability penalty (critical: -50, high: -30, medium: -15, low: -5)
        vuln_penalty = 0
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'unknown').lower()
            if severity == 'critical':
                vuln_penalty += 50
            elif severity == 'high':
                vuln_penalty += 30
            elif severity == 'medium':
                vuln_penalty += 15
            else:
                vuln_penalty += 5
        
        score += max(0, 50 - vuln_penalty)
        
        # Outdated packages (penalty for >10% outdated)
        if len(outdated) == 0:
            score += 30.0
        else:
            # Assume 100 total packages for calculation
            outdated_pct = len(outdated) / max(100, len(outdated) * 10)
            if outdated_pct <= 0.1:
                score += 30.0
            else:
                score += 30.0 * max(0, 1.0 - (outdated_pct - 0.1) / 0.2)
        
        # License compliance
        if len(license_issues) == 0:
            score += 20.0
        else:
            score += 20.0 * max(0, 1.0 - len(license_issues) / 10)
        
        return max(0.0, min(100.0, round(score, 2)))