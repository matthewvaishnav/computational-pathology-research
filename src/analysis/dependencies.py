"""
Dependency Auditor for HistoCore Project Optimization Analysis System.

Analyzes dependency security, health, and license compatibility.
"""

import json
import logging
import subprocess
import toml
from pathlib import Path
from typing import List, Dict, Any, Optional

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
            DependencyAnalysis with security and health metrics
        """
        logger.info("Starting dependency analysis...")
        
        # Parse dependency files
        dependencies = self._parse_dependencies()
        total_deps = len(dependencies)
        
        # CVE scanning
        vulnerabilities = self._scan_vulnerabilities()
        
        # Outdated packages
        outdated_packages = self._find_outdated_packages()
        
        # License issues
        license_issues = self._validate_licenses()
        
        # Calculate overall score
        score = self._calculate_dependency_score(
            total_deps, len(vulnerabilities), len(outdated_packages), len(license_issues)
        )
        
        return DependencyAnalysis(
            total_dependencies=total_deps,
            vulnerabilities=vulnerabilities,
            outdated_packages=outdated_packages,
            license_issues=license_issues,
            score=score
        )
    
    def _parse_dependencies(self) -> List[Dict[str, str]]:
        """Parse dependencies from requirements.txt and pyproject.toml."""
        dependencies = []
        
        # Parse requirements.txt
        req_file = self.project_path / 'requirements.txt'
        if req_file.exists():
            try:
                with open(req_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Parse package name and version
                            if '==' in line:
                                name, version = line.split('==', 1)
                            elif '>=' in line:
                                name, version = line.split('>=', 1)
                            elif '<=' in line:
                                name, version = line.split('<=', 1)
                            elif '>' in line:
                                name, version = line.split('>', 1)
                            elif '<' in line:
                                name, version = line.split('<', 1)
                            else:
                                name, version = line, 'latest'
                            
                            dependencies.append({
                                'name': name.strip(),
                                'version': version.strip(),
                                'source': 'requirements.txt'
                            })
            except Exception as e:
                logger.warning(f"Failed to parse requirements.txt: {e}")
        
        # Parse pyproject.toml
        pyproject_file = self.project_path / 'pyproject.toml'
        if pyproject_file.exists():
            try:
                with open(pyproject_file, 'r', encoding='utf-8') as f:
                    data = toml.load(f)
                
                # Parse dependencies from different sections
                sections = [
                    ('project', 'dependencies'),
                    ('tool', 'poetry', 'dependencies'),
                    ('build-system', 'requires')
                ]
                
                for section_path in sections:
                    current = data
                    for key in section_path:
                        if isinstance(current, dict) and key in current:
                            current = current[key]
                        else:
                            current = None
                            break
                    
                    if current and isinstance(current, list):
                        for dep in current:
                            if isinstance(dep, str):
                                # Parse package name and version constraint
                                if '>=' in dep:
                                    name, version = dep.split('>=', 1)
                                elif '==' in dep:
                                    name, version = dep.split('==', 1)
                                elif '~=' in dep:
                                    name, version = dep.split('~=', 1)
                                else:
                                    name, version = dep, 'latest'
                                
                                dependencies.append({
                                    'name': name.strip(),
                                    'version': version.strip(),
                                    'source': 'pyproject.toml'
                                })
                    elif current and isinstance(current, dict):
                        for name, version_spec in current.items():
                            if name != 'python':  # Skip Python version
                                dependencies.append({
                                    'name': name,
                                    'version': str(version_spec) if version_spec else 'latest',
                                    'source': 'pyproject.toml'
                                })
            
            except Exception as e:
                logger.warning(f"Failed to parse pyproject.toml: {e}")
        
        return dependencies
    
    def _scan_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Scan for CVEs using pip-audit and safety."""
        vulnerabilities = []
        
        # Try pip-audit first
        try:
            result = subprocess.run(
                ['pip-audit', '--format=json', '--desc'],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                cwd=self.project_path
            )
            
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                
                for vuln in data.get('vulnerabilities', []):
                    vulnerabilities.append({
                        'package': vuln.get('package', 'unknown'),
                        'version': vuln.get('installed_version', 'unknown'),
                        'vulnerability_id': vuln.get('id', 'unknown'),
                        'description': vuln.get('description', ''),
                        'severity': self._map_severity(vuln.get('aliases', [])),
                        'fixed_versions': vuln.get('fix_versions', []),
                        'source': 'pip-audit'
                    })
        
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"pip-audit failed: {e}")
        
        # Try safety as fallback
        if not vulnerabilities:
            try:
                result = subprocess.run(
                    ['safety', 'check', '--json'],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                    cwd=self.project_path
                )
                
                if result.returncode != 0 and result.stdout:
                    # safety returns non-zero when vulnerabilities found
                    data = json.loads(result.stdout)
                    
                    for vuln in data:
                        vulnerabilities.append({
                            'package': vuln.get('package_name', 'unknown'),
                            'version': vuln.get('installed_version', 'unknown'),
                            'vulnerability_id': vuln.get('vulnerability_id', 'unknown'),
                            'description': vuln.get('advisory', ''),
                            'severity': 'medium',  # safety doesn't provide severity
                            'fixed_versions': [vuln.get('spec', '')],
                            'source': 'safety'
                        })
            
            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"safety check failed: {e}")
        
        return vulnerabilities[:50]  # Limit to 50 vulnerabilities
    
    def _map_severity(self, aliases: List[str]) -> str:
        """Map CVE aliases to severity levels."""
        # Look for CVSS scores or severity indicators
        for alias in aliases:
            alias_lower = alias.lower()
            if 'critical' in alias_lower or 'cvss:9' in alias_lower or 'cvss:10' in alias_lower:
                return 'critical'
            elif 'high' in alias_lower or 'cvss:7' in alias_lower or 'cvss:8' in alias_lower:
                return 'high'
            elif 'medium' in alias_lower or 'cvss:4' in alias_lower or 'cvss:5' in alias_lower or 'cvss:6' in alias_lower:
                return 'medium'
            elif 'low' in alias_lower or 'cvss:1' in alias_lower or 'cvss:2' in alias_lower or 'cvss:3' in alias_lower:
                return 'low'
        
        return 'medium'  # Default severity
    
    def _find_outdated_packages(self) -> List[str]:
        """Find outdated packages using pip list --outdated."""
        outdated = []
        
        try:
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                cwd=self.project_path
            )
            
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                
                for package in data:
                    package_name = package.get('name', 'unknown')
                    current_version = package.get('version', 'unknown')
                    latest_version = package.get('latest_version', 'unknown')
                    
                    outdated.append(f"{package_name} ({current_version} -> {latest_version})")
        
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to check outdated packages: {e}")
        
        return outdated[:30]  # Limit to 30 packages
    
    def _validate_licenses(self) -> List[str]:
        """Validate license compatibility using pip-licenses."""
        license_issues = []
        
        # Define problematic licenses
        problematic_licenses = {
            'GPL-2.0', 'GPL-3.0', 'AGPL-3.0', 'LGPL-2.1', 'LGPL-3.0',
            'SSPL-1.0', 'OSL-3.0', 'EPL-1.0', 'EPL-2.0', 'MPL-2.0'
        }
        
        try:
            result = subprocess.run(
                ['pip-licenses', '--format=json'],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                cwd=self.project_path
            )
            
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                
                for package in data:
                    package_name = package.get('Name', 'unknown')
                    license_name = package.get('License', 'unknown')
                    
                    # Check for problematic licenses
                    if any(prob_license in license_name for prob_license in problematic_licenses):
                        license_issues.append(f"{package_name}: {license_name}")
                    
                    # Check for unknown/missing licenses
                    elif license_name.lower() in ['unknown', 'none', '', 'null']:
                        license_issues.append(f"{package_name}: Unknown license")
        
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to check licenses: {e}")
        
        return license_issues[:20]  # Limit to 20 issues
    
    def _calculate_dependency_score(
        self,
        total_deps: int,
        vuln_count: int,
        outdated_count: int,
        license_issues_count: int
    ) -> float:
        """
        Calculate dependency health score (0-100).
        
        Scoring:
        - Vulnerabilities: -20 points per critical, -10 per high, -5 per medium, -2 per low
        - Outdated packages: -1 point per outdated package
        - License issues: -5 points per problematic license
        - Base score: 100
        """
        score = 100.0
        
        # Penalize vulnerabilities (assume medium severity if not specified)
        score -= vuln_count * 5
        
        # Penalize outdated packages
        if total_deps > 0:
            outdated_ratio = outdated_count / total_deps
            score -= outdated_ratio * 20  # Up to 20 points for all outdated
        
        # Penalize license issues
        score -= license_issues_count * 5
        
        return max(0.0, min(100.0, round(score, 2)))
