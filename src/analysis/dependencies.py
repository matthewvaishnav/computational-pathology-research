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

from .models import DependencyAnalysis

logger = logging.getLogger(__name__)


class DependencyAuditor:
    """Analyzes dependency security and health."""

    def __init__(self, project_path: str):
        """
        Initialize auditor.

        Args:
            project_path: Path to project root directory

        Raises:
            ValueError: If project_path is empty or None
            FileNotFoundError: If project_path does not exist
        """
        if not project_path or not project_path.strip():
            raise ValueError("project_path cannot be empty")

        self.project_path = Path(project_path).resolve()

        if not self.project_path.exists():
            raise FileNotFoundError(f"Project path does not exist: {self.project_path}")

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

        # Dependency bloat detection
        unused_deps, redundant_deps = self._detect_dependency_bloat(dependencies)

        # License issues
        license_issues = self._validate_licenses()

        # Calculate overall score
        score = self._calculate_dependency_score(
            total_deps,
            len(vulnerabilities),
            len(outdated_packages),
            len(license_issues),
            len(unused_deps),
            len(redundant_deps),
        )

        return DependencyAnalysis(
            total_dependencies=total_deps,
            vulnerabilities=vulnerabilities,
            outdated_packages=outdated_packages,
            license_issues=license_issues,
            score=score,
            unused_dependencies=unused_deps,
            redundant_dependencies=redundant_deps,
            security_report=self._generate_security_report(vulnerabilities, outdated_packages),
        )

    def _parse_dependencies(self) -> List[Dict[str, str]]:
        """Parse dependencies from requirements.txt and pyproject.toml."""
        dependencies = []

        # Parse requirements.txt
        req_file = self.project_path / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # Parse package name and version
                            if "==" in line:
                                name, version = line.split("==", 1)
                            elif ">=" in line:
                                name, version = line.split(">=", 1)
                            elif "<=" in line:
                                name, version = line.split("<=", 1)
                            elif ">" in line:
                                name, version = line.split(">", 1)
                            elif "<" in line:
                                name, version = line.split("<", 1)
                            else:
                                name, version = line, "latest"

                            dependencies.append(
                                {
                                    "name": name.strip(),
                                    "version": version.strip(),
                                    "source": "requirements.txt",
                                }
                            )
            except Exception as e:
                logger.warning(f"Failed to parse requirements.txt: {e}")

        # Parse pyproject.toml
        pyproject_file = self.project_path / "pyproject.toml"
        if pyproject_file.exists():
            try:
                with open(pyproject_file, "r", encoding="utf-8") as f:
                    data = toml.load(f)

                # Parse dependencies from different sections
                sections = [
                    ("project", "dependencies"),
                    ("tool", "poetry", "dependencies"),
                    ("build-system", "requires"),
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
                                if ">=" in dep:
                                    name, version = dep.split(">=", 1)
                                elif "==" in dep:
                                    name, version = dep.split("==", 1)
                                elif "~=" in dep:
                                    name, version = dep.split("~=", 1)
                                else:
                                    name, version = dep, "latest"

                                dependencies.append(
                                    {
                                        "name": name.strip(),
                                        "version": version.strip(),
                                        "source": "pyproject.toml",
                                    }
                                )
                    elif current and isinstance(current, dict):
                        for name, version_spec in current.items():
                            if name != "python":  # Skip Python version
                                dependencies.append(
                                    {
                                        "name": name,
                                        "version": str(version_spec) if version_spec else "latest",
                                        "source": "pyproject.toml",
                                    }
                                )

            except Exception as e:
                logger.warning(f"Failed to parse pyproject.toml: {e}")

        return dependencies

    def _scan_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Scan for CVEs using pip-audit and safety."""
        vulnerabilities = []

        # Try pip-audit first
        try:
            result = subprocess.run(
                ["pip-audit", "--format=json", "--desc"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                cwd=self.project_path,
            )

            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)

                for vuln in data.get("vulnerabilities", []):
                    vulnerabilities.append(
                        {
                            "package": vuln.get("package", "unknown"),
                            "version": vuln.get("installed_version", "unknown"),
                            "vulnerability_id": vuln.get("id", "unknown"),
                            "description": vuln.get("description", ""),
                            "severity": self._map_severity(vuln.get("aliases", [])),
                            "fixed_versions": vuln.get("fix_versions", []),
                            "source": "pip-audit",
                        }
                    )

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"pip-audit failed: {e}")

        # Try safety as fallback
        if not vulnerabilities:
            try:
                result = subprocess.run(
                    ["safety", "check", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                    cwd=self.project_path,
                )

                if result.returncode != 0 and result.stdout:
                    # safety returns non-zero when vulnerabilities found
                    data = json.loads(result.stdout)

                    for vuln in data:
                        vulnerabilities.append(
                            {
                                "package": vuln.get("package_name", "unknown"),
                                "version": vuln.get("installed_version", "unknown"),
                                "vulnerability_id": vuln.get("vulnerability_id", "unknown"),
                                "description": vuln.get("advisory", ""),
                                "severity": "medium",  # safety doesn't provide severity
                                "fixed_versions": [vuln.get("spec", "")],
                                "source": "safety",
                            }
                        )

            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"safety check failed: {e}")

        return vulnerabilities[:50]  # Limit to 50 vulnerabilities

    def _map_severity(self, aliases: List[str]) -> str:
        """Map CVE aliases to severity levels."""
        # Look for CVSS scores or severity indicators
        for alias in aliases:
            alias_lower = alias.lower()
            if "critical" in alias_lower or "cvss:9" in alias_lower or "cvss:10" in alias_lower:
                return "critical"
            elif "high" in alias_lower or "cvss:7" in alias_lower or "cvss:8" in alias_lower:
                return "high"
            elif (
                "medium" in alias_lower
                or "cvss:4" in alias_lower
                or "cvss:5" in alias_lower
                or "cvss:6" in alias_lower
            ):
                return "medium"
            elif (
                "low" in alias_lower
                or "cvss:1" in alias_lower
                or "cvss:2" in alias_lower
                or "cvss:3" in alias_lower
            ):
                return "low"

        return "medium"  # Default severity

    def _find_outdated_packages(self) -> List[str]:
        """
        Find outdated packages using pip list --outdated.

        Prioritizes security updates over feature updates by checking
        for security patches in new versions.
        """
        outdated = []

        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                cwd=self.project_path,
            )

            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)

                # Sort packages by priority (security updates first)
                security_updates = []
                feature_updates = []

                for package in data:
                    package_name = package.get("name", "unknown")
                    current_version = package.get("version", "unknown")
                    latest_version = package.get("latest_version", "unknown")
                    package_type = package.get("latest_filetype", "wheel")

                    # Check if this might be a security update
                    is_security_update = self._is_security_update(
                        package_name, current_version, latest_version
                    )

                    package_info = {
                        "name": package_name,
                        "current_version": current_version,
                        "latest_version": latest_version,
                        "type": package_type,
                        "is_security": is_security_update,
                        "upgrade_command": f"pip install --upgrade {package_name}=={latest_version}",
                    }

                    if is_security_update:
                        security_updates.append(package_info)
                    else:
                        feature_updates.append(package_info)

                # Format output with security updates first
                for package in security_updates:
                    outdated.append(
                        f"🔒 {package['name']} ({package['current_version']} -> {package['latest_version']}) [SECURITY]"
                    )

                for package in feature_updates:
                    outdated.append(
                        f"{package['name']} ({package['current_version']} -> {package['latest_version']})"
                    )

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to check outdated packages: {e}")

        return outdated[:30]  # Limit to 30 packages

    def _is_security_update(
        self, package_name: str, current_version: str, latest_version: str
    ) -> bool:
        """
        Determine if an update is likely a security patch.

        Uses heuristics to identify security updates:
        - Patch version bumps (x.y.z -> x.y.z+1)
        - Known security-focused packages
        - Version patterns indicating security fixes
        """
        try:
            # Parse version numbers
            current_parts = [int(x) for x in current_version.split(".") if x.isdigit()]
            latest_parts = [int(x) for x in latest_version.split(".") if x.isdigit()]

            if len(current_parts) >= 3 and len(latest_parts) >= 3:
                # Check if it's a patch version bump (major.minor.patch)
                if (
                    current_parts[0] == latest_parts[0]
                    and current_parts[1] == latest_parts[1]
                    and latest_parts[2] > current_parts[2]
                ):
                    return True

            # Security-focused packages that should be prioritized
            security_packages = {
                "cryptography",
                "pycryptodome",
                "requests",
                "urllib3",
                "certifi",
                "pillow",
                "lxml",
                "pyyaml",
                "jinja2",
                "werkzeug",
                "flask",
                "django",
                "sqlalchemy",
                "psycopg2",
                "pymongo",
                "redis",
                "paramiko",
                "pyopenssl",
                "bcrypt",
                "passlib",
            }

            if package_name.lower() in security_packages:
                return True

            # Check for security-related version patterns
            version_indicators = ["security", "sec", "cve", "vuln", "patch", "fix"]
            latest_lower = latest_version.lower()

            for indicator in version_indicators:
                if indicator in latest_lower:
                    return True

        except (ValueError, IndexError):
            # If version parsing fails, assume it's not a security update
            pass

        return False

    def _detect_dependency_bloat(
        self, dependencies: List[Dict[str, str]]
    ) -> tuple[List[str], List[str]]:
        """
        Detect unused and redundant dependencies.

        Returns:
            Tuple of (unused_dependencies, redundant_dependencies)
        """
        unused_deps = []
        redundant_deps = []

        try:
            # Get list of imported packages from codebase
            imported_packages = self._get_imported_packages()

            # Check for unused dependencies
            for dep in dependencies:
                dep_name = dep["name"].lower()

                # Skip common build/dev dependencies
                if dep_name in {
                    "pip",
                    "setuptools",
                    "wheel",
                    "build",
                    "twine",
                    "pytest",
                    "coverage",
                }:
                    continue

                # Check if package is imported anywhere
                is_used = False

                # Check direct import
                if dep_name in imported_packages:
                    is_used = True

                # Check common name variations
                name_variations = [
                    dep_name.replace("-", "_"),  # package-name -> package_name
                    dep_name.replace("_", "-"),  # package_name -> package-name
                    dep_name.split("-")[0],  # first part of hyphenated name
                    dep_name.split("_")[0],  # first part of underscored name
                ]

                for variation in name_variations:
                    if variation in imported_packages:
                        is_used = True
                        break

                # Check for common package aliases
                package_aliases = {
                    "pillow": "pil",
                    "pyyaml": "yaml",
                    "beautifulsoup4": "bs4",
                    "python-dateutil": "dateutil",
                    "msgpack-python": "msgpack",
                    "pycryptodome": "crypto",
                }

                if dep_name in package_aliases:
                    if package_aliases[dep_name] in imported_packages:
                        is_used = True

                if not is_used:
                    unused_deps.append(f"{dep['name']} (from {dep['source']})")

            # Detect redundant packages (overlapping functionality)
            redundant_deps = self._find_redundant_packages(dependencies)

        except Exception as e:
            logger.warning(f"Failed to detect dependency bloat: {e}")

        return unused_deps[:20], redundant_deps[:10]  # Limit results

    def _get_imported_packages(self) -> set[str]:
        """Get set of all imported package names from the codebase."""
        imported = set()

        try:
            src_dir = self.project_path / "src"
            if not src_dir.exists():
                return imported

            python_files = list(src_dir.rglob("*.py"))

            for py_file in python_files:
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Parse import statements
                    import ast

                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                # Extract top-level package name
                                package = alias.name.split(".")[0]
                                imported.add(package.lower())

                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                # Extract top-level package name
                                package = node.module.split(".")[0]
                                imported.add(package.lower())

                except (SyntaxError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to parse {py_file}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Failed to scan imports: {e}")

        return imported

    def _find_redundant_packages(self, dependencies: List[Dict[str, str]]) -> List[str]:
        """Find packages with overlapping functionality."""
        redundant = []

        # Define groups of packages with overlapping functionality
        overlapping_groups = [
            # HTTP clients
            {"requests", "urllib3", "httpx", "aiohttp"},
            # JSON libraries
            {"simplejson", "ujson", "orjson"},
            # Date/time libraries
            {"python-dateutil", "arrow", "pendulum"},
            # Image processing
            {"pillow", "opencv-python", "imageio", "scikit-image"},
            # Data manipulation
            {"pandas", "polars", "dask"},
            # Testing frameworks
            {"pytest", "unittest2", "nose", "nose2"},
            # Async libraries
            {"asyncio", "trio", "curio"},
            # Serialization
            {"pickle", "dill", "cloudpickle"},
            # Configuration
            {"configparser", "pyyaml", "toml", "json5"},
            # Logging
            {"loguru", "structlog", "colorlog"},
            # CLI frameworks
            {"click", "argparse", "fire", "typer"},
            # Web frameworks
            {"flask", "fastapi", "django", "tornado"},
            # Database ORMs
            {"sqlalchemy", "peewee", "tortoise-orm"},
            # Validation
            {"pydantic", "marshmallow", "cerberus", "schema"},
        ]

        dep_names = {dep["name"].lower() for dep in dependencies}

        for group in overlapping_groups:
            found_in_group = group.intersection(dep_names)
            if len(found_in_group) > 1:
                redundant.append(f"Overlapping functionality: {', '.join(sorted(found_in_group))}")

        return redundant

    def _generate_security_report(
        self, vulnerabilities: List[Dict[str, Any]], outdated_packages: List[str]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive security report with upgrade paths and CVSS scores.

        Returns:
            Security report with vulnerabilities, upgrade commands, and workarounds
        """
        report = {
            "summary": {
                "total_vulnerabilities": len(vulnerabilities),
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
            },
            "vulnerabilities_by_severity": {"critical": [], "high": [], "medium": [], "low": []},
            "upgrade_commands": [],
            "workarounds": [],
            "security_updates": [],
        }

        # Categorize vulnerabilities by severity
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "medium")
            report["summary"][f"{severity}_count"] += 1

            # Add CVSS score if available
            cvss_score = self._extract_cvss_score(vuln.get("vulnerability_id", ""))

            vuln_detail = {
                "package": vuln.get("package", "unknown"),
                "version": vuln.get("version", "unknown"),
                "vulnerability_id": vuln.get("vulnerability_id", "unknown"),
                "description": vuln.get("description", ""),
                "cvss_score": cvss_score,
                "fixed_versions": vuln.get("fixed_versions", []),
                "upgrade_command": self._generate_upgrade_command(
                    vuln.get("package", ""), vuln.get("fixed_versions", [])
                ),
                "workaround": self._suggest_workaround(vuln),
            }

            report["vulnerabilities_by_severity"][severity].append(vuln_detail)

        # Generate upgrade commands for security updates
        security_packages = []
        for package_info in outdated_packages:
            if "🔒" in package_info and "[SECURITY]" in package_info:
                # Extract package name from formatted string
                package_name = package_info.split(" ")[1]  # Skip emoji
                security_packages.append(package_name)

        if security_packages:
            report["upgrade_commands"].append(
                {
                    "type": "security_batch_update",
                    "command": f"pip install --upgrade {' '.join(security_packages)}",
                    "description": "Batch update all packages with security fixes",
                    "packages": security_packages,
                }
            )

        # Add individual upgrade commands for critical vulnerabilities
        for vuln in report["vulnerabilities_by_severity"]["critical"]:
            if vuln["upgrade_command"]:
                report["upgrade_commands"].append(
                    {
                        "type": "critical_fix",
                        "command": vuln["upgrade_command"],
                        "description": f"Fix critical vulnerability {vuln['vulnerability_id']} in {vuln['package']}",
                        "package": vuln["package"],
                        "vulnerability_id": vuln["vulnerability_id"],
                    }
                )

        # Generate workarounds for unpatchable issues
        unpatchable_vulns = [
            vuln
            for vuln in vulnerabilities
            if not vuln.get("fixed_versions") or len(vuln.get("fixed_versions", [])) == 0
        ]

        for vuln in unpatchable_vulns:
            workaround = self._suggest_workaround(vuln)
            if workaround:
                report["workarounds"].append(
                    {
                        "package": vuln.get("package", "unknown"),
                        "vulnerability_id": vuln.get("vulnerability_id", "unknown"),
                        "workaround": workaround,
                        "risk_level": vuln.get("severity", "medium"),
                    }
                )

        return report

    def _extract_cvss_score(self, vulnerability_id: str) -> Optional[float]:
        """Extract CVSS score from vulnerability ID or description."""
        try:
            # Try to extract CVSS score from CVE databases (simplified)
            if "CVE-" in vulnerability_id:
                # In a real implementation, this would query CVE databases
                # For now, return a placeholder based on severity patterns
                if "critical" in vulnerability_id.lower():
                    return 9.0
                elif "high" in vulnerability_id.lower():
                    return 7.5
                elif "medium" in vulnerability_id.lower():
                    return 5.0
                elif "low" in vulnerability_id.lower():
                    return 2.0
        except Exception:
            pass

        return None

    def _generate_upgrade_command(
        self, package_name: str, fixed_versions: List[str]
    ) -> Optional[str]:
        """Generate pip upgrade command for a vulnerable package."""
        if not package_name or not fixed_versions:
            return None

        # Use the latest fixed version
        latest_fix = fixed_versions[-1] if fixed_versions else None
        if latest_fix:
            # Clean version string (remove operators like >=, etc.)
            clean_version = latest_fix.replace(">=", "").replace(">", "").replace("==", "").strip()
            return f"pip install --upgrade {package_name}>={clean_version}"

        return f"pip install --upgrade {package_name}"

    def _suggest_workaround(self, vulnerability: Dict[str, Any]) -> Optional[str]:
        """Suggest workarounds for vulnerabilities that can't be easily patched."""
        package = vulnerability.get("package", "").lower()
        vuln_id = vulnerability.get("vulnerability_id", "")
        description = vulnerability.get("description", "").lower()

        # Common workarounds based on vulnerability types
        if "injection" in description or "sql" in description:
            return "Use parameterized queries and input validation. Consider using an ORM with built-in protections."

        elif "xss" in description or "cross-site" in description:
            return "Implement proper input sanitization and output encoding. Use Content Security Policy (CSP) headers."

        elif "deserialization" in description or "pickle" in description:
            return "Avoid deserializing untrusted data. Use safe serialization formats like JSON instead of pickle."

        elif "path traversal" in description or "directory" in description:
            return "Validate and sanitize file paths. Use allowlists for permitted directories and filenames."

        elif package in ["pillow", "pil"]:
            return "Validate image files before processing. Consider using image processing in sandboxed environments."

        elif package in ["requests", "urllib3"]:
            return "Verify SSL certificates and use the latest TLS versions. Implement request timeouts."

        elif package in ["pyyaml", "yaml"]:
            return "Use yaml.safe_load() instead of yaml.load(). Avoid loading YAML from untrusted sources."

        elif "denial of service" in description or "dos" in description:
            return (
                "Implement rate limiting and resource usage monitoring. Set appropriate timeouts."
            )

        # Generic workarounds
        if not vulnerability.get("fixed_versions"):
            return "Consider using an alternative package or implementing additional security controls around this dependency."

        return None

    def _validate_licenses(self) -> List[str]:
        """Validate license compatibility using pip-licenses."""
        license_issues = []

        # Define problematic licenses
        problematic_licenses = {
            "GPL-2.0",
            "GPL-3.0",
            "AGPL-3.0",
            "LGPL-2.1",
            "LGPL-3.0",
            "SSPL-1.0",
            "OSL-3.0",
            "EPL-1.0",
            "EPL-2.0",
            "MPL-2.0",
        }

        try:
            result = subprocess.run(
                ["pip-licenses", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                cwd=self.project_path,
            )

            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)

                for package in data:
                    package_name = package.get("Name", "unknown")
                    license_name = package.get("License", "unknown")

                    # Check for problematic licenses
                    if any(prob_license in license_name for prob_license in problematic_licenses):
                        license_issues.append(f"{package_name}: {license_name}")

                    # Check for unknown/missing licenses
                    elif license_name.lower() in ["unknown", "none", "", "null"]:
                        license_issues.append(f"{package_name}: Unknown license")

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to check licenses: {e}")

        return license_issues[:20]  # Limit to 20 issues

    def _calculate_dependency_score(
        self,
        total_deps: int,
        vuln_count: int,
        outdated_count: int,
        license_issues_count: int,
        unused_count: int = 0,
        redundant_count: int = 0,
    ) -> float:
        """
        Calculate dependency health score (0-100).

        Scoring:
        - Vulnerabilities: -20 points per critical, -10 per high, -5 per medium, -2 per low
        - Outdated packages: -1 point per outdated package
        - License issues: -5 points per problematic license
        - Unused dependencies: -2 points per unused dependency
        - Redundant packages: -3 points per redundant group
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

        # Penalize unused dependencies
        score -= unused_count * 2

        # Penalize redundant packages
        score -= redundant_count * 3

        return max(0.0, min(100.0, round(score, 2)))
