"""
Deployment Validator for HistoCore Project Optimization Analysis System.

Analyzes deployment readiness including Docker, Kubernetes, and CI/CD.
"""

import logging
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any

from src.analysis.models import DeploymentAnalysis


logger = logging.getLogger(__name__)


class DeploymentValidator:
    """Analyzes deployment readiness."""
    
    def __init__(self, project_path: str):
        """
        Initialize validator.
        
        Args:
            project_path: Path to project root directory
        """
        self.project_path = Path(project_path).resolve()
        
    def analyze(self) -> DeploymentAnalysis:
        """
        Run deployment analysis.
        
        Returns:
            DeploymentAnalysis with readiness metrics
        """
        logger.info("Starting deployment analysis...")
        
        # Dockerfile validation
        dockerfile_score = self._validate_dockerfile()
        
        # Kubernetes readiness
        k8s_score = self._validate_k8s_manifests()
        
        # CI/CD completeness
        ci_cd_score = self._assess_ci_cd_pipeline()
        
        # Monitoring score
        monitoring_score = self._calculate_monitoring_score()
        
        # Overall deployment score
        score = self._calculate_deployment_score(
            dockerfile_score, k8s_score, ci_cd_score, monitoring_score
        )
        
        return DeploymentAnalysis(
            dockerfile_score=dockerfile_score,
            k8s_readiness=k8s_score,
            ci_cd_completeness=ci_cd_score,
            monitoring_score=monitoring_score,
            score=score
        )
    
    def _validate_dockerfile(self) -> float:
        """
        Validate Dockerfile best practices.
        
        Checks:
        - Multi-stage builds for smaller images
        - Layer caching optimization (COPY before RUN)
        - Security (non-root user, no secrets)
        - Essential directives (FROM, WORKDIR, CMD)
        
        Returns:
            Score 0-100 based on best practices compliance
        """
        dockerfile = self.project_path / 'Dockerfile'
        
        if not dockerfile.exists():
            logger.info("No Dockerfile found")
            return 0.0
        
        try:
            content = dockerfile.read_text()
            lines = content.split('\n')
            score = 0.0
            
            # Essential directives (reduced points - these are baseline)
            checks = [
                ('FROM', 'Has base image', 8),
                ('WORKDIR', 'Sets working directory', 4),
                ('COPY', 'Copies application files', 4),
                ('RUN', 'Installs dependencies', 4),
                ('CMD', 'Has startup command', 4),
            ]
            
            for keyword, description, points in checks:
                if keyword in content:
                    score += points
                    logger.debug(f"✓ {description}")
                else:
                    logger.debug(f"✗ {description}")
            
            # Multi-stage build (smaller images) - IMPORTANT
            from_count = content.count('FROM')
            if from_count > 1:
                score += 20
                logger.debug(f"✓ Multi-stage build ({from_count} stages)")
            else:
                logger.debug("✗ Single-stage build (consider multi-stage)")
            
            # Layer caching optimization
            # Check if requirements/dependencies copied before app code
            copy_indices = [i for i, line in enumerate(lines) if 'COPY' in line]
            run_indices = [i for i, line in enumerate(lines) if 'RUN' in line and 'pip install' in line.lower()]
            
            if copy_indices and run_indices:
                # Good: COPY requirements.txt before RUN pip install
                req_copy = any('requirements' in lines[i].lower() or 'pyproject' in lines[i].lower() 
                              for i in copy_indices)
                if req_copy and any(c < r for c in copy_indices for r in run_indices):
                    score += 15
                    logger.debug("✓ Layer caching optimized (deps before code)")
                else:
                    logger.debug("✗ Layer caching not optimized")
            
            # Security: non-root user - CRITICAL
            if 'USER' in content and 'USER root' not in content:
                score += 25
                logger.debug("✓ Runs as non-root user")
            else:
                logger.debug("✗ Running as root (security risk)")
            
            # Security: no hardcoded secrets - CRITICAL
            secret_patterns = ['password', 'api_key', 'secret', 'token']
            has_secrets = any(pattern in content.lower() for pattern in secret_patterns)
            if not has_secrets:
                score += 10
                logger.debug("✓ No hardcoded secrets detected")
            else:
                logger.warning("✗ Potential hardcoded secrets detected")
                score -= 10  # Penalty for secrets
            
            # Health check
            if 'HEALTHCHECK' in content:
                score += 10
                logger.debug("✓ Has health check")
            
            # Minimal layers (combine RUN commands)
            run_count = content.count('\nRUN')
            if run_count <= 3:
                score += 10
                logger.debug(f"✓ Minimal RUN layers ({run_count})")
            elif run_count > 10:
                logger.debug(f"✗ Too many RUN layers ({run_count}), consider combining")
            
            return max(0.0, min(100.0, score))
        
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to validate Dockerfile: {e}")
            return 0.0
    
    def _validate_k8s_manifests(self) -> float:
        """Validate Kubernetes manifests for best practices."""
        k8s_dir = self.project_path / 'k8s'
        
        if not k8s_dir.exists():
            logger.info("No k8s directory found")
            return 0.0
        
        score = 0.0
        yaml_files = list(k8s_dir.glob('*.yaml')) + list(k8s_dir.glob('*.yml'))
        
        if not yaml_files:
            return 0.0
        
        # Check for essential K8s resources
        resource_types = set()
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    docs = yaml.safe_load_all(f)
                    for doc in docs:
                        if doc and 'kind' in doc:
                            resource_types.add(doc['kind'])
            except (yaml.YAMLError, OSError) as e:
                logger.warning(f"Failed to parse {yaml_file}: {e}")
                continue
        
        # Score based on resource types
        essential_resources = {
            'Deployment': 30,
            'Service': 20,
            'ConfigMap': 15,
            'Secret': 15,
            'Ingress': 10,
            'PersistentVolumeClaim': 10,
        }
        
        for resource, points in essential_resources.items():
            if resource in resource_types:
                score += points
                logger.debug(f"✓ Has {resource}")
        
        return min(100.0, score)
    
    def _assess_ci_cd_pipeline(self) -> float:
        """Assess CI/CD pipeline completeness."""
        workflows_dir = self.project_path / '.github' / 'workflows'
        
        if not workflows_dir.exists():
            logger.info("No GitHub workflows found")
            return 0.0
        
        score = 0.0
        yaml_files = list(workflows_dir.glob('*.yaml')) + list(workflows_dir.glob('*.yml'))
        
        if not yaml_files:
            return 0.0
        
        # Check for CI/CD stages
        all_content = ""
        for yaml_file in yaml_files:
            try:
                all_content += yaml_file.read_text()
            except (OSError, UnicodeDecodeError):
                continue
        
        # CI/CD stages checklist
        stages = [
            ('checkout', 'Code checkout', 15),
            ('test', 'Testing stage', 25),
            ('build', 'Build stage', 20),
            ('docker', 'Docker build', 15),
            ('deploy', 'Deployment stage', 15),
            ('security', 'Security scanning', 10),
        ]
        
        for keyword, description, points in stages:
            if keyword in all_content.lower():
                score += points
                logger.debug(f"✓ {description}")
        
        return min(100.0, score)
    
    def _calculate_monitoring_score(self) -> float:
        """Calculate monitoring and logging setup score (placeholder)."""
        # TODO: Check for monitoring configuration
        logger.info("Monitoring assessment not yet implemented")
        return 50.0  # Default neutral score
    
    def _calculate_deployment_score(
        self,
        dockerfile_score: float,
        k8s_score: float,
        ci_cd_score: float,
        monitoring_score: float
    ) -> float:
        """
        Calculate overall deployment score (0-100).
        
        Weights:
        - CI/CD: 40%
        - Dockerfile: 25%
        - Kubernetes: 25%
        - Monitoring: 10%
        """
        score = (
            ci_cd_score * 0.40 +
            dockerfile_score * 0.25 +
            k8s_score * 0.25 +
            monitoring_score * 0.10
        )
        
        return round(score, 2)