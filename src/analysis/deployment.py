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
        k8s_score = self._assess_k8s_readiness()
        
        # CI/CD completeness
        cicd_score = self._assess_cicd_completeness()
        
        # Monitoring score (placeholder)
        monitoring_score = self._assess_monitoring()
        
        # Overall deployment score
        score = self._calculate_deployment_score(
            dockerfile_score, k8s_score, cicd_score, monitoring_score
        )
        
        return DeploymentAnalysis(
            dockerfile_score=dockerfile_score,
            k8s_readiness=k8s_score,
            ci_cd_completeness=cicd_score,
            monitoring_score=monitoring_score,
            score=score
        )
    
    def _validate_dockerfile(self) -> float:
        """Validate Dockerfile best practices."""
        dockerfile = self.project_path / 'Dockerfile'
        
        if not dockerfile.exists():
            logger.info("No Dockerfile found")
            return 0.0
        
        try:
            content = dockerfile.read_text()
            score = 0.0
            
            # Best practices checklist
            checks = [
                ('FROM', 'Has base image', 20),
                ('WORKDIR', 'Sets working directory', 15),
                ('COPY', 'Copies application files', 15),
                ('RUN', 'Installs dependencies', 15),
                ('EXPOSE', 'Exposes ports', 10),
                ('CMD', 'Has startup command', 15),
                ('USER', 'Runs as non-root user', 10),
            ]
            
            for keyword, description, points in checks:
                if keyword in content:
                    score += points
                    logger.debug(f"✓ {description}")
                else:
                    logger.debug(f"✗ {description}")
            
            # Multi-stage build bonus
            if content.count('FROM') > 1:
                score += 10
                logger.debug("✓ Multi-stage build")
            
            return min(100.0, score)
        
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to validate Dockerfile: {e}")
            return 0.0
    
    def _assess_k8s_readiness(self) -> float:
        """Assess Kubernetes deployment readiness."""
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
    
    def _assess_cicd_completeness(self) -> float:
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
    
    def _assess_monitoring(self) -> float:
        """Assess monitoring and logging setup (placeholder)."""
        # TODO: Check for monitoring configuration
        logger.info("Monitoring assessment not yet implemented")
        return 50.0  # Default neutral score
    
    def _calculate_deployment_score(
        self,
        dockerfile_score: float,
        k8s_score: float,
        cicd_score: float,
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
            cicd_score * 0.40 +
            dockerfile_score * 0.25 +
            k8s_score * 0.25 +
            monitoring_score * 0.10
        )
        
        return round(score, 2)