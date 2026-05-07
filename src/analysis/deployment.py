"""
Deployment Validator for HistoCore Project Optimization Analysis System.

Analyzes deployment readiness including Docker, Kubernetes, and CI/CD.
"""

import logging
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any

from .models import DeploymentAnalysis


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
        """
        Validate Kubernetes manifests for best practices.
        
        Checks:
        - Essential resources (Deployment, Service, ConfigMap, etc.)
        - Resource limits (CPU, memory)
        - Health checks (liveness, readiness probes)
        - Security (non-root, read-only filesystem)
        
        Returns:
            Score 0-100 based on K8s best practices
        """
        k8s_dir = self.project_path / 'k8s'
        
        if not k8s_dir.exists():
            logger.info("No k8s directory found")
            return 0.0
        
        yaml_files = list(k8s_dir.glob('*.yaml')) + list(k8s_dir.glob('*.yml'))
        
        if not yaml_files:
            return 0.0
        
        score = 0.0
        resource_types = set()
        deployments = []
        
        # Parse all manifests
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    docs = yaml.safe_load_all(f)
                    for doc in docs:
                        if doc and 'kind' in doc:
                            resource_types.add(doc['kind'])
                            if doc['kind'] == 'Deployment':
                                deployments.append(doc)
            except (yaml.YAMLError, OSError) as e:
                logger.warning(f"Failed to parse {yaml_file}: {e}")
                continue
        
        # Score based on resource types (20 pts)
        essential_resources = {
            'Deployment': 10,
            'Service': 5,
            'ConfigMap': 5,
        }
        
        for resource, points in essential_resources.items():
            if resource in resource_types:
                score += points
                logger.debug(f"✓ Has {resource}")
        
        # Validate Deployment best practices (80 pts)
        if deployments:
            deployment_score = self._validate_deployment_best_practices(deployments)
            score += deployment_score
        else:
            logger.debug("✗ No Deployment resources found")
        
        return min(100.0, score)
    
    def _validate_deployment_best_practices(self, deployments: list) -> float:
        """
        Validate Deployment manifest best practices.
        
        Args:
            deployments: List of Deployment manifest dicts
            
        Returns:
            Score 0-80 based on best practices
        """
        score = 0.0
        total_deployments = len(deployments)
        
        has_resource_limits = 0
        has_liveness_probe = 0
        has_readiness_probe = 0
        has_security_context = 0
        has_replicas = 0
        
        for deployment in deployments:
            try:
                spec = deployment.get('spec', {})
                template = spec.get('template', {})
                pod_spec = template.get('spec', {})
                containers = pod_spec.get('containers', [])
                
                # Check replicas (high availability)
                replicas = spec.get('replicas', 1)
                if replicas >= 2:
                    has_replicas += 1
                
                # Check if ANY container has these features
                deployment_has_limits = False
                deployment_has_liveness = False
                deployment_has_readiness = False
                
                for container in containers:
                    # Resource limits
                    resources = container.get('resources', {})
                    if 'limits' in resources and 'requests' in resources:
                        deployment_has_limits = True
                    
                    # Liveness probe
                    if 'livenessProbe' in container:
                        deployment_has_liveness = True
                    
                    # Readiness probe
                    if 'readinessProbe' in container:
                        deployment_has_readiness = True
                
                if deployment_has_limits:
                    has_resource_limits += 1
                if deployment_has_liveness:
                    has_liveness_probe += 1
                if deployment_has_readiness:
                    has_readiness_probe += 1
                
                # Security context
                security_context = pod_spec.get('securityContext', {})
                if security_context.get('runAsNonRoot') or security_context.get('runAsUser'):
                    has_security_context += 1
                    
            except (KeyError, TypeError) as e:
                logger.warning(f"Failed to validate deployment: {e}")
                continue
        
        # Calculate percentages and assign points
        if total_deployments > 0:
            if has_resource_limits / total_deployments >= 0.5:
                score += 25
                logger.debug("✓ Resource limits configured")
            else:
                logger.debug("✗ Missing resource limits")
            
            if has_liveness_probe / total_deployments >= 0.5:
                score += 20
                logger.debug("✓ Liveness probes configured")
            else:
                logger.debug("✗ Missing liveness probes")
            
            if has_readiness_probe / total_deployments >= 0.5:
                score += 20
                logger.debug("✓ Readiness probes configured")
            else:
                logger.debug("✗ Missing readiness probes")
            
            if has_security_context / total_deployments >= 0.5:
                score += 5
                logger.debug("✓ Security context configured")
            else:
                logger.debug("✗ Missing security context")
            
            if has_replicas / total_deployments >= 0.5:
                score += 10
                logger.debug("✓ High availability (replicas >= 2)")
            else:
                logger.debug("✗ Low replica count")
        
        return score
    
    def _assess_ci_cd_pipeline(self) -> float:
        """
        Assess CI/CD pipeline completeness.
        
        Checks:
        - Build, test, deploy stages
        - Security scanning (bandit, safety, etc.)
        - Code quality checks (linting, type checking)
        - Artifact management
        
        Returns:
            Score 0-100 based on CI/CD completeness
        """
        workflows_dir = self.project_path / '.github' / 'workflows'
        
        if not workflows_dir.exists():
            logger.info("No GitHub workflows found")
            return 0.0
        
        yaml_files = list(workflows_dir.glob('*.yaml')) + list(workflows_dir.glob('*.yml'))
        
        if not yaml_files:
            return 0.0
        
        score = 0.0
        all_content = ""
        workflow_configs = []
        
        # Parse all workflows
        for yaml_file in yaml_files:
            try:
                content = yaml_file.read_text()
                all_content += content.lower()
                
                # Parse YAML for structured analysis
                with open(yaml_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if config:
                        workflow_configs.append(config)
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
                logger.warning(f"Failed to parse {yaml_file}: {e}")
                continue
        
        # Essential CI/CD stages (60 pts)
        stages = [
            ('checkout', 'Code checkout', 10),
            ('test', 'Testing stage', 20),
            ('build', 'Build stage', 15),
            ('deploy', 'Deployment stage', 15),
        ]
        
        for keyword, description, points in stages:
            if keyword in all_content:
                score += points
                logger.debug(f"✓ {description}")
            else:
                logger.debug(f"✗ Missing {description}")
        
        # Security and quality checks (40 pts)
        quality_checks = [
            (['bandit', 'security', 'safety', 'snyk'], 'Security scanning', 15),
            (['lint', 'pylint', 'flake8', 'ruff'], 'Code linting', 10),
            (['mypy', 'type', 'pyright'], 'Type checking', 5),
            (['docker', 'container'], 'Container build', 5),
            (['artifact', 'upload'], 'Artifact management', 5),
        ]
        
        for keywords, description, points in quality_checks:
            if any(kw in all_content for kw in keywords):
                score += points
                logger.debug(f"✓ {description}")
            else:
                logger.debug(f"✗ Missing {description}")
        
        return min(100.0, score)
    
    def _calculate_monitoring_score(self) -> float:
        """
        Calculate monitoring and logging setup score.
        
        Checks for:
        - Prometheus metrics endpoints
        - Logging configuration
        - Health check endpoints
        - Distributed tracing setup
        
        Returns:
            Score from 0-100
        """
        score = 0.0
        
        try:
            # Check for Prometheus metrics (25 points)
            prometheus_files = list(self.project_path.rglob('*prometheus*.py'))
            prometheus_files += list(self.project_path.rglob('*metrics*.py'))
            
            if prometheus_files:
                score += 25.0
                logger.debug("Found Prometheus/metrics files")
            
            # Check for logging configuration (25 points)
            logging_configs = []
            for pattern in ['logging.yaml', 'logging.json', 'logging.conf', 'logconfig.py']:
                logging_configs.extend(list(self.project_path.rglob(pattern)))
            
            if logging_configs:
                score += 25.0
                logger.debug("Found logging configuration")
            
            # Check for health check endpoints (25 points)
            health_check_found = False
            for py_file in self.project_path.rglob('*.py'):
                try:
                    content = py_file.read_text()
                    if any(endpoint in content for endpoint in ['/health', '/healthz', '/ready', '/liveness']):
                        health_check_found = True
                        break
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            if health_check_found:
                score += 25.0
                logger.debug("Found health check endpoints")
            
            # Check for distributed tracing (25 points)
            tracing_keywords = ['opentelemetry', 'jaeger', 'zipkin', 'trace', 'span']
            tracing_found = False
            
            for py_file in self.project_path.rglob('*.py'):
                try:
                    content = py_file.read_text()
                    if any(keyword in content.lower() for keyword in tracing_keywords):
                        tracing_found = True
                        break
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            if tracing_found:
                score += 25.0
                logger.debug("Found distributed tracing setup")
        
        except Exception as e:
            logger.debug(f"Monitoring score calculation error: {e}")
            return 50.0  # Default neutral score
        
        return score
    
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
