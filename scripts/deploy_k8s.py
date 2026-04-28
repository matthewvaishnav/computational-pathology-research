#!/usr/bin/env python3
"""
Kubernetes Deployment Script for Medical AI Platform

Deploys the complete Medical AI platform to Kubernetes with monitoring,
auto-scaling, and production-ready configuration.
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KubernetesDeployer:
    """Deploys Medical AI platform to Kubernetes."""
    
    def __init__(self, namespace: str = "medical-ai", context: str = None):
        """Initialize deployer.
        
        Args:
            namespace: Kubernetes namespace
            context: Kubernetes context to use
        """
        self.namespace = namespace
        self.context = context
        self.k8s_dir = Path(__file__).parent.parent / "k8s"
        
        logger.info(f"Kubernetes deployer initialized")
        logger.info(f"Namespace: {self.namespace}")
        logger.info(f"Context: {self.context or 'default'}")
        logger.info(f"K8s configs: {self.k8s_dir}")
    
    def run_kubectl(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run kubectl command."""
        cmd = ["kubectl"]
        
        if self.context:
            cmd.extend(["--context", self.context])
        
        cmd.extend(args)
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0 and check:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"Error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
        
        return result
    
    def check_prerequisites(self) -> bool:
        """Check deployment prerequisites."""
        logger.info("Checking prerequisites...")
        
        # Check kubectl
        try:
            result = self.run_kubectl(["version", "--client"])
            logger.info("✅ kubectl is available")
        except Exception as e:
            logger.error(f"❌ kubectl not found: {e}")
            return False
        
        # Check cluster connectivity
        try:
            result = self.run_kubectl(["cluster-info"])
            logger.info("✅ Kubernetes cluster is accessible")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Kubernetes cluster: {e}")
            return False
        
        # Check Docker image
        if not self.check_docker_image():
            logger.warning("⚠️ Docker image not found, will need to build")
        
        return True
    
    def check_docker_image(self) -> bool:
        """Check if Docker image exists."""
        try:
            result = subprocess.run(
                ["docker", "images", "medical-ai:latest", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if "medical-ai:latest" in result.stdout:
                logger.info("✅ Docker image medical-ai:latest found")
                return True
            else:
                logger.warning("⚠️ Docker image medical-ai:latest not found")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Cannot check Docker image: {e}")
            return False
    
    def build_docker_image(self):
        """Build Docker image."""
        logger.info("Building Docker image...")
        
        dockerfile_content = '''
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-production.txt .
RUN pip install --no-cache-dir -r requirements-production.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "scripts/start_production_api.py"]
'''
        
        # Write Dockerfile
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        
        # Build image
        try:
            result = subprocess.run(
                ["docker", "build", "-t", "medical-ai:latest", "."],
                cwd=Path(__file__).parent.parent,
                check=True
            )
            logger.info("✅ Docker image built successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Docker build failed: {e}")
            raise
    
    def deploy_namespace(self):
        """Deploy namespace and resource quotas."""
        logger.info("Deploying namespace...")
        
        self.run_kubectl([
            "apply", "-f", str(self.k8s_dir / "namespace.yaml")
        ])
        
        logger.info("✅ Namespace deployed")
    
    def deploy_secrets_and_config(self):
        """Deploy secrets and configuration."""
        logger.info("Deploying secrets and configuration...")
        
        # Deploy ConfigMaps
        self.run_kubectl([
            "apply", "-f", str(self.k8s_dir / "configmap.yaml")
        ])
        
        # Deploy Secrets
        self.run_kubectl([
            "apply", "-f", str(self.k8s_dir / "secrets.yaml")
        ])
        
        logger.info("✅ Secrets and configuration deployed")
    
    def deploy_database(self):
        """Deploy PostgreSQL database."""
        logger.info("Deploying PostgreSQL database...")
        
        self.run_kubectl([
            "apply", "-f", str(self.k8s_dir / "postgres.yaml")
        ])
        
        # Wait for database to be ready
        logger.info("Waiting for database to be ready...")
        self.wait_for_deployment("postgres", timeout=300)
        
        logger.info("✅ Database deployed")
    
    def deploy_api(self):
        """Deploy API application."""
        logger.info("Deploying API application...")
        
        self.run_kubectl([
            "apply", "-f", str(self.k8s_dir / "api-deployment.yaml")
        ])
        
        # Wait for API to be ready
        logger.info("Waiting for API to be ready...")
        self.wait_for_deployment("medical-ai-api", timeout=600)
        
        logger.info("✅ API deployed")
    
    def deploy_services(self):
        """Deploy services and ingress."""
        logger.info("Deploying services...")
        
        self.run_kubectl([
            "apply", "-f", str(self.k8s_dir / "services.yaml")
        ])
        
        logger.info("✅ Services deployed")
    
    def deploy_rbac(self):
        """Deploy RBAC configuration."""
        logger.info("Deploying RBAC...")
        
        self.run_kubectl([
            "apply", "-f", str(self.k8s_dir / "rbac.yaml")
        ])
        
        logger.info("✅ RBAC deployed")
    
    def deploy_monitoring(self):
        """Deploy monitoring stack."""
        logger.info("Deploying monitoring stack...")
        
        # Deploy Prometheus
        self.run_kubectl([
            "apply", "-f", str(self.k8s_dir / "monitoring" / "prometheus.yaml")
        ])
        
        # Deploy Grafana
        self.run_kubectl([
            "apply", "-f", str(self.k8s_dir / "monitoring" / "grafana.yaml")
        ])
        
        # Deploy Alertmanager
        self.run_kubectl([
            "apply", "-f", str(self.k8s_dir / "monitoring" / "alertmanager.yaml")
        ])
        
        logger.info("✅ Monitoring stack deployed")
    
    def deploy_autoscaling(self):
        """Deploy auto-scaling configuration."""
        logger.info("Deploying auto-scaling...")
        
        # Deploy HPA
        self.run_kubectl([
            "apply", "-f", str(self.k8s_dir / "hpa.yaml")
        ])
        
        # Deploy VPA (if available)
        try:
            self.run_kubectl([
                "apply", "-f", str(self.k8s_dir / "vpa.yaml")
            ])
            logger.info("✅ VPA deployed")
        except subprocess.CalledProcessError:
            logger.warning("⚠️ VPA not available, skipping")
        
        logger.info("✅ Auto-scaling deployed")
    
    def wait_for_deployment(self, deployment_name: str, timeout: int = 300):
        """Wait for deployment to be ready."""
        logger.info(f"Waiting for deployment {deployment_name} to be ready...")
        
        try:
            self.run_kubectl([
                "wait", "--for=condition=available",
                f"deployment/{deployment_name}",
                f"--namespace={self.namespace}",
                f"--timeout={timeout}s"
            ])
            logger.info(f"✅ Deployment {deployment_name} is ready")
        except subprocess.CalledProcessError:
            logger.error(f"❌ Deployment {deployment_name} failed to become ready")
            raise
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get deployment status."""
        logger.info("Getting deployment status...")
        
        try:
            # Get pods
            result = self.run_kubectl([
                "get", "pods", f"--namespace={self.namespace}", "-o", "json"
            ])
            
            import json
            pods_data = json.loads(result.stdout)
            
            # Get services
            result = self.run_kubectl([
                "get", "services", f"--namespace={self.namespace}", "-o", "json"
            ])
            
            services_data = json.loads(result.stdout)
            
            status = {
                "pods": len(pods_data.get("items", [])),
                "services": len(services_data.get("items", [])),
                "ready_pods": len([
                    pod for pod in pods_data.get("items", [])
                    if pod.get("status", {}).get("phase") == "Running"
                ])
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {}
    
    def test_deployment(self):
        """Test the deployment."""
        logger.info("Testing deployment...")
        
        try:
            # Port forward to test API
            logger.info("Setting up port forward for testing...")
            
            # Get API pod
            result = self.run_kubectl([
                "get", "pods", f"--namespace={self.namespace}",
                "-l", "app=medical-ai-api",
                "-o", "jsonpath={.items[0].metadata.name}"
            ])
            
            pod_name = result.stdout.strip()
            
            if not pod_name:
                logger.error("❌ No API pods found")
                return False
            
            logger.info(f"Testing API pod: {pod_name}")
            
            # Test health endpoint
            result = self.run_kubectl([
                "exec", pod_name, f"--namespace={self.namespace}",
                "--", "curl", "-f", "http://localhost:8000/health"
            ])
            
            if result.returncode == 0:
                logger.info("✅ API health check passed")
                return True
            else:
                logger.error("❌ API health check failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Deployment test failed: {e}")
            return False
    
    def deploy_all(self):
        """Deploy complete Medical AI platform."""
        logger.info("🚀 Starting complete Medical AI platform deployment")
        
        try:
            # Check prerequisites
            if not self.check_prerequisites():
                logger.error("❌ Prerequisites check failed")
                return False
            
            # Build Docker image if needed
            if not self.check_docker_image():
                self.build_docker_image()
            
            # Deploy components in order
            self.deploy_namespace()
            self.deploy_rbac()
            self.deploy_secrets_and_config()
            self.deploy_database()
            self.deploy_api()
            self.deploy_services()
            self.deploy_monitoring()
            self.deploy_autoscaling()
            
            # Get status
            status = self.get_deployment_status()
            logger.info(f"Deployment status: {status}")
            
            # Test deployment
            if self.test_deployment():
                logger.info("🎉 Medical AI platform deployed successfully!")
                self.print_access_info()
                return True
            else:
                logger.error("❌ Deployment test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False
    
    def print_access_info(self):
        """Print access information."""
        logger.info("\n" + "="*50)
        logger.info("🎯 MEDICAL AI PLATFORM ACCESS INFO")
        logger.info("="*50)
        
        try:
            # Get LoadBalancer IP
            result = self.run_kubectl([
                "get", "service", "medical-ai-api-service",
                f"--namespace={self.namespace}",
                "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"
            ])
            
            external_ip = result.stdout.strip()
            
            if external_ip:
                logger.info(f"🌐 External API URL: http://{external_ip}")
                logger.info(f"📊 Health Check: http://{external_ip}/health")
                logger.info(f"📚 API Docs: http://{external_ip}/docs")
            else:
                logger.info("🔄 LoadBalancer IP pending, use port-forward:")
                logger.info(f"   kubectl port-forward -n {self.namespace} service/medical-ai-api-internal 8000:8000")
                logger.info("   Then access: http://localhost:8000")
            
            # Monitoring access
            logger.info(f"📈 Grafana: kubectl port-forward -n {self.namespace} service/grafana 3000:3000")
            logger.info(f"🔍 Prometheus: kubectl port-forward -n {self.namespace} service/prometheus 9090:9090")
            
            logger.info("="*50)
            
        except Exception as e:
            logger.warning(f"Could not get access info: {e}")


def main():
    """Main deployment function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Medical AI platform to Kubernetes")
    parser.add_argument(
        '--namespace',
        default='medical-ai',
        help='Kubernetes namespace'
    )
    parser.add_argument(
        '--context',
        help='Kubernetes context to use'
    )
    parser.add_argument(
        '--build-image',
        action='store_true',
        help='Force rebuild Docker image'
    )
    
    args = parser.parse_args()
    
    deployer = KubernetesDeployer(
        namespace=args.namespace,
        context=args.context
    )
    
    if args.build_image:
        deployer.build_docker_image()
    
    success = deployer.deploy_all()
    
    if success:
        logger.info("✅ Deployment completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Deployment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()