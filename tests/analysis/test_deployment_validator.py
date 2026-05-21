"""
Unit tests for Deployment Validator.

Tests Dockerfile parsing, Kubernetes manifest validation, and CI/CD pipeline assessment.
Requirements: 6.1, 6.2, 6.3
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.analysis.deployment import DeploymentValidator
from src.analysis.models import DeploymentAnalysis


class TestDeploymentValidator:
    """Test suite for DeploymentValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.validator = DeploymentValidator(str(self.project_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test validator initialization."""
        validator = DeploymentValidator("/path/to/project")
        assert validator.project_path == Path("/path/to/project").resolve()

    def test_analyze_returns_deployment_analysis(self):
        """Test that analyze() returns a DeploymentAnalysis object."""
        with patch.object(self.validator, "_validate_dockerfile", return_value=85.0):
            with patch.object(self.validator, "_validate_k8s_manifests", return_value=75.0):
                with patch.object(self.validator, "_assess_ci_cd_pipeline", return_value=90.0):
                    with patch.object(
                        self.validator, "_calculate_monitoring_score", return_value=80.0
                    ):
                        result = self.validator.analyze()

        assert isinstance(result, DeploymentAnalysis)
        assert result.dockerfile_score == 85.0
        assert result.k8s_readiness == 75.0
        assert result.ci_cd_completeness == 90.0
        assert result.monitoring_score == 80.0
        assert isinstance(result.score, float)
        assert 0 <= result.score <= 100


class TestDockerfileValidation:
    """Test Dockerfile validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.validator = DeploymentValidator(str(self.project_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_dockerfile(self, content: str):
        """Create a Dockerfile with specified content."""
        dockerfile_path = self.project_path / "Dockerfile"
        dockerfile_path.write_text(content)
        return dockerfile_path

    def test_validate_dockerfile_no_file(self):
        """Test Dockerfile validation when no Dockerfile exists."""
        score = self.validator._validate_dockerfile()
        assert score == 0.0

    def test_validate_dockerfile_good_practices(self):
        """Test Dockerfile validation with good practices."""
        good_dockerfile = """
# Multi-stage build
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "app.py"]
"""
        self.create_dockerfile(good_dockerfile)

        score = self.validator._validate_dockerfile()

        # Should get high score for good practices
        assert score > 80.0

    def test_validate_dockerfile_poor_practices(self):
        """Test Dockerfile validation with poor practices."""
        poor_dockerfile = """
FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3 python3-pip
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD python app.py
"""
        self.create_dockerfile(poor_dockerfile)

        score = self.validator._validate_dockerfile()

        # Should get low score for poor practices
        assert score < 50.0

    @patch("subprocess.run")
    def test_validate_dockerfile_with_hadolint(self, mock_run):
        """Test Dockerfile validation using hadolint."""
        # Mock hadolint output
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = """
[
  {
    "file": "Dockerfile",
    "line": 1,
    "level": "error",
    "code": "DL3006",
    "message": "Always tag the version of an image explicitly"
  }
]
"""

        dockerfile_content = "FROM python:latest\nCOPY . /app"
        self.create_dockerfile(dockerfile_content)

        score = self.validator._validate_dockerfile()

        # Should call hadolint
        mock_run.assert_called()
        assert isinstance(score, float)
        assert 0 <= score <= 100


class TestKubernetesValidation:
    """Test Kubernetes manifest validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.validator = DeploymentValidator(str(self.project_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_k8s_manifest(self, filename: str, content: dict):
        """Create a Kubernetes manifest file."""
        k8s_dir = self.project_path / "k8s"
        k8s_dir.mkdir(exist_ok=True)

        manifest_path = k8s_dir / filename
        manifest_path.write_text(yaml.dump(content))
        return manifest_path

    def test_validate_k8s_manifests_no_directory(self):
        """Test K8s validation when no k8s directory exists."""
        score = self.validator._validate_k8s_manifests()
        assert score == 0.0

    def test_validate_k8s_manifests_good_deployment(self):
        """Test K8s validation with good deployment manifest."""
        good_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "app-deployment"},
            "spec": {
                "replicas": 3,
                "selector": {"matchLabels": {"app": "myapp"}},
                "template": {
                    "metadata": {"labels": {"app": "myapp"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "app",
                                "image": "myapp:v1.0.0",
                                "resources": {
                                    "requests": {"cpu": "100m", "memory": "128Mi"},
                                    "limits": {"cpu": "500m", "memory": "512Mi"},
                                },
                                "livenessProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "initialDelaySeconds": 30,
                                },
                                "readinessProbe": {
                                    "httpGet": {"path": "/ready", "port": 8000},
                                    "initialDelaySeconds": 5,
                                },
                            }
                        ]
                    },
                },
            },
        }

        self.create_k8s_manifest("deployment.yaml", good_deployment)

        score = self.validator._validate_k8s_manifests()

        # Should get high score for good practices
        assert score > 80.0

    def test_validate_k8s_manifests_poor_deployment(self):
        """Test K8s validation with poor deployment manifest."""
        poor_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "app-deployment"},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "myapp"}},
                "template": {
                    "metadata": {"labels": {"app": "myapp"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "app",
                                "image": "myapp:latest",  # No resource limits, no health checks
                            }
                        ]
                    },
                },
            },
        }

        self.create_k8s_manifest("deployment.yaml", poor_deployment)

        score = self.validator._validate_k8s_manifests()

        # Should get low score for poor practices
        assert score < 50.0

    def test_validate_k8s_manifests_invalid_yaml(self):
        """Test K8s validation with invalid YAML."""
        k8s_dir = self.project_path / "k8s"
        k8s_dir.mkdir()

        # Create invalid YAML file
        invalid_yaml = k8s_dir / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content: [")

        score = self.validator._validate_k8s_manifests()

        # Should handle invalid YAML gracefully
        assert isinstance(score, float)
        assert 0 <= score <= 100


class TestCICDPipelineAssessment:
    """Test CI/CD pipeline assessment functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.validator = DeploymentValidator(str(self.project_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_github_workflow(self, filename: str, content: dict):
        """Create a GitHub Actions workflow file."""
        workflows_dir = self.project_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        workflow_path = workflows_dir / filename
        workflow_path.write_text(yaml.dump(content))
        return workflow_path

    def test_assess_ci_cd_pipeline_no_workflows(self):
        """Test CI/CD assessment when no workflows exist."""
        score = self.validator._assess_ci_cd_pipeline()
        assert score == 0.0

    def test_assess_ci_cd_pipeline_comprehensive_workflow(self):
        """Test CI/CD assessment with comprehensive workflow."""
        comprehensive_workflow = {
            "name": "CI/CD Pipeline",
            "on": ["push", "pull_request"],
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {"name": "Set up Python", "uses": "actions/setup-python@v4"},
                        {"name": "Install dependencies", "run": "pip install -r requirements.txt"},
                        {"name": "Run tests", "run": "pytest"},
                        {"name": "Run linting", "run": "pylint src/"},
                        {"name": "Security scan", "run": "bandit -r src/"},
                        {"name": "Build Docker image", "run": "docker build -t myapp ."},
                        {"name": "Deploy to staging", "run": "kubectl apply -f k8s/"},
                    ],
                }
            },
        }

        self.create_github_workflow("ci-cd.yml", comprehensive_workflow)

        score = self.validator._assess_ci_cd_pipeline()

        # Should get high score for comprehensive pipeline
        assert score > 80.0

    def test_assess_ci_cd_pipeline_basic_workflow(self):
        """Test CI/CD assessment with basic workflow."""
        basic_workflow = {
            "name": "Basic CI",
            "on": ["push"],
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {"name": "Run tests", "run": "python -m pytest"},
                    ],
                }
            },
        }

        self.create_github_workflow("basic.yml", basic_workflow)

        score = self.validator._assess_ci_cd_pipeline()

        # Should get moderate score for basic pipeline
        assert 30.0 <= score <= 70.0

    def test_assess_ci_cd_pipeline_invalid_yaml(self):
        """Test CI/CD assessment with invalid YAML."""
        workflows_dir = self.project_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True)

        # Create invalid YAML file
        invalid_yaml = workflows_dir / "invalid.yml"
        invalid_yaml.write_text("invalid: yaml: content: [")

        score = self.validator._assess_ci_cd_pipeline()

        # Should handle invalid YAML gracefully
        assert isinstance(score, float)
        assert 0 <= score <= 100


class TestMonitoringScore:
    """Test monitoring score calculation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.validator = DeploymentValidator(str(self.project_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_calculate_monitoring_score_placeholder(self):
        """Test monitoring score calculation (currently placeholder)."""
        score = self.validator._calculate_monitoring_score()

        # Currently returns 50.0 as placeholder
        assert score == 50.0
        assert isinstance(score, float)


class TestDeploymentScoreCalculation:
    """Test deployment score calculation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DeploymentValidator("/test/project")

    def test_calculate_deployment_score_perfect(self):
        """Test deployment score calculation with perfect metrics."""
        score = self.validator._calculate_deployment_score(
            dockerfile_score=100.0, k8s_score=100.0, ci_cd_score=100.0, monitoring_score=100.0
        )

        # Perfect scores should result in 100
        assert score == 100.0

    def test_calculate_deployment_score_poor(self):
        """Test deployment score calculation with poor metrics."""
        score = self.validator._calculate_deployment_score(
            dockerfile_score=0.0, k8s_score=0.0, ci_cd_score=0.0, monitoring_score=0.0
        )

        # Poor scores should result in 0
        assert score == 0.0

    def test_calculate_deployment_score_mixed(self):
        """Test deployment score calculation with mixed metrics."""
        score = self.validator._calculate_deployment_score(
            dockerfile_score=80.0, k8s_score=60.0, ci_cd_score=90.0, monitoring_score=70.0
        )

        # Expected: (80 + 60 + 90 + 70) / 4 = 75.0
        expected_score = (80.0 + 60.0 + 90.0 + 70.0) / 4
        assert score == expected_score

    def test_calculate_deployment_score_bounds(self):
        """Test deployment score calculation stays within bounds."""
        # Test with values outside normal range
        score = self.validator._calculate_deployment_score(
            dockerfile_score=150.0,  # Over 100
            k8s_score=-10.0,  # Below 0
            ci_cd_score=50.0,
            monitoring_score=50.0,
        )

        # Should be clamped to 0-100 range
        assert 0.0 <= score <= 100.0


class TestIntegrationWithMockData:
    """Integration tests with mock deployment data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.validator = DeploymentValidator(str(self.project_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_analysis_with_mock_data(self):
        """Test complete analysis workflow with mocked data."""
        # Mock all the validation methods
        with patch.object(self.validator, "_validate_dockerfile", return_value=85.0):
            with patch.object(self.validator, "_validate_k8s_manifests", return_value=75.0):
                with patch.object(self.validator, "_assess_ci_cd_pipeline", return_value=90.0):
                    with patch.object(
                        self.validator, "_calculate_monitoring_score", return_value=80.0
                    ):

                        result = self.validator.analyze()

        # Verify all fields are populated
        assert result.dockerfile_score == 85.0
        assert result.k8s_readiness == 75.0
        assert result.ci_cd_completeness == 90.0
        assert result.monitoring_score == 80.0

        # Verify score calculation
        expected_score = (85.0 + 75.0 + 90.0 + 80.0) / 4
        assert result.score == expected_score

    def test_analysis_with_excellent_deployment(self):
        """Test analysis with excellent deployment practices."""
        with patch.object(self.validator, "_validate_dockerfile", return_value=95.0):
            with patch.object(self.validator, "_validate_k8s_manifests", return_value=90.0):
                with patch.object(self.validator, "_assess_ci_cd_pipeline", return_value=95.0):
                    with patch.object(
                        self.validator, "_calculate_monitoring_score", return_value=85.0
                    ):

                        result = self.validator.analyze()

        # Excellent deployment should result in high score
        assert result.score > 90.0

    def test_analysis_with_poor_deployment(self):
        """Test analysis with poor deployment practices."""
        with patch.object(self.validator, "_validate_dockerfile", return_value=20.0):
            with patch.object(self.validator, "_validate_k8s_manifests", return_value=15.0):
                with patch.object(self.validator, "_assess_ci_cd_pipeline", return_value=25.0):
                    with patch.object(
                        self.validator, "_calculate_monitoring_score", return_value=30.0
                    ):

                        result = self.validator.analyze()

        # Poor deployment should result in low score
        assert result.score < 30.0


if __name__ == "__main__":
    pytest.main([__file__])
