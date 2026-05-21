"""
Tests for W&B Collaborative Workspace

Test suite for the enhanced Weights & Biases integration with collaborative
workspace features, team management, and experiment templates.
"""

import json
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.features.research.experiment.wandb_integration import (
    ExperimentTemplate,
    ProjectVisibility,
    WandBCollaborativeWorkspace,
    WandBTracker,
    WorkspaceConfig,
    WorkspaceRole,
)


class TestWandBCollaborativeWorkspace:
    """Test W&B collaborative workspace functionality."""

    @pytest.fixture
    def workspace_config(self):
        """Workspace configuration fixture."""
        return WorkspaceConfig(
            entity="test-team",
            project="test-project",
            workspace_name="test-workspace",
            description="Test workspace for unit tests",
            visibility=ProjectVisibility.TEAM,
            tags=["test", "unit-test"],
        )

    @pytest.fixture
    def mock_wandb_api(self):
        """Mock W&B API fixture."""
        with patch("src.research_platform.wandb_integration.WandBApi") as mock_api:
            mock_instance = Mock()
            mock_api.return_value = mock_instance

            # Mock project
            mock_project = Mock()
            mock_project.name = "test-project"
            mock_instance.project.return_value = mock_project

            # Mock runs
            mock_run = Mock()
            mock_run.id = "test-run-123"
            mock_run.name = "test-experiment"
            mock_run.state = "finished"
            mock_run.url = "https://wandb.ai/test-team/test-project/runs/test-run-123"
            mock_run.summary = {"accuracy": 0.95, "loss": 0.05}
            mock_run.config = {"learning_rate": 0.001, "batch_size": 32}
            mock_run.tags = ["test"]
            mock_run.created_at = datetime.now()
            mock_run.user = Mock()
            mock_run.user.username = "test-user"

            mock_instance.run.return_value = mock_run
            mock_instance.runs.return_value = [mock_run]

            yield mock_instance

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    def test_workspace_initialization(self, workspace_config, mock_wandb_api):
        """Test workspace initialization."""
        workspace = WandBCollaborativeWorkspace(workspace_config)

        assert workspace.config == workspace_config
        assert workspace.api is not None
        assert len(workspace.experiment_templates) > 0
        assert "foundation_model_training" in workspace.experiment_templates

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    @patch("src.research_platform.wandb_integration.wandb")
    def test_create_experiment_from_template(self, mock_wandb, workspace_config, mock_wandb_api):
        """Test creating experiment from template."""
        workspace = WandBCollaborativeWorkspace(workspace_config)

        # Mock wandb.init
        mock_run = Mock()
        mock_run.id = "test-run-456"
        mock_wandb.init.return_value = mock_run

        # Create experiment from template
        run_id = workspace.create_experiment_from_template(
            template_name="foundation_model_training",
            experiment_name="test-foundation-experiment",
            config_overrides={"training.epochs": 50},
            tags=["custom-tag"],
            notes="Custom experiment notes",
        )

        assert run_id == "test-run-456"
        mock_wandb.init.assert_called_once()

        # Check call arguments
        call_args = mock_wandb.init.call_args
        assert call_args[1]["project"] == workspace_config.project
        assert call_args[1]["entity"] == workspace_config.entity
        assert call_args[1]["name"] == "test-foundation-experiment"
        assert "foundation_model" in call_args[1]["tags"]
        assert "custom-tag" in call_args[1]["tags"]

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    def test_add_collaborator(self, workspace_config, mock_wandb_api):
        """Test adding collaborator to workspace."""
        workspace = WandBCollaborativeWorkspace(workspace_config)

        # Add collaborator
        success = workspace.add_collaborator(
            username="test-collaborator", role=WorkspaceRole.MEMBER, email="test@example.com"
        )

        assert success is True
        assert "test-collaborator" in workspace.collaborators

        collaborator = workspace.collaborators["test-collaborator"]
        assert collaborator.username == "test-collaborator"
        assert collaborator.role == WorkspaceRole.MEMBER
        assert collaborator.email == "test@example.com"

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    @patch("src.research_platform.wandb_integration.wandb")
    def test_create_shared_dashboard(self, mock_wandb, workspace_config, mock_wandb_api):
        """Test creating shared dashboard."""
        workspace = WandBCollaborativeWorkspace(workspace_config)

        # Mock current run
        mock_run = Mock()
        mock_wandb.run = mock_run
        workspace.current_run = mock_run

        # Mock artifact
        mock_artifact = Mock()
        mock_wandb.Artifact.return_value = mock_artifact

        # Create dashboard
        dashboard_url = workspace.create_shared_dashboard(
            dashboard_name="team-dashboard",
            description="Team collaboration dashboard",
            metrics=["accuracy", "loss", "f1_score"],
        )

        assert "wandb.ai" in dashboard_url
        mock_wandb.Artifact.assert_called_once()
        mock_wandb.log_artifact.assert_called_once()

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    def test_share_experiment(self, workspace_config, mock_wandb_api):
        """Test sharing experiment with collaborators."""
        workspace = WandBCollaborativeWorkspace(workspace_config)

        # Share experiment
        success = workspace.share_experiment(
            run_id="test-run-123",
            collaborators=["user1", "user2"],
            message="Please review this experiment",
        )

        assert success is True
        mock_wandb_api.run.assert_called_once_with("test-team/test-project/test-run-123")

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    @patch("src.research_platform.wandb_integration.wandb")
    def test_create_model_registry_entry(self, mock_wandb, workspace_config, mock_wandb_api):
        """Test creating model registry entry."""
        workspace = WandBCollaborativeWorkspace(workspace_config)

        # Mock artifact
        mock_artifact = Mock()
        mock_wandb.Artifact.return_value = mock_artifact

        # Mock current run
        mock_run = Mock()
        workspace.current_run = mock_run

        with tempfile.NamedTemporaryFile(suffix=".pth") as temp_file:
            # Create model registry entry
            registry_url = workspace.create_model_registry_entry(
                model_name="test-model",
                model_version="v1.0",
                model_path=temp_file.name,
                description="Test model for unit tests",
                metrics={"accuracy": 0.95, "f1_score": 0.92},
                tags=["test", "model"],
            )

            assert "wandb.ai" in registry_url
            assert "test-model" in registry_url
            mock_wandb.Artifact.assert_called_once()

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    def test_create_experiment_comparison(self, workspace_config, mock_wandb_api):
        """Test creating experiment comparison."""
        workspace = WandBCollaborativeWorkspace(workspace_config)

        # Create comparison
        comparison_url = workspace.create_experiment_comparison(
            run_ids=["run1", "run2", "run3"],
            comparison_name="model-comparison",
            metrics=["accuracy", "loss"],
            description="Comparing different model architectures",
        )

        assert "wandb.ai" in comparison_url
        # Should have called API for each run
        assert mock_wandb_api.run.call_count == 3

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    def test_get_workspace_activity(self, workspace_config, mock_wandb_api):
        """Test getting workspace activity."""
        workspace = WandBCollaborativeWorkspace(workspace_config)

        # Get activity
        activity = workspace.get_workspace_activity(days=7)

        assert len(activity) > 0
        assert activity[0]["type"] == "run_created"
        assert "run_id" in activity[0]
        assert "user" in activity[0]

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    def test_export_workspace_data(self, workspace_config, mock_wandb_api):
        """Test exporting workspace data."""
        workspace = WandBCollaborativeWorkspace(workspace_config)

        # Add some test data
        workspace.add_collaborator("test-user", WorkspaceRole.MEMBER)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            # Export data
            success = workspace.export_workspace_data(temp_file.name)

            assert success is True

            # Verify exported data
            with open(temp_file.name, "r") as f:
                exported_data = json.load(f)

            assert "config" in exported_data
            assert "collaborators" in exported_data
            assert "experiment_templates" in exported_data
            assert exported_data["config"]["project"] == workspace_config.project

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    def test_get_collaboration_stats(self, workspace_config, mock_wandb_api):
        """Test getting collaboration statistics."""
        workspace = WandBCollaborativeWorkspace(workspace_config)

        # Add test collaborators
        workspace.add_collaborator("admin-user", WorkspaceRole.ADMIN)
        workspace.add_collaborator("member-user", WorkspaceRole.MEMBER)
        workspace.add_collaborator("viewer-user", WorkspaceRole.VIEWER)

        # Get stats
        stats = workspace.get_collaboration_stats()

        assert stats["total_collaborators"] == 3
        assert stats["collaborators_by_role"]["admin"] == 1
        assert stats["collaborators_by_role"]["member"] == 1
        assert stats["collaborators_by_role"]["viewer"] == 1
        assert "total_experiments" in stats
        assert "experiment_templates" in stats

    def test_experiment_template_structure(self):
        """Test experiment template structure."""
        template = ExperimentTemplate(
            name="test-template",
            description="Test template",
            config_template={"param1": "value1"},
            required_metrics=["accuracy"],
            optional_metrics=["precision"],
            tags=["test"],
        )

        assert template.name == "test-template"
        assert template.description == "Test template"
        assert template.config_template["param1"] == "value1"
        assert "accuracy" in template.required_metrics
        assert "precision" in template.optional_metrics
        assert "test" in template.tags


class TestWandBTrackerLegacy:
    """Test legacy W&B tracker for backward compatibility."""

    @pytest.fixture
    def mock_wandb(self):
        """Mock wandb module."""
        with patch("src.research_platform.wandb_integration.wandb") as mock:
            mock_run = Mock()
            mock_run.name = "test-run"
            mock.init.return_value = mock_run
            mock.run = mock_run
            yield mock

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    @patch("src.research_platform.wandb_integration.WandBApi")
    def test_legacy_tracker_initialization(self, mock_api, mock_wandb):
        """Test legacy tracker initialization."""
        tracker = WandBTracker(project="test-project", entity="test-entity")

        assert tracker.config.project == "test-project"
        assert tracker.config.entity == "test-entity"

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    @patch("src.research_platform.wandb_integration.WandBApi")
    def test_legacy_init_run(self, mock_api, mock_wandb):
        """Test legacy init_run method."""
        tracker = WandBTracker()

        tracker.init_run(name="test-run", config={"lr": 0.001}, tags=["test"], notes="Test run")

        mock_wandb.init.assert_called_once()
        call_args = mock_wandb.init.call_args[1]
        assert call_args["name"] == "test-run"
        assert call_args["config"]["lr"] == 0.001
        assert "test" in call_args["tags"]

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    @patch("src.research_platform.wandb_integration.WandBApi")
    def test_legacy_log_methods(self, mock_api, mock_wandb):
        """Test legacy logging methods."""
        tracker = WandBTracker()

        # Test log_metrics
        tracker.log_metrics({"accuracy": 0.95}, step=100)
        mock_wandb.log.assert_called_with({"accuracy": 0.95}, step=100)

        # Test log_image
        mock_image = Mock()
        tracker.log_image("test_image", mock_image)
        mock_wandb.Image.assert_called_with(mock_image)

        # Test log_table
        tracker.log_table("test_table", [["a", 1], ["b", 2]], ["col1", "col2"])
        mock_wandb.Table.assert_called_with(data=[["a", 1], ["b", 2]], columns=["col1", "col2"])

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    @patch("src.research_platform.wandb_integration.WandBApi")
    def test_legacy_sweep_methods(self, mock_api, mock_wandb):
        """Test legacy sweep methods."""
        tracker = WandBTracker()

        # Test sweep creation
        sweep_config = {"method": "bayes", "parameters": {"lr": {"values": [0.001, 0.01, 0.1]}}}

        mock_wandb.sweep.return_value = "test-sweep-123"
        sweep_id = tracker.sweep(sweep_config)

        assert sweep_id == "test-sweep-123"
        mock_wandb.sweep.assert_called_with(sweep_config, project=tracker.config.project)

        # Test agent
        mock_function = Mock()
        tracker.agent("test-sweep-123", mock_function, count=5)
        mock_wandb.agent.assert_called_with("test-sweep-123", function=mock_function, count=5)


class TestWorkspaceIntegration:
    """Integration tests for workspace functionality."""

    @patch("src.research_platform.wandb_integration.WANDB_AVAILABLE", True)
    @patch("src.research_platform.wandb_integration.WandBApi")
    @patch("src.research_platform.wandb_integration.wandb")
    def test_full_collaboration_workflow(self, mock_wandb, mock_api, workspace_config):
        """Test complete collaboration workflow."""
        workspace = WandBCollaborativeWorkspace(workspace_config)

        # Setup mock run
        mock_run = Mock()
        mock_run.id = "workflow-run-123"
        mock_wandb.init.return_value = mock_run
        workspace.current_run = mock_run

        # 1. Add collaborators
        workspace.add_collaborator("researcher1", WorkspaceRole.MEMBER)
        workspace.add_collaborator("supervisor", WorkspaceRole.ADMIN)

        # 2. Create experiment from template
        run_id = workspace.create_experiment_from_template(
            template_name="foundation_model_training",
            experiment_name="collaborative-experiment",
            config_overrides={"training.epochs": 100},
        )

        # 3. Share experiment
        workspace.share_experiment(
            run_id=run_id, collaborators=["researcher1", "supervisor"], message="Ready for review"
        )

        # 4. Create dashboard
        dashboard_url = workspace.create_shared_dashboard(
            dashboard_name="project-dashboard",
            description="Main project dashboard",
            metrics=["accuracy", "loss", "f1_score"],
        )

        # 5. Get collaboration stats
        stats = workspace.get_collaboration_stats()

        # Verify workflow completed
        assert len(workspace.collaborators) == 2
        assert run_id == "workflow-run-123"
        assert "wandb.ai" in dashboard_url
        assert stats["total_collaborators"] == 2


if __name__ == "__main__":
    pytest.main([__file__])
