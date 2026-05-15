"""
Weights & Biases Integration

Enhanced experiment tracking with W&B including collaborative workspaces,
team management, project organization, and advanced collaboration features.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading

try:
    import wandb
    from wandb.apis.public import Api as WandBApi

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class WorkspaceRole(Enum):
    """Workspace role types."""

    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class ProjectVisibility(Enum):
    """Project visibility levels."""

    PRIVATE = "private"
    PUBLIC = "public"
    TEAM = "team"


@dataclass
class WorkspaceConfig:
    """Configuration for W&B workspace."""

    entity: str  # Team or user entity
    project: str
    workspace_name: str
    description: Optional[str] = None
    visibility: ProjectVisibility = ProjectVisibility.TEAM
    tags: Optional[List[str]] = None
    default_config: Optional[Dict[str, Any]] = None


@dataclass
class CollaboratorInfo:
    """Information about workspace collaborator."""

    username: str
    email: Optional[str]
    role: WorkspaceRole
    added_at: datetime
    last_active: Optional[datetime] = None


@dataclass
class ExperimentTemplate:
    """Template for standardized experiments."""

    name: str
    description: str
    config_template: Dict[str, Any]
    required_metrics: List[str]
    optional_metrics: List[str]
    tags: List[str]
    notes_template: Optional[str] = None


class WandBCollaborativeWorkspace:
    """
    Enhanced W&B integration with collaborative workspace features.

    Provides team collaboration, project organization, experiment templates,
    shared dashboards, and advanced experiment management capabilities.
    """

    def __init__(self, config: WorkspaceConfig):
        """Initialize collaborative workspace."""
        if not WANDB_AVAILABLE:
            raise ImportError("Weights & Biases not available. Install with: pip install wandb")

        self.config = config
        self.api = WandBApi()
        self.current_run = None
        self.collaborators: Dict[str, CollaboratorInfo] = {}
        self.experiment_templates: Dict[str, ExperimentTemplate] = {}
        self.shared_artifacts: Dict[str, Any] = {}

        # Initialize workspace
        self._initialize_workspace()

        logger.info("W&B collaborative workspace initialized: %s/%s", config.entity, config.project)

    def _initialize_workspace(self) -> None:
        """Initialize workspace and project settings."""
        try:
            # Check if project exists
            try:
                project = self.api.project(f"{self.config.entity}/{self.config.project}")
                logger.info("Connected to existing project: %s", project.name)
            except Exception:
                # Project doesn't exist, will be created on first run
                logger.info("Project will be created on first run: %s", self.config.project)

            # Load collaborators if project exists
            self._load_collaborators()

            # Load experiment templates
            self._load_experiment_templates()

        except Exception as e:
            logger.error("Failed to initialize workspace: %s", e)

    def _load_collaborators(self) -> None:
        """Load collaborator information."""
        try:
            # Note: W&B API doesn't directly expose team member info
            # This would typically be managed through W&B team settings
            # For now, we'll maintain our own collaborator tracking
            pass
        except Exception as e:
            logger.warning("Failed to load collaborators: %s", e)

    def _load_experiment_templates(self) -> None:
        """Load experiment templates from workspace."""
        try:
            # Load templates from W&B artifacts or local storage
            # This is a placeholder for template management
            self._create_default_templates()
        except Exception as e:
            logger.warning("Failed to load experiment templates: %s", e)

    def _create_default_templates(self) -> None:
        """Create default experiment templates."""
        # Foundation model training template
        foundation_template = ExperimentTemplate(
            name="foundation_model_training",
            description="Template for foundation model training experiments",
            config_template={
                "model": {"architecture": "resnet50", "pretrained": True, "num_classes": 2},
                "training": {
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "epochs": 100,
                    "optimizer": "adam",
                },
                "data": {"dataset": "camelyon16", "augmentation": True, "validation_split": 0.2},
            },
            required_metrics=["accuracy", "loss", "val_accuracy", "val_loss"],
            optional_metrics=["precision", "recall", "f1_score", "auc"],
            tags=["foundation_model", "training"],
            notes_template="Foundation model training experiment for {dataset} dataset",
        )

        # Model evaluation template
        evaluation_template = ExperimentTemplate(
            name="model_evaluation",
            description="Template for model evaluation experiments",
            config_template={
                "model": {"checkpoint_path": "", "model_version": ""},
                "evaluation": {
                    "test_dataset": "",
                    "batch_size": 64,
                    "metrics": ["accuracy", "precision", "recall", "f1", "auc"],
                },
            },
            required_metrics=["test_accuracy", "test_precision", "test_recall"],
            optional_metrics=["confusion_matrix", "roc_curve", "pr_curve"],
            tags=["evaluation", "testing"],
            notes_template="Model evaluation on {test_dataset}",
        )

        # Hyperparameter sweep template
        sweep_template = ExperimentTemplate(
            name="hyperparameter_sweep",
            description="Template for hyperparameter optimization",
            config_template={
                "method": "bayes",
                "metric": {"goal": "maximize", "name": "val_accuracy"},
                "parameters": {
                    "learning_rate": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-1},
                    "batch_size": {"values": [16, 32, 64, 128]},
                    "optimizer": {"values": ["adam", "sgd", "rmsprop"]},
                },
            },
            required_metrics=["val_accuracy", "val_loss"],
            optional_metrics=["train_accuracy", "train_loss"],
            tags=["hyperparameter_sweep", "optimization"],
            notes_template="Hyperparameter optimization for {model_type}",
        )

        self.experiment_templates = {
            "foundation_model_training": foundation_template,
            "model_evaluation": evaluation_template,
            "hyperparameter_sweep": sweep_template,
        }

    def create_experiment_from_template(
        self,
        template_name: str,
        experiment_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Create experiment from template.

        Args:
            template_name: Name of the experiment template
            experiment_name: Name for the new experiment
            config_overrides: Configuration overrides
            tags: Additional tags
            notes: Custom notes

        Returns:
            Run ID of created experiment
        """
        if template_name not in self.experiment_templates:
            raise ValueError(f"Template not found: {template_name}")

        template = self.experiment_templates[template_name]

        # Merge configuration
        config = template.config_template.copy()
        if config_overrides:
            config.update(config_overrides)

        # Merge tags
        experiment_tags = template.tags.copy()
        if tags:
            experiment_tags.extend(tags)

        # Format notes
        experiment_notes = notes or template.notes_template
        if experiment_notes and config_overrides:
            try:
                experiment_notes = experiment_notes.format(**config_overrides)
            except KeyError:
                pass  # Keep original notes if formatting fails

        # Initialize run
        run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=experiment_name,
            config=config,
            tags=experiment_tags,
            notes=experiment_notes,
            reinit=True,
        )

        self.current_run = run

        # Log template information
        wandb.log({"template_used": template_name, "template_description": template.description})

        logger.info(
            "Created experiment from template %s: %s (%s)", template_name, experiment_name, run.id
        )

        return run.id

    def add_collaborator(
        self, username: str, role: WorkspaceRole, email: Optional[str] = None
    ) -> bool:
        """
        Add collaborator to workspace.

        Args:
            username: W&B username
            role: Role for the collaborator
            email: Optional email address

        Returns:
            True if successful, False otherwise
        """
        try:
            # Note: Actual team member management is done through W&B web interface
            # This maintains our local tracking
            collaborator = CollaboratorInfo(
                username=username, email=email, role=role, added_at=datetime.now()
            )

            self.collaborators[username] = collaborator

            # Log collaborator addition
            if self.current_run:
                wandb.log({"collaborator_added": username, "collaborator_role": role.value})

            logger.info("Added collaborator: %s (%s)", username, role.value)
            return True

        except Exception as e:
            logger.error("Failed to add collaborator %s: %s", username, e)
            return False

    def create_shared_dashboard(
        self,
        dashboard_name: str,
        description: str,
        metrics: List[str],
        charts: Optional[List[Dict]] = None,
    ) -> str:
        """
        Create shared dashboard for team collaboration.

        Args:
            dashboard_name: Name of the dashboard
            description: Dashboard description
            metrics: List of metrics to include
            charts: Optional chart configurations

        Returns:
            Dashboard URL
        """
        try:
            # Create dashboard configuration
            dashboard_config = {
                "name": dashboard_name,
                "description": description,
                "metrics": metrics,
                "charts": charts or [],
                "created_at": datetime.now().isoformat(),
                "created_by": wandb.run.entity if wandb.run else self.config.entity,
            }

            # Log dashboard configuration as artifact
            if self.current_run:
                artifact = wandb.Artifact(
                    name=f"dashboard_{dashboard_name}", type="dashboard_config"
                )

                with artifact.new_file("dashboard_config.json") as f:
                    json.dump(dashboard_config, f, indent=2)

                wandb.log_artifact(artifact)

            # Generate dashboard URL (placeholder)
            dashboard_url = f"https://wandb.ai/{self.config.entity}/{self.config.project}/workspace"

            logger.info("Created shared dashboard: %s", dashboard_name)
            return dashboard_url

        except Exception as e:
            logger.error("Failed to create dashboard %s: %s", dashboard_name, e)
            raise

    def share_experiment(
        self, run_id: str, collaborators: List[str], message: Optional[str] = None
    ) -> bool:
        """
        Share experiment with collaborators.

        Args:
            run_id: ID of the run to share
            collaborators: List of collaborator usernames
            message: Optional message to include

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get run information
            run = self.api.run(f"{self.config.entity}/{self.config.project}/{run_id}")

            # Create sharing notification
            sharing_info = {
                "run_id": run_id,
                "run_name": run.name,
                "shared_with": collaborators,
                "shared_by": wandb.run.entity if wandb.run else self.config.entity,
                "shared_at": datetime.now().isoformat(),
                "message": message,
                "run_url": run.url,
            }

            # Log sharing event
            if self.current_run:
                wandb.log({"experiment_shared": True, "shared_with_count": len(collaborators)})

            # In a real implementation, this would send notifications
            # to the specified collaborators

            logger.info("Shared experiment %s with %d collaborators", run_id, len(collaborators))
            return True

        except Exception as e:
            logger.error("Failed to share experiment %s: %s", run_id, e)
            return False

    def create_model_registry_entry(
        self,
        model_name: str,
        model_version: str,
        model_path: str,
        description: str,
        metrics: Dict[str, float],
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Create entry in shared model registry.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            model_path: Path to model file
            description: Model description
            metrics: Performance metrics
            tags: Optional tags

        Returns:
            Model registry URL
        """
        try:
            # Create model artifact
            artifact = wandb.Artifact(
                name=f"{model_name}",
                type="model",
                description=description,
                metadata={
                    "version": model_version,
                    "metrics": metrics,
                    "tags": tags or [],
                    "created_at": datetime.now().isoformat(),
                },
            )

            # Add model file
            artifact.add_file(model_path)

            # Log artifact
            if self.current_run:
                wandb.log_artifact(artifact)
            else:
                # Create temporary run for artifact logging
                with wandb.init(
                    project=self.config.project,
                    entity=self.config.entity,
                    job_type="model_registry",
                ) as run:
                    run.log_artifact(artifact)

            # Generate registry URL
            registry_url = f"https://wandb.ai/{self.config.entity}/{self.config.project}/artifacts/model/{model_name}"

            logger.info("Added model to registry: %s v%s", model_name, model_version)
            return registry_url

        except Exception as e:
            logger.error("Failed to add model to registry: %s", e)
            raise

    def create_experiment_comparison(
        self,
        run_ids: List[str],
        comparison_name: str,
        metrics: List[str],
        description: Optional[str] = None,
    ) -> str:
        """
        Create experiment comparison for team review.

        Args:
            run_ids: List of run IDs to compare
            comparison_name: Name for the comparison
            metrics: Metrics to compare
            description: Optional description

        Returns:
            Comparison URL
        """
        try:
            # Collect run information
            runs_info = []
            for run_id in run_ids:
                try:
                    run = self.api.run(f"{self.config.entity}/{self.config.project}/{run_id}")
                    run_metrics = {}

                    # Extract specified metrics
                    for metric in metrics:
                        if metric in run.summary:
                            run_metrics[metric] = run.summary[metric]

                    runs_info.append(
                        {
                            "run_id": run_id,
                            "run_name": run.name,
                            "metrics": run_metrics,
                            "config": dict(run.config),
                            "tags": run.tags,
                            "url": run.url,
                        }
                    )
                except Exception as e:
                    logger.warning("Failed to get info for run %s: %s", run_id, e)

            # Create comparison artifact
            comparison_data = {
                "name": comparison_name,
                "description": description,
                "runs": runs_info,
                "compared_metrics": metrics,
                "created_at": datetime.now().isoformat(),
                "created_by": wandb.run.entity if wandb.run else self.config.entity,
            }

            # Log comparison
            if self.current_run:
                artifact = wandb.Artifact(
                    name=f"comparison_{comparison_name}", type="experiment_comparison"
                )

                with artifact.new_file("comparison.json") as f:
                    json.dump(comparison_data, f, indent=2)

                wandb.log_artifact(artifact)

            # Generate comparison URL
            comparison_url = (
                f"https://wandb.ai/{self.config.entity}/{self.config.project}/workspace"
            )

            logger.info("Created experiment comparison: %s", comparison_name)
            return comparison_url

        except Exception as e:
            logger.error("Failed to create experiment comparison: %s", e)
            raise

    def setup_automated_reporting(
        self,
        report_name: str,
        schedule: str,  # cron-like schedule
        metrics: List[str],
        recipients: List[str],
    ) -> str:
        """
        Setup automated reporting for team updates.

        Args:
            report_name: Name of the report
            schedule: Schedule for report generation
            metrics: Metrics to include in report
            recipients: List of recipient usernames

        Returns:
            Report configuration ID
        """
        try:
            report_config = {
                "name": report_name,
                "schedule": schedule,
                "metrics": metrics,
                "recipients": recipients,
                "created_at": datetime.now().isoformat(),
                "created_by": wandb.run.entity if wandb.run else self.config.entity,
                "project": self.config.project,
                "entity": self.config.entity,
            }

            # Store report configuration
            if self.current_run:
                artifact = wandb.Artifact(name=f"report_config_{report_name}", type="report_config")

                with artifact.new_file("report_config.json") as f:
                    json.dump(report_config, f, indent=2)

                wandb.log_artifact(artifact)

            # In a real implementation, this would set up actual scheduled reporting
            report_id = f"report_{report_name}_{int(time.time())}"

            logger.info("Setup automated reporting: %s", report_name)
            return report_id

        except Exception as e:
            logger.error("Failed to setup automated reporting: %s", e)
            raise

    def get_workspace_activity(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get recent workspace activity.

        Args:
            days: Number of days to look back

        Returns:
            List of activity events
        """
        try:
            # Get recent runs
            runs = self.api.runs(
                f"{self.config.entity}/{self.config.project}",
                filters={"created_at": {"$gte": datetime.now() - timedelta(days=days)}},
            )

            activity = []
            for run in runs:
                activity.append(
                    {
                        "type": "run_created",
                        "run_id": run.id,
                        "run_name": run.name,
                        "user": run.user.username if run.user else "unknown",
                        "created_at": run.created_at,
                        "state": run.state,
                        "tags": run.tags,
                    }
                )

            # Sort by creation time
            activity.sort(key=lambda x: x["created_at"], reverse=True)

            logger.debug("Retrieved %d activity events", len(activity))
            return activity

        except Exception as e:
            logger.error("Failed to get workspace activity: %s", e)
            return []

    def export_workspace_data(self, export_path: str) -> bool:
        """
        Export workspace data for backup or migration.

        Args:
            export_path: Path to export file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Collect workspace data
            workspace_data = {
                "config": asdict(self.config),
                "collaborators": {
                    username: asdict(collab) for username, collab in self.collaborators.items()
                },
                "experiment_templates": {
                    name: asdict(template) for name, template in self.experiment_templates.items()
                },
                "exported_at": datetime.now().isoformat(),
                "exported_by": wandb.run.entity if wandb.run else self.config.entity,
            }

            # Add recent runs information
            try:
                runs = self.api.runs(f"{self.config.entity}/{self.config.project}")
                workspace_data["runs"] = [
                    {
                        "id": run.id,
                        "name": run.name,
                        "state": run.state,
                        "created_at": run.created_at.isoformat() if run.created_at else None,
                        "tags": run.tags,
                        "url": run.url,
                    }
                    for run in runs[:100]  # Limit to recent 100 runs
                ]
            except Exception as e:
                logger.warning("Failed to export runs data: %s", e)
                workspace_data["runs"] = []

            # Write to file
            with open(export_path, "w") as f:
                json.dump(workspace_data, f, indent=2, default=str)

            logger.info("Exported workspace data to %s", export_path)
            return True

        except Exception as e:
            logger.error("Failed to export workspace data: %s", e)
            return False

    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics for the workspace."""
        try:
            stats = {
                "total_collaborators": len(self.collaborators),
                "collaborators_by_role": {},
                "total_experiments": 0,
                "active_experiments": 0,
                "shared_artifacts": len(self.shared_artifacts),
                "experiment_templates": len(self.experiment_templates),
            }

            # Count collaborators by role
            for collab in self.collaborators.values():
                role = collab.role.value
                stats["collaborators_by_role"][role] = (
                    stats["collaborators_by_role"].get(role, 0) + 1
                )

            # Get experiment counts
            try:
                runs = self.api.runs(f"{self.config.entity}/{self.config.project}")
                stats["total_experiments"] = len(runs)
                stats["active_experiments"] = len([r for r in runs if r.state == "running"])
            except Exception as e:
                logger.warning("Failed to get experiment counts: %s", e)

            return stats

        except Exception as e:
            logger.error("Failed to get collaboration stats: %s", e)
            return {}


# Legacy compatibility class
class WandBTracker(WandBCollaborativeWorkspace):
    """
    Legacy W&B tracker class for backward compatibility.

    Extends the collaborative workspace with the original simple interface.
    """

    def __init__(self, project: str = "medical-ai", entity: Optional[str] = None):
        """Initialize with legacy interface."""
        config = WorkspaceConfig(
            entity=entity or "histocore", project=project, workspace_name=f"{project}_workspace"
        )
        super().__init__(config)

    def init_run(
        self, name: str = None, config: Dict[str, Any] = None, tags: list = None, notes: str = None
    ):
        """Initialize W&B run (legacy interface)."""
        run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
        )

        self.current_run = run
        logger.info("W&B run initialized: %s", run.name)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics."""
        wandb.log(metrics, step=step)

    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """Log artifact."""
        artifact = wandb.Artifact(name=artifact_path, type=artifact_type)
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)

    def log_model(self, model_path: str):
        """Log model."""
        wandb.save(model_path)

    def log_image(self, key: str, image):
        """Log image."""
        wandb.log({key: wandb.Image(image)})

    def log_table(self, key: str, data: list, columns: list):
        """Log table."""
        table = wandb.Table(data=data, columns=columns)
        wandb.log({key: table})

    def finish(self):
        """Finish W&B run."""
        wandb.finish()

    def sweep(self, sweep_config: Dict[str, Any]) -> str:
        """Create hyperparameter sweep."""
        sweep_id = wandb.sweep(sweep_config, project=self.config.project)
        logger.info("Sweep created: %s", sweep_id)
        return sweep_id

    def agent(self, sweep_id: str, function, count: int = None):
        """Run sweep agent."""
        wandb.agent(sweep_id, function=function, count=count)
