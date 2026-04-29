"""
Weights & Biases Integration

Experiment tracking with W&B.
"""

import logging
from typing import Any, Dict, Optional

import wandb


class WandBTracker:
    """
    W&B experiment tracker

    Features:
    - Real-time monitoring
    - Hyperparameter sweeps
    - Model comparison
    - Collaborative workspaces
    """

    def __init__(self, project: str = "medical-ai", entity: Optional[str] = None):
        self.project = project
        self.entity = entity
        self.logger = logging.getLogger(__name__)

    def init_run(
        self, name: str = None, config: Dict[str, Any] = None, tags: list = None, notes: str = None
    ):
        """Initialize W&B run"""

        wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
        )

        self.logger.info(f"W&B run initialized: {wandb.run.name}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics"""
        wandb.log(metrics, step=step)

    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """Log artifact"""

        artifact = wandb.Artifact(name=artifact_path, type=artifact_type)
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)

    def log_model(self, model_path: str):
        """Log model"""
        wandb.save(model_path)

    def log_image(self, key: str, image):
        """Log image"""
        wandb.log({key: wandb.Image(image)})

    def log_table(self, key: str, data: list, columns: list):
        """Log table"""
        table = wandb.Table(data=data, columns=columns)
        wandb.log({key: table})

    def finish(self):
        """Finish W&B run"""
        wandb.finish()

    def sweep(self, sweep_config: Dict[str, Any]) -> str:
        """Create hyperparameter sweep"""

        sweep_id = wandb.sweep(sweep_config, project=self.project)
        self.logger.info(f"Sweep created: {sweep_id}")

        return sweep_id

    def agent(self, sweep_id: str, function, count: int = None):
        """Run sweep agent"""
        wandb.agent(sweep_id, function=function, count=count)
