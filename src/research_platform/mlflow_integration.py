"""
MLflow Integration

Experiment tracking with MLflow.
"""

import mlflow
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class MLflowTracker:
    """
    MLflow experiment tracker
    
    Features:
    - Experiment logging
    - Model registry
    - Hyperparameter tracking
    - Artifact management
    """
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = 'medical-ai'):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        
        self.logger = logging.getLogger(__name__)
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """Start MLflow run"""
        
        run = mlflow.start_run(run_name=run_name, tags=tags)
        self.logger.info(f"Started run: {run.info.run_id}")
        
        return run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, artifact_path: Path):
        """Log artifact"""
        mlflow.log_artifact(str(artifact_path))
    
    def log_model(self, model, artifact_path: str = 'model'):
        """Log model"""
        mlflow.pytorch.log_model(model, artifact_path)
    
    def register_model(self, model_uri: str, model_name: str) -> str:
        """Register model"""
        
        result = mlflow.register_model(model_uri, model_name)
        self.logger.info(f"Registered model: {model_name} v{result.version}")
        
        return result.version
    
    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()
    
    def load_model(self, model_uri: str):
        """Load model"""
        return mlflow.pytorch.load_model(model_uri)
