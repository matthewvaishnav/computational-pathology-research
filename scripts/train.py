"""
Supervised training script for multimodal and baseline pathology models.

Usage:
    # Train multimodal model
    python train.py model=multimodal task=classification
    
    # Train baseline (single modality)
    python train.py model=baseline model.modality=wsi task=classification
    
    # Train with custom config
    python train.py model=multimodal task=classification training.num_epochs=100
    
    # Train with specific data directory
    python train.py data.data_dir=/path/to/data model=multimodal
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data import MultimodalDataset, collate_multimodal
from src.models import ClassificationHead, MultimodalFusionModel, get_baseline_model
from src.training import SupervisedTrainer
from src.utils.monitoring import get_logger

logger = get_logger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(config: DictConfig) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Hydra configuration
        
    Returns:
        Dictionary of dataloaders
    """
    data_dir = Path(config.data.data_dir)
    
    # Dataset configurations
    dataset_configs = {
        "train": {
            "wsi_enabled": config.data.wsi_enabled,
            "genomic_enabled": config.data.genomic_enabled,
            "clinical_text_enabled": config.data.clinical_text_enabled,
            "max_text_length": config.data.max_text_length,
        },
        "val": {
            "wsi_enabled": config.data.wsi_enabled,
            "genomic_enabled": config.data.genomic_enabled,
            "clinical_text_enabled": config.data.clinical_text_enabled,
            "max_text_length": config.data.max_text_length,
        },
        "test": {
            "wsi_enabled": config.data.wsi_enabled,
            "genomic_enabled": config.data.genomic_enabled,
            "clinical_text_enabled": config.data.clinical_text_enabled,
            "max_text_length": config.data.max_text_length,
        },
    }
    
    # Create datasets
    datasets = {}
    for split in ["train", "val", "test"]:
        try:
            datasets[split] = MultimodalDataset(
                data_dir=data_dir,
                split=split,
                config=dataset_configs[split],
            )
            logger.info(f"Loaded {split} dataset: {len(datasets[split])} samples")
        except FileNotFoundError as e:
            logger.warning(f"Could not load {split} dataset: {e}")
            datasets[split] = None
    
    # Create dataloaders
    dataloaders = {}
    for split in ["train", "val", "test"]:
        if datasets[split] is not None and len(datasets[split]) > 0:
            dataloaders[split] = DataLoader(
                datasets[split],
                batch_size=config.training.batch_size,
                shuffle=(split == "train"),
                num_workers=config.data.num_workers,
                collate_fn=collate_multimodal,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=(split == "train"),
            )
        else:
            dataloaders[split] = None
    
    return dataloaders


def create_model(config: DictConfig) -> tuple:
    """
    Create model and task head based on configuration.
    
    Args:
        config: Hydra configuration
        
    Returns:
        Tuple of (model, task_head)
    """
    model_type = config.model.name
    num_classes = config.task.num_classes
    
    if model_type == "multimodal":
        # Full multimodal fusion model
        model = MultimodalFusionModel(
            embed_dim=config.model.embed_dim,
            dropout=config.model.dropout,
        )
        logger.info(
            f"Created MultimodalFusionModel with embed_dim={config.model.embed_dim}"
        )
    
    elif model_type == "baseline":
        # Single modality baseline
        modality = config.model.modality
        model = get_baseline_model(
            model_type="single_modality",
            modality=modality,
            embed_dim=config.model.embed_dim,
        )
        logger.info(
            f"Created SingleModalityModel (modality={modality}) "
            f"with embed_dim={config.model.embed_dim}"
        )
    
    elif model_type == "late_fusion":
        # Late fusion baseline
        model = get_baseline_model(
            model_type="late_fusion",
            embed_dim=config.model.embed_dim,
        )
        logger.info(f"Created LateFusionModel with embed_dim={config.model.embed_dim}")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create task head (classification or survival)
    if config.task.name == "classification":
        task_head = ClassificationHead(
            input_dim=config.model.embed_dim,
            num_classes=num_classes,
            hidden_dim=config.task.get("head_hidden_dim", 128),
            dropout=config.model.dropout,
        )
        logger.info(f"Created ClassificationHead with {num_classes} classes")
    
    elif config.task.name == "survival":
        from src.models import SurvivalPredictionHead
        task_head = SurvivalPredictionHead(
            input_dim=config.model.embed_dim,
            num_time_bins=config.task.num_time_bins,
        )
        logger.info(f"Created SurvivalPredictionHead")
    
    else:
        raise ValueError(f"Unknown task type: {config.task.name}")
    
    return model, task_head


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Set seed for reproducibility
    set_seed(cfg.training.seed)
    
    # Create output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(cfg)
    
    if dataloaders["train"] is None:
        logger.error("No training data found. Please check data directory.")
        return
    
    if dataloaders["val"] is None:
        logger.warning("No validation data found. Training without validation.")
    
    # Create model
    model, task_head = create_model(cfg)
    
    # Count parameters
    model_params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in task_head.parameters())
    logger.info(f"Model parameters: {model_params:,}")
    logger.info(f"Task head parameters: {head_params:,}")
    logger.info(f"Total parameters: {model_params + head_params:,}")
    
    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        task_head=task_head,
        num_classes=cfg.task.num_classes,
        device=cfg.training.device,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        use_amp=cfg.training.use_amp,
        grad_clip_norm=cfg.training.grad_clip_norm,
        early_stopping_patience=cfg.training.early_stopping_patience,
        checkpoint_dir=str(output_dir / "checkpoints"),
        log_dir=str(output_dir / "tensorboard"),
    )
    
    # Train
    if dataloaders["val"] is not None:
        history = trainer.fit(
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            num_epochs=cfg.training.num_epochs,
        )
        
        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
    else:
        # Train without validation
        logger.warning("Training without validation - early stopping disabled")
        for epoch in range(cfg.training.num_epochs):
            # Manual training loop without validation
            pass
    
    # Evaluate on test set if available
    if dataloaders["test"] is not None:
        logger.info("Evaluating on test set...")
        
        # Load best checkpoint
        best_checkpoint_path = output_dir / "checkpoints" / "checkpoint_best.pth"
        if best_checkpoint_path.exists():
            trainer.load_checkpoint(str(best_checkpoint_path))
        
        test_metrics = trainer.evaluate(dataloaders["test"])
        
        # Save test results
        results_path = output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Saved test results to {results_path}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
