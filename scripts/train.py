"""
Supervised training script for multimodal and baseline pathology models.

Usage:
    # Train multimodal model
    python scripts/train.py model=multimodal task=classification

    # Train baseline (single modality)
    python scripts/train.py model=baseline model.modality=wsi task=classification

    # Train with custom config
    python scripts/train.py model=multimodal task=classification training.num_epochs=100

    # Train with specific data directory
    python scripts/train.py data.data_dir=/path/to/data model=multimodal
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import MultimodalDataset, collate_multimodal
from src.models import ClassificationHead, MultimodalFusionModel, get_baseline_model
from src.training import SupervisedTrainer
from src.utils.monitoring import get_logger

logger = get_logger(__name__)


def _patch_hydra_help_for_python_314():
    """Patch Hydra's argparse help construction for Python 3.14+."""
    if sys.version_info < (3, 14):
        return

    import argparse
    import importlib

    hydra_utils = importlib.import_module("hydra._internal.utils")
    hydra_main_module = importlib.import_module("hydra.main")

    if getattr(hydra_utils, "_py314_help_patch_applied", False):
        return

    def patched_get_args_parser() -> argparse.ArgumentParser:
        from hydra import __version__

        parser = argparse.ArgumentParser(add_help=False, description="Hydra")
        parser.add_argument("--help", "-h", action="store_true", help="Application's help")
        parser.add_argument("--hydra-help", action="store_true", help="Hydra's help")
        parser.add_argument(
            "--version",
            action="version",
            help="Show Hydra's version and exit",
            version=f"Hydra {__version__}",
        )
        parser.add_argument(
            "overrides",
            nargs="*",
            help="Any key=value arguments to override config values (use dots for.nested=overrides)",
        )
        parser.add_argument(
            "--cfg",
            "-c",
            choices=["job", "hydra", "all"],
            help="Show config instead of running [job|hydra|all]",
        )
        parser.add_argument(
            "--resolve",
            action="store_true",
            help="Used in conjunction with --cfg, resolve config interpolations before printing.",
        )
        parser.add_argument("--package", "-p", help="Config package to show")
        parser.add_argument("--run", "-r", action="store_true", help="Run a job")
        parser.add_argument(
            "--multirun",
            "-m",
            action="store_true",
            help="Run multiple jobs with the configured launcher and sweeper",
        )
        parser.add_argument(
            "--shell-completion",
            "-sc",
            action="store_true",
            help=f"Install or Uninstall shell completion:\n{hydra_utils._get_completion_help()}",
        )
        parser.add_argument(
            "--config-path",
            "-cp",
            help="""Overrides the config_path specified in hydra.main().
                    The config_path is absolute or relative to the Python file declaring @hydra.main()""",
        )
        parser.add_argument(
            "--config-name",
            "-cn",
            help="Overrides the config_name specified in hydra.main()",
        )
        parser.add_argument(
            "--config-dir",
            "-cd",
            help="Adds an additional config dir to the config search path",
        )
        parser.add_argument(
            "--experimental-rerun",
            help="Rerun a job from a previous config pickle",
        )

        info_choices = [
            "all",
            "config",
            "defaults",
            "defaults-tree",
            "plugins",
            "searchpath",
        ]
        parser.add_argument(
            "--info",
            "-i",
            const="all",
            nargs="?",
            action="store",
            choices=info_choices,
            help=f"Print Hydra information [{'|'.join(info_choices)}]",
        )
        return parser

    hydra_utils.get_args_parser = patched_get_args_parser
    hydra_main_module.get_args_parser = patched_get_args_parser
    hydra_utils._py314_help_patch_applied = True


_patch_hydra_help_for_python_314()


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
        logger.info(f"Created MultimodalFusionModel with embed_dim={config.model.embed_dim}")

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
    history = trainer.fit(
        train_loader=dataloaders["train"],
        num_epochs=cfg.training.num_epochs,
        val_loader=dataloaders["val"],  # None if no validation data
    )

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    # Evaluate on test set if available
    if dataloaders["test"] is not None:
        logger.info("Evaluating on test set...")

        # Load best checkpoint if available (only exists when validation was used)
        best_checkpoint_path = output_dir / "checkpoints" / "checkpoint_best.pth"
        if dataloaders["val"] is not None and best_checkpoint_path.exists():
            trainer.load_checkpoint(str(best_checkpoint_path))
            logger.info("Loaded best checkpoint for evaluation")
        else:
            logger.info("Using latest checkpoint for evaluation (no validation split)")

        test_metrics = trainer.evaluate(dataloaders["test"])

        # Save test results
        results_path = output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Saved test results to {results_path}")

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
