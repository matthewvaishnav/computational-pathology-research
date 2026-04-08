"""Regression tests for binary classification handling in experiment scripts."""

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.evaluate import ModelEvaluator
from experiments.train import MultimodalTrainer
from src.models import ClassificationHead, MultimodalFusionModel


def _create_binary_loader(batch_size=2, num_samples=4):
    """Create a small multimodal loader for binary classification tests."""
    data = []
    for i in range(num_samples):
        data.append(
            {
                "wsi_features": torch.randn(6, 1024),
                "wsi_mask": torch.ones(6, dtype=torch.bool),
                "genomic": torch.randn(2000),
                "clinical_text": torch.randint(1, 30000, (64,)),
                "clinical_mask": torch.ones(64, dtype=torch.bool),
                "label": torch.tensor(i % 2),
            }
        )

    def collate_fn(batch):
        return {
            "wsi_features": torch.stack([b["wsi_features"] for b in batch]),
            "wsi_mask": torch.stack([b["wsi_mask"] for b in batch]),
            "genomic": torch.stack([b["genomic"] for b in batch]),
            "clinical_text": torch.stack([b["clinical_text"] for b in batch]),
            "clinical_mask": torch.stack([b["clinical_mask"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
        }

    return DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def _create_experiment_model(embed_dim=32):
    """Create a small multimodal model for script-level tests."""
    return MultimodalFusionModel(
        embed_dim=embed_dim,
        wsi_config={
            "input_dim": 1024,
            "hidden_dim": 64,
            "output_dim": embed_dim,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.1,
            "pooling": "attention",
        },
        genomic_config={
            "input_dim": 2000,
            "hidden_dims": [128],
            "output_dim": embed_dim,
            "dropout": 0.1,
            "use_batch_norm": True,
        },
        clinical_config={
            "vocab_size": 30000,
            "embed_dim": 32,
            "hidden_dim": 64,
            "output_dim": embed_dim,
            "num_heads": 4,
            "num_layers": 1,
            "max_seq_length": 64,
            "dropout": 0.1,
            "pooling": "mean",
        },
        fusion_config={
            "embed_dim": embed_dim,
            "num_heads": 4,
            "dropout": 0.1,
            "modalities": ["wsi", "genomic", "clinical"],
        },
    )


def test_multimodal_trainer_supports_single_logit_binary_validation():
    """The legacy experiment trainer should support BCE-style binary heads."""
    loader = _create_binary_loader()
    model = _create_experiment_model()
    task_head = ClassificationHead(input_dim=32, num_classes=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = MultimodalTrainer(
            model=model,
            task_head=task_head,
            train_loader=loader,
            val_loader=loader,
            config={
                "task_type": "classification",
                "num_classes": 1,
                "learning_rate": 1e-3,
                "num_epochs": 1,
            },
            device="cpu",
            checkpoint_dir=Path(tmpdir) / "checkpoints",
            log_dir=Path(tmpdir) / "logs",
        )

        metrics = trainer.validate()

    assert metrics["val_loss"] > 0
    assert "val_auc" in metrics
    assert 0.0 <= metrics["val_accuracy"] <= 1.0


def test_multimodal_trainer_supports_two_logit_binary_validation():
    """The legacy experiment trainer should use CE/softmax for two-logit binary heads."""
    loader = _create_binary_loader()
    model = _create_experiment_model()
    task_head = ClassificationHead(input_dim=32, num_classes=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = MultimodalTrainer(
            model=model,
            task_head=task_head,
            train_loader=loader,
            val_loader=loader,
            config={
                "task_type": "classification",
                "num_classes": 2,
                "learning_rate": 1e-3,
                "num_epochs": 1,
            },
            device="cpu",
            checkpoint_dir=Path(tmpdir) / "checkpoints",
            log_dir=Path(tmpdir) / "logs",
        )

        metrics = trainer.validate()

    assert metrics["val_loss"] > 0
    assert "val_auc" in metrics
    assert 0.0 <= metrics["val_accuracy"] <= 1.0


def test_model_evaluator_supports_single_logit_binary_metrics(tmp_path):
    """The legacy evaluator should compute binary metrics from a single logit."""
    loader = _create_binary_loader()
    model = _create_experiment_model()
    task_head = ClassificationHead(input_dim=32, num_classes=1)

    evaluator = ModelEvaluator(
        model=model,
        task_head=task_head,
        test_loader=loader,
        device="cpu",
        output_dir=tmp_path / "single_logit_eval",
        config={"task_type": "classification", "num_classes": 1, "generate_plots": False},
    )

    metrics = evaluator.compute_metrics()

    assert "auc" in metrics
    assert metrics["accuracy"] >= 0.0
    assert evaluator.all_probs.ndim == 1


def test_model_evaluator_supports_two_logit_binary_metrics(tmp_path):
    """The legacy evaluator should compute binary metrics from the positive softmax class."""
    loader = _create_binary_loader()
    model = _create_experiment_model()
    task_head = ClassificationHead(input_dim=32, num_classes=2)

    evaluator = ModelEvaluator(
        model=model,
        task_head=task_head,
        test_loader=loader,
        device="cpu",
        output_dir=tmp_path / "two_logit_eval",
        config={"task_type": "classification", "num_classes": 2, "generate_plots": False},
    )

    metrics = evaluator.compute_metrics()

    assert "auc" in metrics
    assert metrics["accuracy"] >= 0.0
    assert evaluator.all_probs.ndim == 1
