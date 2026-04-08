"""Regression tests for the statistical analysis helpers."""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from experiments.statistical_analysis import AblationStudy, run_cross_validation
from src.data.loaders import MultimodalDataset
from src.models import ClassificationHead, MultimodalFusionModel


@pytest.fixture
def temp_multimodal_cv_data():
    """Create a small multimodal dataset with variable-length modalities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        (data_dir / "wsi_features").mkdir()
        (data_dir / "genomic").mkdir()
        (data_dir / "clinical_text").mkdir()

        samples = []
        for i in range(6):
            with h5py.File(data_dir / "wsi_features" / f"patient_{i}_wsi.h5", "w") as f:
                f.create_dataset(
                    "features", data=np.random.randn(5 + i, 512).astype(np.float32)
                )

            np.save(
                data_dir / "genomic" / f"patient_{i}_genomic.npy",
                np.random.randn(1000).astype(np.float32),
            )
            np.save(
                data_dir / "clinical_text" / f"patient_{i}_clinical.npy",
                np.random.randint(1, 1000, size=16 + i).astype(np.int64),
            )

            samples.append(
                {
                    "patient_id": f"patient_{i}",
                    "wsi_file": f"patient_{i}_wsi.h5",
                    "genomic_file": f"patient_{i}_genomic.npy",
                    "clinical_file": f"patient_{i}_clinical.npy",
                    "label": i % 2,
                }
            )

        with open(data_dir / "train_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"samples": samples}, f)

        yield data_dir


def test_run_cross_validation_supports_multimodal_dataset(temp_multimodal_cv_data):
    """Cross-validation should infer collate_multimodal for MultimodalDataset."""
    dataset = MultimodalDataset(
        temp_multimodal_cv_data,
        "train",
        DictConfig(
            {
                "wsi_enabled": True,
                "genomic_enabled": True,
                "clinical_text_enabled": True,
                "max_text_length": 32,
            }
        ),
    )

    class WrappedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = MultimodalFusionModel(
                embed_dim=16,
                wsi_config={
                    "input_dim": 512,
                    "hidden_dim": 32,
                    "output_dim": 16,
                    "num_heads": 4,
                    "num_layers": 1,
                },
                genomic_config={
                    "input_dim": 1000,
                    "hidden_dims": [64],
                    "output_dim": 16,
                    "dropout": 0.1,
                    "use_batch_norm": True,
                },
                clinical_config={
                    "vocab_size": 1000,
                    "embed_dim": 16,
                    "hidden_dim": 32,
                    "output_dim": 16,
                    "num_heads": 4,
                    "num_layers": 1,
                    "max_seq_length": 32,
                    "dropout": 0.1,
                    "pooling": "mean",
                },
                fusion_config={
                    "embed_dim": 16,
                    "num_heads": 4,
                    "dropout": 0.1,
                    "modalities": ["wsi", "genomic", "clinical"],
                },
            )
            self.classification_head = ClassificationHead(input_dim=16, num_classes=2)

        def forward(self, batch):
            return self.backbone(batch)

    results = run_cross_validation(
        model_factory=WrappedModel,
        dataset=dataset,
        n_folds=2,
        batch_size=2,
        device="cpu",
        n_bootstrap=10,
    )

    assert results["n_folds"] == 2
    assert len(results["fold_metrics"]) == 2
    assert "mean_accuracy" in results
    assert "mean_f1" in results


def test_run_cross_validation_supports_logits_returning_wrapper(temp_multimodal_cv_data):
    """Cross-validation should not double-apply classification heads."""
    dataset = MultimodalDataset(
        temp_multimodal_cv_data,
        "train",
        DictConfig(
            {
                "wsi_enabled": True,
                "genomic_enabled": True,
                "clinical_text_enabled": True,
                "max_text_length": 32,
            }
        ),
    )

    class WrappedLogitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = MultimodalFusionModel(
                embed_dim=16,
                wsi_config={
                    "input_dim": 512,
                    "hidden_dim": 32,
                    "output_dim": 16,
                    "num_heads": 4,
                    "num_layers": 1,
                },
                genomic_config={
                    "input_dim": 1000,
                    "hidden_dims": [64],
                    "output_dim": 16,
                    "dropout": 0.1,
                    "use_batch_norm": True,
                },
                clinical_config={
                    "vocab_size": 1000,
                    "embed_dim": 16,
                    "hidden_dim": 32,
                    "output_dim": 16,
                    "num_heads": 4,
                    "num_layers": 1,
                    "max_seq_length": 32,
                    "dropout": 0.1,
                    "pooling": "mean",
                },
                fusion_config={
                    "embed_dim": 16,
                    "num_heads": 4,
                    "dropout": 0.1,
                    "modalities": ["wsi", "genomic", "clinical"],
                },
            )
            self.classification_head = ClassificationHead(input_dim=16, num_classes=2)

        def forward(self, batch):
            embeddings = self.backbone(batch)
            return self.classification_head(embeddings)

    results = run_cross_validation(
        model_factory=WrappedLogitModel,
        dataset=dataset,
        n_folds=2,
        batch_size=2,
        device="cpu",
        n_bootstrap=10,
    )

    assert results["n_folds"] == 2
    assert len(results["fold_metrics"]) == 2


def test_ablation_study_supports_logits_returning_wrapper():
    """AblationStudy should handle wrapper models that already return logits."""
    class TinyLogitWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classification_head = ClassificationHead(input_dim=4, num_classes=2)

        def forward(self, batch):
            return torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float32)

    dataloader = DataLoader(
        [
            {
                "wsi_features": torch.randn(3, 4),
                "wsi_mask": torch.ones(3, dtype=torch.bool),
                "genomic": None,
                "clinical_text": None,
                "label": torch.tensor(0),
            },
            {
                "wsi_features": torch.randn(3, 4),
                "wsi_mask": torch.ones(3, dtype=torch.bool),
                "genomic": None,
                "clinical_text": None,
                "label": torch.tensor(1),
            },
        ],
        batch_size=2,
        collate_fn=lambda batch: {
            "wsi_features": torch.stack([item["wsi_features"] for item in batch]),
            "wsi_mask": torch.stack([item["wsi_mask"] for item in batch]),
            "genomic": None,
            "clinical_text": None,
            "label": torch.stack([item["label"] for item in batch]),
        },
    )

    study = AblationStudy(
        model_factory=TinyLogitWrapper,
        dataset=None,
        ablation_components=["wsi"],
        device="cpu",
        n_bootstrap=10,
    )

    metrics = study.evaluate_model(TinyLogitWrapper(), dataloader)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
