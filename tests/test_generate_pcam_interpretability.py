"""Targeted tests for the PCam interpretability workflow."""

from pathlib import Path

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import experiments.generate_pcam_interpretability as pcam_interpretability


class _TinyPCamDataset(Dataset):
    """Minimal image/label dataset for interpretability workflow tests."""

    def __init__(self, num_samples: int = 4):
        self.images = torch.linspace(0.0, 1.0, steps=num_samples * 3 * 8 * 8).view(num_samples, 3, 8, 8)
        self.labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)[:num_samples]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return {"image": self.images[index], "label": self.labels[index]}


class _ToyFeatureExtractor(nn.Module):
    """Small differentiable image-to-feature model used in tests."""

    def __init__(self, model_name: str = "resnet18", pretrained: bool = False, feature_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(3, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        return self.linear(pooled)


class _ToyEncoder(nn.Module):
    """Simple encoder that averages sequence features and projects them."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 6, output_dim: int = 5, **_: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask
        if features.dim() == 3:
            features = features.mean(dim=1)
        return self.linear(features)


class _ToyClassificationHead(nn.Module):
    """Simple classification head that preserves the hidden-layer flag."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 8,
        num_classes: int = 1,
        dropout: float = 0.1,
        use_hidden_layer: bool = True,
    ):
        super().__init__()
        self.use_hidden_layer = use_hidden_layer
        if use_hidden_layer:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, num_classes))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


def test_collect_pcam_embeddings_respects_max_samples():
    dataloader = DataLoader(_TinyPCamDataset(), batch_size=2, shuffle=False)
    collected = pcam_interpretability.collect_pcam_embeddings(
        feature_extractor=_ToyFeatureExtractor(),
        encoder=_ToyEncoder(),
        head=_ToyClassificationHead(),
        dataloader=dataloader,
        device=torch.device("cpu"),
        max_samples=3,
    )

    assert collected["embeddings"].shape == (3, 5)
    assert collected["labels"].tolist() == [0, 1, 0]
    assert collected["probabilities"].shape == (3,)


def test_generate_pcam_interpretability_artifacts_writes_summary_and_plots(tmp_path):
    dataloader = DataLoader(_TinyPCamDataset(), batch_size=2, shuffle=False)

    summary = pcam_interpretability.generate_pcam_interpretability_artifacts(
        feature_extractor=_ToyFeatureExtractor(),
        encoder=_ToyEncoder(),
        head=_ToyClassificationHead(),
        dataloader=dataloader,
        output_dir=str(tmp_path),
        device=torch.device("cpu"),
        max_samples=4,
        top_k=3,
        metadata={"checkpoint_path": "checkpoints/pcam/best_model.pth", "split": "test"},
    )

    assert summary["num_samples"] == 4
    assert summary["embedding_dim"] == 5
    assert summary["metadata"]["split"] == "test"
    assert len(summary["top_saliency_features"]) == 3

    expected_files = [
        "pcam_embeddings_pca.png",
        "pcam_embeddings_tsne.png",
        "feature_saliency_topk.png",
        "feature_saliency_topk.json",
        "interpretability_summary.json",
    ]
    for filename in expected_files:
        assert (tmp_path / filename).exists()

    with open(tmp_path / "interpretability_summary.json", "r", encoding="utf-8") as handle:
        saved_summary = json.load(handle)
    assert saved_summary["artifacts"]["pca_plot"].endswith("pcam_embeddings_pca.png")
    assert saved_summary["metadata"]["checkpoint_path"].endswith("best_model.pth")


def test_load_pcam_models_from_checkpoint_respects_no_hidden_layer_config(tmp_path, monkeypatch):
    monkeypatch.setattr(pcam_interpretability, "ResNetFeatureExtractor", _ToyFeatureExtractor)
    monkeypatch.setattr(pcam_interpretability, "WSIEncoder", _ToyEncoder)
    monkeypatch.setattr(pcam_interpretability, "ClassificationHead", _ToyClassificationHead)

    config = {
        "experiment": {"name": "pcam-test"},
        "model": {
            "embed_dim": 5,
            "feature_extractor": {"model": "resnet18", "feature_dim": 4},
            "wsi": {"input_dim": 4, "hidden_dim": 6, "num_heads": 1, "num_layers": 1, "pooling": "mean"},
        },
        "task": {"classification": {"hidden_dims": [], "dropout": 0.1}},
        "training": {"dropout": 0.1},
    }

    feature_extractor = _ToyFeatureExtractor(feature_dim=4)
    encoder = _ToyEncoder(input_dim=4, hidden_dim=6, output_dim=5)
    head = _ToyClassificationHead(input_dim=5, num_classes=1, dropout=0.1, use_hidden_layer=False)

    checkpoint_path = tmp_path / "pcam_checkpoint.pth"
    torch.save(
        {
            "config": config,
            "feature_extractor_state_dict": feature_extractor.state_dict(),
            "encoder_state_dict": encoder.state_dict(),
            "head_state_dict": head.state_dict(),
        },
        checkpoint_path,
    )

    loaded_feature_extractor, loaded_encoder, loaded_head, loaded_config = (
        pcam_interpretability.load_pcam_models_from_checkpoint(
            str(checkpoint_path), torch.device("cpu")
        )
    )

    assert loaded_config["experiment"]["name"] == "pcam-test"
    assert loaded_head.use_hidden_layer is False

    images = torch.randn(2, 3, 8, 8)
    logits = loaded_head(loaded_encoder(loaded_feature_extractor(images).unsqueeze(1)))
    assert logits.shape == (2, 1)
