"""Targeted tests for the PCam interpretability workflow."""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import experiments.generate_pcam_interpretability as pcam_interpretability


class _TinyPCamDataset(Dataset):
    """Minimal image/label dataset for interpretability workflow tests."""

    def __init__(self, num_samples: int = 4):
        self.images = torch.linspace(0.0, 1.0, steps=num_samples * 3 * 8 * 8).view(
            num_samples, 3, 8, 8
        )
        self.labels = torch.tensor([index % 2 for index in range(num_samples)], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return {"image": self.images[index], "label": self.labels[index]}


class _ToyFeatureExtractor(nn.Module):
    """Small differentiable image-to-feature model used in tests."""

    def __init__(
        self, model_name: str = "resnet18", pretrained: bool = False, feature_dim: int = 4
    ):
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
        "interpretability_report.md",
    ]
    for filename in expected_files:
        assert (tmp_path / filename).exists()

    with open(tmp_path / "interpretability_summary.json", "r", encoding="utf-8") as handle:
        saved_summary = json.load(handle)
    assert saved_summary["artifacts"]["pca_plot"].endswith("pcam_embeddings_pca.png")
    assert saved_summary["artifacts"]["report"].endswith("interpretability_report.md")
    assert saved_summary["summary_path"].endswith("interpretability_summary.json")
    assert saved_summary["metadata"]["checkpoint_path"].endswith("best_model.pth")
    assert saved_summary["metadata"]["split"] == "test"
    assert Path(summary["report_path"]).name == "interpretability_report.md"
    assert summary["artifacts"]["report"].endswith("interpretability_report.md")
    assert Path(summary["summary_path"]).name == "interpretability_summary.json"
    report_text = (tmp_path / "interpretability_report.md").read_text(encoding="utf-8")
    assert "PCam Interpretability Report" in report_text
    assert "Top Saliency Features" in report_text


def test_generate_pcam_interpretability_artifacts_rejects_too_few_samples(tmp_path):
    dataloader = DataLoader(_TinyPCamDataset(num_samples=1), batch_size=1, shuffle=False)

    with pytest.raises(ValueError, match="at least 2 samples"):
        pcam_interpretability.generate_pcam_interpretability_artifacts(
            feature_extractor=_ToyFeatureExtractor(),
            encoder=_ToyEncoder(),
            head=_ToyClassificationHead(),
            dataloader=dataloader,
            output_dir=str(tmp_path),
            device=torch.device("cpu"),
            max_samples=1,
            top_k=3,
        )


def test_generate_pcam_interpretability_artifacts_passes_max_samples_to_saliency(
    tmp_path, monkeypatch
):
    captured = {}

    class _StubAnalyzer:
        def __init__(self, output_dir: str):
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def plot_pca(self, embeddings, labels, title, filename):
            del embeddings, labels, title
            path = self.output_dir / filename
            path.write_text("pca", encoding="utf-8")
            return str(path), [0.75, 0.25]

        def plot_tsne(self, embeddings, labels, title, filename):
            del embeddings, labels, title
            path = self.output_dir / filename
            path.write_text("tsne", encoding="utf-8")
            return str(path)

    def _fake_saliency(
        feature_extractor,
        encoder,
        head,
        dataloader,
        device,
        output_dir,
        top_k,
        num_steps=16,
        max_samples=16,
    ):
        del feature_extractor, encoder, head, dataloader, device, num_steps
        captured["top_k"] = top_k
        captured["max_samples"] = max_samples
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "feature_saliency_topk.png"
        json_path = output_dir / "feature_saliency_topk.json"
        plot_path.write_text("plot", encoding="utf-8")
        json_path.write_text("{}", encoding="utf-8")
        return {
            "plot_path": str(plot_path),
            "json_path": str(json_path),
            "top_features": [{"feature_index": 0, "importance": 1.0}],
        }

    monkeypatch.setattr(pcam_interpretability, "EmbeddingAnalyzer", _StubAnalyzer)
    monkeypatch.setattr(pcam_interpretability, "compute_pcam_feature_saliency", _fake_saliency)

    dataloader = DataLoader(_TinyPCamDataset(num_samples=8), batch_size=2, shuffle=False)

    summary = pcam_interpretability.generate_pcam_interpretability_artifacts(
        feature_extractor=_ToyFeatureExtractor(),
        encoder=_ToyEncoder(),
        head=_ToyClassificationHead(),
        dataloader=dataloader,
        output_dir=str(tmp_path),
        device=torch.device("cpu"),
        max_samples=7,
        top_k=4,
    )

    assert captured["max_samples"] == 7
    assert captured["top_k"] == 4
    assert summary["num_samples"] == 7


def test_generate_pcam_interpretability_summary_preserves_top_k_metadata(tmp_path):
    dataloader = DataLoader(_TinyPCamDataset(num_samples=6), batch_size=2, shuffle=False)

    summary = pcam_interpretability.generate_pcam_interpretability_artifacts(
        feature_extractor=_ToyFeatureExtractor(),
        encoder=_ToyEncoder(),
        head=_ToyClassificationHead(),
        dataloader=dataloader,
        output_dir=str(tmp_path),
        device=torch.device("cpu"),
        max_samples=6,
        top_k=5,
        metadata={"checkpoint_path": "checkpoints/pcam/best_model.pth", "top_k": 5},
    )

    assert summary["metadata"]["top_k"] == 5

    with open(tmp_path / "interpretability_summary.json", "r", encoding="utf-8") as handle:
        saved_summary = json.load(handle)
    assert saved_summary["metadata"]["top_k"] == 5


def test_generate_pcam_interpretability_serializes_path_metadata(tmp_path):
    dataloader = DataLoader(_TinyPCamDataset(num_samples=6), batch_size=2, shuffle=False)

    summary = pcam_interpretability.generate_pcam_interpretability_artifacts(
        feature_extractor=_ToyFeatureExtractor(),
        encoder=_ToyEncoder(),
        head=_ToyClassificationHead(),
        dataloader=dataloader,
        output_dir=str(tmp_path),
        device=torch.device("cpu"),
        max_samples=6,
        top_k=4,
        metadata={
            "checkpoint_path": Path("checkpoints/pcam/best_model.pth"),
            "seed": 7,
            "fold": np.int64(2),
        },
    )

    assert summary["metadata"]["checkpoint_path"] == "checkpoints/pcam/best_model.pth"
    assert summary["metadata"]["seed"] == 7
    assert summary["metadata"]["fold"] == 2

    with open(tmp_path / "interpretability_summary.json", "r", encoding="utf-8") as handle:
        saved_summary = json.load(handle)
    assert saved_summary["metadata"]["checkpoint_path"] == "checkpoints/pcam/best_model.pth"
    assert saved_summary["metadata"]["seed"] == 7
    assert saved_summary["metadata"]["fold"] == 2


def test_compute_pcam_feature_saliency_rejects_non_positive_top_k(tmp_path):
    dataloader = DataLoader(_TinyPCamDataset(), batch_size=2, shuffle=False)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        pcam_interpretability.compute_pcam_feature_saliency(
            feature_extractor=_ToyFeatureExtractor(),
            encoder=_ToyEncoder(),
            head=_ToyClassificationHead(),
            dataloader=dataloader,
            device=torch.device("cpu"),
            output_dir=tmp_path,
            top_k=0,
        )


def test_compute_pcam_feature_saliency_collects_across_multiple_batches(tmp_path, monkeypatch):
    dataloader = DataLoader(_TinyPCamDataset(num_samples=5), batch_size=2, shuffle=False)
    captured = {}

    class _CapturingSaliency:
        def __init__(self, device):
            self.device = device

        def compute_integrated_gradients(self, model, batch, num_steps):
            del model, num_steps
            captured["num_samples"] = batch["wsi_features"].shape[0]
            captured["labels"] = batch["label"].detach().cpu().tolist()
            return {"wsi": torch.ones_like(batch["wsi_features"]).detach().cpu().numpy()}

    monkeypatch.setattr(pcam_interpretability, "SaliencyMap", _CapturingSaliency)

    artifacts = pcam_interpretability.compute_pcam_feature_saliency(
        feature_extractor=_ToyFeatureExtractor(),
        encoder=_ToyEncoder(),
        head=_ToyClassificationHead(),
        dataloader=dataloader,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        max_samples=5,
        top_k=2,
    )

    assert captured["num_samples"] == 5
    assert captured["labels"] == [0, 1, 0, 1, 0]
    assert Path(artifacts["plot_path"]).exists()
    assert Path(artifacts["json_path"]).exists()


def test_load_pcam_models_from_checkpoint_respects_no_hidden_layer_config(tmp_path, monkeypatch):
    monkeypatch.setattr(pcam_interpretability, "ResNetFeatureExtractor", _ToyFeatureExtractor)
    monkeypatch.setattr(pcam_interpretability, "WSIEncoder", _ToyEncoder)
    monkeypatch.setattr(pcam_interpretability, "ClassificationHead", _ToyClassificationHead)

    config = {
        "experiment": {"name": "pcam-test"},
        "model": {
            "embed_dim": 5,
            "feature_extractor": {"model": "resnet18", "feature_dim": 4},
            "wsi": {
                "input_dim": 4,
                "hidden_dim": 6,
                "num_heads": 1,
                "num_layers": 1,
                "pooling": "mean",
            },
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
