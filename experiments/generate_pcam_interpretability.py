"""Generate interpretability artifacts for trained PCam checkpoints."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path for imports.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pcam_dataset import PCamDataset, get_pcam_transforms
from src.models.encoders import WSIEncoder
from src.models.feature_extractors import ResNetFeatureExtractor
from src.models.heads import ClassificationHead
from src.utils.interpretability import EmbeddingAnalyzer, SaliencyMap

logger = logging.getLogger(__name__)


class _PCamFeatureClassifier(nn.Module):
    """Wrap the PCam encoder/head to expose a batch-based saliency interface."""

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["wsi_features"]
        if features.dim() == 2:
            features = features.unsqueeze(1)

        mask = batch.get("wsi_mask")
        if mask is not None:
            embeddings = self.encoder(features, mask=mask)
        else:
            embeddings = self.encoder(features)

        return self.head(embeddings)


def load_pcam_models_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[nn.Module, nn.Module, nn.Module, Dict[str, Any]]:
    """Load the trained PCam feature extractor, encoder, head, and config."""
    checkpoint = torch.load(checkpoint_path, map_location=str(device))
    config = checkpoint.get("config")
    if config is None:
        raise RuntimeError("Checkpoint does not contain config; cannot reconstruct PCam models.")

    model_config = config["model"]
    feature_extractor_config = model_config["feature_extractor"]
    wsi_config = model_config["wsi"]
    classification_config = config["task"]["classification"]

    feature_extractor = ResNetFeatureExtractor(
        model_name=feature_extractor_config["model"],
        pretrained=False,
        feature_dim=feature_extractor_config.get("feature_dim"),
    )
    encoder = WSIEncoder(
        input_dim=wsi_config["input_dim"],
        hidden_dim=wsi_config["hidden_dim"],
        output_dim=model_config["embed_dim"],
        num_heads=wsi_config["num_heads"],
        num_layers=wsi_config["num_layers"],
        pooling=wsi_config.get("pooling", "mean"),
        dropout=config["training"].get("dropout", 0.1),
    )
    hidden_dims = classification_config.get("hidden_dims", [128])
    use_hidden_layer = len(hidden_dims) > 0
    hidden_dim = hidden_dims[0] if use_hidden_layer else 128
    head = ClassificationHead(
        input_dim=model_config["embed_dim"],
        hidden_dim=hidden_dim,
        num_classes=1,
        dropout=classification_config["dropout"],
        use_hidden_layer=use_hidden_layer,
    )

    feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    head.load_state_dict(checkpoint["head_state_dict"])

    feature_extractor = feature_extractor.to(device).eval()
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()

    return feature_extractor, encoder, head, config


def collect_pcam_embeddings(
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Collect embeddings, labels, and probabilities from a PCam loader."""
    embeddings = []
    labels = []
    probabilities = []
    collected = 0

    feature_extractor.eval()
    encoder.eval()
    head.eval()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            batch_labels = batch["label"].to(device)

            features = feature_extractor(images)
            batch_embeddings = encoder(features.unsqueeze(1))
            logits = head(batch_embeddings)
            batch_probabilities = torch.sigmoid(logits).squeeze(-1)

            remaining = None if max_samples is None else max_samples - collected
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None:
                batch_embeddings = batch_embeddings[:remaining]
                batch_labels = batch_labels[:remaining]
                batch_probabilities = batch_probabilities[:remaining]

            embeddings.append(batch_embeddings.cpu())
            labels.append(batch_labels.cpu())
            probabilities.append(batch_probabilities.cpu())
            collected += batch_embeddings.size(0)

            if max_samples is not None and collected >= max_samples:
                break

    if not embeddings:
        raise ValueError("No samples were collected for interpretability analysis.")

    return {
        "embeddings": torch.cat(embeddings, dim=0).numpy(),
        "labels": torch.cat(labels, dim=0).numpy(),
        "probabilities": torch.cat(probabilities, dim=0).numpy(),
    }


def compute_pcam_feature_saliency(
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    top_k: int = 20,
    num_steps: int = 16,
    max_samples: int = 16,
) -> Dict[str, Any]:
    """Generate a top-feature saliency plot over extracted PCam image features."""
    try:
        batch = next(iter(dataloader))
    except StopIteration as exc:
        raise ValueError("Interpretability dataloader is empty.") from exc

    images = batch["image"].to(device)[:max_samples]
    labels = batch["label"].to(device)[:max_samples]

    with torch.no_grad():
        features = feature_extractor(images)

    wrapper = _PCamFeatureClassifier(encoder, head).to(device).eval()
    saliency = SaliencyMap(device=device)
    attributions = saliency.compute_integrated_gradients(
        wrapper,
        {
            "wsi_features": features.unsqueeze(1),
            "wsi_mask": torch.ones(features.size(0), 1, dtype=torch.bool, device=device),
            "label": labels,
        },
        num_steps=num_steps,
    )

    if "wsi" not in attributions:
        raise RuntimeError("Expected WSI attributions were not produced.")

    saliency_scores = np.abs(attributions["wsi"]).mean(axis=(0, 1))
    top_k = min(top_k, saliency_scores.shape[0])
    top_indices = np.argsort(saliency_scores)[-top_k:][::-1]
    top_scores = saliency_scores[top_indices]

    plot_path = output_dir / "feature_saliency_topk.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(top_k), top_scores, color="steelblue")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([str(i) for i in top_indices], rotation=45, ha="right")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Mean |Integrated Gradient|")
    ax.set_title("Top PCam Feature Saliency")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    json_path = output_dir / "feature_saliency_topk.json"
    top_features = [
        {"feature_index": int(index), "importance": float(score)}
        for index, score in zip(top_indices, top_scores)
    ]
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"top_features": top_features}, handle, indent=2)

    return {
        "plot_path": str(plot_path),
        "json_path": str(json_path),
        "top_features": top_features,
    }


def generate_pcam_interpretability_artifacts(
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    output_dir: str,
    device: torch.device,
    max_samples: int = 128,
    top_k: int = 20,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate PCam interpretability artifacts and a small JSON manifest."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analyzer = EmbeddingAnalyzer(output_dir=str(output_path))
    collected = collect_pcam_embeddings(
        feature_extractor, encoder, head, dataloader, device=device, max_samples=max_samples
    )

    pca_path, explained_variance = analyzer.plot_pca(
        collected["embeddings"],
        collected["labels"],
        title="PCam Embedding PCA",
        filename="pcam_embeddings_pca.png",
    )
    tsne_path = analyzer.plot_tsne(
        collected["embeddings"],
        collected["labels"],
        title="PCam Embedding t-SNE",
        filename="pcam_embeddings_tsne.png",
    )
    saliency_artifacts = compute_pcam_feature_saliency(
        feature_extractor,
        encoder,
        head,
        dataloader,
        device=device,
        output_dir=output_path,
        top_k=top_k,
    )

    unique_labels, counts = np.unique(collected["labels"], return_counts=True)
    summary = {
        "num_samples": int(collected["embeddings"].shape[0]),
        "embedding_dim": int(collected["embeddings"].shape[1]),
        "label_counts": {str(label): int(count) for label, count in zip(unique_labels, counts)},
        "mean_probability": float(collected["probabilities"].mean()),
        "artifacts": {
            "pca_plot": str(pca_path),
            "tsne_plot": str(tsne_path),
            "saliency_plot": saliency_artifacts["plot_path"],
            "saliency_json": saliency_artifacts["json_path"],
        },
        "explained_variance": explained_variance,
        "top_saliency_features": saliency_artifacts["top_features"],
        "metadata": metadata or {},
    }

    summary_path = output_path / "interpretability_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_path"] = str(summary_path)

    return summary


def build_pcam_dataloader(
    data_root: str, split: str, batch_size: int, num_workers: int
) -> DataLoader:
    """Create the PCam dataloader used by the interpretability workflow."""
    dataset = PCamDataset(
        root_dir=data_root,
        split=split,
        transform=get_pcam_transforms(split=split, augmentation=False),
        download=False,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interpretability artifacts for PCam")
    parser.add_argument("--checkpoint", required=True, help="Path to trained PCam checkpoint")
    parser.add_argument(
        "--data-root", default="data/pcam", help="Root directory of the PCam dataset"
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--output-dir",
        default="results/pcam/interpretability",
        help="Directory to store interpretability artifacts",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; using CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    feature_extractor, encoder, head, config = load_pcam_models_from_checkpoint(
        args.checkpoint, device
    )
    dataloader = build_pcam_dataloader(
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    summary = generate_pcam_interpretability_artifacts(
        feature_extractor=feature_extractor,
        encoder=encoder,
        head=head,
        dataloader=dataloader,
        output_dir=args.output_dir,
        device=device,
        max_samples=args.max_samples,
        top_k=args.top_k,
        metadata={
            "checkpoint_path": args.checkpoint,
            "data_root": args.data_root,
            "split": args.split,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "experiment_name": config.get("experiment", {}).get("name"),
        },
    )

    logger.info("Generated PCam interpretability artifacts:")
    for name, path in summary["artifacts"].items():
        logger.info("  %s: %s", name, path)
    logger.info("  summary: %s", summary["summary_path"])
    logger.info(
        "Interpretability analysis reflects model behavior on the selected PCam split only; "
        "it does not constitute clinical validation."
    )


if __name__ == "__main__":
    main()
