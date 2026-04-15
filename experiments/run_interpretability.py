"""
Run Interpretability Analysis

This script loads a trained model and generates comprehensive interpretability
visualizations including:
- Cross-modal attention visualization
- Temporal attention for WSI patches
- Gradient-based and Integrated Gradients saliency maps
- t-SNE and PCA embedding visualizations
- Modality correlation analysis
- Summary HTML report

Usage:
    python experiments/run_interpretability.py

Output:
    - results/interpretability/*.png: Visualization images
    - results/interpretability/summary.html: HTML report with all visualizations
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ClassificationHead, MultimodalFusionModel
from src.utils.interpretability import AttentionVisualizer, EmbeddingAnalyzer, SaliencyMap

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
MODEL_PATH = "models/quick_demo_model.pth"
OUTPUT_DIR = "results/interpretability"


class SyntheticMultimodalDataset(Dataset):
    """Synthetic dataset for interpretability analysis."""

    def __init__(self, num_samples=50, num_classes=3, missing_rate=0.15):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.missing_rate = missing_rate
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = self.labels[idx].item()

        # WSI features
        if np.random.rand() > self.missing_rate:
            num_patches = np.random.randint(30, 80)
            wsi_features = torch.randn(num_patches, 1024) + label * 2.0
        else:
            wsi_features = None

        # Genomic features
        if np.random.rand() > self.missing_rate:
            genomic = torch.randn(2000) + label * 1.5
        else:
            genomic = None

        # Clinical text
        if np.random.rand() > self.missing_rate:
            seq_len = np.random.randint(30, 100)
            clinical_text = torch.randint(1, 30000, (seq_len,))
            clinical_text[:10] = clinical_text[:10] + label * 2000
            clinical_text = torch.clamp(clinical_text, 1, 29999)
        else:
            clinical_text = None

        return {
            "wsi_features": wsi_features,
            "genomic": genomic,
            "clinical_text": clinical_text,
            "label": self.labels[idx],
        }


def collate_fn(batch):
    """Custom collate function."""
    batch_size = len(batch)
    labels = torch.stack([item["label"] for item in batch])

    # WSI
    wsi_list = [item["wsi_features"] for item in batch]
    max_patches = max((wsi.shape[0] for wsi in wsi_list if wsi is not None), default=0)

    if max_patches > 0:
        wsi_padded = torch.zeros(batch_size, max_patches, 1024)
        wsi_mask = torch.zeros(batch_size, max_patches, dtype=torch.bool)
        for i, wsi in enumerate(wsi_list):
            if wsi is not None:
                length = wsi.shape[0]
                wsi_padded[i, :length] = wsi
                wsi_mask[i, :length] = True
    else:
        wsi_padded = None
        wsi_mask = None

    # Genomic
    genomic_list = [item["genomic"] for item in batch if item["genomic"] is not None]
    if len(genomic_list) > 0:
        genomic = torch.zeros(batch_size, 2000)
        for i, item in enumerate(batch):
            if item["genomic"] is not None:
                genomic[i] = item["genomic"]
    else:
        genomic = None

    # Clinical text
    clinical_list = [item["clinical_text"] for item in batch if item["clinical_text"] is not None]
    if len(clinical_list) > 0:
        max_len = max(text.shape[0] for text in clinical_list)
        clinical_text = torch.zeros(batch_size, max_len, dtype=torch.long)
        clinical_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, item in enumerate(batch):
            if item["clinical_text"] is not None:
                length = item["clinical_text"].shape[0]
                clinical_text[i, :length] = item["clinical_text"]
                clinical_mask[i, :length] = True
    else:
        clinical_text = None
        clinical_mask = None

    return {
        "wsi_features": wsi_padded,
        "wsi_mask": wsi_mask,
        "genomic": genomic,
        "clinical_text": clinical_text,
        "clinical_mask": clinical_mask,
        "label": labels,
    }


def generate_summary_html(visualizations: dict, output_path: str) -> None:
    """Generate HTML summary page with all visualizations."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Interpretability Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 40px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
        }}
        .visualization {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        .visualization h3 {{
            color: #666;
            margin-top: 0;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 30px;
        }}
        .metric {{
            background: white;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
        }}
        .timestamp {{
            color: #999;
            font-size: 12px;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Interpretability Report</h1>

        <h2>Overview</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{visualizations.get('num_samples', 'N/A')}</div>
                <div class="metric-label">Samples Analyzed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{visualizations.get('num_classes', 'N/A')}</div>
                <div class="metric-label">Classes</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(visualizations.get('visualizations', []))}</div>
                <div class="metric-label">Visualizations</div>
            </div>
        </div>

        <h2>Attention Analysis</h2>
        <div class="grid">
"""

    # Add attention visualizations
    for viz_name in ["modality_attention", "temporal_attention"]:
        if viz_name in visualizations:
            html_content += f"""
            <div class="card">
                <h3>{viz_name.replace('_', ' ').title()}</h3>
                <img src="{viz_name}.png" alt="{viz_name}">
            </div>
"""

    html_content += """
        </div>

        <h2>Saliency Analysis</h2>
        <div class="grid">
"""

    # Add saliency visualizations
    for viz_name in ["saliency_wsi", "saliency_genomic", "saliency_clinical"]:
        if viz_name in visualizations:
            html_content += f"""
            <div class="card">
                <h3>{viz_name.replace('_', ' ').title()}</h3>
                <img src="{viz_name}.png" alt="{viz_name}">
            </div>
"""

    html_content += """
        </div>

        <h2>Embedding Analysis</h2>
        <div class="grid">
"""

    # Add embedding visualizations
    for viz_name in ["tsne", "pca"]:
        if viz_name in visualizations:
            html_content += f"""
            <div class="card">
                <h3>{viz_name.upper()} Visualization</h3>
                <img src="{viz_name}_visualization.png" alt="{viz_name}">
            </div>
"""

    html_content += """
        </div>

        <h2>Modality Correlation</h2>
"""

    if "modality_correlation" in visualizations:
        html_content += """
        <div class="visualization">
            <img src="modality_correlation.png" alt="Modality Correlation">
        </div>
"""

    html_content += f"""
        <div class="timestamp">
            Report generated on {visualizations.get('timestamp', 'Unknown')}
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html_content)


def main():
    """Main interpretability analysis pipeline."""
    print("=" * 60)
    print("MODEL INTERPRETABILITY ANALYSIS")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print("\n[1/6] Loading model...")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please train a model first using run_quick_demo.py")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Initialize model with same architecture as training
    model = MultimodalFusionModel(embed_dim=128).to(device)
    classifier = ClassificationHead(input_dim=128, num_classes=3).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    model.eval()
    classifier.eval()

    print(f"  Loaded model from {MODEL_PATH}")
    print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")

    # Create dataset
    print("\n[2/6] Creating dataset...")
    dataset = SyntheticMultimodalDataset(num_samples=50, num_classes=3, missing_rate=0.15)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    print(f"  Dataset size: {len(dataset)} samples")

    # Initialize interpretability tools
    attention_viz = AttentionVisualizer(output_dir=OUTPUT_DIR)
    saliency_map = SaliencyMap(device=device)
    embedding_analyzer = EmbeddingAnalyzer(output_dir=OUTPUT_DIR)

    # Collect embeddings and data
    print("\n[3/6] Collecting embeddings and computing saliency...")

    all_embeddings = []
    all_labels = []
    all_modality_embeddings = {}
    all_slide_embeddings = []

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        labels = batch["label"]
        batch.pop("label")

        with torch.no_grad():
            fused_emb, mod_emb = model(batch, return_modality_embeddings=True)

        all_embeddings.append(fused_emb.cpu())
        all_labels.append(labels.cpu())

        # Collect per-modality embeddings
        for mod_name, mod_emb in mod_emb.items():
            if mod_emb is not None:
                if mod_name not in all_modality_embeddings:
                    all_modality_embeddings[mod_name] = []
                all_modality_embeddings[mod_name].append(mod_emb.cpu())

        # Collect slide embeddings for temporal attention
        if batch.get("wsi_features") is not None:
            wsi_encoder = model.wsi_encoder
            wsi_emb = wsi_encoder(batch["wsi_features"], mask=batch.get("wsi_mask"))
            all_slide_embeddings.append(wsi_emb.cpu())

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Process modality embeddings
    modality_emb_dict = {}
    for mod_name, emb_list in all_modality_embeddings.items():
        modality_emb_dict[mod_name] = torch.cat(emb_list, dim=0)

    print(f"  Collected embeddings: {all_embeddings.shape}")

    # Store visualization paths
    visualizations = {"num_samples": len(all_embeddings), "num_classes": 3, "visualizations": []}

    # 1. Attention Visualization
    print("\n[4/6] Generating attention visualizations...")

    # Modality attention
    if modality_emb_dict:
        viz_path = attention_viz.plot_modality_attention(modality_emb_dict)
        print(f"  Saved: {viz_path}")
        visualizations["visualizations"].append("modality_attention")
        visualizations["modality_attention"] = "modality_attention.png"

    # Temporal attention
    if all_slide_embeddings:
        slide_emb = torch.cat(all_slide_embeddings, dim=0)
        # Take subset for visualization
        if slide_emb.shape[0] > 0:
            # Use first batch's slide embeddings
            sample_slide_emb = slide_emb[: min(8, slide_emb.shape[0])]
            viz_path = attention_viz.plot_temporal_attention(sample_slide_emb)
            print(f"  Saved: {viz_path}")
            visualizations["visualizations"].append("temporal_attention")
            visualizations["temporal_attention"] = "temporal_attention.png"

    # 2. Saliency Maps
    print("\n[5/6] Computing saliency maps...")

    # Get a batch for saliency computation
    saliency_batch = next(iter(dataloader))
    saliency_batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in saliency_batch.items()
    }

    # Gradient-based saliency
    try:
        saliency_maps = saliency_map.compute_gradient_saliency(model, saliency_batch)
        for mod_name, saliency in saliency_maps.items():
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(saliency.mean(axis=0), linewidth=2)
            ax.fill_between(range(len(saliency.mean(axis=0))), saliency.mean(axis=0), alpha=0.3)
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("Saliency Magnitude")
            ax.set_title(f"{mod_name.capitalize()} Saliency Profile")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            filepath = os.path.join(OUTPUT_DIR, f"saliency_{mod_name}.png")
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {filepath}")
            visualizations["visualizations"].append(f"saliency_{mod_name}")
            visualizations[f"saliency_{mod_name}"] = f"saliency_{mod_name}.png"
    except Exception as e:
        print(f"  Warning: Saliency computation failed: {e}")

    # 3. Embedding Analysis
    print("\n[6/6] Generating embedding visualizations...")

    # Check for NaN
    if np.isnan(all_embeddings).any():
        print("  Warning: NaN found in embeddings, replacing with zeros")
        all_embeddings = np.nan_to_num(all_embeddings, nan=0.0)

    # t-SNE
    viz_path = embedding_analyzer.plot_tsne(
        all_embeddings,
        all_labels,
        title="t-SNE of Multimodal Embeddings",
        filename="tsne_visualization.png",
    )
    print(f"  Saved: {viz_path}")
    visualizations["visualizations"].append("tsne")
    visualizations["tsne"] = "tsne_visualization.png"

    # PCA
    viz_path, variance = embedding_analyzer.plot_pca(
        all_embeddings,
        all_labels,
        title="PCA of Multimodal Embeddings",
        filename="pca_visualization.png",
    )
    print(f"  Saved: {viz_path}")
    print(f"  Explained variance: PC1={variance['PC1']:.2%}, PC2={variance['PC2']:.2%}")
    visualizations["visualizations"].append("pca")
    visualizations["pca"] = "pca_visualization.png"

    # Modality correlation
    if len(modality_emb_dict) > 1:
        viz_path, corr_matrix = embedding_analyzer.compute_modality_correlation(modality_emb_dict)
        print(f"  Saved: {viz_path}")
        print(f"  Correlation matrix:\n{corr_matrix}")
        visualizations["visualizations"].append("modality_correlation")
        visualizations["modality_correlation"] = "modality_correlation.png"

    # Generate HTML summary
    print("\n[Summary] Generating HTML report...")
    visualizations["timestamp"] = str(np.datetime64("now"))

    html_path = os.path.join(OUTPUT_DIR, "summary.html")
    generate_summary_html(visualizations, html_path)
    print(f"  Saved: {html_path}")

    print("\n" + "=" * 60)
    print("INTERPRETABILITY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated visualizations:")
    for viz in visualizations.get("visualizations", []):
        print(f"  - {viz}")
    print(f"\nHTML summary: {html_path}")
    print("=" * 60)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main()
