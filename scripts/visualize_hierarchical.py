"""
Visualize hierarchical pooling regions.

Creates visualizations of spatial clustering and region assignments:
- Scatter plot of patches colored by region
- Region centroids
- Attention weights per region
- Interactive plots (optional)

Usage:
    # Visualize from coordinates
    python scripts/visualize_hierarchical.py --coords data/coords.npy --num-regions 16 --output viz/regions.png
    
    # Interactive plot
    python scripts/visualize_hierarchical.py --coords data/coords.npy --num-regions 16 --interactive
    
    # With attention weights
    python scripts/visualize_hierarchical.py --coords data/coords.npy --attention data/attention.npy --output viz/regions.png

Reference:
- TransnnMIL v2.0: Hierarchical + Topology (2027)
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.models.hierarchical_pooling import HierarchicalPooling


def visualize_regions_matplotlib(
    coords: np.ndarray,
    assignments: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    attention: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title: str = "Hierarchical Regions",
):
    """
    Visualize regions using matplotlib.

    Args:
        coords: Patch coordinates [N, 2]
        assignments: Region assignments [N] (region index per patch)
        centroids: Region centroids [num_regions, 2]
        attention: Attention weights [N] (importance per patch)
        output_path: Path to save figure
        title: Plot title
    """
    num_regions = assignments.max() + 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Region assignments
    ax = axes[0]
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=assignments,
        cmap="tab20",
        s=30,
        alpha=0.6,
    )
    plt.colorbar(scatter, ax=ax, label="Region")

    # Plot centroids
    if centroids is not None:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c="red",
            marker="X",
            s=200,
            edgecolors="black",
            linewidths=2,
            label="Centroids",
            zorder=10,
        )
        ax.legend()

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"{title} (N={len(coords)}, R={num_regions})")
    ax.grid(True, alpha=0.3)

    # 2. Attention weights (if available)
    ax = axes[1]
    if attention is not None:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=attention,
            cmap="viridis",
            s=30,
            alpha=0.7,
        )
        plt.colorbar(scatter, ax=ax, label="Attention Weight")
        ax.set_title("Attention Weights")
    else:
        # Region size distribution
        region_sizes = np.bincount(assignments)
        ax.bar(range(num_regions), region_sizes, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Region")
        ax.set_ylabel("Number of patches")
        ax.set_title("Region Size Distribution")
        ax.grid(True, alpha=0.3, axis="y")

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_regions_interactive(
    coords: np.ndarray,
    assignments: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    attention: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title: str = "Hierarchical Regions",
):
    """
    Create interactive region visualization using plotly.

    Args:
        coords: Patch coordinates [N, 2]
        assignments: Region assignments [N]
        centroids: Region centroids [num_regions, 2]
        attention: Attention weights [N]
        output_path: Path to save HTML
        title: Plot title
    """
    if not PLOTLY_AVAILABLE:
        print("Error: plotly not available. Install with: pip install plotly")
        return

    num_regions = assignments.max() + 1

    # Create traces for each region
    traces = []
    for region_id in range(num_regions):
        mask = assignments == region_id
        region_coords = coords[mask]

        # Color by attention if available
        if attention is not None:
            region_attention = attention[mask]
            marker_color = region_attention
            colorscale = "Viridis"
        else:
            marker_color = region_id
            colorscale = None

        trace = go.Scatter(
            x=region_coords[:, 0],
            y=region_coords[:, 1],
            mode="markers",
            marker=dict(
                size=8,
                color=marker_color,
                colorscale=colorscale,
                opacity=0.6,
            ),
            text=[f"Region {region_id}, Patch {i}" for i in range(len(region_coords))],
            hoverinfo="text",
            name=f"Region {region_id}",
        )
        traces.append(trace)

    # Add centroids
    if centroids is not None:
        centroid_trace = go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode="markers",
            marker=dict(
                size=15,
                color="red",
                symbol="x",
                line=dict(width=2, color="black"),
            ),
            text=[f"Centroid {i}" for i in range(len(centroids))],
            hoverinfo="text",
            name="Centroids",
        )
        traces.append(centroid_trace)

    # Create figure
    fig = go.Figure(data=traces)

    fig.update_layout(
        title=f"{title} (N={len(coords)}, R={num_regions})",
        xaxis=dict(title="X coordinate", showgrid=True),
        yaxis=dict(title="Y coordinate", showgrid=True),
        hovermode="closest",
        showlegend=True,
    )

    if output_path:
        fig.write_html(output_path)
        print(f"✓ Saved: {output_path}")
    else:
        fig.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize hierarchical regions")

    # Input options
    parser.add_argument("--coords", type=str, required=True, help="Path to coordinates (.npy)")
    parser.add_argument("--features", type=str, help="Path to features (.npy)")
    parser.add_argument("--attention", type=str, help="Path to attention weights (.npy)")

    # Model options
    parser.add_argument("--num-regions", type=int, default=16, help="Number of regions")
    parser.add_argument(
        "--clustering",
        type=str,
        default="learnable",
        choices=["learnable", "kmeans", "grid"],
        help="Clustering method",
    )

    # Visualization options
    parser.add_argument("--output", type=str, help="Output path (.png or .html)")
    parser.add_argument("--interactive", action="store_true", help="Create interactive plot")
    parser.add_argument("--title", type=str, default="Hierarchical Regions", help="Plot title")

    args = parser.parse_args()

    # Load data
    print(f"Loading coordinates: {args.coords}")
    coords = np.load(args.coords)
    coords_tensor = torch.from_numpy(coords).float().unsqueeze(0)  # [1, N, 2]

    # Load features if provided
    if args.features:
        print(f"Loading features: {args.features}")
        features = np.load(args.features)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)  # [1, N, D]
        feature_dim = features.shape[1]
    else:
        # Dummy features
        feature_dim = 1024
        features_tensor = torch.randn(1, coords.shape[0], feature_dim)

    # Load attention if provided
    attention = None
    if args.attention:
        print(f"Loading attention: {args.attention}")
        attention = np.load(args.attention)

    # Build hierarchical pooling
    print(f"Building hierarchical pooling (num_regions={args.num_regions})")
    hierarchical = HierarchicalPooling(
        feature_dim=feature_dim,
        num_regions=args.num_regions,
        hidden_dim=512,
        clustering_method=args.clustering,
        pooling_method="attention",
    )
    hierarchical.eval()

    # Forward pass to get assignments
    with torch.no_grad():
        _ = hierarchical(features_tensor, coords_tensor)

        # Get assignments
        if args.clustering == "learnable":
            # Compute distances to learned centers
            distances = torch.cdist(coords_tensor, hierarchical.cluster_centers.unsqueeze(0))
            assignments = distances.argmin(dim=-1).squeeze(0).numpy()
            centroids = hierarchical.cluster_centers.numpy()
        elif args.clustering == "kmeans":
            # Use k-means assignments
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=args.num_regions, random_state=42)
            assignments = kmeans.fit_predict(coords)
            centroids = kmeans.cluster_centers_
        elif args.clustering == "grid":
            # Grid-based assignments
            x_bins = int(np.sqrt(args.num_regions))
            y_bins = args.num_regions // x_bins

            x_edges = np.linspace(coords[:, 0].min(), coords[:, 0].max(), x_bins + 1)
            y_edges = np.linspace(coords[:, 1].min(), coords[:, 1].max(), y_bins + 1)

            x_indices = np.digitize(coords[:, 0], x_edges) - 1
            y_indices = np.digitize(coords[:, 1], y_edges) - 1

            x_indices = np.clip(x_indices, 0, x_bins - 1)
            y_indices = np.clip(y_indices, 0, y_bins - 1)

            assignments = y_indices * x_bins + x_indices

            # Compute grid centroids
            centroids = []
            for i in range(args.num_regions):
                mask = assignments == i
                if mask.sum() > 0:
                    centroid = coords[mask].mean(axis=0)
                    centroids.append(centroid)
            centroids = np.array(centroids) if centroids else None

    # Visualize
    print("Creating visualization...")

    if args.interactive:
        visualize_regions_interactive(
            coords=coords,
            assignments=assignments,
            centroids=centroids,
            attention=attention,
            output_path=args.output,
            title=args.title,
        )
    else:
        visualize_regions_matplotlib(
            coords=coords,
            assignments=assignments,
            centroids=centroids,
            attention=attention,
            output_path=args.output,
            title=args.title,
        )

    print("✓ Done")


if __name__ == "__main__":
    main()
