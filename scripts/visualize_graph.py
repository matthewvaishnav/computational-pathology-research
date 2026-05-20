"""
Visualize k-NN graphs for spatial topology.

Creates visualizations of k-NN graphs built from patch coordinates:
- Scatter plot of patches with edges
- Degree distribution
- Edge length distribution
- Interactive plots (optional)

Usage:
    # Visualize from coordinates
    python scripts/visualize_graph.py --coords data/coords.npy --k 8 --output viz/graph.png
    
    # Visualize from cached graph
    python scripts/visualize_graph.py --cache data/graph_cache --slide slide_001 --output viz/graph.png
    
    # Interactive plot
    python scripts/visualize_graph.py --coords data/coords.npy --k 8 --interactive

Reference:
- TransnnMIL v2.0: Hierarchical + Topology (2027)
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available. Install with: pip install networkx")

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.models.transnnmil.graph_cache import GraphCache
from src.models.transnnmil.topology_branch import KNNGraphBuilder


def visualize_graph_matplotlib(
    coords: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title: str = "k-NN Graph",
):
    """
    Visualize k-NN graph using matplotlib.

    Args:
        coords: Patch coordinates [N, 2]
        edge_index: Edge indices [2, E]
        edge_attr: Optional edge features [E, 2] (distance, similarity)
        output_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Graph visualization
    ax = axes[0, 0]
    ax.scatter(coords[:, 0], coords[:, 1], c="blue", s=20, alpha=0.6, label="Patches")

    # Draw edges
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[:, i]
        x = [coords[src, 0], coords[tgt, 0]]
        y = [coords[src, 1], coords[tgt, 1]]
        ax.plot(x, y, "gray", alpha=0.2, linewidth=0.5)

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"{title} (N={len(coords)}, E={edge_index.shape[1]})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Degree distribution
    ax = axes[0, 1]
    degrees = np.bincount(edge_index[0])
    ax.hist(degrees, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Count")
    ax.set_title(f"Degree Distribution (mean={degrees.mean():.1f})")
    ax.grid(True, alpha=0.3)

    # 3. Edge length distribution
    ax = axes[1, 0]
    if edge_attr is not None:
        edge_lengths = edge_attr[:, 0]  # Distance feature
        ax.hist(edge_lengths, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Edge length")
        ax.set_ylabel("Count")
        ax.set_title(f"Edge Length Distribution (mean={edge_lengths.mean():.3f})")
        ax.grid(True, alpha=0.3)
    else:
        # Compute edge lengths
        edge_lengths = []
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[:, i]
            dist = np.linalg.norm(coords[src] - coords[tgt])
            edge_lengths.append(dist)
        edge_lengths = np.array(edge_lengths)
        ax.hist(edge_lengths, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Edge length")
        ax.set_ylabel("Count")
        ax.set_title(f"Edge Length Distribution (mean={edge_lengths.mean():.3f})")
        ax.grid(True, alpha=0.3)

    # 4. Edge similarity distribution (if available)
    ax = axes[1, 1]
    if edge_attr is not None and edge_attr.shape[1] > 1:
        edge_similarity = edge_attr[:, 1]  # Similarity feature
        ax.hist(edge_similarity, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"Edge Similarity Distribution (mean={edge_similarity.mean():.3f})")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "Edge similarity not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Edge Similarity Distribution")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_graph_interactive(
    coords: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title: str = "k-NN Graph",
):
    """
    Create interactive k-NN graph visualization using plotly.

    Args:
        coords: Patch coordinates [N, 2]
        edge_index: Edge indices [2, E]
        edge_attr: Optional edge features [E, 2]
        output_path: Path to save HTML
        title: Plot title
    """
    if not PLOTLY_AVAILABLE:
        print("Error: plotly not available. Install with: pip install plotly")
        return

    # Create edge traces
    edge_traces = []
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[:, i]
        x = [coords[src, 0], coords[tgt, 0], None]
        y = [coords[src, 1], coords[tgt, 1], None]

        edge_trace = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(width=0.5, color="gray"),
            hoverinfo="none",
            showlegend=False,
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_trace = go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers",
        marker=dict(size=8, color="blue", opacity=0.6),
        text=[f"Node {i}" for i in range(len(coords))],
        hoverinfo="text",
        name="Patches",
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=f"{title} (N={len(coords)}, E={edge_index.shape[1]})",
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


def visualize_networkx(
    coords: np.ndarray,
    edge_index: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "k-NN Graph",
):
    """
    Visualize using networkx layout algorithms.

    Args:
        coords: Patch coordinates [N, 2]
        edge_index: Edge indices [2, E]
        output_path: Path to save figure
        title: Plot title
    """
    if not NETWORKX_AVAILABLE:
        print("Error: networkx not available. Install with: pip install networkx")
        return

    # Create networkx graph
    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))

    # Add edges
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[:, i]
        G.add_edge(src, tgt)

    # Use actual coordinates as positions
    pos = {i: coords[i] for i in range(len(coords))}

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color="blue", alpha=0.6, ax=ax)

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"{title} (N={len(coords)}, E={edge_index.shape[1]})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize k-NN graphs")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--coords", type=str, help="Path to coordinates (.npy)")
    input_group.add_argument("--cache", type=str, help="Path to graph cache directory")

    # Graph construction options
    parser.add_argument("--k", type=int, default=8, help="Number of neighbors (default: 8)")
    parser.add_argument(
        "--use-faiss", action="store_true", help="Use FAISS approximate k-NN"
    )
    parser.add_argument("--features", type=str, help="Path to features (.npy) for edge similarity")

    # Cache options
    parser.add_argument("--slide", type=str, help="Slide ID (required if using --cache)")

    # Visualization options
    parser.add_argument("--output", type=str, help="Output path (.png or .html)")
    parser.add_argument("--interactive", action="store_true", help="Create interactive plot")
    parser.add_argument("--networkx", action="store_true", help="Use networkx visualization")
    parser.add_argument("--title", type=str, default="k-NN Graph", help="Plot title")

    args = parser.parse_args()

    # Load data
    if args.coords:
        # Build graph from coordinates
        print(f"Loading coordinates: {args.coords}")
        coords = np.load(args.coords)
        coords_tensor = torch.from_numpy(coords).float()

        # Load features if provided
        features_tensor = None
        if args.features:
            print(f"Loading features: {args.features}")
            features = np.load(args.features)
            features_tensor = torch.from_numpy(features).float()

        # Build graph
        print(f"Building k-NN graph (k={args.k}, use_faiss={args.use_faiss})")
        builder = KNNGraphBuilder(k=args.k, use_faiss=args.use_faiss)
        edge_index, edge_attr = builder(coords_tensor, features_tensor)

        # Convert to numpy
        edge_index = edge_index.cpu().numpy()
        edge_attr = edge_attr.cpu().numpy() if edge_attr is not None else None

    elif args.cache:
        # Load from cache
        if not args.slide:
            parser.error("--slide required when using --cache")

        print(f"Loading from cache: {args.cache}")
        cache = GraphCache(cache_dir=args.cache, k=args.k)

        # Load graph
        edge_index, edge_attr = cache.load_graph(args.slide)
        edge_index = edge_index.cpu().numpy()
        edge_attr = edge_attr.cpu().numpy() if edge_attr is not None else None

        # Load coordinates from metadata (if available)
        # For now, use edge_index to infer node positions
        # In practice, you'd store coords in cache too
        print("Warning: Coordinates not in cache, using spring layout")
        num_nodes = edge_index.max() + 1
        coords = np.random.rand(num_nodes, 2)  # Placeholder

    # Visualize
    print("Creating visualization...")

    if args.interactive:
        visualize_graph_interactive(
            coords=coords,
            edge_index=edge_index,
            edge_attr=edge_attr,
            output_path=args.output,
            title=args.title,
        )
    elif args.networkx:
        visualize_networkx(
            coords=coords,
            edge_index=edge_index,
            output_path=args.output,
            title=args.title,
        )
    else:
        visualize_graph_matplotlib(
            coords=coords,
            edge_index=edge_index,
            edge_attr=edge_attr,
            output_path=args.output,
            title=args.title,
        )

    print("✓ Done")


if __name__ == "__main__":
    main()
