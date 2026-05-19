"""
TransnnMIL v2.0 Visualization Demo

Demonstrates visualization capabilities:
1. Hierarchical region assignments
2. k-NN graph topology
3. Attention heatmaps
4. Combined multi-view visualization
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.transnnmil_v2 import TransnnMILv2


def generate_synthetic_data(num_patches=500, feature_dim=512):
    """Generate synthetic WSI-like data for visualization."""
    # Create features with some structure
    features = torch.randn(1, num_patches, feature_dim)
    
    # Create coordinates in [0, 1] x [0, 1] with spatial structure
    # Simulate tissue regions
    coords = torch.zeros(1, num_patches, 2)
    
    # Region 1: Top-left (tumor-like)
    n1 = num_patches // 3
    coords[0, :n1, 0] = torch.rand(n1) * 0.4 + 0.05
    coords[0, :n1, 1] = torch.rand(n1) * 0.4 + 0.05
    
    # Region 2: Top-right (stroma-like)
    n2 = num_patches // 3
    coords[0, n1:n1+n2, 0] = torch.rand(n2) * 0.4 + 0.55
    coords[0, n1:n1+n2, 1] = torch.rand(n2) * 0.4 + 0.05
    
    # Region 3: Bottom (normal-like)
    n3 = num_patches - n1 - n2
    coords[0, n1+n2:, 0] = torch.rand(n3) * 0.9 + 0.05
    coords[0, n1+n2:, 1] = torch.rand(n3) * 0.4 + 0.55
    
    return features, coords


def visualize_hierarchical_regions(model, coords, save_path=None):
    """Visualize hierarchical region assignments."""
    # Get region assignments
    hierarchical = model.hierarchical
    assignments = hierarchical(coords)  # [1, N, R]
    
    # Get cluster centers
    centers = hierarchical.get_centers()  # [R, 2]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Hard assignments (argmax)
    region_ids = assignments[0].argmax(dim=1).cpu().numpy()
    scatter1 = ax1.scatter(
        coords[0, :, 0].cpu(),
        coords[0, :, 1].cpu(),
        c=region_ids,
        cmap='tab20',
        s=30,
        alpha=0.6
    )
    ax1.scatter(
        centers[:, 0].cpu(),
        centers[:, 1].cpu(),
        c='red',
        marker='X',
        s=200,
        edgecolors='black',
        linewidths=2,
        label='Cluster Centers'
    )
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Hierarchical Regions (Hard Assignment)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Region ID')
    
    # Plot 2: Soft assignments (entropy)
    # High entropy = patch belongs to multiple regions
    probs = torch.softmax(assignments[0], dim=1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).cpu().numpy()
    scatter2 = ax2.scatter(
        coords[0, :, 0].cpu(),
        coords[0, :, 1].cpu(),
        c=entropy,
        cmap='viridis',
        s=30,
        alpha=0.6
    )
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('Assignment Uncertainty (Entropy)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Entropy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved hierarchical visualization to {save_path}")
    
    return fig


def visualize_graph_topology(model, features, coords, save_path=None):
    """Visualize k-NN graph topology."""
    # Build k-NN graph
    topology = model.topology
    edge_index = topology._build_knn_graph(coords[0])  # [2, E]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Graph structure
    coords_np = coords[0].cpu().numpy()
    edges = edge_index.T.cpu().numpy()
    
    # Draw edges
    for i, j in edges[:500]:  # Limit edges for visibility
        ax1.plot(
            [coords_np[i, 0], coords_np[j, 0]],
            [coords_np[i, 1], coords_np[j, 1]],
            'gray',
            alpha=0.1,
            linewidth=0.5
        )
    
    # Draw nodes
    ax1.scatter(
        coords_np[:, 0],
        coords_np[:, 1],
        c='blue',
        s=20,
        alpha=0.6,
        zorder=2
    )
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title(f'k-NN Graph (k={topology.k_neighbors})')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Node degree distribution
    degrees = torch.zeros(coords.shape[1])
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    
    scatter2 = ax2.scatter(
        coords_np[:, 0],
        coords_np[:, 1],
        c=degrees.cpu().numpy(),
        cmap='hot',
        s=30,
        alpha=0.6
    )
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('Node Degree (Connectivity)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Degree')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved graph visualization to {save_path}")
    
    return fig


def visualize_attention_heatmap(model, features, coords, save_path=None):
    """Visualize TransMIL attention weights."""
    # Get attention weights from TransMIL
    # Note: This requires modifying TransMIL to return attention weights
    # For demo, we'll simulate attention weights
    
    # Simulate attention (higher for patches near center)
    center = coords[0].mean(dim=0)
    distances = torch.norm(coords[0] - center, dim=1)
    attention = torch.exp(-distances * 5)  # Exponential decay
    attention = attention / attention.sum()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Attention heatmap
    scatter1 = ax1.scatter(
        coords[0, :, 0].cpu(),
        coords[0, :, 1].cpu(),
        c=attention.cpu(),
        cmap='hot',
        s=50,
        alpha=0.7
    )
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Attention Heatmap')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Attention Weight')
    
    # Plot 2: Top-k attention patches
    topk = 50
    top_indices = torch.topk(attention, topk).indices
    
    ax2.scatter(
        coords[0, :, 0].cpu(),
        coords[0, :, 1].cpu(),
        c='lightgray',
        s=20,
        alpha=0.3,
        label='All Patches'
    )
    ax2.scatter(
        coords[0, top_indices, 0].cpu(),
        coords[0, top_indices, 1].cpu(),
        c=attention[top_indices].cpu(),
        cmap='hot',
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidths=1,
        label=f'Top-{topk} Patches'
    )
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title(f'Top-{topk} Attended Patches')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    
    return fig


def visualize_combined(model, features, coords, save_path=None):
    """Create combined multi-view visualization."""
    # Get all components
    hierarchical = model.hierarchical
    topology = model.topology
    
    assignments = hierarchical(coords)
    region_ids = assignments[0].argmax(dim=1).cpu().numpy()
    edge_index = topology._build_knn_graph(coords[0])
    
    # Simulate attention
    center = coords[0].mean(dim=0)
    distances = torch.norm(coords[0] - center, dim=1)
    attention = torch.exp(-distances * 5)
    attention = attention / attention.sum()
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    coords_np = coords[0].cpu().numpy()
    
    # Plot 1: Hierarchical regions
    scatter1 = axes[0, 0].scatter(
        coords_np[:, 0],
        coords_np[:, 1],
        c=region_ids,
        cmap='tab20',
        s=30,
        alpha=0.6
    )
    axes[0, 0].set_title('Hierarchical Regions')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 0], label='Region ID')
    
    # Plot 2: Graph topology
    edges = edge_index.T.cpu().numpy()
    for i, j in edges[:300]:
        axes[0, 1].plot(
            [coords_np[i, 0], coords_np[j, 0]],
            [coords_np[i, 1], coords_np[j, 1]],
            'gray',
            alpha=0.1,
            linewidth=0.5
        )
    axes[0, 1].scatter(
        coords_np[:, 0],
        coords_np[:, 1],
        c='blue',
        s=20,
        alpha=0.6
    )
    axes[0, 1].set_title(f'k-NN Graph (k={topology.k_neighbors})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Attention heatmap
    scatter3 = axes[1, 0].scatter(
        coords_np[:, 0],
        coords_np[:, 1],
        c=attention.cpu(),
        cmap='hot',
        s=50,
        alpha=0.7
    )
    axes[1, 0].set_title('Attention Heatmap')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[1, 0], label='Attention')
    
    # Plot 4: Combined view (regions + attention)
    scatter4 = axes[1, 1].scatter(
        coords_np[:, 0],
        coords_np[:, 1],
        c=region_ids,
        s=attention.cpu() * 500,  # Size by attention
        cmap='tab20',
        alpha=0.6,
        edgecolors='black',
        linewidths=0.5
    )
    axes[1, 1].set_title('Combined: Regions + Attention')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=axes[1, 1], label='Region ID')
    
    # Set labels
    for ax in axes.flat:
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined visualization to {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='TransnnMIL v2.0 Visualization Demo')
    parser.add_argument('--num_patches', type=int, default=500, help='Number of patches')
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension')
    parser.add_argument('--num_regions', type=int, default=16, help='Number of regions')
    parser.add_argument('--k_neighbors', type=int, default=8, help='k-NN neighbors')
    parser.add_argument('--output_dir', type=str, default='visualizations/', help='Output directory')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("TransnnMIL v2.0 Visualization Demo")
    print("=" * 60)
    
    # Generate synthetic data
    print(f"\n1. Generating synthetic data ({args.num_patches} patches)...")
    features, coords = generate_synthetic_data(args.num_patches, args.feature_dim)
    
    # Initialize model
    print(f"2. Initializing TransnnMIL v2.0...")
    print(f"   - Regions: {args.num_regions}")
    print(f"   - k-NN neighbors: {args.k_neighbors}")
    model = TransnnMILv2(
        feature_dim=args.feature_dim,
        num_classes=2,
        num_regions=args.num_regions,
        k_neighbors=args.k_neighbors,
        gnn_type='gat'
    )
    model.eval()
    
    # Visualize hierarchical regions
    print(f"\n3. Visualizing hierarchical regions...")
    fig1 = visualize_hierarchical_regions(
        model, coords,
        save_path=output_dir / 'hierarchical_regions.png'
    )
    
    # Visualize graph topology
    print(f"4. Visualizing graph topology...")
    fig2 = visualize_graph_topology(
        model, features, coords,
        save_path=output_dir / 'graph_topology.png'
    )
    
    # Visualize attention
    print(f"5. Visualizing attention heatmap...")
    fig3 = visualize_attention_heatmap(
        model, features, coords,
        save_path=output_dir / 'attention_heatmap.png'
    )
    
    # Combined visualization
    print(f"6. Creating combined visualization...")
    fig4 = visualize_combined(
        model, features, coords,
        save_path=output_dir / 'combined_view.png'
    )
    
    print(f"\n✓ All visualizations saved to {output_dir}/")
    print("\nGenerated files:")
    print(f"  - hierarchical_regions.png")
    print(f"  - graph_topology.png")
    print(f"  - attention_heatmap.png")
    print(f"  - combined_view.png")
    
    if args.show:
        print("\nShowing plots...")
        plt.show()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
