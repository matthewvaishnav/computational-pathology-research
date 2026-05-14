"""
Visualize hierarchical pooling region assignments.

Creates scatter plots showing:
- Patch coordinates colored by assigned region
- Cluster center positions
- Soft vs hard assignments

Usage:
    python scripts/visualize_hierarchical.py --num_clusters 16 --num_patches 500
    python scripts/visualize_hierarchical.py --method kmeans --output viz/regions.png
"""

import argparse
from pathlib import Path
import sys
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.hierarchical_pooling import (
    LearnableClusterCenters,
    KMeansClusterer,
    GridClusterer,
)


def visualize_assignments(
    coords: torch.Tensor,
    assignments: torch.Tensor,
    centers: torch.Tensor,
    method: str,
    output_path: Optional[Path] = None,
    show_soft: bool = False,
):
    """
    Visualize region assignments.
    
    Args:
        coords: Patch coordinates [num_patches, 2]
        assignments: Soft assignments [num_patches, num_clusters]
        centers: Cluster centers [num_clusters, 2]
        method: Clustering method name
        output_path: Optional save path
        show_soft: If True, show soft assignment strength
    """
    # Convert to numpy
    coords_np = coords.detach().cpu().numpy()
    assignments_np = assignments.detach().cpu().numpy()
    centers_np = centers.detach().cpu().numpy()
    
    # Hard assignments (argmax)
    hard_assign = assignments_np.argmax(axis=1)
    
    # Create figure
    fig, axes = plt.subplots(1, 2 if show_soft else 1, figsize=(12 if show_soft else 6, 5))
    if not show_soft:
        axes = [axes]
    
    # Plot 1: Hard assignments
    ax = axes[0]
    scatter = ax.scatter(
        coords_np[:, 0],
        coords_np[:, 1],
        c=hard_assign,
        cmap='tab20',
        s=20,
        alpha=0.6,
    )
    
    # Plot cluster centers
    ax.scatter(
        centers_np[:, 0],
        centers_np[:, 1],
        c='red',
        marker='X',
        s=200,
        edgecolors='black',
        linewidths=2,
        label='Centers',
    )
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'{method}: Hard Assignments')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 2: Soft assignment strength (optional)
    if show_soft:
        ax = axes[1]
        
        # Max assignment probability (confidence)
        max_prob = assignments_np.max(axis=1)
        
        scatter = ax.scatter(
            coords_np[:, 0],
            coords_np[:, 1],
            c=max_prob,
            cmap='viridis',
            s=20,
            alpha=0.6,
        )
        
        # Plot centers
        ax.scatter(
            centers_np[:, 0],
            centers_np[:, 1],
            c='red',
            marker='X',
            s=200,
            edgecolors='black',
            linewidths=2,
        )
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'{method}: Assignment Confidence')
        plt.colorbar(scatter, ax=ax, label='Max Probability')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def compare_methods(
    coords: torch.Tensor,
    num_clusters: int,
    output_dir: Optional[Path] = None,
):
    """
    Compare all clustering methods side-by-side.
    
    Args:
        coords: Patch coordinates [num_patches, 2]
        num_clusters: Number of clusters
        output_dir: Optional output directory
    """
    methods = {
        'Learnable': LearnableClusterCenters(num_clusters),
        'K-Means': KMeansClusterer(num_clusters),
        'Grid': GridClusterer(num_clusters),
    }
    
    # Fit k-means
    methods['K-Means'].fit(coords)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (name, clusterer) in zip(axes, methods.items()):
        # Get assignments
        assignments = clusterer(coords.unsqueeze(0)).squeeze(0)
        centers = clusterer.get_centers()
        
        # Convert to numpy
        coords_np = coords.detach().cpu().numpy()
        assignments_np = assignments.detach().cpu().numpy()
        centers_np = centers.detach().cpu().numpy()
        
        # Hard assignments
        hard_assign = assignments_np.argmax(axis=1)
        
        # Plot
        ax.scatter(
            coords_np[:, 0],
            coords_np[:, 1],
            c=hard_assign,
            cmap='tab20',
            s=20,
            alpha=0.6,
        )
        
        # Centers
        ax.scatter(
            centers_np[:, 0],
            centers_np[:, 1],
            c='red',
            marker='X',
            s=200,
            edgecolors='black',
            linewidths=2,
        )
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize hierarchical pooling')
    parser.add_argument('--num_clusters', type=int, default=16,
                       help='Number of clusters (default: 16)')
    parser.add_argument('--num_patches', type=int, default=500,
                       help='Number of patches to visualize (default: 500)')
    parser.add_argument('--method', type=str, default='learnable',
                       choices=['learnable', 'kmeans', 'grid', 'compare'],
                       help='Clustering method (default: learnable)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Softmax temperature (default: 1.0)')
    parser.add_argument('--show_soft', action='store_true',
                       help='Show soft assignment confidence')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: show plot)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Generate random coordinates [0, 1]
    coords = torch.rand(args.num_patches, 2)
    
    # Output path
    output_path = Path(args.output) if args.output else None
    
    # Compare all methods
    if args.method == 'compare':
        compare_methods(coords, args.num_clusters, output_path.parent if output_path else None)
        return
    
    # Single method
    if args.method == 'learnable':
        clusterer = LearnableClusterCenters(
            num_clusters=args.num_clusters,
            temperature=args.temperature,
        )
        method_name = 'Learnable'
    
    elif args.method == 'kmeans':
        clusterer = KMeansClusterer(
            num_clusters=args.num_clusters,
            temperature=args.temperature,
        )
        clusterer.fit(coords)
        method_name = 'K-Means'
    
    elif args.method == 'grid':
        clusterer = GridClusterer(
            num_clusters=args.num_clusters,
            temperature=args.temperature,
        )
        method_name = 'Grid'
    
    # Get assignments
    assignments = clusterer(coords.unsqueeze(0)).squeeze(0)
    centers = clusterer.get_centers()
    
    # Visualize
    visualize_assignments(
        coords,
        assignments,
        centers,
        method_name,
        output_path,
        args.show_soft,
    )
    
    # Print stats
    hard_assign = assignments.argmax(dim=1)
    unique, counts = torch.unique(hard_assign, return_counts=True)
    
    print(f"\n{method_name} Statistics:")
    print(f"  Num clusters: {args.num_clusters}")
    print(f"  Num patches: {args.num_patches}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Patches per region (mean ± std): {counts.float().mean():.1f} ± {counts.float().std():.1f}")
    print(f"  Min/max patches: {counts.min()}/{counts.max()}")


if __name__ == '__main__':
    main()
