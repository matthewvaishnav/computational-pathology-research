"""
Test dataset visualization functionality.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path("src/data").resolve()))
from pcam_dataset import PCamDataset

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300

# Output directory
OUTPUT_DIR = Path("results/pcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("TESTING DATASET VISUALIZATION")
print("=" * 60)

# Load dataset
print("\n[Test 1] Loading PCam dataset...")
try:
    dataset = PCamDataset(root_dir="data/pcam", split="test", download=False)

    # Sample indices
    total_samples = len(dataset)
    SAMPLE_SIZE = min(64, total_samples)
    sample_indices = np.random.choice(total_samples, size=SAMPLE_SIZE, replace=False)

    # Collect samples
    samples = []
    labels = []
    for idx in sample_indices:
        sample = dataset[idx]
        samples.append(sample["image"])
        labels.append(sample["label"].item())

    samples = torch.stack(samples)
    labels = np.array(labels)

    print(f"✓ Loaded {len(samples)} samples from PCam dataset")
    print(f"  Class distribution: Normal={np.sum(labels == 0)}, Metastatic={np.sum(labels == 1)}")

    DATASET_LOADED = True
except Exception as e:
    print(f"✗ Could not load dataset: {e}")
    DATASET_LOADED = False

# Test 2: Display 8x8 grid of sample images
if DATASET_LOADED:
    print("\n[Test 2] Generating sample image grid...")
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    fig.suptitle("PCam Sample Images (8x8 Grid)", fontsize=16, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            img = samples[i]
            # Denormalize for display if needed
            if img.max() <= 1.0:
                img = img.numpy().transpose(1, 2, 0)
            else:
                img = img.numpy().transpose(1, 2, 0) / 255.0

            ax.imshow(img)
            ax.set_title(f"L={labels[i]}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_grid.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f'✓ Saved sample grid to {OUTPUT_DIR / "sample_grid.png"}')

# Test 3: Plot class distribution
if DATASET_LOADED:
    print("\n[Test 3] Generating class distribution plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    class_names = ["Normal (0)", "Metastatic (1)"]
    class_counts = [np.sum(labels == 0), np.sum(labels == 1)]

    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(class_names, class_counts, color=colors, edgecolor="black", linewidth=1.2)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("PCam Class Distribution (Sample)", fontsize=14, fontweight="bold")

    # Add count labels on bars
    for bar, count in zip(bars, class_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f'✓ Saved class distribution to {OUTPUT_DIR / "class_distribution.png"}')

# Test 4: Compute and display image statistics
if DATASET_LOADED:
    print("\n[Test 4] Computing image statistics...")
    # Compute mean and std per channel
    all_images = samples.numpy()  # [N, 3, 96, 96]
    mean_per_channel = all_images.mean(axis=(0, 2, 3))
    std_per_channel = all_images.std(axis=(0, 2, 3))

    print(f"  Mean per channel (R, G, B): {mean_per_channel}")
    print(f"  Std per channel (R, G, B): {std_per_channel}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    channels = ["Red", "Green", "Blue"]
    colors_rgb = ["#e74c3c", "#2ecc71", "#3498db"]

    # Mean plot
    ax1 = axes[0]
    bars1 = ax1.bar(channels, mean_per_channel, color=colors_rgb, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("Mean Value", fontsize=11)
    ax1.set_title("Mean Pixel Values per Channel", fontsize=12, fontweight="bold")
    ax1.set_ylim([0, 1])
    for bar, val in zip(bars1, mean_per_channel):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Std plot
    ax2 = axes[1]
    bars2 = ax2.bar(channels, std_per_channel, color=colors_rgb, edgecolor="black", linewidth=1.2)
    ax2.set_ylabel("Standard Deviation", fontsize=11)
    ax2.set_title("Std Deviation per Channel", fontsize=12, fontweight="bold")
    ax2.set_ylim([0, 0.5])
    for bar, val in zip(bars2, std_per_channel):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "image_statistics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f'✓ Saved image statistics to {OUTPUT_DIR / "image_statistics.png"}')

# Summary
print("\n" + "=" * 60)
print("DATASET VISUALIZATION TEST SUMMARY")
print("=" * 60)
saved_plots = sorted(OUTPUT_DIR.glob("*.png"))
print(f"\nTotal plots in output directory: {len(saved_plots)}")
print(f"Output directory: {OUTPUT_DIR}")

if DATASET_LOADED:
    print("\n✓ All dataset visualization tests passed!")
else:
    print("\n⚠ Dataset visualization tests skipped (dataset not available)")
