#!/bin/bash
# Download real PatchCamelyon dataset from Zenodo
# Total size: ~7GB compressed, ~7GB extracted

set -e

# Create data directory
mkdir -p data/pcam_real
cd data/pcam_real

echo "Downloading PatchCamelyon dataset from Zenodo..."
echo "This will download ~7GB of data. Please be patient."
echo ""

# Download all splits
echo "[1/6] Downloading training images..."
wget -c https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz

echo "[2/6] Downloading training labels..."
wget -c https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz

echo "[3/6] Downloading validation images..."
wget -c https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz

echo "[4/6] Downloading validation labels..."
wget -c https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz

echo "[5/6] Downloading test images..."
wget -c https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz

echo "[6/6] Downloading test labels..."
wget -c https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz

echo ""
echo "Download complete! Extracting files..."
echo ""

# Extract all files
gunzip -v *.gz

echo ""
echo "Extraction complete!"
echo ""
echo "Dataset statistics:"
echo "  Train: 262,144 images"
echo "  Valid: 32,768 images"
echo "  Test:  32,768 images"
echo "  Total: 327,680 images"
echo ""
echo "Files are located in: data/pcam_real/"
echo ""
echo "Next step: Run training with:"
echo "  python experiments/train_pcam.py --config experiments/configs/pcam_rtx4070_laptop.yaml"
