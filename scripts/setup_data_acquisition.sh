#!/bin/bash
# Setup script for data acquisition
# This script installs dependencies and prepares the environment

set -e  # Exit on error

echo "=========================================="
echo "Medical AI Data Acquisition Setup"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install required packages
echo ""
echo "Installing required packages..."
pip install --upgrade pip
pip install openai biopython requests tqdm pillow

# Optional: Install Kaggle CLI
read -p "Install Kaggle CLI for PANDA dataset? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    pip install kaggle
    echo "Kaggle CLI installed. Set up credentials:"
    echo "  1. Go to https://www.kaggle.com/account"
    echo "  2. Create new API token"
    echo "  3. Save to ~/.kaggle/kaggle.json"
fi

# Create data directory structure
echo ""
echo "Creating data directory structure..."
mkdir -p data/multi_disease/{breast,lung,colon,melanoma,prostate}
mkdir -p data/vision_language
mkdir -p data/metadata

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Set OpenAI API key: export OPENAI_API_KEY=your_key_here"
echo "  2. Download datasets: python scripts/download_public_datasets.py"
echo "  3. Verify downloads: python scripts/verify_datasets.py"
echo "  4. Generate captions: python scripts/generate_captions_gpt4v.py"
echo ""
echo "See QUICKSTART_DATA_ACQUISITION.md for detailed instructions"
