#!/bin/bash
# HistoCore Linux/Mac Installer
# Run: bash install.sh

set -e

echo "============================================================"
echo "   HistoCore Linux/Mac Installer"
echo "   Production-grade computational pathology framework"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found!"
    echo ""
    echo "Install Python 3.9+:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  macOS: brew install python@3.11"
    echo "  Or download from: https://python.org/downloads/"
    exit 1
fi

echo "[OK] Python found"
python3 --version

# Check pip
if ! python3 -m pip --version &> /dev/null; then
    echo "[ERROR] pip not found!"
    echo "Install pip: sudo apt install python3-pip"
    exit 1
fi

echo "[OK] pip found"
echo ""

# Ask for confirmation
read -p "Install HistoCore? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled"
    exit 0
fi

echo ""
echo "============================================================"
echo "   Installing Dependencies"
echo "============================================================"
echo ""

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (recommended) (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        source venv/bin/activate
    fi
    
    echo "[OK] Virtual environment created and activated"
    echo "     To activate later: source venv/bin/activate"
else
    echo "[SKIP] Virtual environment"
fi

# Upgrade pip
echo ""
echo "[2/5] Upgrading pip..."
python3 -m pip install --upgrade pip

# Install dependencies
echo ""
echo "[3/5] Installing core dependencies..."
python3 -m pip install -r requirements.txt

# Install package
echo ""
echo "[4/5] Installing HistoCore package..."
python3 -m pip install -e .

# Verify installation
echo ""
echo "[5/5] Verifying installation..."
python3 -c "import torch; import numpy; import cv2; print('[OK] All core packages installed')" || {
    echo ""
    echo "[WARNING] Some packages may not be installed correctly"
    echo "Try running: pip3 install -r requirements.txt"
    exit 1
}

echo ""
echo "============================================================"
echo "   Installation Complete!"
echo "============================================================"
echo ""
echo "Next Steps:"
echo ""
echo "1. Launch GUI:"
echo "   python3 histocore.py"
echo ""
echo "2. Use CLI:"
echo "   histocore analyze slide.svs --output results/"
echo "   histocore demo --quick"
echo ""
echo "3. Launch Web Interface:"
echo "   histocore web"
echo ""
echo "4. Read Documentation:"
echo "   https://github.com/matthewvaishnav/histocore"
echo ""

# Make histocore.py executable
chmod +x histocore.py 2>/dev/null || true

echo "Tip: If you created a virtual environment, activate it with:"
echo "     source venv/bin/activate"
echo ""
