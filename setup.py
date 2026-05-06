#!/usr/bin/env python3
"""
HistoCore Setup Script

Simple installation and setup for HistoCore.
"""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install core dependencies."""
    print("📦 Installing HistoCore dependencies...")
    
    # Install in development mode
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])
    
    print("✅ Installation complete!")
    print()
    print("🚀 Quick start:")
    print("  histocore train --dataset pcam --model nnmil")
    print("  jupyter notebook examples/quickstart.ipynb")
    print()
    print("📚 Documentation: https://github.com/matthewvaishnav/histocore")

def main():
    if not Path("pyproject.toml").exists():
        print("❌ Error: Run this script from the HistoCore root directory")
        sys.exit(1)
    
    install_dependencies()

if __name__ == "__main__":
    main()