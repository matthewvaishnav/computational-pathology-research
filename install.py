#!/usr/bin/env python3
"""
HistoCore One-Click Installer
Just run: python install.py
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print installation header"""
    print("\n" + "="*60)
    print("🔬 HistoCore Installer")
    print("Production-grade computational pathology framework")
    print("="*60 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 9):
        print("❌ Python 3.9+ required")
        print("💡 Download from: https://python.org/downloads/")
        return False
    
    print("✅ Python version OK")
    return True

def check_system():
    """Check system information"""
    print(f"💻 System: {platform.system()} {platform.release()}")
    print(f"🏗️  Architecture: {platform.machine()}")
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🎮 CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - CPU mode only")
    except ImportError:
        print("⚠️  PyTorch not installed yet")

def install_dependencies():
    """Install all dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    print("🔧 Upgrading pip...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
    except subprocess.CalledProcessError:
        print("⚠️  Could not upgrade pip")
    
    # Install from requirements.txt
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        print("📋 Installing from requirements.txt...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✅ Core dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        print("⚠️  requirements.txt not found")
        print("📦 Installing minimal dependencies...")
        
        minimal_deps = [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "numpy>=1.24.0",
            "opencv-python>=4.7.0",
            "scikit-learn>=1.2.0",
            "matplotlib>=3.6.0",
            "click>=8.0.0",
            "tqdm>=4.64.0"
        ]
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + minimal_deps)
            print("✅ Minimal dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install minimal dependencies: {e}")
            return False
    
    # Install package in development mode
    print("🔧 Installing HistoCore package...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", "."
        ])
        print("✅ HistoCore package installed")
    except subprocess.CalledProcessError:
        print("⚠️  Could not install package in development mode")
    
    return True

def verify_installation():
    """Verify installation was successful"""
    print("\n🔍 Verifying installation...")
    
    required_packages = [
        "torch",
        "torchvision", 
        "numpy",
        "cv2",
        "sklearn",
        "matplotlib",
        "click",
        "tqdm"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        return False
    
    print("\n✅ All required packages installed!")
    return True

def show_next_steps():
    """Show next steps after installation"""
    print("\n" + "="*60)
    print("🎉 Installation Complete!")
    print("="*60)
    print("\n📚 Next Steps:")
    print()
    print("1. 🖥️  Launch GUI:")
    print("   python histocore.py")
    print()
    print("2. 💻 Use CLI:")
    print("   histocore analyze slide.svs --output results/")
    print("   histocore demo --quick")
    print()
    print("3. 🌐 Launch Web Interface:")
    print("   histocore web")
    print()
    print("4. 📖 Read Documentation:")
    print("   https://github.com/matthewvaishnav/histocore")
    print()
    print("5. 🎓 Try Tutorial:")
    print("   python examples/quickstart.py")
    print()

def main():
    """Main installer function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check system
    check_system()
    
    # Ask for confirmation
    print("\n" + "="*60)
    response = input("📦 Install HistoCore? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Installation cancelled")
        return 0
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed")
        return 1
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Installation incomplete")
        print("💡 Try: pip install -r requirements.txt")
        return 1
    
    # Show next steps
    show_next_steps()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
