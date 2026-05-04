#!/usr/bin/env python3
"""
HistoCore Quick Installer
One-command setup for all platforms
"""

import os
import sys
import subprocess
import platform

def check_python():
    """Check Python version"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        print(f"Current: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install core dependencies"""
    print("📦 Installing dependencies...")
    
    deps = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "flask>=3.0.0",
        "click>=8.0.0",
        "PyQt6>=6.4.0",
        "scikit-learn>=1.2.0",
        "opencv-python>=4.7.0",
        "h5py>=3.8.0",
        "tqdm>=4.65.0"
    ]
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade"
        ] + deps, check=True)
        print("✅ Core dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def install_optional():
    """Install optional dependencies"""
    print("📦 Installing optional dependencies...")
    
    optional = [
        "openslide-python>=1.2.0",
        "pydicom>=2.3.0",
        "jupyter>=1.0.0"
    ]
    
    for dep in optional:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], check=True, capture_output=True)
            print(f"✅ {dep.split('>=')[0]}")
        except subprocess.CalledProcessError:
            print(f"⚠️  {dep.split('>=')[0]} (optional)")

def create_shortcuts():
    """Create desktop shortcuts"""
    system = platform.system()
    
    if system == "Windows":
        # Windows shortcut
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            path = os.path.join(desktop, "HistoCore.lnk")
            target = sys.executable
            wDir = os.getcwd()
            icon = target
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(path)
            shortcut.Targetpath = target
            shortcut.Arguments = "histocore.py"
            shortcut.WorkingDirectory = wDir
            shortcut.IconLocation = icon
            shortcut.save()
            
            print("✅ Windows shortcut created")
        except ImportError:
            print("⚠️  Windows shortcut (install pywin32)")
    
    elif system == "Darwin":
        # macOS alias
        print("💡 macOS: Add to Applications manually")
    
    elif system == "Linux":
        # Linux desktop file
        desktop_dir = os.path.expanduser("~/.local/share/applications")
        os.makedirs(desktop_dir, exist_ok=True)
        
        desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=HistoCore
Comment=Computational pathology framework
Exec={sys.executable} {os.path.join(os.getcwd(), 'histocore.py')}
Icon=applications-science
Terminal=false
Categories=Science;Education;
"""
        
        with open(os.path.join(desktop_dir, "histocore.desktop"), "w") as f:
            f.write(desktop_content)
        
        print("✅ Linux desktop file created")

def test_installation():
    """Test installation"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import torch
        import numpy
        import matplotlib
        import flask
        import click
        print("✅ Core imports work")
        
        # Test GUI (optional)
        try:
            import PyQt6
            print("✅ GUI available")
        except ImportError:
            print("⚠️  GUI not available")
        
        # Test medical imaging (optional)
        try:
            import openslide
            print("✅ WSI support available")
        except ImportError:
            print("⚠️  WSI support limited")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Main installer"""
    print("🔬 HistoCore Quick Installer")
    print("=" * 40)
    
    # Check Python
    if not check_python():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Install optional
    install_optional()
    
    # Create shortcuts
    create_shortcuts()
    
    # Test installation
    if not test_installation():
        return 1
    
    print("\n✅ Installation complete!")
    print("\n🚀 Quick start:")
    print("  python histocore.py")
    print("  python -m src.cli.main --help")
    print("  python -m src.web.app")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())