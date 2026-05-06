#!/usr/bin/env python3
"""
Simple package builder for HistoCore (no pip required)
"""

import os
import shutil
import tarfile
import zipfile
from pathlib import Path

def create_source_distribution():
    """Create source distribution manually."""
    
    print("📦 Creating HistoCore source distribution...")
    
    # Clean previous builds
    for path in ["dist", "build", "histocore.egg-info"]:
        if os.path.exists(path):
            shutil.rmtree(path)
    
    # Create dist directory
    os.makedirs("dist", exist_ok=True)
    
    # Package info
    name = "histocore"
    version = "0.1.0"
    
    # Create source archive
    archive_name = f"{name}-{version}.tar.gz"
    
    with tarfile.open(f"dist/{archive_name}", "w:gz") as tar:
        # Add essential files
        files_to_include = [
            "README.md",
            "LICENSE", 
            "setup_pypi.py",
            "pyproject.toml",
            "requirements-core.txt",
            "src/",
            "configs/",
            "examples/",
        ]
        
        for file_path in files_to_include:
            if os.path.exists(file_path):
                tar.add(file_path, arcname=f"{name}-{version}/{file_path}")
                print(f"  ✅ Added {file_path}")
    
    print(f"✅ Created {archive_name}")
    print(f"📁 Size: {os.path.getsize(f'dist/{archive_name}') // 1024} KB")
    
    return f"dist/{archive_name}"

def show_manual_upload_instructions(archive_path):
    """Show manual upload instructions."""
    
    print("\n" + "="*50)
    print("📋 MANUAL PYPI UPLOAD INSTRUCTIONS")
    print("="*50)
    
    print("\n1. 🌐 Go to https://pypi.org/account/register/")
    print("   Create account with your email")
    
    print("\n2. 📤 Go to https://pypi.org/project/manage/")
    print("   Click 'Upload files'")
    
    print(f"\n3. 📦 Upload this file: {archive_path}")
    print("   Drag and drop or browse to select")
    
    print("\n4. ✅ Once uploaded:")
    print("   pip install histocore")
    print("   import histocore")
    
    print("\n🎯 RESULT:")
    print("   - Global pip install works")
    print("   - Download metrics start tracking") 
    print("   - Professional PyPI presence")
    print("   - Resume line: 'Published open source package'")

def main():
    """Build package for manual upload."""
    
    if not Path("README.md").exists():
        print("❌ Run from HistoCore root directory")
        return
    
    archive_path = create_source_distribution()
    show_manual_upload_instructions(archive_path)

if __name__ == "__main__":
    main()