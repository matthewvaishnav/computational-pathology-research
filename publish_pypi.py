#!/usr/bin/env python3
"""
Publish HistoCore to PyPI
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run command and handle errors."""
    print(f"🔄 {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        print(f"Error: {result.stderr}")
        sys.exit(1)
    print(f"✅ {description} complete")
    return result.stdout

def check_prerequisites():
    """Check if required tools are installed."""
    print("🔍 Checking prerequisites...")
    
    # Check if twine is installed
    try:
        subprocess.run(["twine", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing twine...")
        subprocess.run([sys.executable, "-m", "pip", "install", "twine", "build"], check=True)
    
    # Check if we're in the right directory
    if not Path("README.md").exists():
        print("❌ Error: Run this script from the HistoCore root directory")
        sys.exit(1)
    
    print("✅ Prerequisites check complete")

def build_package():
    """Build the package."""
    # Clean previous builds
    run_command("rm -rf dist/ build/ *.egg-info/", "Cleaning previous builds")
    
    # Build package
    run_command(f"{sys.executable} -m build", "Building package")

def upload_to_pypi():
    """Upload to PyPI."""
    print("\n📦 Ready to upload to PyPI")
    print("Choose upload target:")
    print("1. TestPyPI (recommended for first time)")
    print("2. PyPI (production)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Upload to TestPyPI
        print("🧪 Uploading to TestPyPI...")
        run_command("twine upload --repository testpypi dist/*", "Uploading to TestPyPI")
        print("\n✅ Upload complete!")
        print("🔗 View at: https://test.pypi.org/project/histocore/")
        print("📦 Install with: pip install -i https://test.pypi.org/simple/ histocore")
        
    elif choice == "2":
        # Upload to PyPI
        confirm = input("⚠️  Upload to production PyPI? This cannot be undone. (yes/no): ")
        if confirm.lower() == "yes":
            run_command("twine upload dist/*", "Uploading to PyPI")
            print("\n✅ Upload complete!")
            print("🔗 View at: https://pypi.org/project/histocore/")
            print("📦 Install with: pip install histocore")
        else:
            print("❌ Upload cancelled")
    else:
        print("❌ Invalid choice")
        sys.exit(1)

def main():
    """Main publishing workflow."""
    print("🚀 HistoCore PyPI Publishing")
    print("=" * 40)
    
    check_prerequisites()
    build_package()
    
    # Show what will be uploaded
    print("\n📋 Package contents:")
    run_command("ls -la dist/", "Listing package files")
    
    upload_to_pypi()

if __name__ == "__main__":
    main()