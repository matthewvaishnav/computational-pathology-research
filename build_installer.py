"""
Build standalone Windows installer executable using PyInstaller.
Creates HistoCore-Installer.exe that bundles Python + dependencies.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        return True
    except ImportError:
        return False


def install_pyinstaller():
    """Install PyInstaller if not present."""
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def create_installer_script():
    """Create the main installer script that will be compiled to .exe"""
    installer_code = '''"""
HistoCore Standalone Installer
Double-click to install HistoCore with all dependencies.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
import tempfile


def print_header():
    """Print installer header."""
    print("=" * 60)
    print("HistoCore Installer")
    print("Computational Pathology Research Platform")
    print("=" * 60)
    print()


def check_python():
    """Check Python version."""
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro} detected")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("ERROR: Python 3.9+ required")
        return False
    
    print("OK: Python version compatible")
    return True


def download_histocore():
    """Download HistoCore from GitHub."""
    print("\\nDownloading HistoCore...")
    
    url = "https://github.com/matthewvaishnav/computational-pathology-research/archive/refs/heads/main.zip"
    temp_dir = tempfile.gettempdir()
    zip_path = os.path.join(temp_dir, "histocore.zip")
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("OK: Download complete")
        return zip_path
    except Exception as e:
        print(f"ERROR: Download failed - {e}")
        return None


def extract_histocore(zip_path):
    """Extract HistoCore archive."""
    print("\\nExtracting files...")
    
    install_dir = os.path.join(os.path.expanduser("~"), "HistoCore")
    
    try:
        # Remove old installation
        if os.path.exists(install_dir):
            shutil.rmtree(install_dir)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(install_dir))
        
        # Rename extracted folder
        extracted = os.path.join(os.path.dirname(install_dir), 
                                "computational-pathology-research-main")
        os.rename(extracted, install_dir)
        
        print(f"OK: Installed to: {install_dir}")
        return install_dir
    except Exception as e:
        print(f"ERROR: Extraction failed - {e}")
        return None


def install_dependencies(install_dir):
    """Install Python dependencies."""
    print("\\nInstalling dependencies...")
    print("This may take several minutes...")
    
    requirements = os.path.join(install_dir, "requirements.txt")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", requirements, "--quiet"
        ])
        print("OK: Dependencies installed")
        return True
    except Exception as e:
        print(f"ERROR: Dependency installation failed - {e}")
        return False


def install_package(install_dir):
    """Install HistoCore package."""
    print("\\nInstalling HistoCore package...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-e", install_dir, "--quiet"
        ])
        print("OK: Package installed")
        return True
    except Exception as e:
        print(f"ERROR: Package installation failed - {e}")
        return False


def create_shortcuts(install_dir):
    """Create desktop shortcuts."""
    print("\\nCreating shortcuts...")
    
    try:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        
        # Create batch file to launch HistoCore
        launcher = os.path.join(install_dir, "launch_histocore.bat")
        with open(launcher, 'w') as f:
            f.write(f'@echo off\\n')
            f.write(f'cd /d "{install_dir}"\\n')
            f.write(f'python histocore.py\\n')
            f.write(f'pause\\n')
        
        # Copy to desktop
        desktop_launcher = os.path.join(desktop, "HistoCore.bat")
        shutil.copy(launcher, desktop_launcher)
        
        print(f"OK: Shortcut created on desktop")
        return True
    except Exception as e:
        print(f"WARNING: Could not create shortcuts - {e}")
        return False


def main():
    """Main installer flow."""
    print_header()
    
    # Check Python
    if not check_python():
        input("\\nPress Enter to exit...")
        sys.exit(1)
    
    # Download
    zip_path = download_histocore()
    if not zip_path:
        input("\\nPress Enter to exit...")
        sys.exit(1)
    
    # Extract
    install_dir = extract_histocore(zip_path)
    if not install_dir:
        input("\\nPress Enter to exit...")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies(install_dir):
        input("\\nPress Enter to exit...")
        sys.exit(1)
    
    # Install package
    if not install_package(install_dir):
        input("\\nPress Enter to exit...")
        sys.exit(1)
    
    # Create shortcuts
    create_shortcuts(install_dir)
    
    # Success
    print("\\n" + "=" * 60)
    print("Installation Complete!")
    print("=" * 60)
    print(f"\\nInstalled to: {install_dir}")
    print("\\nTo launch HistoCore:")
    print("  1. Double-click 'HistoCore.bat' on your desktop")
    print(f"  2. Or run: python {os.path.join(install_dir, 'histocore.py')}")
    print("\\nFor help: https://github.com/matthewvaishnav/computational-pathology-research")
    
    input("\\nPress Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n\\nInstallation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n\\nFATAL ERROR: {e}")
        input("\\nPress Enter to exit...")
        sys.exit(1)
'''
    
    with open("installer_main.py", "w", encoding="utf-8") as f:
        f.write(installer_code)
    
    print("OK: Created installer script")


def build_exe():
    """Build the executable using PyInstaller."""
    print("\nBuilding executable...")
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",                    # Single executable
        "--console",                    # Show console for debugging
        "--name=HistoCore-Installer",   # Output name
        "--icon=NONE",                  # No icon (add later if desired)
        "--clean",                      # Clean cache
        "installer_main.py"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("OK: Executable built successfully")
        return True
    except Exception as e:
        print(f"ERROR: Build failed - {e}")
        return False


def cleanup():
    """Clean up temporary files."""
    print("\nCleaning up...")
    
    files_to_remove = ["installer_main.py", "HistoCore-Installer.spec"]
    dirs_to_remove = ["build"]
    
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
    
    for d in dirs_to_remove:
        if os.path.exists(d):
            shutil.rmtree(d)
    
    print("OK: Cleanup complete")


def main():
    """Main build process."""
    print("=" * 60)
    print("HistoCore Installer Builder")
    print("=" * 60)
    print()
    
    # Check/install PyInstaller
    if not check_pyinstaller():
        install_pyinstaller()
    
    # Create installer script
    create_installer_script()
    
    # Build executable
    if not build_exe():
        sys.exit(1)
    
    # Cleanup
    cleanup()
    
    # Success
    print("\n" + "=" * 60)
    print("Build Complete!")
    print("=" * 60)
    print("\nExecutable location: dist/HistoCore-Installer.exe")
    print("\nDistribute this .exe file to users.")
    print("Users can double-click to install HistoCore.")
    print("\nNote: Windows Defender may flag unsigned executables.")
    print("See WINDOWS_DEFENDER_FIX.md for solutions.")


if __name__ == "__main__":
    main()
