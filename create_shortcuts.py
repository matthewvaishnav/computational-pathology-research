#!/usr/bin/env python3
"""
Create desktop shortcuts for HistoCore
Windows, macOS, Linux support
"""

import os
import sys
import platform
from pathlib import Path

def create_windows_shortcut():
    """Create Windows desktop shortcut"""
    
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        shortcut_path = os.path.join(desktop, "HistoCore.lnk")
        
        # Get Python executable and script paths
        python_exe = sys.executable
        script_path = os.path.join(os.getcwd(), "histocore.py")
        icon_path = python_exe  # Use Python icon
        
        # Create shortcut
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = python_exe
        shortcut.Arguments = f'"{script_path}"'
        shortcut.WorkingDirectory = os.getcwd()
        shortcut.IconLocation = icon_path
        shortcut.Description = "HistoCore - Computational Pathology Framework"
        shortcut.save()
        
        print(f"✅ Windows shortcut created: {shortcut_path}")
        
        # Also create Start Menu shortcut
        start_menu = winshell.start_menu()
        start_shortcut_path = os.path.join(start_menu, "HistoCore.lnk")
        
        start_shortcut = shell.CreateShortCut(start_shortcut_path)
        start_shortcut.Targetpath = python_exe
        start_shortcut.Arguments = f'"{script_path}"'
        start_shortcut.WorkingDirectory = os.getcwd()
        start_shortcut.IconLocation = icon_path
        start_shortcut.Description = "HistoCore - Computational Pathology Framework"
        start_shortcut.save()
        
        print(f"✅ Start Menu shortcut created: {start_shortcut_path}")
        
        return True
        
    except ImportError:
        print("❌ Windows shortcuts require: pip install pywin32 winshell")
        return False
    except Exception as e:
        print(f"❌ Windows shortcut failed: {e}")
        return False

def create_macos_alias():
    """Create macOS application alias"""
    
    try:
        # Create .command file for macOS
        desktop = os.path.expanduser("~/Desktop")
        command_file = os.path.join(desktop, "HistoCore.command")
        
        script_content = f"""#!/bin/bash
cd "{os.getcwd()}"
{sys.executable} histocore.py
"""
        
        with open(command_file, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(command_file, 0o755)
        
        print(f"✅ macOS command file created: {command_file}")
        
        # Try to create proper .app bundle
        app_dir = os.path.join(desktop, "HistoCore.app")
        contents_dir = os.path.join(app_dir, "Contents")
        macos_dir = os.path.join(contents_dir, "MacOS")
        
        os.makedirs(macos_dir, exist_ok=True)
        
        # Create Info.plist
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>HistoCore</string>
    <key>CFBundleIdentifier</key>
    <string>com.histocore.app</string>
    <key>CFBundleName</key>
    <string>HistoCore</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
</dict>
</plist>"""
        
        with open(os.path.join(contents_dir, "Info.plist"), 'w') as f:
            f.write(plist_content)
        
        # Create executable
        executable_path = os.path.join(macos_dir, "HistoCore")
        with open(executable_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(executable_path, 0o755)
        
        print(f"✅ macOS app bundle created: {app_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ macOS alias failed: {e}")
        return False

def create_linux_desktop_file():
    """Create Linux desktop file"""
    
    try:
        # Create desktop file
        desktop_dir = os.path.expanduser("~/.local/share/applications")
        os.makedirs(desktop_dir, exist_ok=True)
        
        desktop_file = os.path.join(desktop_dir, "histocore.desktop")
        
        desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=HistoCore
Comment=Production-grade computational pathology framework
Exec={sys.executable} {os.path.join(os.getcwd(), 'histocore.py')}
Icon=applications-science
Terminal=false
Categories=Science;Education;Medical;
StartupNotify=true
"""
        
        with open(desktop_file, 'w') as f:
            f.write(desktop_content)
        
        # Make executable
        os.chmod(desktop_file, 0o755)
        
        print(f"✅ Linux desktop file created: {desktop_file}")
        
        # Also create desktop shortcut
        desktop = os.path.expanduser("~/Desktop")
        if os.path.exists(desktop):
            desktop_shortcut = os.path.join(desktop, "histocore.desktop")
            with open(desktop_shortcut, 'w') as f:
                f.write(desktop_content)
            os.chmod(desktop_shortcut, 0o755)
            print(f"✅ Desktop shortcut created: {desktop_shortcut}")
        
        return True
        
    except Exception as e:
        print(f"❌ Linux desktop file failed: {e}")
        return False

def create_batch_files():
    """Create batch/shell files for easy launching"""
    
    system = platform.system()
    
    if system == "Windows":
        # Create .bat file
        bat_content = f"""@echo off
cd /d "{os.getcwd()}"
"{sys.executable}" histocore.py
pause
"""
        
        with open("HistoCore.bat", 'w') as f:
            f.write(bat_content)
        
        print("✅ Windows batch file created: HistoCore.bat")
        
    else:
        # Create shell script
        sh_content = f"""#!/bin/bash
cd "{os.getcwd()}"
{sys.executable} histocore.py
"""
        
        with open("histocore.sh", 'w') as f:
            f.write(sh_content)
        
        os.chmod("histocore.sh", 0o755)
        print("✅ Shell script created: histocore.sh")

def create_file_associations():
    """Create file associations for WSI files"""
    
    system = platform.system()
    
    if system == "Windows":
        print("💡 Windows file associations:")
        print("   Run as administrator:")
        print(f'   assoc .svs=HistoCoreWSI')
        print(f'   ftype HistoCoreWSI="{sys.executable}" "{os.path.join(os.getcwd(), "histocore.py")}" "%1"')
        
    elif system == "Darwin":
        print("💡 macOS file associations:")
        print("   Add to Info.plist in .app bundle:")
        print("   <key>CFBundleDocumentTypes</key>")
        print("   <array><dict><key>CFBundleTypeExtensions</key>")
        print("   <array><string>svs</string><string>tiff</string></array>")
        
    elif system == "Linux":
        print("💡 Linux file associations:")
        print("   Add to .desktop file:")
        print("   MimeType=image/tiff;application/x-aperio-svs;")

def main():
    """Create shortcuts for current platform"""
    
    print("🔗 Creating HistoCore Shortcuts")
    print("=" * 32)
    
    system = platform.system()
    print(f"Platform: {system}")
    
    success = False
    
    if system == "Windows":
        success = create_windows_shortcut()
    elif system == "Darwin":
        success = create_macos_alias()
    elif system == "Linux":
        success = create_linux_desktop_file()
    else:
        print(f"❌ Unsupported platform: {system}")
    
    # Create batch files for all platforms
    create_batch_files()
    
    # Show file association info
    create_file_associations()
    
    print(f"\n📋 Quick Launch Options:")
    print(f"   Desktop shortcut: {'✅' if success else '❌'}")
    print(f"   Batch/shell file: ✅")
    print(f"   Direct command: python histocore.py")
    
    if success:
        print(f"\n🎉 Shortcuts created successfully!")
        print(f"   Double-click desktop icon to launch HistoCore")
    else:
        print(f"\n💡 Use batch file or direct command to launch")

if __name__ == "__main__":
    main()