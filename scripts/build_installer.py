"""
Build HistoCore installer packages
Windows .exe, macOS .app, Linux .deb
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_windows_exe():
    """Build Windows executable with PyInstaller"""
    
    print("🔨 Building Windows executable...")
    
    # PyInstaller spec
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['../histocore.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('../src/web/templates', 'src/web/templates'),
        ('../src/gui', 'src/gui'),
        ('../src/cli', 'src/cli'),
        ('../requirements.txt', '.'),
    ],
    hiddenimports=[
        'PyQt6',
        'matplotlib',
        'flask',
        'click',
        'numpy',
        'torch',
        'torchvision',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='HistoCore',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico'
)
"""
    
    # Write spec file
    with open('histocore.spec', 'w') as f:
        f.write(spec_content)
    
    # Build executable
    try:
        subprocess.run(['pyinstaller', 'histocore.spec', '--clean'], check=True)
        print("✅ Windows executable built: dist/HistoCore.exe")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to build Windows executable")
        return False

def create_windows_installer():
    """Create Windows installer with NSIS"""
    
    nsis_script = """
!define APPNAME "HistoCore"
!define COMPANYNAME "HistoCore"
!define DESCRIPTION "Production-grade computational pathology framework"
!define VERSIONMAJOR 1
!define VERSIONMINOR 0
!define VERSIONBUILD 0

!define HELPURL "https://github.com/matthewvaishnav/histocore"
!define UPDATEURL "https://github.com/matthewvaishnav/histocore/releases"
!define ABOUTURL "https://github.com/matthewvaishnav/histocore"

!define INSTALLSIZE 500000

RequestExecutionLevel admin

InstallDir "$PROGRAMFILES\\${APPNAME}"

Name "${APPNAME}"
Icon "assets\\icon.ico"
outFile "HistoCore-Setup.exe"

!include LogicLib.nsh

page components
page directory
page instfiles

!macro VerifyUserIsAdmin
UserInfo::GetAccountType
pop $0
${If} $0 != "admin"
    messageBox mb_iconstop "Administrator rights required!"
    setErrorLevel 740
    quit
${EndIf}
!macroend

function .onInit
    setShellVarContext all
    !insertmacro VerifyUserIsAdmin
functionEnd

section "install"
    setOutPath $INSTDIR
    file /r "dist\\HistoCore.exe"
    
    writeUninstaller "$INSTDIR\\uninstall.exe"
    
    createDirectory "$SMPROGRAMS\\${APPNAME}"
    createShortCut "$SMPROGRAMS\\${APPNAME}\\${APPNAME}.lnk" "$INSTDIR\\HistoCore.exe" "" "$INSTDIR\\HistoCore.exe"
    createShortCut "$DESKTOP\\${APPNAME}.lnk" "$INSTDIR\\HistoCore.exe" "" "$INSTDIR\\HistoCore.exe"
    
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayName" "${APPNAME}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "UninstallString" "$INSTDIR\\uninstall.exe"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "InstallLocation" "$INSTDIR"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayIcon" "$INSTDIR\\HistoCore.exe"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "Publisher" "${COMPANYNAME}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "HelpLink" "${HELPURL}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "URLUpdateInfo" "${UPDATEURL}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "URLInfoAbout" "${ABOUTURL}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayVersion" "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONBUILD}"
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "VersionMajor" ${VERSIONMAJOR}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "VersionMinor" ${VERSIONMINOR}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "NoModify" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "NoRepair" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "EstimatedSize" ${INSTALLSIZE}
sectionEnd

section "uninstall"
    delete "$INSTDIR\\HistoCore.exe"
    delete "$INSTDIR\\uninstall.exe"
    rmDir "$INSTDIR"
    
    delete "$SMPROGRAMS\\${APPNAME}\\${APPNAME}.lnk"
    rmDir "$SMPROGRAMS\\${APPNAME}"
    delete "$DESKTOP\\${APPNAME}.lnk"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}"
sectionEnd
"""
    
    with open('installer.nsi', 'w') as f:
        f.write(nsis_script)
    
    try:
        subprocess.run(['makensis', 'installer.nsi'], check=True)
        print("✅ Windows installer created: HistoCore-Setup.exe")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create Windows installer (NSIS required)")
        return False

def create_desktop_file():
    """Create Linux desktop file"""
    
    desktop_content = """[Desktop Entry]
Version=1.0
Type=Application
Name=HistoCore
Comment=Production-grade computational pathology framework
Exec=/usr/local/bin/histocore
Icon=histocore
Terminal=false
Categories=Science;Education;
"""
    
    os.makedirs('dist/linux', exist_ok=True)
    with open('dist/linux/histocore.desktop', 'w') as f:
        f.write(desktop_content)
    
    print("✅ Linux desktop file created")

def main():
    """Build installers for all platforms"""
    
    print("🚀 Building HistoCore installers...")
    
    # Create assets directory
    os.makedirs('assets', exist_ok=True)
    
    # Create placeholder icon (replace with real icon)
    icon_content = """# Placeholder icon file
# Replace with actual .ico file for Windows
# Replace with actual .icns file for macOS
# Replace with actual .png file for Linux
"""
    with open('assets/icon.ico', 'w') as f:
        f.write(icon_content)
    
    success_count = 0
    
    # Windows
    if sys.platform == 'win32':
        if build_windows_exe():
            success_count += 1
            if create_windows_installer():
                success_count += 1
    
    # Linux
    create_desktop_file()
    success_count += 1
    
    print(f"\n✅ Built {success_count} installer components")
    print("📦 Next steps:")
    print("  - Windows: Run HistoCore-Setup.exe")
    print("  - macOS: Create .app bundle with py2app")
    print("  - Linux: Create .deb/.rpm packages")

if __name__ == '__main__':
    main()