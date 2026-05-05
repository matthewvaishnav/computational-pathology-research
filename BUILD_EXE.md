# Building the Windows Installer Executable

This guide explains how to build `HistoCore-Installer.exe` - a standalone Windows installer.

## Quick Build

```bash
python build_installer.py
```

This creates `dist/HistoCore-Installer.exe` - a single-file installer users can download and run.

## What It Does

The installer executable:
1. Checks Python version (3.9+ required)
2. Downloads HistoCore from GitHub
3. Extracts to `~/HistoCore`
4. Installs all dependencies
5. Creates desktop shortcut
6. Ready to use

## Requirements

- Python 3.9+
- PyInstaller (auto-installed if missing)

## Distribution

After building:
1. Upload `dist/HistoCore-Installer.exe` to GitHub Releases
2. Users download and double-click to install
3. No Python knowledge required

## Windows Defender

Unsigned executables trigger Windows Defender warnings. Solutions:

### For Users
See `WINDOWS_DEFENDER_FIX.md` for workarounds.

### For Developers (Code Signing)

To eliminate warnings, sign the executable:

```bash
# Requires code signing certificate ($100-400/year)
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com dist/HistoCore-Installer.exe
```

**Certificate providers:**
- DigiCert: https://www.digicert.com/code-signing
- Sectigo: https://sectigo.com/ssl-certificates-tls/code-signing
- GlobalSign: https://www.globalsign.com/en/code-signing-certificate

**Free alternative for open source:**
- SignPath.io: https://signpath.io/ (free for OSS projects)

## Advanced Options

### Add Icon

1. Create/download `icon.ico`
2. Edit `build_installer.py`:
   ```python
   "--icon=icon.ico",  # Instead of "--icon=NONE"
   ```

### Show Console (for debugging)

Edit `build_installer.py`:
```python
"--console",  # Instead of "--windowed"
```

### Customize Install Location

Edit `installer_main.py` in `build_installer.py`:
```python
install_dir = "C:\\Program Files\\HistoCore"  # Custom location
```

## Testing

Before distributing:

1. Build the executable
2. Copy to clean Windows VM
3. Run installer
4. Verify installation works
5. Test launching HistoCore

## Troubleshooting

### Build fails with "PyInstaller not found"
```bash
pip install pyinstaller
```

### Executable too large
Use UPX compression:
```bash
pip install pyinstaller[encryption]
# Edit build_installer.py, add:
"--upx-dir=C:\\path\\to\\upx",
```

### Antivirus false positive
- Sign the executable (see above)
- Submit to antivirus vendors for whitelisting
- Use alternative distribution (Python installer)

## CI/CD Integration

Add to `.github/workflows/release.yml`:

```yaml
- name: Build Windows Installer
  run: python build_installer.py
  
- name: Upload Installer
  uses: actions/upload-artifact@v3
  with:
    name: windows-installer
    path: dist/HistoCore-Installer.exe
```

## Size Optimization

Current size: ~10-15 MB (depends on Python version)

To reduce:
1. Use `--onefile` (already enabled)
2. Exclude unnecessary modules:
   ```python
   "--exclude-module=tkinter",
   "--exclude-module=matplotlib",
   ```
3. Use UPX compression (see above)

## Alternative: MSI Installer

For enterprise deployment, create MSI:

```bash
pip install cx_Freeze
python setup_msi.py bdist_msi
```

See: https://cx-freeze.readthedocs.io/

## Support

Issues: https://github.com/matthewvaishnav/computational-pathology-research/issues
