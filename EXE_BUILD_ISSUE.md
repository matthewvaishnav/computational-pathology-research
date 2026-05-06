# EXE Build Issue - Pip Hanging

## Problem

`pip install` commands hang indefinitely on this system. Affects:
- `pip install pyinstaller`
- `pip show pyinstaller`
- Any pip operation after initial output

## Root Cause

Likely one of:
1. Network/proxy configuration issue
2. Corrupted pip cache
3. Antivirus blocking pip
4. Windows Defender scanning packages

## Workarounds

### Option 1: Fix Pip (Recommended)

Try these in order:

```bash
# Clear pip cache
python -m pip cache purge

# Upgrade pip
python -m pip install --upgrade pip --no-cache-dir

# Try with timeout
python -m pip install pyinstaller --timeout 15

# Try with different index
python -m pip install pyinstaller --index-url https://pypi.org/simple/

# Disable cache
python -m pip install pyinstaller --no-cache-dir
```

### Option 2: Use Pre-built PyInstaller

```bash
# Download PyInstaller wheel manually
# From: https://pypi.org/project/pyinstaller/#files

# Install from wheel
python -m pip install pyinstaller-6.11.1-py3-none-win_amd64.whl
```

### Option 3: Alternative Exe Builders

Instead of PyInstaller, use:

**cx_Freeze:**
```bash
pip install cx_Freeze
python setup_cx.py build
```

**Nuitka:**
```bash
pip install nuitka
python -m nuitka --onefile --windows-console-mode=attach installer_main.py
```

**py2exe:**
```bash
pip install py2exe
python setup_py2exe.py
```

### Option 4: GitHub Actions Build

Build exe in CI where pip works:

```yaml
# .github/workflows/build-installer.yml
name: Build Installer
on: [push]
jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install pyinstaller
      - run: python build_installer.py
      - uses: actions/upload-artifact@v3
        with:
          name: installer
          path: dist/HistoCore-Installer.exe
```

## Diagnosis Steps

```bash
# Check network
ping pypi.org

# Check proxy
echo %HTTP_PROXY%
echo %HTTPS_PROXY%

# Check antivirus logs
# Windows Security > Virus & threat protection > Protection history

# Test pip with verbose output
python -m pip install pyinstaller -vvv

# Check for hanging processes
tasklist | findstr python
tasklist | findstr pip
```

## Temporary Solution

For now, users can:
1. Use `install.py` (Python installer)
2. Use `install.bat` (batch installer)
3. Use Docker image
4. Manual installation

The exe installer is a convenience feature, not required.

## Next Steps

1. Diagnose pip hang on this system
2. Once fixed, run `python build_installer.py`
3. Upload `dist/HistoCore-Installer.exe` to GitHub Releases
4. Update README with download link

## Status

- [x] Installer scripts created (install.py, install.bat, install.sh)
- [x] Exe builder script created (build_installer.py)
- [ ] Exe built (blocked by pip hang)
- [ ] Exe uploaded to releases

## Contact

If you encounter this issue, report at:
https://github.com/matthewvaishnav/computational-pathology-research/issues
