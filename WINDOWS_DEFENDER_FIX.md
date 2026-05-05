# Windows Defender Fix

If Windows Defender blocks the installer, follow these steps:

## Quick Fix (Recommended)

**Option 1: Use Python installer instead of .bat**
```bash
python install.py
```
This avoids Defender's batch file restrictions.

## If Defender Blocks Python Scripts

**Option 2: Manual Installation**
```bash
# 1. Open PowerShell or Command Prompt
# 2. Navigate to the HistoCore directory
cd path\to\histocore

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package
pip install -e .

# 5. Launch
python histocore.py
```

## Allow Installer Through Defender

If you trust the source and want to use install.bat:

1. **Click "More info"** when Defender blocks
2. **Click "Run anyway"**

OR

1. Open **Windows Security**
2. Go to **Virus & threat protection**
3. Click **Manage settings**
4. Scroll to **Exclusions**
5. Click **Add or remove exclusions**
6. Add the HistoCore folder

## Why Does This Happen?

Windows Defender flags batch files that:
- Download files from internet
- Install software
- Modify system settings

This is normal for installers. HistoCore is open-source and safe.

## Verify Safety

Check the installer source code:
- `install.bat` - Windows installer
- `install.py` - Cross-platform installer
- `install.sh` - Linux/Mac installer

All code is visible on GitHub:
https://github.com/matthewvaishnav/computational-pathology-research

## Alternative: Use Docker

Completely bypass installation issues:

```bash
# Pull image
docker pull matthewvaishnav/histocore:latest

# Run
docker run -p 5000:5000 matthewvaishnav/histocore
```

## Still Having Issues?

1. Check Python is installed: `python --version`
2. Check pip is installed: `pip --version`
3. Try manual installation (Option 2 above)
4. Open an issue: https://github.com/matthewvaishnav/computational-pathology-research/issues

## For IT Administrators

To deploy HistoCore in enterprise environments:

1. **Code Sign the installer** (requires certificate)
2. **Use Group Policy** to whitelist the installer
3. **Deploy via SCCM/Intune** with pre-approved packages
4. **Use Docker/Kubernetes** for containerized deployment

Contact: https://github.com/matthewvaishnav/computational-pathology-research/issues
