@echo off
REM HistoCore Windows Installer
REM Just double-click this file to install

echo ============================================================
echo    HistoCore Windows Installer
echo    Production-grade computational pathology framework
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.9+ from:
    echo https://python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [OK] Python found
python --version

REM Check pip
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip not found!
    pause
    exit /b 1
)

echo [OK] pip found
echo.

REM Ask for confirmation
set /p CONFIRM="Install HistoCore? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Installation cancelled
    pause
    exit /b 0
)

echo.
echo ============================================================
echo    Installing Dependencies
echo ============================================================
echo.

REM Upgrade pip
echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo [2/4] Installing core dependencies...
python -m pip install -r requirements.txt

REM Install package
echo.
echo [3/4] Installing HistoCore package...
python -m pip install -e .

REM Verify installation
echo.
echo [4/4] Verifying installation...
python -c "import torch; import numpy; import cv2; print('[OK] All core packages installed')"

if errorlevel 1 (
    echo.
    echo [WARNING] Some packages may not be installed correctly
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ============================================================
echo    Installation Complete!
echo ============================================================
echo.
echo Next Steps:
echo.
echo 1. Launch GUI:
echo    python histocore.py
echo.
echo 2. Use CLI:
echo    histocore analyze slide.svs --output results/
echo    histocore demo --quick
echo.
echo 3. Launch Web Interface:
echo    histocore web
echo.
echo 4. Read Documentation:
echo    https://github.com/matthewvaishnav/histocore
echo.
pause
