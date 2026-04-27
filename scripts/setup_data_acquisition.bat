@echo off
REM Setup script for data acquisition (Windows)
REM This script installs dependencies and prepares the environment

echo ==========================================
echo Medical AI Data Acquisition Setup
echo ==========================================

REM Check Python version
echo.
echo Checking Python version...
python --version

REM Install required packages
echo.
echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install openai biopython requests tqdm pillow

REM Optional: Install Kaggle CLI
echo.
set /p kaggle_install="Install Kaggle CLI for PANDA dataset? (y/n): "
if /i "%kaggle_install%"=="y" (
    python -m pip install kaggle
    echo Kaggle CLI installed. Set up credentials:
    echo   1. Go to https://www.kaggle.com/account
    echo   2. Create new API token
    echo   3. Save to %%USERPROFILE%%\.kaggle\kaggle.json
)

REM Create data directory structure
echo.
echo Creating data directory structure...
mkdir data\multi_disease\breast 2>nul
mkdir data\multi_disease\lung 2>nul
mkdir data\multi_disease\colon 2>nul
mkdir data\multi_disease\melanoma 2>nul
mkdir data\multi_disease\prostate 2>nul
mkdir data\vision_language 2>nul
mkdir data\metadata 2>nul

echo.
echo Setup complete!
echo.
echo Next steps:
echo   1. Set OpenAI API key: set OPENAI_API_KEY=your_key_here
echo   2. Download datasets: python scripts\download_public_datasets.py
echo   3. Verify downloads: python scripts\verify_datasets.py
echo   4. Generate captions: python scripts\generate_captions_gpt4v.py
echo.
echo See QUICKSTART_DATA_ACQUISITION.md for detailed instructions

pause
