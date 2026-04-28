@echo off
REM Setup CUDA-enabled Python environment for optimized training

echo ========================================
echo CUDA Environment Setup
echo ========================================
echo.
echo Current Python: %PYTHON%
python --version
echo.

REM Check if CUDA PyTorch is available
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: PyTorch not found or CUDA not available
    echo.
    echo SETUP INSTRUCTIONS:
    echo.
    echo 1. Install Python 3.11 or 3.12 ^(not 3.14^)
    echo    Download from: https://www.python.org/downloads/
    echo.
    echo 2. Create virtual environment:
    echo    python -m venv venv_cuda
    echo.
    echo 3. Activate environment:
    echo    venv_cuda\Scripts\activate
    echo.
    echo 4. Install CUDA PyTorch:
    echo    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    echo.
    echo 5. Install other dependencies:
    echo    pip install -r requirements.txt
    echo.
    echo 6. Verify CUDA:
    echo    python -c "import torch; print(torch.cuda.is_available())"
    echo.
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo CUDA Environment Ready!
echo ========================================
echo.
echo You can now run optimized training:
echo   scripts\run_optimized_training.bat
echo.
echo Or benchmark performance:
echo   python scripts\benchmark_optimizations.py
echo.
echo ========================================
pause
