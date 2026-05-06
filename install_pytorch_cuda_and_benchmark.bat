@echo off
echo ========================================
echo Installing PyTorch with CUDA 11.8
echo ========================================
echo.

echo Step 1: Uninstalling CPU-only PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo Step 2: Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo Step 3: Verifying CUDA installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo ========================================
echo Running Benchmark System Validation
echo ========================================
python experiments/benchmark_system/run_benchmark.py validate

echo.
echo ========================================
echo Installation and validation complete!
echo ========================================
echo.
echo To run quick benchmark (3-4 hours):
echo   python experiments/benchmark_system/run_benchmark.py run --mode quick
echo.
echo To run full benchmark (20-40+ hours):
echo   python experiments/benchmark_system/run_benchmark.py run --mode full
echo.
pause
