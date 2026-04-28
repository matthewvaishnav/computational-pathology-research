@echo off
REM Run aggressive PCam training to beat competitors
REM Expected: 95-96% test AUC (vs current 93.7%)

echo ========================================
echo AGGRESSIVE PCAM TRAINING
echo ========================================
echo.
echo Goal: Push test AUC from 93.7%% to 95-96%%+
echo Strategy: Larger model (ResNet50), deeper architecture, stronger regularization
echo.
echo Configuration:
echo - Model: ResNet50 (2048-dim) with 3-layer transformer
echo - Batch size: 64 (vs 16)
echo - Epochs: 50 (vs 20)
echo - Dropout: 0.5 (vs 0.3)
echo - Weight decay: 0.01 (vs 0.001)
echo - Attention pooling (vs mean)
echo.

REM Activate virtual environment
call venv311\Scripts\activate.bat

REM Run training
python experiments\train_pcam.py ^
    --config experiments\configs\pcam_aggressive.yaml ^
    --device cuda

echo.
echo ========================================
echo Training complete!
echo Check logs/pcam_aggressive/ for results
echo ========================================
pause
