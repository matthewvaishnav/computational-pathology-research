@echo off
REM Run optimized PCam training with all performance enhancements
REM Expected: 8-12x faster than baseline (15-30 min vs 2.5 hours)

echo ========================================
echo PCam Optimized Training
echo ========================================
echo.
echo Optimizations enabled:
echo   - Batch size: 128 (8x larger)
echo   - Mixed precision (AMP)
echo   - torch.compile (max-autotune)
echo   - Channels last memory format
echo   - Persistent workers
echo   - cuDNN benchmark
echo.
echo Expected speedup: 8-12x
echo Estimated time: 15-30 minutes
echo ========================================
echo.

python experiments/train_pcam.py --config experiments/configs/pcam_full_20_epochs_optimized.yaml

echo.
echo ========================================
echo Training complete!
echo ========================================
