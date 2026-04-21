@echo off
REM Full 5-fold cross-validation with GPU venv
REM Estimated time: 30-40 hours on RTX 4070 Laptop

echo Activating GPU virtual environment...
call venv_gpu\Scripts\activate.bat

echo.
echo ========================================
echo FULL CROSS-VALIDATION
echo ========================================
echo This will take approximately 30-40 hours
echo Press Ctrl+C to cancel, or
pause

echo.
echo Running full 5-fold cross-validation...
python scripts/cross_validate_pcam.py ^
  --data-root data/pcam_real ^
  --output-dir results/pcam_cv_full ^
  --n-folds 5 ^
  --num-epochs 20 ^
  --batch-size 128 ^
  --learning-rate 1e-3 ^
  --weight-decay 1e-4 ^
  --num-workers 4 ^
  --bootstrap-samples 1000 ^
  --use-amp ^
  --seed 42

echo.
echo Cross-validation complete!
pause
