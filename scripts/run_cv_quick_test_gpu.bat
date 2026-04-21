@echo off
REM Quick test of cross-validation with GPU venv

echo Activating GPU virtual environment...
call venv_gpu\Scripts\activate.bat

echo.
echo Running cross-validation quick test...
python scripts/cross_validate_pcam.py ^
  --data-root data/pcam_real ^
  --output-dir results/pcam_cv_test ^
  --n-folds 3 ^
  --num-epochs 3 ^
  --batch-size 128 ^
  --learning-rate 1e-3 ^
  --weight-decay 1e-4 ^
  --num-workers 4 ^
  --bootstrap-samples 100 ^
  --use-amp ^
  --seed 42 ^
  --subset-size 5000

echo.
echo Cross-validation test complete!
pause
