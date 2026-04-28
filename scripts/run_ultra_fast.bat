@echo off
echo ============================================================
echo ULTRA FAST Training - Sub-1 Hour
echo ============================================================
echo.
echo Testing maximum batch size...
call venv311\Scripts\activate.bat
python scripts\test_max_batch_size.py
echo.
echo Starting ultra-fast training...
echo Expected time: 20-30 minutes
echo.
python experiments\train_pcam.py --config experiments\configs\pcam_ultra_fast.yaml
echo.
echo ============================================================
echo Training complete!
echo ============================================================
pause
