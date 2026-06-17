@echo off
echo Starting Phikon feature extraction for PANDA dataset...
echo This will take approximately 8-10 hours for 10,616 slides
echo.
echo Output directory: D:\panda\features_phikon
echo.
python scripts/extract_panda_features_openslide.py --data_dir D:\panda --output_dir D:\panda\features_phikon --model phikon --max_patches 600 --batch_size 32
echo.
echo Extraction complete!
pause
