#!/bin/bash
# Quick test of cross-validation with small subset

python scripts/cross_validate_pcam.py \
  --data-root data/pcam_real \
  --output-dir results/pcam_cv_test \
  --n-folds 3 \
  --num-epochs 3 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --bootstrap-samples 100 \
  --use-amp \
  --seed 42 \
  --subset-size 5000
