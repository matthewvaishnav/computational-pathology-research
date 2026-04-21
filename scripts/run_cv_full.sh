#!/bin/bash
# Full 5-fold cross-validation on complete PCam dataset
# Estimated time: 30-40 hours on RTX 4070 Laptop (6 hours per fold)

python scripts/cross_validate_pcam.py \
  --data-root data/pcam_real \
  --output-dir results/pcam_cv_full \
  --n-folds 5 \
  --num-epochs 20 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --bootstrap-samples 1000 \
  --use-amp \
  --seed 42
