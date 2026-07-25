# PatchCamelyon benchmark results

**Date:** 2026-04-09  
**Status:** Completed single-split benchmark  
**Hardware:** RTX 4070 Laptop GPU, 8 GB VRAM

## Result

A ResNet18-based patch classifier was trained on the full PatchCamelyon training
split and evaluated on the official 32,768-patch test split.

| Metric | Estimate | 95% bootstrap interval |
|---|---:|---:|
| Accuracy | 0.8526 | 0.8483–0.8563 |
| ROC AUC | 0.9394 | 0.9369–0.9418 |
| F1 | 0.8507 | 0.8464–0.8543 |
| Macro precision | 0.8718 | 0.8680–0.8751 |
| Macro recall | 0.8526 | 0.8486–0.8561 |

The bootstrap resampled test patches. It quantifies patch-level sampling
uncertainty under this split; it does not account for model-training variation,
patients, slides, institutions, or external-domain shift.

## Per-class performance at threshold 0.5

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Normal patch | 0.787 | 0.966 | 0.868 |
| Tumour-labelled patch | 0.956 | 0.739 | 0.834 |

```text
              Predicted
              Normal  Tumour
Actual Normal  15,837     554
Actual Tumour   4,276  12,101
```

These are patch-classification errors. They are not counts of patients,
diagnoses, cancers missed, or clinical decisions.

## Architecture and training record

- ResNet18 ImageNet-pretrained feature extractor
- transformer encoder and classification head
- approximately 12 million parameters
- 20 epochs
- batch size 128
- AdamW, learning rate 1e-3, weight decay 1e-4
- mixed-precision training

The documented run took approximately six hours. Separate engineering work
explored faster training configurations; those speed measurements are systems
results rather than evidence of scientific superiority.

## What this result supports

- the repository can train and evaluate a neural network on the complete PCam
  benchmark;
- the resulting model has strong patch-level discrimination on one official
  split;
- the evaluation pipeline reports confusion-matrix metrics and bootstrap
  intervals;
- the benchmark is suitable as an engineering and patch-classification sanity
  check for the broader research program.

## What this result does not support

This result does not establish:

- state-of-the-art performance;
- statistically significant superiority over published models;
- whole-slide or patient-level performance;
- clinical utility, clinical readiness, or diagnostic performance;
- improved outcomes or reduced missed diagnoses;
- external generalization;
- calibration at a clinically meaningful operating point.

The previous table ranking this run against values collected from unrelated
papers has been removed. Those studies did not share a demonstrated common
split, preprocessing pipeline, tuning budget, seeds, predictions, or paired
statistical comparison, so they cannot support a valid leaderboard or
significance claim.

## Threshold analysis boundary

The historical threshold analysis selected operating points using the same test
predictions on which they were reported. It is retained only as a retrospective
sensitivity-specificity illustration. See `docs/THRESHOLD_OPTIMIZATION.md` for
the corrected interpretation.

A confirmatory operating-point analysis must select the threshold on validation
data, freeze it, and evaluate it once on untouched test data.

## Limitations

1. Single official train/validation/test partition.
2. One trained-model run; training-seed uncertainty is not included.
3. Patch-level labels and outcomes only.
4. No whole-slide aggregation or patient-level endpoint.
5. No independent external dataset.
6. No matched reimplementation of comparison models.
7. No clinical workflow or pathologist comparison.

## Reproduction

```bash
python experiments/train_pcam.py \
  --config experiments/configs/pcam_rtx4070_laptop.yaml \
  --data-root data/pcam_real \
  --output-dir checkpoints/pcam_real

python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam_real/best_model.pth \
  --data-root data/pcam_real \
  --output-dir results/pcam_real \
  --batch-size 64 \
  --bootstrap-samples 1000
```

Primary artifacts:

- `results/pcam_real/metrics.json`
- `results/pcam_real/confusion_matrix.png`
- `results/pcam_real/roc_curve.png`
