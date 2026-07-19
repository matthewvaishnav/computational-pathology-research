# Linear Baseline Consistency Audit

## Question

Are old_style linear_projection_k4 (bec06eb4) and new_style linear_keep_k4
(ec2a509f) equivalent representations? If not, what explains the scanner
accuracy mismatch (0.2000 vs 0.7071)?

## Definitions

### Old-style (bec06eb4): per-scanner mean directions + QR orthonormalization

1. Fit StandardScaler on ALL features (fit + test).
2. Compute per-scanner mean vectors: for each scanner, mean(features) - grand_mean.
   This produces 5 directions (one per scanner).
3. Take first k=4 direction rows, QR-orthonormalize to get ~4 orthonormal vectors.
4. Project out: features - features @ Q.T @ Q.
5. Return result (no removed branch computed).

Note: The 5 scanner means span a 4-dimensional affine subspace (they sum to zero).
Taking any 4 of the 5 means and QR-orthonormalizing gives the full scanner-mean
subspace. So k=4 removes essentially all first-order scanner-centroid information.

### New-style (ec2a509f): logistic regression coefficient SVD

1. Fit StandardScaler on FIT features only, transform all.
2. Fit LogisticRegression(scanner) on fit set.
3. Center the (5, 768) coefficient matrix, compute SVD.
4. Take top k=4 right singular vectors (max 4 due to rank after centering).
5. keep = features - features @ basis.T @ basis.
6. removed = features @ basis.T @ basis.

Note: SVD of logistic regression coefficients finds directions in feature space
that the classifier uses for discrimination. These may not span the full
scanner-mean subspace.

### Key difference

The old-style approach removes the full scanner-centroid subspace (all first-order
scanner structure). The new-style approach removes the most discriminative
directions for ONE particular linear classifier, which may leave residual
scanner signal that a fresh probe classifier can exploit.

## Results (5-fold means)

### old_style_linear_projection_k4
- Scanner acc: 0.2000 +- 0.0000
- Category acc: 0.4015 +- 0.0553

### new_style_linear_keep_k4
- Scanner acc: 0.7071 +- 0.0413
- Category acc: 0.4011 +- 0.0538

### new_style_linear_removed_k4
- Scanner acc: 0.8639 +- 0.0226
- Category acc: 0.1364 +- 0.0240

### new_style_keep_on_old_standardization
- Scanner acc: 0.7010 +- 0.0368
- Category acc: 0.3984 +- 0.0563

## Feature-Space Comparison (5-fold means)

- Mean L2 difference (old vs new keep): 8.89
- Mean cosine similarity (old vs new keep): 0.936771
- Old mean norm: 25.15
- New keep mean norm: 26.87
- New removed mean norm: 2.26
- Entrywise old/new keep variance-similarity ratio: 0.8648

Variance-similarity definition: 1 - Var(old_style_linear_projection_k4 -
new_style_linear_keep_k4) / Var(old_style_linear_projection_k4), where
Var is computed over all matrix entries. This is a feature-space similarity
diagnostic only; it is not centered variance removed, per-sample projected
feature energy removed, or scanner-centroid-offset variance.

## Conclusion

old_style linear_projection_k4 and new_style linear_keep_k4 are NOT equivalent representations (scanner acc 0.2000 vs 0.7071).

The old-style per-scanner-mean approach removes the full scanner-centroid subspace, which eliminates essentially all first-order scanner signal. The new-style logistic-regression-SVD approach removes only the 4 most discriminative directions for one particular classifier, leaving residual scanner signal that a fresh probe can exploit.

## Bounded Interpretation

This is a consistency audit. It identifies the implementation difference
between two linear projection baselines used in different experiments. It
does not claim clinical validation, diagnostic performance, or deployment
readiness.

Lower scanner probe accuracy means stronger scanner suppression. Under this
reading, old_style linear_projection_k4 (scanner acc 0.2000) is a stronger
raw scanner-removal baseline than the previously reported true_pair_biological
branch (scanner acc 0.3614). Paired-acquisition should not be claimed to beat
the strongest linear scanner-removal baseline on raw scanner suppression or
raw category preservation.

The old-style baseline is a stronger scanner-removal baseline because it
directly targets the scanner centroids. The new-style baseline is weaker at
scanner removal because it relies on one classifier's decision boundaries.

For future experiments, the old-style (per-scanner-mean) approach should be
the default linear baseline, as it more completely removes first-order
scanner information. The new-style approach understates the power of a
simple linear baseline.

Prior conclusions about structured separation remain useful, but the raw
scanner-removal comparison must favor old-style projection. The remaining
paired-acquisition distinction is structural: it learns an explicit
acquisition branch carrying scanner signal, while old-style projection
removes scanner-centroid signal without a learned acquisition branch.

## Implications for Previous Experiments

- The biological-label preservation audit using linear_projection_k4 remains
  the stronger scanner-removal baseline.
- Scanner-confounded and heldout transfer interpretations should prefer the
  old-style linear baseline when discussing strongest simple linear scanner
  removal.
- The linear residual branch-separation audit ec2a509f used a weaker
  logistic-SVD linear split. Its frontier result is still informative for
  that split, but it should not be treated as the strongest possible linear
  scanner-subspace decomposition.

## Follow-up Needed

This consistency audit reconciles the mismatch, but it does not fully compare
old-style keep/residual branch separation. A follow-up old-style residual
decomposition audit may be needed:

- old_style_keep_k4
- old_style_removed_k4
- category leakage in old_style_removed
- scanner signal in old_style_removed
- category/scanner contrast versus true_pair_biological/acquisition


## Validation

- Metric rows: 20
- Feature comparison rows: 5
- Variants evaluated: 4
- Folds: 5

## Output Files

- linear_baseline_consistency_metrics.csv
- linear_baseline_feature_comparison.csv
- linear_baseline_consistency_report.md
- experiment_design.json
- run_log.txt

## Readiness

Ready after validation; no staging or commit performed by this cleanup.
