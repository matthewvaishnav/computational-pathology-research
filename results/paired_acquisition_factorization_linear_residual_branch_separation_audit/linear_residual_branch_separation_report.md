# Linear Residual Branch-Separation Audit

## Branch

experiment/linear-residual-branch-separation-audit

## Question

Does paired-acquisition produce cleaner biological/acquisition separation than a
linear scanner-subspace decomposition (keep vs removed components)?

## Dataset

- Canine cutaneous SCC, DINOv2-Base frozen features
- 4025 patches, 44 samples, 5 scanners (cs2, gt450, nz20, nz210, p1000), 805 regions
- 7 categories: Bone, Cartilage, Dermis, Epidermis, Inflamm/Necrosis, SCC, Subcutis
- 5 folds, 5 neural seeds (911-915)

## Formulas

Linear decomposition (per fold, fit-set standardized features):

1. Fit logistic regression scanner classifier on fit set.
2. SVD of centered coefficient matrix to obtain orthonormal scanner subspace basis.
3. `linear_keep_kN = features - features @ basis[:N].T @ basis[:N]`  (subspace removed)
4. `linear_removed_kN = features @ basis[:N].T @ basis[:N]`  (subspace projection)

For k at or above the fitted scanner subspace rank (4), all larger k values
produce identical results (rank saturation).

Fitted scanner subspace rank: 4 (all 5 folds).

Paired-acquisition factorization (pre-computed, frozen hyperparameters):

- true_pair_biological: biological branch from true-pair factorization.
- true_pair_acquisition: acquisition branch from true-pair factorization.
- Shuffled-sample variants use deranged sample-pair training.

## Representations

13 non-neural representations:
- original_frozen_features (1)
- linear_keep_kN for N in {1, 2, 4, 8, 16, 32} (6)
- linear_removed_kN for N in {1, 2, 4, 8, 16, 32} (6)

4 neural-factorization representations:
- true_pair_biological, true_pair_acquisition
- shuffled_sample_biological, shuffled_sample_acquisition

## Row-Count Design

- 13 simple x 5 folds = 65 rows
- 4 neural x 5 folds x 5 neural seeds = 100 rows
- Total: 165 rows

## Key Metrics

Mean across all folds and neural seeds.

### original_frozen_features
- Scanner balanced accuracy: 0.8641
- Category balanced accuracy: 0.4016
- Neighborhood purity k=1: 0.9600

### true_pair_biological
- Scanner balanced accuracy: 0.3614
- Category balanced accuracy: 0.3860
- Neighborhood purity k=1: 0.9730

### true_pair_acquisition
- Scanner balanced accuracy: 0.8651
- Category balanced accuracy: 0.3456
- Neighborhood purity k=1: 0.5737

### linear_keep_k4
- Scanner balanced accuracy: 0.7071
- Category balanced accuracy: 0.4005
- Neighborhood purity k=1: 0.9619

### linear_removed_k4
- Scanner balanced accuracy: 0.8574
- Category balanced accuracy: 0.1366
- Neighborhood purity k=1: 0.2684

Additional k values and full means with standard deviations are in
linear_residual_summary.csv. Neighborhood purity columns are present
in both the raw and summary files.

## Branch Separation Analysis

### Scanner suppression in the "good" branch (lower is better)

- true_pair_biological: 0.3614 (strong suppression)
- linear_keep_k4: 0.7071 (moderate suppression)
- original_frozen_features: 0.8641 (no suppression)

Neural factorization suppresses scanner recoverability much more strongly in the
biological branch than linear keep does, while preserving substantial category
structure (0.3860 vs 0.4005).

### Scanner signal in the "removed" branch (higher is better)

- true_pair_acquisition: 0.8651
- linear_removed_k4: 0.8574

Both branches carry strong scanner signal. The difference is small (0.0077).

### Category leakage in the "removed" branch (lower is better)

- linear_removed_k4: 0.1366 (low leakage)
- true_pair_acquisition: 0.3456 (higher leakage)

Linear removed keeps category signal much more cleanly out of the removed branch
than paired-acquisition keeps it out of the acquisition branch. The difference
is 0.2090. The linear removed category accuracy (0.1366) is near the 7-class
chance rate of approximately 0.14, suggesting minimal category leakage.

### Neighborhood purity in the "good" branch (higher is better)

- true_pair_biological: k=1 0.9730, k=5 0.8973, k=10 0.7508
- linear_keep_k4: k=1 0.9619, k=5 0.8728, k=10 0.7318
- original_frozen_features: k=1 0.9600, k=5 0.8713, k=10 0.7307

true_pair_biological has slightly higher same-category neighborhood purity than
linear_keep_k4 or frozen features, despite having much lower scanner signal. This
suggests the biological branch preserves category-relevant structure while
suppressing scanner information.

### Shuffled-sample controls

- shuffled_sample_biological: scanner 0.4093, category 0.3228, purity k=1 0.9272
- shuffled_sample_acquisition: scanner 0.8302, category 0.3871, purity k=1 0.7309

The shuffled-sample biological branch has lower category signal (0.3228 vs 0.3860)
and lower neighborhood purity (0.9272 vs 0.9730) than true_pair_biological. This
confirms that true pair structure matters for preserving category-relevant
information in the biological branch.

## Key Questions

1. Does true_pair_biological have lower scanner signal than frozen features?
   Yes (0.3614 vs 0.8641). Neural factorization suppresses scanner strongly.

2. Does true_pair_acquisition carry scanner signal?
   Yes (0.8651). The acquisition branch captures scanner information.

3. Does true_pair_acquisition have lower category signal than biological?
   Yes (0.3456 vs 0.3860). But the acquisition branch still carries substantial
   category signal (0.3456), well above the 7-class chance rate of approximately
   0.14.

4. Does linear_removed_k4 carry scanner signal?
   Yes (0.8574). The removed scanner subspace component captures scanner
   information comparably to the paired acquisition branch.

5. Does linear_removed_k4 also leak category signal?
   Minimally (0.1366). This is near the 7-class chance rate, suggesting the
   linear removed branch carries very little category structure.

6. Which produces cleaner branch separation?
   On "removed branch category leakage": linear decomposition is cleaner
   (0.1366 vs 0.3456). On "keep branch scanner suppression": neural
   factorization is cleaner (0.3614 vs 0.7071). Neither strictly dominates.

7. Is the linear baseline sufficient to explain paired-acquisition behavior?
   No. Linear decomposition cannot simultaneously match the scanner suppression
   of the biological branch (0.3614) while keeping category signal high (0.3860).
   The two approaches occupy different points on the separation frontier.

## Separation Front

The audit reveals a separation-frontier tradeoff:

- Paired-acquisition suppresses scanner recoverability much more strongly in the
  biological branch than linear keep, while preserving substantial category
  structure (0.3860 vs 0.4005) and achieving slightly higher neighborhood purity
  (0.9730 vs 0.9619).

- Linear residual decomposition keeps category signal much more cleanly out of
  the removed scanner branch than paired-acquisition keeps it out of the
  acquisition branch (0.1366 vs 0.3456).

Therefore, linear residual decomposition is not sufficient to explain the
paired-acquisition behavior, but paired-acquisition also does not strictly
dominate the linear split. The two approaches occupy different points on the
separation frontier.

## Bounded Interpretation

This is a branch-separation audit. It does not claim clinical validation,
diagnostic performance, patient-care utility, deployment readiness, or that
scanner bias is solved.

The audit shows a separation-frontier tradeoff. Paired-acquisition suppresses
scanner recoverability much more strongly in the biological branch than the
linear keep branch, while preserving substantial category structure. Linear
residual decomposition keeps category signal much more cleanly out of the
removed scanner branch than paired-acquisition keeps it out of the acquisition
branch. Therefore, linear residual decomposition is not sufficient to explain
the paired-acquisition behavior, but paired-acquisition also does not strictly
dominate the linear split. The two approaches occupy different points on the
separation frontier.

## File Notes

Neighborhood purity metrics (k=1, k=5, k=10) are stored in
linear_residual_raw_metrics.csv and linear_residual_summary.csv. There is no
separate linear_residual_neighborhood_purity.csv file; purity metrics are
collocated with the other metrics in the raw and summary files.

## Validation

- Total raw rows: 165
- 0 duplicate representation/fold/neural_seed rows
- 0 nonfinite values in scanner_balanced_accuracy, category_balanced_accuracy,
  scanner_macro_f1, category_macro_f1, category_weighted_f1
- 17 representations present (13 simple + 4 neural)
- All 6 linear keep k-values present: 1, 2, 4, 8, 16, 32
- All 6 linear removed k-values present: 1, 2, 4, 8, 16, 32
- All 4 neural representations present
- Formulas for linear_keep and linear_removed documented in Formulas section
- No previous result files modified
- git diff --check clean

## Output Files

- linear_residual_raw_metrics.csv (165 rows)
- linear_residual_summary.csv
- linear_residual_branch_contrasts.csv (80 rows)
- linear_residual_branch_separation_report.md
- experiment_design.json
- run_log.txt

## Readiness

Ready to commit.
