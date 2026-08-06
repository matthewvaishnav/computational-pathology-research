# Representation-geometry instrument calibration

## Scope and frozen scientific state

The primary unseen-identity gate remains closed. The first post-hoc
representation-geometry diagnostic remains `diagnostic_failed`: its oracle
positive control failed both the nonlinear-probe criterion and part of the
independent-decoder composite, so that diagnostic could not distinguish
representation failure from instrument failure. Perfect unseen-identity
retrieval and successful counterfactual transfer remain observations, not proof
of canonical biological-factor recovery.

This branch calibrates diagnostic instruments rather than testing the
crossed-target representation. It reruns only the eight oracle conditions and
never trains a crossed-target model. No calibration result can retroactively
change either previous result. All evidence remains synthetic and is not
pathology-domain validation.

## Frozen inputs

Before calibration, and again after it, the runner verifies:

- primary unseen-identity result file SHA-256
  `091700113bddde4abe6b4dd0891a26a31c92c6ebf8815b422747f04c58c35b75`;
- failed diagnostic file SHA-256
  `c7b2c24dfdccbd17d9a3084f76e5b89f458287936b0b3d2782e9bb9c430d9e6d`;
- failed diagnostic internal SHA-256
  `432fffa59c58d9e279eb2be10129b608c073cafe3d0e3ff602ff8a9965fa8e55`;
- the primary schema and closed gate, failed diagnostic schema and status, and
  the complete 16-run failed-diagnostic grid.

The runner never modifies either input or an existing output directory.

## Residual regression probe

The calibrated nonlinear regressor is a strict extension of standardized
Ridge:

`prediction(x) = ridge(x) + residual_mlp(x)`.

Input and target scalers and Ridge (`alpha=1e-3`) are fit only on probe-training
identities. Ridge predicts standardized targets. The residual MLP has two GELU
hidden layers of width 32 and a final layer initialized to exactly zero. Epoch
zero therefore equals Ridge and is a valid checkpoint. AdamW uses learning rate
`1e-3`, weight decay `1e-4`, at most 500 full-batch epochs, patience 50, and
validation improvement tolerance `1e-7`. Only identity-disjoint validation loss
selects a checkpoint; unseen identities never affect selection. The epoch-zero
prediction must equal Ridge within `1e-7`.

Fixed elementary datasets use 80 training and 40 unseen identities, five
scanner observations per identity, and eight biological dimensions:

- identity map, seed 6101: `x = b + epsilon`, with deterministic
  `epsilon ~ N(0, 1e-4)`;
- invertible affine, seed 6102: `x = bA + c + epsilon`, where orthogonal factors
  and singular values evenly spaced from 0.8 to 1.2 make `A` full-rank and
  conditioned;
- mild nonlinear invertible, seed 6103:
  `x = tanh(1.25 b) Q + epsilon`, with deterministic orthogonal `Q`;
- permuted-target negative control, seed 6104: identity-map features paired
  with a fixed derangement of identity targets.

Identity and affine Ridge R² must exceed 0.95 and the residual result may not be
worse beyond `1e-7`. The nonlinear case must improve test R² by at least 0.05
under probe seeds 7203 and 7204. Neither negative-control model may reach the
frozen `R² >= 0.80` threshold.

For each of the eight oracle reruns, the original and probe-split Ridge metrics
must reproduce the failed diagnostic within `1e-6 + 1e-5 * abs(reference)`.
Every Ridge-positive oracle representation must remain at or above R² 0.80
under the residual probe; otherwise oracle calibration fails.

## Scanner classifier

The existing width-32 shallow classifier is retained. Scanner-free features are
identical across scanner views of each biological identity and must remain at or
below chance plus 0.10 for seeds 7301, 7302, and 7303. Scanner-positive features
concatenate biological variation with a scanner one-hot component scaled by 4;
every repeat must reach balanced accuracy and macro F1 of at least 0.90.

Oracle representations also receive three paired observed/null fits. The null
independently permutes scanner labels within every probe-training and validation
identity while retaining scanner counts. Leakage requires median observed
balanced accuracy above chance plus 0.10 and every observed initialization above
its paired null. No single initialization is decisive, and real-representation
leakage is reported rather than used to validate classifier capacity.

## Residual decoder

The independent decoder nests the same standardized linear Ridge baseline. Its
residual network uses two GELU layers of width 128, matching the frozen nonlinear
renderer hidden width, zero output initialization, AdamW `1e-3`, weight decay
`1e-4`, 500 epochs, patience 50, and epoch-zero eligibility. It never updates a
factorizer.

The primary capacity control predicts standardized observations from true
biological plus true acquisition latents for both dataset seeds and both
renderers. It must beat each fixed negative control by at least 20% in
observation-mean-square-normalized MSE. The negative controls are scanner latent
alone, the correct biology with the cyclically wrong scanner, and a deterministic
permutation of unseen biological identities. Wrong-scanner inputs retain biology
exactly and change only acquisition.

Oracle biological representation plus true acquisition latent is reported as a
second capacity diagnostic, but this branch does not invent a universal
representation-decoding threshold from it. A later branch may freeze a
comparison criterion only if this instrument suite calibrates successfully.

## Status policy

`complete_instrument_calibration_passed` requires all elementary regression
controls, scanner controls, true-factor decoder controls under both renderers,
all Ridge-positive oracle preservation checks, deterministic reproductions, and
unchanged input hashes. Failures are reported as
`regression_probe_calibration_failed`, `scanner_probe_calibration_failed`,
`decoder_calibration_failed`, `oracle_representation_calibration_failed`, or
`instrument_calibration_failed`. No status is a crossed-target representation
claim.
