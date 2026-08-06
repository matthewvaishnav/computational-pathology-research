# Residual-probe nonlinear-capacity calibration v2

## Scope

The primary unseen-identity gate remains closed. The first representation-
geometry diagnostic remains failed, and instrument-calibration v1 remains
`regression_probe_calibration_failed`. This v2 branch calibrates only the frozen
residual regression probe's ability to learn a beneficial nonlinear correction.
It initializes and trains no factorizer, oracle, crossed-target model, scanner
classifier, or observation decoder.

The passed v1 scanner, true-factor decoder, and oracle Ridge-preservation
observations are imported from the immutable v1 artifact. No v2 result edits or
replaces v1. Perfect retrieval and counterfactual transfer remain observations,
not proof of canonical biological recovery. All evidence is synthetic and not
pathology-domain validation.

## Frozen inputs and inherited evidence

The runner verifies before and after execution:

- primary unseen-identity SHA-256
  `091700113bddde4abe6b4dd0891a26a31c92c6ebf8815b422747f04c58c35b75`;
- failed geometry diagnostic SHA-256
  `c7b2c24dfdccbd17d9a3084f76e5b89f458287936b0b3d2782e9bb9c430d9e6d`;
- v1 calibration file SHA-256
  `72b00083d789141c9abde67986a36765c8c9127ec49981e4ab0edeecb8f2d634`;
- v1 internal SHA-256
  `a571e854df12f4c68053576abbf5b90b28ade51fa7c7cab45e9a4dfe84529225`.

It also requires the unchanged v1 failed status, passed scanner and true-factor
decoder controls, eight preserved Ridge-positive oracle records, and zero
crossed-target records. Those observations are copied into
`inherited_v1_evidence`; none is recomputed.

## Treatment of the v1 nonlinear control

The original transformation `x = tanh(1.25 b) Q + epsilon` remains a failed,
unresolved nonlinear inversion challenge. Rotated inverse-tanh geometry gave
Ridge R² about 0.869 and residual R² about 0.868. It is not reclassified as a
pass, deleted, or weakened. In v2 it is not treated as a valid standalone
capacity positive control and is not evidence that the residual probe lacks all
nonlinear capacity.

## Frozen probe

V2 imports the v1 implementation unchanged:

`prediction(x) = standardized_ridge(x) + residual_mlp(x)`.

The residual MLP has two width-32 GELU hidden layers, a zero-initialized final
layer, AdamW learning rate `1e-3`, weight decay `1e-4`, at most 500 full-batch
epochs, patience 50, and epoch zero eligibility. Input and target scalers and
Ridge are fit only on probe-training identities. Only identity-disjoint
validation loss selects checkpoints. Unseen-test identities do not affect
generation calibration or model selection.

Each v2 dataset has 512 training identities and 256 independently generated
unseen-test identities. Twenty percent of training identities form validation.
Every identity has five scanner labels, and all five rows contain identical
features and targets. Scanner repeats never cross identity boundaries.

## Fixed controls

Control A, Ridge preservation, uses an identity map with seed 9101. Ridge test
R² must be at least 0.95; residual R² may not be lower by more than `1e-7`; and
epoch zero must match Ridge within `1e-7`.

Control B, teacher in hypothesis class, uses generation seed 9102. Base features
`u` are standard normal. A full-rank linear teacher has singular values from 0.8
to 1.2. The nonlinear teacher is a frozen two-hidden-layer GELU network of width
8, initialized once with seed 9151; its parameters never train. Linear and
nonlinear components are centered and scaled using a separate deterministic
65,536-sample generation distribution. Targets are

`y = normalized_linear(u) + 0.75 * normalized_residual_teacher(u) + epsilon`,

with fixed identity-level Gaussian noise standard deviation 0.002. The width-8
teacher lies inside the width-32 student's hypothesis class. Probe seeds are
7203 and 7204. Each must select after epoch zero, improve validation loss,
improve unseen-test R² by at least 0.05, and reach unseen-test R² at least 0.90.

Control C, analytic interaction, uses generation seed 9103 and
`beta = 0.50`. Output dimensions use the fixed interaction pairs
`(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,0)`:

`y_j = linear_j(u) + 0.50 * u_a * u_b + epsilon`.

The same noise standard deviation 0.002 and probe seeds 7203/7204 apply. Each
repeat must improve validation loss, improve test R² by at least 0.03, and reach
test R² at least 0.85.

Control D, seed 9104, deterministically permutes targets by identity before
scanner repetition. Ridge and residual test R² must both remain below 0.80.

Teacher amplitude, interaction strength, generation seeds, architecture, and
thresholds are frozen before execution and are not adjusted after outcomes.

## Status and composite adjudication

Allowed v2 statuses are
`complete_residual_nonlinear_capacity_calibrated`,
`ridge_preservation_failed`, `teacher_residual_control_failed`,
`analytic_interaction_control_failed`, `negative_control_failed`,
`inherited_evidence_verification_failed`, and
`residual_capacity_calibration_failed`.

Only a complete v2 pass plus verified immutable v1 scanner, decoder, and oracle
evidence may produce
`complete_instrument_families_calibrated_v2`. That field is explicitly a
post-hoc composite adjudication across immutable artifacts. It does not validate
the crossed-target representation or retroactively change either source result.
