# Calibrated unseen-identity representation geometry v2

## Purpose

This protocol is the first post-hoc interpretation of the crossed-target synthetic representation using independently calibrated instrument families. It reruns exactly eight `crossed_target_prototype` factorizer conditions: dataset seeds 4301 and 5301, linear and nonlinear renderers, and model seeds 2201 and 2202. It does not rerun oracle models, PA-NF, reconstruction-only models, or any calibration dataset.

The original unseen-identity primary gate remains closed. The first representation-geometry diagnostic remains `diagnostic_failed`; instrument-calibration v1 remains `regression_probe_calibration_failed` as a standalone aggregate; and residual-capacity calibration v2 established the post-hoc composite adjudication `complete_instrument_families_calibrated_v2`. This diagnostic does not retroactively change any prior result.

## Frozen execution boundary

Before and after execution, the runner verifies the file and internal hashes of the primary result, failed geometry result, v1 calibration, and v2 calibration. It reconstructs the primary configuration and requires all seven operational metrics to reproduce within `1e-6 + 1e-5 * abs(reference)`. It also reproduces all six training, validation, and unseen-test split hashes from the failed diagnostic. Any mismatch is an execution failure.

All probe and decoder scalers use probe-training identities only. Validation identities select checkpoints. Entirely unseen test identities cannot affect fitting or checkpoint selection.

## Calibrated measurements

The frozen Ridge probe uses alpha 0.001 and a biological-recovery threshold of variance-weighted R² at least 0.80. The unchanged calibrated residual probe runs at seeds 7203 and 7204; stable recovery requires both seeds to reach 0.80. A nonlinear recovery may cross the threshold without an additional 0.05 improvement requirement.

The calibrated scanner classifier runs observed and identity-aware permutation-null fits at seeds 7301, 7302, and 7303. Hidden scanner leakage requires median observed balanced accuracy greater than chance plus 0.10, every observed result above its paired null, and verified inherited scanner controls. One initialization cannot establish leakage.

Acquisition-to-biology exclusion requires the maximum residual R² across seeds 7203 and 7204 to be at most 0.10. Cross-scanner retrieval requires both overall and worst ordered-pair top-1 accuracy to be at least 0.90.

The calibrated independent decoder uses learned biological code with the true target-scanner acquisition latent. It is informative only when its observation-mean-square-normalized MSE is at least 20% lower than each of four within-run negatives: target acquisition alone, a cyclic wrong scanner latent, identity-permuted learned biology, and learned biology alone. The condition-specific true-factor positive-control NMSE is imported from immutable v1 evidence and is not recomputed. No new closeness-to-oracle threshold is introduced.

## Interpretation

Transferable geometry requires exact replication, original two-axis transfer, stable residual biological recovery, nonlinear scanner exclusion, acquisition biological exclusion, scanner-prototype invariance, retrieval success, and independent decoder informativeness in the same run. Decoder-dependent geometry is suspected only when replication, original transfer, and retrieval pass; hidden scanner leakage is absent; and both stable canonical recovery and independent decoder informativeness fail. Other combinations are unresolved at run level.

Poor representation performance is a scientific outcome, not an execution failure. The aggregate may report transferable geometry, decoder dependence, hidden scanner leakage, or mixed representation geometry. The failure status is reserved for frozen-input, inherited-calibration, replication, split, finite-output, or internal execution failures.

## Claim limits

This protocol tests synthetic representations only. It does not establish pathology-domain validity or clinical, scanner-vendor, stain, site, cohort, or endpoint generalization. Retrieval, counterfactual transfer, canonical factor recovery, scanner exclusion, and independent decoding are distinct properties; success in one does not substitute for another.
