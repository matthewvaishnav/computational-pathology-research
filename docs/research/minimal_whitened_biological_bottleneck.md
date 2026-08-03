# Minimal and whitened biological bottleneck

## Question

The calibrated crossed-target diagnostic found the same phenotype in all eight synthetic runs: exact operational replication, perfect cross-scanner retrieval, informative independent decoding, acquisition exclusion, prototype invariance, and no calibrated scanner leakage, but no stable recovery of the known biological latent at R² ≥ 0.80. This experiment tests whether the learned code is an overcomplete nonlinear chart that can be made linearly equivalent to the generator factor by dimensional minimality, identity-level covariance whitening, or their interaction.

This is a synthetic identifiability-mechanism experiment. The true biological dimension of eight is known only because the data generator is synthetic. The experiment does not propose a pathology-facing architecture and does not establish pathology, clinical, vendor, stain, site, cohort, or endpoint generalization.

## Frozen boundaries

The successful calibrated diagnostic is frozen by its complete file SHA-256 and internal result hash. Its mixed status and all four upstream artifact hashes are verified before and after execution. The incomplete first attempt is not admissible evidence. The original unseen-identity primary gate remains closed, and no prior result or threshold is reinterpreted.

The smoke grid contains dataset seeds 4301 and 5301, linear and nonlinear renderers, model seeds 2201 and 2202, and four predeclared families. The 32-dimensional unwhitened baseline reproduces the prior crossed-target architecture and objective. The only factorial changes are biological-code dimension (32 or 8) and biological-whitening weight (0 or 0.10). Scanner prototypes, decoder, FiLM mechanism, losses, regularization, optimizer, epochs, dataset sizes, noise, and bootstrap configuration remain frozen.

## Whitening objective

At each optimization step, every training scanner view is encoded and codes are averaged within each training identity. With centered identity means `Z`, covariance is `ZᵀZ / (N - 1)`. The diagonal penalty is the mean squared deviation of covariance diagonals from one; the off-diagonal penalty is the mean squared off-diagonal covariance. Their sum is multiplied by the fixed weight 0.10 in both whitened families.

Distinct training identities must exceed the biological-code dimension. Probe-validation identities are not a special fitting set, unseen identities never enter the whitening objective, and known biological latent arrays are evaluation-only. The existing marginal variance-floor term remains active in all families.

## Evaluation and interpretation

Every run uses the unchanged calibrated residual probes, repeated observed/null scanner classifier, acquisition probe, retrieval diagnostic, and independent decoder controls. Ridge remains the canonical linear-equivalence test because whitening cannot remove an arbitrary orthogonal rotation, but a full-rank linear rotation remains recoverable by Ridge. Recovery does not imply identification of named semantic axes.

Covariance geometry is evaluated independently on probe-training, probe-validation, and entirely unseen identities. Generalized whitening requires finite metrics, full numerical rank when sample count makes it feasible, mean absolute off-diagonal covariance at most 0.10, and every covariance diagonal between 0.50 and 1.50. These geometric criteria do not substitute for biological recovery.

Mechanism success requires both residual-probe seeds to reach R² ≥ 0.80, preservation of operational transfer, scanner exclusion, acquisition exclusion, prototype invariance, overall and worst-pair retrieval, and independent decoder informativeness. Whitened families must additionally generalize the covariance criterion. The complete factorial pattern determines whether dimensionality, whitening, their interaction, multiple mechanisms, neither mechanism, or an operational trade-off is supported. Poor scientific performance is a scientific status rather than an execution failure.
