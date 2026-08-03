# Routed paired-consensus biological bottleneck

## Frozen context

This experiment does not alter the frozen `complete_consensus_anchor_operational_tradeoff`
result. The frozen task benchmark remains unsupported, and the nonlinear, interaction, and
classification representation failures remain unresolved. Only the previously admissible
linear biological task is a primary endpoint here.

The auxiliary-anchor campaign allowed a biological code to feed a separate consensus head
while the decoder continued to consume the original code. This experiment removes that
possible bypass: the single representation returned by the biological encoder is also the
content supplied to the decoder and every diagnostic consumer.

## Isolated families

| Family | Biological dimension | Consensus weight |
|---|---:|---:|
| `crossed_target_baseline_32` | 32 | 0 |
| `routed_dimension_control_64` | 64 | 0 |
| `routed_consensus_bottleneck_64` | 64 | 0.25 |

The two 64-dimensional families have identical model structure. The routed-consensus family
adds only `0.25 * MSE(biological_code, detached_standardized_consensus)`. There is no
auxiliary head, private pre-consensus code, encoder-to-decoder skip, decoder-only content
representation, hidden-feature concatenation, or stop-gradient decoder route.

The same biological tensor feeds reconstruction, crossed decoding, biological consistency,
variance regularization, the optional consensus loss, task probes, retrieval, scanner probes,
independent decoding, and counterfactual decoding/re-encoding.

## Target integrity and preflight

Targets reuse the frozen paired-observation construction exactly. Scanner means and target
scaling are fitted only with the 40 factorizer-training identities. Each scanner contributes
once to an identity consensus, each training identity contributes once to target scaling, and
the detached target is repeated across all five views. No biological latent, downstream label,
teacher, or class enters construction or factorizer training.

All target hashes must equal the frozen auxiliary-anchor hashes. The four frozen preflight
conditions must reproduce before any factorizer is initialized. A mismatch fails closed.

## Invariance and scanner evidence

The frozen auxiliary experiment's absolute view-variance threshold of `1e-4` is retained only
as `legacy_absolute_view_variance_passed`. The primary routed criterion is scale normalized:

`mean within-identity cross-scanner variance / mean between-identity variance <= 0.01`

and the maximum identity-level normalized ratio must be at most `0.05`. This postulated
normalized criterion does not revise or reinterpret the frozen absolute-variance result.

Scanner exclusion remains governed by the frozen three seeds 7301--7303. A primary leakage
flag triggers the predeclared confirmation seeds 7304--7308, each with an identity-aware paired
permutation null. Confirmation is mechanistic evidence and cannot erase the primary flag.

## Claim boundaries

Success would show that routing a paired-consensus representation through the decoder improves
accessible biological sufficiency in this synthetic paired-acquisition setting. It would not
establish canonical generator coordinates or pathology, clinical, vendor, stain, site, cohort,
or endpoint validity.

Failure rejects this routed-consensus mechanism, not biological representation learning in
general. The normalized invariance analysis does not change the frozen auxiliary-anchor status,
the frozen task-benchmark status, or any previous threshold or result.
