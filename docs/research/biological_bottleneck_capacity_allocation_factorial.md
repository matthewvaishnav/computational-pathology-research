# Biological-bottleneck capacity-allocation factorial

## Frozen context

The routed-consensus result remains `complete_mixed_routed_consensus_effects`. Most of its
linear-task improvement was reproduced by an unsupervised 64-dimensional control, but the
previous 32-to-64 comparison also raised total parameters from 44,296 to 52,520. It therefore
confounded biological-code allocation with total encoder/decoder capacity.

This final synthetic architecture-attribution control separates those factors within two
approximately matched parameter bands. It contains no consensus target, consensus loss,
auxiliary head, biological supervision, or downstream-task supervision.

## Fixed factorial

For biological dimension `B` and shared hidden width `H`, the scanner-prototype model has

`P(B,H) = H^2 + (153 + 2B)H + B + 104`

trainable parameters. The predeclared families are:

| Family | B | H | Budget | Parameters |
|---|---:|---:|---|---:|
| `b32_h128_low_budget` | 32 | 128 | low | 44,296 |
| `b64_h112_low_budget` | 64 | 112 | low | 44,184 |
| `b32_h145_high_budget` | 32 | 145 | high | 52,626 |
| `b64_h128_high_budget` | 64 | 128 | high | 52,520 |

The low-budget difference is 112 parameters and the high-budget difference is 106, each below
0.5%. Network depth, GELU activations, LayerNorm placement, FiLM decoder, scanner prototypes,
all loss terms, optimizer, epochs, data, noise, identities, and scanners remain fixed.

## Attribution and diagnostics

Dimension effects compare 64 versus 32 dimensions separately within each budget. Parameter
effects compare high versus low budget separately within each biological dimension. Every
contrast is paired by dataset seed, renderer, and model seed across 16 conditions. A material
effect requires median gains of at least 0.05 for both full-budget R2 and label-efficiency area,
at least 12 positive paired changes for both metrics, and intact execution integrity.

Label-free spectral diagnostics summarize identity-averaged codes on training, validation, and
unseen identities. The PCA accessibility audit fits PCA only on labeled probe-training
identities, reports every predeclared feasible component count, and never uses test performance
to select components. A separate alpha diagnostic selects from the fixed grid by validation
performance only and does not replace the frozen alpha-0.001 endpoint.

Scanner leakage remains governed by the frozen three-seed criterion. A primary flag triggers
five additional predeclared seeds, each with its paired identity-aware permutation null.

## Claim boundaries

The admissible task remains synthetic. A supported factor would identify an architecture effect
within this generator, not canonical coordinates, pathology-domain validity, or clinical,
vendor, stain, site, cohort, or endpoint generalization. Scanner leakage is reported as a
separate trade-off and does not invalidate biological task information by itself.

This is the final synthetic architecture-attribution control in this line. After it, the next
major evidence stage should use real paired-scanner pathology features rather than another
synthetic width, bottleneck, or loss-weight sweep. No frozen result, status, threshold, or task
definition is changed by this audit.
