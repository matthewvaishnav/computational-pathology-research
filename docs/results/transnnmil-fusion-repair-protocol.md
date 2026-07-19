# TransnnMIL fusion repair: controlled evaluation protocol

Status: draft protocol for PR #41. No performance improvement is claimed.

## Established defect

The historical two-branch fusion supplies one TransMIL query token and one nnMIL key/value token to multi-head attention. With one key, softmax attention is exactly one and the output is invariant to the TransMIL query. Historical checkpoints remain reproducible and are not modified.

## Models

Use the same extracted Phikon features, train/validation split, seed, class weights, optimizer, scheduler, epoch budget, and early-stopping rule for:

1. `transnnmil` — historical implementation;
2. `transnnmil_concat_experimental` — concatenation plus projection;
3. `transnnmil_gate_experimental` — sample-specific normalized gate;
4. `transnnmil_branch_attention_experimental` — branch identity embeddings, two-token self-attention, and learned pooling;
5. standalone TransMIL and nnMIL controls using their existing implementations.

The experimental names are intentionally neutral. None is called corrected or improved before empirical validation.

## Primary endpoint

PANDA validation quadratic weighted kappa under a locked split, followed by repeated-seed reporting. Report every seed and the mean, standard deviation, minimum, and maximum. Do not select the best fusion method using the final reported validation set without a separate confirmation split.

## Required diagnostics

- branch ablation: replace each projected branch vector with zero at inference;
- gradient norms entering each projected branch during training;
- learned branch-pooling weights for gate and attention variants;
- fraction of slides with branch weight above 0.9;
- agreement and disagreement among individual branch predictions and fused predictions;
- parameter count and wall-clock training cost;
- checkpoint save/load equivalence.

## Collapse criterion

A fusion variant is considered practically collapsed when one branch receives more than 0.9 normalized weight for at least 90% of validation slides or when ablating one branch changes fewer than 1% of predictions while ablating the other materially degrades performance.

## Interpretation

A useful fusion result must outperform both individual branches under the same protocol and remain stable across seeds. Beating the historical implementation alone is insufficient because the historical implementation contains a known fusion defect. A simple concat or gate control matching the attention variant would argue against claiming attention-specific novelty.

## Current safety boundary

Topology is disabled for all experimental variants until the historical unregistered forward-created topology projection is repaired and independently tested. Hierarchical mode is permitted only with coordinates and requires separate smoke coverage before use in a reported experiment.

## Merge gate

PR #41 remains a draft until its CI workflow passes and the targeted regression suite confirms historical query invariance, dual-branch gradient flow, factory construction, topology rejection, and checkpoint round-trip equivalence. Model-performance claims require the controlled experiment above and are not a merge prerequisite for retaining the defect regression and experimental code.
