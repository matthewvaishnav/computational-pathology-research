# SCORPION capacity-matched claim boundary

Status: validated 175-cell evidence package.

Execution commit: `0adea50f1ef22865969109f1834a3c175e3f8b43`.

## Supported within the registered paired-acquisition protocol

- Relative to the equal-parameter two-branch control with scanner objectives
  disabled, the full model reduced biological-branch linear scanner-probe
  balanced accuracy by -0.310833 (fold-aware 95% interval
  -0.334612 to -0.285787). Every fold mean was
  negative (-0.332400 to -0.274667).
- Average and worst same-region retrieval were preserved within the
  preregistered 0.02 noninferiority margin. Their near-zero point contrasts
  (0.000042 and 0.000000) are
  not improvements; retrieval was near ceiling and is correspondingly
  insensitive to small differences.
- Full-model acquisition-branch scanner recoverability exceeded 0.2 chance by
  0.656500 (interval 0.623918 to
  0.687200); this supports retention of scanner information
  in the explicit acquisition branch under this protocol.
- Removing scanner-dependence supervision materially weakened scanner
  suppression: full minus ablation was -0.316417
  (interval -0.339280 to -0.292783).
- Removing the acquisition classifier did not produce an interval-supported
  biological-branch scanner difference, but reduced acquisition-branch scanner
  recoverability: full minus ablation was 0.067000
  (interval 0.052355 to 0.082449).

## Unsupported necessity claims and regressions

- The adversarial objective was not necessary for scanner suppression in this
  experiment. Full minus no-adversary scanner recoverability was
  0.099417 (interval 0.083918 to
  0.117067), meaning the no-adversary ablation had lower
  recoverability under this configuration.
- Cross-covariance suppression was not shown necessary for scanner suppression:
  full minus no-cross-covariance was -0.001417
  (interval -0.011404 to
  0.007840). Small cosine changes do not prove
  biological preservation.
- The historical `paired_reference` comparator is not capacity matched. It has
  1,051,648 parameters versus 1,550,026 for every two-branch variant; capacity
  claims use `two_branch_no_scanner_objectives`.

## Prohibited extrapolations

This package does not establish pure biological factors, complete scanner
invariance, complete disentanglement, information-theoretic independence,
causality beyond the paired intervention, clinical utility, diagnostic
improvement, patient benefit, deployment readiness, or universal necessity of
any objective. It does not treat cosine as proof of biological preservation and
does not restore the public paper.
