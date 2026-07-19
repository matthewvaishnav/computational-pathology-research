# Acquisition Factor Swapping Audit Report

## Run Status

- Dataset: external multi-scanner canine cutaneous SCC
- Backbone: DINOv2-Base
- Branch: experiment/acquisition-factor-swapping-audit
- Variants: acq_dim16_stronger_xcov, acq_dim8_default, true_pair
- Folds: 0, 1, 2, 3, 4
- Seeds: 911, 912, 913, 914, 915
- Swap types: A (same sample/diff scanner), B (same category/diff sample), C (diff category/diff scanner), D (random)
- Runtime seconds: 0.0
- N probe rows: 35315
- N NN rows: 35315
- N decoder rows: 35315

## Architecture / Artifact Availability

1. **Decoder/composition path available**: YES. The ScorpionProjection model includes a decoder
   (decoder.0: Linear(264->512), decoder.2: Linear(512->768)) that maps
   concat(biological, acquisition) back to the original DINOv2 feature space.
   All checkpoint files contain complete decoder weights.

2. **Direct swapped representations constructed**: YES. For each swap pair, we computed
   z_swap = decoder(concat(bio_i, acq_j)) where bio_i comes from the biological branch
   of the bio-source sample and acq_j comes from the acquisition branch of the
   acq-source sample.

3. **Branch-space/probe-space proxy also used**: YES. In addition to decoder-space
   metrics, we also compute branch-space probe metrics (scanner probe on acquisition
   branch, category probe on biological branch) to provide complementary evidence.

4. **Artifact sources**:
   - true_pair: pair_integrity experiment (acquisition_dim=64)
   - acq_dim8_default: frontier sweep (acquisition_dim=8, xcov=0.05)
   - acq_dim16_stronger_xcov: frontier sweep (acquisition_dim=16, xcov=0.20)

## Key Metrics by Variant and Swap Type

| Variant | Swap Type | Scanner Follow Rate | Category Pres. Rate | Acq Cat Leakage | Bio Scn Leakage |
|---|---:|---:|---:|---:|
| acq_dim16_stronger_xcov | A: Same sample, diff scanner | 0.8757 | 0.4171 | 0.4171 | 0.0320 |
| acq_dim16_stronger_xcov | B: Same category, diff sample | 0.8337 | 0.3718 | 0.3718 | 0.0445 |
| acq_dim16_stronger_xcov | C: Diff category, diff scanner | 0.8558 | 0.3795 | 0.1252 | 0.0379 |
| acq_dim16_stronger_xcov | D: Random acquisition | 0.8529 | 0.4169 | 0.2183 | 0.2127 |
| acq_dim8_default | A: Same sample, diff scanner | 0.8699 | 0.4147 | 0.4147 | 0.0352 |
| acq_dim8_default | B: Same category, diff sample | 0.8375 | 0.3979 | 0.3979 | 0.0445 |
| acq_dim8_default | C: Diff category, diff scanner | 0.8400 | 0.3825 | 0.1165 | 0.0391 |
| acq_dim8_default | D: Random acquisition | 0.8582 | 0.4044 | 0.2196 | 0.1985 |
| true_pair | A: Same sample, diff scanner | 0.8712 | 0.4112 | 0.4112 | 0.0312 |
| true_pair | B: Same category, diff sample | 0.8498 | 0.4229 | 0.4229 | 0.0335 |
| true_pair | C: Diff category, diff scanner | 0.8526 | 0.3814 | 0.1209 | 0.0423 |
| true_pair | D: Random acquisition | 0.8611 | 0.4211 | 0.2280 | 0.1957 |

## Branch-Space Nearest-Neighbor Purity

| Variant | Swap Type | Bio-Space Cat Purity K1 | Bio-Space Cat Purity K5 | Acq-Space Scn Purity K1 | Acq-Space Scn Purity K5 |
|---|---:|---:|---:|---:|
| acq_dim16_stronger_xcov | A: Same sample, diff scanner | 0.9811 | 0.9220 | 0.9093 | 0.8600 |
| acq_dim16_stronger_xcov | B: Same category, diff sample | 0.9827 | 0.9090 | 0.8811 | 0.8200 |
| acq_dim16_stronger_xcov | C: Diff category, diff scanner | 0.9700 | 0.9029 | 0.8902 | 0.8337 |
| acq_dim16_stronger_xcov | D: Random acquisition | 0.9797 | 0.9239 | 0.8834 | 0.8385 |
| acq_dim8_default | A: Same sample, diff scanner | 0.9859 | 0.9258 | 0.9011 | 0.8553 |
| acq_dim8_default | B: Same category, diff sample | 0.9827 | 0.9029 | 0.8715 | 0.8264 |
| acq_dim8_default | C: Diff category, diff scanner | 0.9725 | 0.9036 | 0.8712 | 0.8228 |
| acq_dim8_default | D: Random acquisition | 0.9813 | 0.9205 | 0.8860 | 0.8324 |
| true_pair | A: Same sample, diff scanner | 0.9861 | 0.9247 | 0.8725 | 0.8347 |
| true_pair | B: Same category, diff sample | 0.9827 | 0.9073 | 0.8486 | 0.7871 |
| true_pair | C: Diff category, diff scanner | 0.9704 | 0.9009 | 0.8652 | 0.8073 |
| true_pair | D: Random acquisition | 0.9864 | 0.9243 | 0.8753 | 0.8210 |

## Decoder-Reconstructed Space Metrics

| Variant | Swap Type | Scanner Follow (Decoder) | Category Pres. (Decoder) |
|---|---:|---:|
| acq_dim16_stronger_xcov | A: Same sample, diff scanner | 0.9011 | 0.9912 |
| acq_dim16_stronger_xcov | B: Same category, diff sample | 0.7390 | 0.9804 |
| acq_dim16_stronger_xcov | C: Diff category, diff scanner | 0.7591 | 0.9656 |
| acq_dim16_stronger_xcov | D: Random acquisition | 0.7718 | 0.9677 |
| acq_dim8_default | A: Same sample, diff scanner | 0.9013 | 0.9917 |
| acq_dim8_default | B: Same category, diff sample | 0.7385 | 0.9789 |
| acq_dim8_default | C: Diff category, diff scanner | 0.7399 | 0.9777 |
| acq_dim8_default | D: Random acquisition | 0.7708 | 0.9784 |
| true_pair | A: Same sample, diff scanner | 0.9443 | 0.9827 |
| true_pair | B: Same category, diff sample | 0.7851 | 0.9353 |
| true_pair | C: Diff category, diff scanner | 0.7996 | 0.6953 |
| true_pair | D: Random acquisition | 0.8184 | 0.7578 |

## Validation Checks

- [PASS] swap_type_A_same_sample_diff_scanner_has_examples
- [PASS] swap_type_B_same_category_diff_sample_has_examples
- [PASS] swap_type_C_diff_category_diff_scanner_has_examples
- [PASS] swap_type_D_random_acquisition_has_examples

## Interpretation

1. **Decoder available**: Yes. The ScorpionProjection architecture includes a decoder that maps concat(bio, acq) -> original feature space. Decoder weights are in all checkpoints. Both decoder-space reconstruction and branch-space probe metrics were computed.
2. **Scanner follows acquisition branch**: The scanner follow rate (probe prediction matches acq-source scanner) averages 0.855 across all variants and swap types. This supports factor-like behavior: scanner information follows the swapped acquisition branch.
3. **Category stays with biological branch**: The category preservation rate (probe prediction matches bio-source category) averages 0.402 across all variants and swap types. Category preservation under swap is limited.
4. **Bottleneck variant comparison**: Evaluated variants: acq_dim16_stronger_xcov, acq_dim8_default, true_pair. Comparison across acquisition dimensions (64, 16, 8) assesses whether bottlenecking improves factor behavior by constraining acquisition branch capacity.
   - acq_dim16_stronger_xcov: scanner_follow_rate = 0.855
   - acq_dim8_default: scanner_follow_rate = 0.851
   - true_pair: scanner_follow_rate = 0.859
   - acq_dim16_stronger_xcov: acq_category_leakage = 0.283
   - acq_dim8_default: acq_category_leakage = 0.287
   - true_pair: acq_category_leakage = 0.296
   - acq_dim16_stronger_xcov: bio_scanner_leakage = 0.082
   - acq_dim8_default: bio_scanner_leakage = 0.079
   - true_pair: bio_scanner_leakage = 0.076
5. **Branch-space NN purity**: Bio-space K=1 category purity averages 0.980. Biological neighbors preserve category under acquisition swap.
   Acq-space K=1 scanner purity averages 0.880. Acquisition neighbors follow scanner identity.
6. **Decoder-space evidence**: Scanner prediction on decoder-reconstructed swapped features matches acq source in 0.806 of cases. The decoder-space reconstruction confirms the branch-space findings.

### Overall Assessment

Validation checks: 4/4 passed.

**Bounded conclusion**: This experiment provides evidence about whether the acquisition branch carries manipulable acquisition information. If scanner_follow_rate is high and category_preservation_rate is high, this supports factor-like behavior — the acquisition branch encodes acquisition-relevant information that can be swapped independently of biological content. If these rates are low, the acquisition branch may function more as a discriminative residual that does not cleanly separate from biological content when recombined.

This experiment does NOT establish clinical validity, diagnostic performance, or deployment readiness. It is a research audit of factorization behavior in a controlled multi-scanner setting.

## Claim Boundary

This experiment uses bounded language. We test whether the acquisition branch
behaves like an acquisition factor rather than only a scanner-discriminative
residual bucket. The experiment does NOT make claims about:

- Clinical validation or diagnostic performance
- Patient-care utility or deployment readiness
- Scanner bias being "solved"
- Universal biological factorization
- Breakthrough proven

## Outputs

- acquisition_swapping_raw_metrics.csv
- acquisition_swapping_summary.csv
- acquisition_swapping_nearest_neighbor_metrics.csv
- acquisition_swapping_probe_metrics.csv
- acquisition_swapping_report.md
- experiment_design.json
- run_log.txt

## Exact Retry Command

```powershell
python experiments/paired_acquisition/run_acquisition_factor_swapping_audit.py --variants acq_dim16_stronger_xcov acq_dim8_default true_pair --folds 0 1 2 3 4 --seeds 911 912 913 914 915 --max-swaps-per-type 150 --device cpu
```
