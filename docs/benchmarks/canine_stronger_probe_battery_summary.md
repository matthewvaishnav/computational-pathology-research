# Canine stronger probe battery

## Purpose

This benchmark attacks the Paired-Acquisition Neural Factorization canine branch-separation result with stronger probes than the default linear identity audit.

The battery evaluates whether scanner/acquisition identity remains suppressed in the biological branch and recoverable in the acquisition branch under multiple probe families.

## Probe setup

Representations:

- raw DINOv2 features;
- paired-reference projected features;
- Paired-Acquisition Neural Factorization biological features;
- Paired-Acquisition Neural Factorization acquisition features.

Targets:

- `scanner_id`, evaluated with `sample_id`-blocked cross-validation;
- `sample_id`, evaluated with `scanner_id`-blocked cross-validation.

Probes run in this completed sweep:

- linear logistic probe;
- random forest probe;
- k-nearest-neighbor probe.

Random-label refits were intentionally skipped in this sweep for speed. Random-label controls were already measured in the identity-audit sweeps and remained near chance.

## Target summary

| Target | Representation | Runs | Accuracy | Accuracy std | Balanced accuracy | Majority baseline |
|---|---|---:|---:|---:|---:|---:|
| sample_id | Raw DINOv2 features | 3 | 0.941946 | 0.039335 | 0.932063 | 0.080745 |
| sample_id | Paired-reference features | 75 | 0.982801 | 0.012843 | 0.979788 | 0.080745 |
| sample_id | Paired-Acquisition Neural Factorization biological features | 75 | 0.989164 | 0.007756 | 0.986343 | 0.080745 |
| sample_id | Paired-Acquisition Neural Factorization acquisition features | 75 | 0.236147 | 0.115624 | 0.226930 | 0.080745 |
| scanner_id | Raw DINOv2 features | 3 | 0.661863 | 0.179844 | 0.661863 | 0.200000 |
| scanner_id | Paired-reference features | 75 | 0.573419 | 0.126075 | 0.573419 | 0.200000 |
| scanner_id | Paired-Acquisition Neural Factorization biological features | 75 | 0.254705 | 0.035569 | 0.254705 | 0.200000 |
| scanner_id | Paired-Acquisition Neural Factorization acquisition features | 75 | 0.970756 | 0.005956 | 0.970756 | 0.200000 |

## Per-probe scanner identity results

| Probe | Raw DINOv2 | Paired reference | Paired-Acquisition Neural Factorization biological | Paired-Acquisition Neural Factorization acquisition |
|---|---:|---:|---:|---:|
| kNN | 0.502112 | 0.437585 | 0.242345 | 0.972730 |
| Linear | 0.856646 | 0.739260 | 0.301297 | 0.967294 |
| Random forest | 0.626832 | 0.543414 | 0.220472 | 0.972243 |

## Per-probe biological sample identity results

| Probe | Raw DINOv2 | Paired reference | Paired-Acquisition Neural Factorization biological | Paired-Acquisition Neural Factorization acquisition |
|---|---:|---:|---:|---:|
| kNN | 0.977391 | 0.998559 | 0.998589 | 0.185381 |
| Linear | 0.948820 | 0.969232 | 0.981486 | 0.393998 |
| Random forest | 0.899627 | 0.980611 | 0.987419 | 0.129063 |

## Interpretation

The stronger probe battery preserves the same separation pattern observed in the default identity audit.

For scanner prediction under sample-blocked cross-validation, Paired-Acquisition Neural Factorization biological features remain close to chance and far below raw DINOv2 and paired-reference features. Averaged across linear, random-forest, and kNN probes, scanner prediction is 0.254705 for the Paired-Acquisition Neural Factorization biological branch, compared with 0.661863 for raw DINOv2, 0.573419 for paired-reference features, and 0.970756 for the Paired-Acquisition Neural Factorization acquisition branch.

For sample prediction under scanner-blocked cross-validation, Paired-Acquisition Neural Factorization biological features preserve biological identity. Averaged across the same probes, sample prediction is 0.989164 for the Paired-Acquisition Neural Factorization biological branch, compared with 0.941946 for raw DINOv2 and 0.236147 for the Paired-Acquisition Neural Factorization acquisition branch.

The acquisition branch shows the opposite behavior: scanner identity is strongly recoverable, while sample identity is much weaker than in the biological branch. This supports the separation interpretation rather than a simple information-deletion interpretation.

## Clean result statement

Under stronger linear, random-forest, and kNN probes across folds and seeds, Paired-Acquisition Neural Factorization biological features remain weakly scanner-decodable but strongly sample-decodable, while Paired-Acquisition Neural Factorization acquisition features remain strongly scanner-decodable and weakly sample-decodable.

## Claim boundary

This is a probe-based representation-identifiability stress test on an external paired-scanner canine SCC dataset. It is not clinical diagnostic validation, regulatory validation, or evidence of prospective patient benefit.
