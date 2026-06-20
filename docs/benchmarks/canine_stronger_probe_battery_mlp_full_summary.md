# Canine full MLP stronger-probe battery

## Purpose

This benchmark closes the nonlinear-probe gap in the canine PathoAlign branch-separation result by running a full MLP probe sweep across folds 0--4 and seeds 911--915.

The test asks whether a nonlinear MLP can recover scanner identity from the PathoAlign biological branch, and whether the acquisition branch remains scanner-decodable.

## Setup

Representations:

- raw DINOv2 features;
- paired-reference projected features;
- PathoAlign biological features;
- PathoAlign acquisition features.

Targets:

- `scanner_id`, evaluated with `sample_id`-blocked cross-validation;
- `sample_id`, evaluated with `scanner_id`-blocked cross-validation.

Probe:

- MLP classifier.

Random-label refits were skipped in this full MLP sweep for speed. Random-label controls were measured in the quick stronger-probe run and the identity-audit sweeps.

## Target summary

| Target | Representation | Runs | Accuracy | Accuracy std | Balanced accuracy | Majority baseline |
|---|---|---:|---:|---:|---:|---:|
| sample_id | Raw DINOv2 features | 1 | 0.939876 | NA | 0.932336 | 0.080745 |
| sample_id | Paired-reference features | 25 | 0.976666 | 0.004133 | 0.973679 | 0.080745 |
| sample_id | PathoAlign biological features | 25 | 0.986137 | 0.002750 | 0.983804 | 0.080745 |
| sample_id | PathoAlign acquisition features | 25 | 0.325684 | 0.018723 | 0.303923 | 0.080745 |
| scanner_id | Raw DINOv2 features | 1 | 0.832050 | NA | 0.832050 | 0.200000 |
| scanner_id | Paired-reference features | 25 | 0.740184 | 0.008488 | 0.740184 | 0.200000 |
| scanner_id | PathoAlign biological features | 25 | 0.206768 | 0.006741 | 0.206768 | 0.200000 |
| scanner_id | PathoAlign acquisition features | 25 | 0.971538 | 0.005863 | 0.971538 | 0.200000 |

## Interpretation

The MLP probe does not break the PathoAlign separation result.

For scanner prediction under sample-blocked cross-validation, the MLP recovers scanner identity from raw DINOv2 features at 0.832050 and from paired-reference features at 0.740184. In contrast, scanner prediction from the PathoAlign biological branch is 0.206768, essentially at the five-class chance baseline of 0.200000. The same MLP predicts scanner identity from the PathoAlign acquisition branch at 0.971538.

For biological sample prediction under scanner-blocked cross-validation, the PathoAlign biological branch remains strongly predictive at 0.986137, exceeding raw DINOv2 at 0.939876 and paired-reference features at 0.976666. The acquisition branch is much weaker at 0.325684.

This gives the expected opposite-branch pattern:

- biological branch: scanner identity near chance, sample identity high;
- acquisition branch: scanner identity very high, sample identity much lower.

## Clean result statement

Across five folds and five seeds, a nonlinear MLP probe cannot meaningfully recover scanner identity from PathoAlign biological features beyond chance, while it strongly recovers scanner identity from PathoAlign acquisition features. The same biological features preserve strong sample identity across scanner-blocked evaluation.

## Claim boundary

This is a nonlinear probe-based representation-identifiability stress test on an external paired-scanner canine SCC dataset. It is not clinical diagnostic validation, regulatory validation, or evidence of prospective patient benefit.
