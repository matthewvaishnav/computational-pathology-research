# PathoAlign Figure 1 benchmark table

## Figure title

**PathoAlign separates biological identity from acquisition identity in pathology representations.**

## One-sentence caption

Raw DINOv2 features entangle scanner and biological identity; PathoAlign produces a biological branch with low scanner decodability and high biological recoverability, and an acquisition branch with high scanner decodability and low biological recoverability.

## Panel A: identity-audit summary

External multi-scanner canine SCC benchmark; folds 0--4 and seeds 911--915 for projected representations.

| Representation | Runs | Scanner probe ↓ | Region R@1 ↑ | Sample R@1 ↑ | Cross-scanner cosine | Effective rank | Zero-var frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw DINOv2 | 1 | 0.878261 | 0.872547 | 0.985839 | 0.918855 | 442.802 | 0.000000 |
| Paired reference | 25 | 0.745660 | 0.939886 | 0.998847 | 0.781365 | 186.481 | 0.000000 |
| PathoAlign biological branch | 25 | **0.299955** | **0.959463** | **0.998966** | 0.835016 | 180.615 | 0.000000 |
| PathoAlign acquisition branch | 25 | **0.968407** | 0.031155 | 0.420661 | 0.308065 | 47.355 | 0.000000 |

## Panel B: stronger-probe stress test

Linear, random-forest, and kNN probes across folds 0--4 and seeds 911--915. Scanner prediction uses sample-blocked cross-validation; sample prediction uses scanner-blocked cross-validation.

| Target | Raw DINOv2 | Paired reference | PathoAlign biological | PathoAlign acquisition |
|---|---:|---:|---:|---:|
| scanner_id | 0.661863 | 0.573419 | **0.254705** | **0.970756** |
| sample_id | 0.941946 | 0.982801 | **0.989164** | 0.236147 |

## Panel C: nonlinear MLP probe stress test

Full MLP sweep across folds 0--4 and seeds 911--915.

| Target | Raw DINOv2 | Paired reference | PathoAlign biological | PathoAlign acquisition |
|---|---:|---:|---:|---:|
| scanner_id | 0.832050 | 0.740184 | **0.206768** | **0.971538** |
| sample_id | 0.939876 | 0.976666 | **0.986137** | 0.325684 |

## Panel D: reading the pattern

| Representation | Scanner/acquisition identity | Biological identity | Interpretation |
|---|---|---|---|
| Raw DINOv2 | High | High | Entangled representation |
| Paired reference | Moderate/high | High | Better retrieval, still scanner-decodable |
| PathoAlign biological | Low | High | Biological branch suppresses acquisition identity while preserving biology |
| PathoAlign acquisition | High | Low | Acquisition branch retains scanner identity and no longer behaves like a biological retrieval space |

## Clean figure claim

Across five folds and five seeds on an external paired-scanner canine SCC benchmark, PathoAlign suppresses scanner decodability in the biological branch while preserving biological retrieval, and retains scanner decodability in the acquisition branch. This separation survives linear, random-forest, kNN, and nonlinear MLP probe attacks.

## Suggested manuscript caption

**Figure 1. PathoAlign separates biological identity from acquisition identity.** On an external multi-scanner canine SCC benchmark, raw DINOv2 representations make both scanner identity and biological identity recoverable, indicating entanglement. PathoAlign produces a biological branch in which scanner-probe accuracy falls from 0.878 to 0.300 while same-region retrieval rises from 0.873 to 0.959. In contrast, the acquisition branch preserves scanner identity at 0.968 while same-region retrieval falls to 0.031. Stronger probe batteries reproduce the same pattern: under linear, random-forest, and kNN probes, scanner prediction is 0.255 from the biological branch and 0.971 from the acquisition branch; under a nonlinear MLP probe, scanner prediction is 0.207 from the biological branch and 0.972 from the acquisition branch. These results support a branch-separation interpretation rather than simple information deletion.

## Claim boundary

This figure summarizes representation-identifiability and branch-separation evidence on an external paired-scanner canine SCC dataset. It is not clinical diagnostic validation, regulatory validation, or evidence of prospective patient benefit.
