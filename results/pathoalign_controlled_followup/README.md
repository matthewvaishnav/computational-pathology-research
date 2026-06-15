# PathoAlign compute-controlled follow-up

This directory records the compact result tables used by the June 15, 2026 paper update.

The experiments were completed locally with the v7 decoupled-resource runner and the v8 exact anchor-repetition runner. The files here contain the corrected aggregate evidence used in the paper:

- compute-controlled point boundaries after fixing floating-point grouping;
- monotone seed-bootstrap boundary intervals and observational-data effects;
- the high-diversity/high-repetition corner of the five-seed exact-schedule pilot;
- matched 10-seed allocation experiments at 6,400 and 12,800 paired presentations;
- seed-blocked global allocation contrasts within each budget;
- matched effects of doubling paired exposure from 6,400 to 12,800 presentations.

The raw seed-level experiment CSV files remain local because they are substantially larger than this compact paper-evidence snapshot. The compact tables preserve the reported means, seed-block bootstrap intervals, positive-block fractions, and exact sign-flip tests. Statistical contrasts treat the seed, rather than the four method-by-`N_u` cells within a seed, as the independent block.

## Controlled boundary protocol

- overlap: `0.975`;
- nonlinear mixing: `true`;
- optimizer steps: `320`;
- observational batch size: `128`;
- paired batch size: `64`;
- seeds: `101` through `120`;
- methods: `hybrid_curriculum`, `pair_consistency`.

## Exact-schedule pilot protocol

- `N_u`: `750`, `1500`;
- `N_p`: `50`, `125`, `225`;
- `R_p`: `32`, `64`, `128`;
- optimizer steps: `320`;
- seeds: `201` through `205`;
- methods: `hybrid_curriculum`, `pair_consistency`.

## Matched-budget protocol

Shared controls:

- `N_u`: `750`, `1500`;
- methods: `hybrid_curriculum`, `pair_consistency`;
- seeds: `301` through `310`;
- optimizer steps: `320`;
- observational batch size: `128`;
- observational presentations: `40,960`;
- pair batch size: `64`;
- overlap: `0.975`;
- nonlinear mixing: `true`.

Budget-specific allocations:

| Paired presentations | Allocations | Pair-loss steps |
|---:|---|---:|
| 6,400 | `50 x 128`, `100 x 64`, `200 x 32` | 100 |
| 12,800 | `50 x 256`, `100 x 128`, `200 x 64` | 200 |

Across the four method-by-`N_u` strata, the mean universal biological scores were:

| Budget | 50 anchors | 100 anchors | 200 anchors |
|---:|---:|---:|---:|
| 6,400 | 0.4081 | 0.4243 | 0.4259 |
| 12,800 | 0.4489 | 0.4595 | 0.4619 |

At 6,400 presentations, the seed-blocked 200-versus-50 contrast was `+0.017865`, with a 95% bootstrap interval of `[0.010560, 0.026079]`, a positive difference in all 10 seed blocks, and exact two-sided sign-flip `p = 0.001953`. At 12,800 presentations, the same contrast was `+0.013050`; the bootstrap interval remained positive, but the exact test was less conclusive (`p = 0.076172`).

The 200-versus-100 contrasts were only `+0.001578` and `+0.002422` at the two budgets. This supports diminishing returns beyond approximately 100 anchors in this tested regime rather than a universal optimum.

Doubling total paired presentations increased recovery by `+0.037357` overall, with a 95% seed-bootstrap interval of `[0.030315, 0.043657]`, positive effects in all 10 seed blocks, and exact `p = 0.001953`. All three allocation-specific doubling effects were positive and had exact `p <= 0.003906`.

No allocation mean at either budget crossed the historical `0.50` recovery threshold. The matched experiments therefore establish resource-allocation and exposure effects, not a recovered phase boundary.
