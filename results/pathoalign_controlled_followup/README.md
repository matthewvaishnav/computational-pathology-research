# PathoAlign compute-controlled follow-up

This directory records the compact result tables used by the June 15, 2026 paper update.

The experiments were completed locally with the v7 decoupled-resource runner and the v8 exact anchor-repetition runner. The files here contain the corrected aggregate evidence used in the paper:

- compute-controlled point boundaries after fixing floating-point grouping;
- monotone seed-bootstrap boundary intervals and observational-data effects;
- the high-diversity/high-repetition corner of the five-seed exact-schedule pilot;
- a 10-seed matched-budget comparison that fixes total paired presentations and pair-loss updates while changing the allocation between unique anchors and repetition;
- paired seed-level contrasts for the matched-budget allocations.

The seed-level raw CSV files are not included in this compact snapshot. The paper therefore treats the compute-controlled observational-data effect as suggestive rather than confirmatory. The matched-budget result is reported as consistent descriptive evidence because most stratum-specific intervals include zero and the individual tests are not treated as confirmatory after multiple-comparison considerations.

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

- allocations: `50 x 128`, `100 x 64`, `200 x 32`;
- total paired presentations: `6400` in every cell;
- pair-loss updates: `100` in every cell;
- `N_u`: `750`, `1500`;
- seeds: `301` through `310`;
- methods: `hybrid_curriculum`, `pair_consistency`.

Across the four method-by-`N_u` strata, mean universal biological scores were `0.4081`, `0.4243`, and `0.4259` for the three allocations. The two higher-diversity allocations exceeded `50 x 128` in all four strata, while `200 x 32` improved only `0.0016` on average over `100 x 64`.
