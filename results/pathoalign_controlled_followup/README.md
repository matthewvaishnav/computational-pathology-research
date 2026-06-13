# PathoAlign compute-controlled follow-up

This directory records the compact result tables used by the June 13, 2026 paper update.

The experiments were completed locally with the v7 decoupled-resource runner and the v8 exact anchor-repetition runner. The files here contain the corrected aggregate evidence used in the paper:

- compute-controlled point boundaries after fixing floating-point grouping;
- monotone seed-bootstrap boundary intervals and observational-data effects;
- the high-diversity/high-repetition corner of the five-seed exact-schedule pilot.

The seed-level raw CSV files are not included in this compact snapshot. The paper therefore treats the compute-controlled observational-data effect as suggestive rather than confirmatory. The exact-schedule pilot is also described as a pilot because total paired presentations still vary with `N_p * R_p`.

Frozen protocol for the controlled boundary study:

- overlap: `0.975`;
- nonlinear mixing: `true`;
- optimizer steps: `320`;
- observational batch size: `128`;
- paired batch size: `64`;
- seeds: `101` through `120`;
- methods: `hybrid_curriculum`, `pair_consistency`.

Frozen pilot protocol:

- `N_u`: `750`, `1500`;
- `N_p`: `50`, `125`, `225`;
- `R_p`: `32`, `64`, `128`;
- optimizer steps: `320`;
- seeds: `201` through `205`;
- methods: `hybrid_curriculum`, `pair_consistency`.
