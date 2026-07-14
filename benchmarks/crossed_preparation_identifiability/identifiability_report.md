# Crossed-preparation identifiability report

## Status and boundary

- Overall status: **identifiable**
- Primary design classification: **fully crossed**
- Input: `example_design_matrix.csv`
- Input SHA-256: `94f7d0f2d8b9ffbbe7274d36d45f4d484e2334665abc3013a6db77a7474d5074`
- Audit fingerprint: `fbc4b04c71667a178ffc51755e59bcf2c9a020d287f13de6a722e829da8b0d64`
- This report audits structural design support only. It does not load representations, train a model, estimate an effect, or make a causal claim.

## Design summary

- Observations: 32
- Biological units: 2
- Blocks: 2
- Sections: 8
- Preparation levels: 2
- Scanners: 2
- Site/workflow levels: 2
- Preparation batches: 2
- Scan batches: 2
- Base physical acquisition identities: 32
- Intentional repeat-acquisition rows beyond base identities: 0
- Repeat-acquisition identifier recorded: yes

## Factor-level inventory

| Factor | Levels | Observations per level | Biological units per level |
|---|---:|---|---|
| biological unit | 2 | BIO_01:16, BIO_02:16 | BIO_01:1, BIO_02:1 |
| preparation | 2 | PREP_A:16, PREP_B:16 | PREP_A:2, PREP_B:2 |
| scanner | 2 | SCN_1:16, SCN_2:16 | SCN_1:2, SCN_2:2 |
| site/workflow | 2 | WF_POST_A:16, WF_POST_B:16 | WF_POST_A:2, WF_POST_B:2 |

## Pairwise crossing

| Factor pair | Observed / possible | Coverage | Minimum observations | Minimum biological units | Components |
|---|---:|---:|---:|---:|---:|
| biological unit x preparation | 4 / 4 | 1.000000 | 8 | 1 | 1 |
| biological unit x scanner | 4 / 4 | 1.000000 | 8 | 1 | 1 |
| biological unit x site/workflow | 4 / 4 | 1.000000 | 8 | 1 | 1 |
| preparation x scanner | 4 / 4 | 1.000000 | 8 | 2 | 1 |
| preparation x site/workflow | 4 / 4 | 1.000000 | 8 | 2 | 1 |
| scanner x site/workflow | 4 / 4 | 1.000000 | 8 | 2 | 1 |

## Higher-order crossing

| Factor product | Observed / possible | Coverage | Minimum observations | Missing combinations |
|---|---:|---:|---:|---|
| biological unit x preparation x scanner | 8 / 8 | 1.000000 | 4 | none |
| preparation x scanner x site/workflow | 8 / 8 | 1.000000 | 4 | none |
| biological unit x preparation x scanner x site/workflow | 16 / 16 | 1.000000 | 2 | none |

Full crossing is assigned only from the complete requested factor product; pairwise completeness alone is insufficient.

## Connectedness and nesting

- Global factor-incidence components: 1
- All factor levels connected: yes
- Exact or partial nesting relationships: none
- Block versus biological unit: exact one-to-one alias; these are not independent replication layers when one-to-one aliased.
- Block versus preparation: fully crossed.
- Technical replicate versus section: exact one-to-one alias; technical repeats never add independent biological support.

## Rank and interaction summary

- Main-effect rank: 5 / 5
- Main-effect row-level residual degrees of freedom: 27
- Unique design rows: 16; unique-design residual degrees of freedom: 11
- Row-level residual degrees of freedom are not independent biological degrees of freedom. Repeated scans can increase row-level n without increasing biological replication, and residual df cannot compensate for two biological units.
- Main-effect aliased columns: none

| Interaction | Structural verdict | Operational validity | Rank | Row-level residual df | Minimum biological units per cell | Minimum direct biological rectangle supporters |
|---|---|---|---:|---:|---:|---:|
| preparation_scanner | directly estimable | randomization/counterbalancing unverified | 6 / 6 | 26 | 2 | 2 |
| scanner_site_workflow | directly estimable | workflow under-specified | 6 / 6 | 26 | 2 | 2 |
| preparation_site_workflow | directly estimable | workflow under-specified | 6 / 6 | 26 | 2 | 2 |

## Per-contrast structural support and operational validity

| Contrast | Structural verdict | Operational validity | Biological units | Blocks | Sections | Bridges | Matched serial-section pairs | Complete rectangles | Preparation batches | Scan batches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Preparation main effect: PREP_A vs PREP_B | directly estimable | randomization/counterbalancing unverified | 2 | 2 | 8 | 16 | 4 | 0 | 2 | 2 |
| Scanner main effect: SCN_1 vs SCN_2 | directly estimable | randomization/counterbalancing unverified | 2 | 2 | 8 | 16 | 0 | 0 | 2 | 2 |
| Post-preparation workflow main effect: WF_POST_A vs WF_POST_B | directly estimable | workflow under-specified | 2 | 2 | 8 | 16 | 0 | 0 | 2 | 2 |
| Preparation x scanner | directly estimable | randomization/counterbalancing unverified | 2 | 2 | 8 | 0 | 0 | 8 | 2 | 2 |
| Scanner x post-preparation workflow | directly estimable | workflow under-specified | 2 | 2 | 8 | 0 | 0 | 8 | 2 | 2 |
| Preparation x post-preparation workflow | directly estimable | workflow under-specified | 2 | 2 | 8 | 0 | 0 | 8 | 2 | 2 |

## Contrast verdicts

| Contrast | Structural/integrated verdict | Operational validity | Boundary |
|---|---|---|---|
| Biology-controlled preparation effect | directly estimable | randomization/counterbalancing unverified | Structural support only; no outcome or effect was measured. |
| Biology/preparation-controlled scanner effect | directly estimable | randomization/counterbalancing unverified | Structural support only; no outcome or effect was measured. |
| Biology/preparation/scanner-controlled site/workflow effect | directly estimable | workflow under-specified | Structural support applies only to the declared repeatable site/workflow exposure; no outcome was measured, and a post-preparation bridge does not identify an upstream site preparation effect. |
| Future scanner-suppressed residual association with preparation | directly estimable | future test only | The design verdict is the worse of the preparation and scanner prerequisites; scanner suppression and residual association remain untested. |
| Future scanner-suppressed residual association with workflow | directly estimable | future test only | The design verdict is the worse of the workflow and scanner prerequisites; scanner suppression and residual association remain untested. |

## Randomization and blocking metadata

- Required structural control fields are present.
- Recognized optional controls present: repeat_acquisition_id
- Recognized optional controls not recorded: operator_id, preparation_order, scanner_order, temporal_window, section_order, section_distance, fold_id, registration_quality
- Randomized execution verified: no; identifiers make a future execution audit possible but do not prove randomization.

### Acquisition-order assessment

| Factor | Status | Matched-stratum direction counts |
|---|---|---|
| biological_unit | counterbalancing demonstrated | BIO_01<BIO_02:8, reverse:8, interleaved:0 |
| block_id | counterbalancing not demonstrated | BLK_01<BLK_02:0, reverse:0, interleaved:0 |
| preparation_condition | counterbalancing demonstrated | PREP_A<PREP_B:8, reverse:8, interleaved:0 |
| scanner | counterbalancing demonstrated | SCN_1<SCN_2:8, reverse:8, interleaved:0 |
| site_workflow | counterbalancing demonstrated | WF_POST_A<WF_POST_B:8, reverse:8, interleaved:0 |
| preparation_batch | counterbalancing not demonstrated | PB_1<PB_2:0, reverse:0, interleaved:0 |
| scan_batch | order metadata insufficient | acquisition_order is comparable within, not between, scan batches |

Order findings are operational diagnostics only; they do not establish causal bias.

### Batch and hierarchy relationships

- Preparation batch versus scan batch: fully crossed (4/4 combinations); independent enough for separate nuisance adjustment structurally.
- Batch counts are bookkeeping/support counts. They are not independent nuisance axes when an exact alias or partial association is reported.

### Workflow boundary

- WF_POST_A and WF_POST_B denote a post-preparation workflow factor only. The operator or operator pool, transfer/storage condition, post-preparation handling, post-processing pipeline, exposure order, timing window, destructive status, and carryover risk require prospective specification.
- Workflow levels may aggregate multiple operational causes; they are not a single causal mechanism and do not identify upstream preparation-site effects.

## Findings

- Blocking findings: none

### Warnings and design qualifications

- `block_biological_unit_one_to_one_alias`: block and biological unit are not independent replication layers
- `optional_control_metadata_not_recorded`: fold_id
- `optional_control_metadata_not_recorded`: operator_id
- `optional_control_metadata_not_recorded`: preparation_order
- `optional_control_metadata_not_recorded`: registration_quality
- `optional_control_metadata_not_recorded`: scanner_order
- `optional_control_metadata_not_recorded`: section_distance
- `optional_control_metadata_not_recorded`: section_order
- `optional_control_metadata_not_recorded`: temporal_window
- `preparation_condition_semantics_require_intervention_stage_documentation`
- `randomization_execution_not_verified`
- `scanner_level_semantics_require_device_instance_documentation`
- `serial_sections_not_identical_pixels`
- `site_workflow_semantics_require_process_stage_documentation`

## Deterministic adversarial and regression tests

- Full suite passed: 50 / 50
- Preserved core regression suite passed: 38 / 38
- Required negative cases passed: 23 / 23
- Required patch regression cases passed: 12 / 12
- Temporary fixtures removed: yes

## Limitations

- Structural identifiability is not statistical power or a power guarantee.
- Balanced crossing does not prove causal attribution.
- Matched serial sections are not identical cells, regions, or pixels.
- A same-block serial-section preparation bridge applies only to interventions physically assignable at or after sectioning.
- Site/workflow effects may aggregate multiple upstream factors.
- Biological heterogeneity can remain within blocks and sections.
- Acquisition, preparation, batch, order, and operator metadata must be recorded prospectively.
- No model quality can recover an effect that the sampling design does not identify.
- A scanner level reused across sites must denote a defensible repeatable acquisition condition; unique devices fixed at sites remain nested.
- The checked example crosses a post-preparation acquisition-workflow label; it does not identify an upstream site preparation effect.
- Randomization metadata availability does not prove randomized execution.
- Row-level residual degrees of freedom are not independent biological degrees of freedom.
- Declared repeat-acquisition identifiers document technical repeats but cannot prove source-file identity without immutable acquisition provenance.

The report supplies design support for future attribution tests only. It does not show that preparation, scanner, or workflow effects exist.
