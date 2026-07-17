# Sampling matrix specification

## Required columns

The CSV header must contain each required column exactly once:

1. `observation_id`
2. `biological_unit`
3. `block_id`
4. `section_id`
5. `preparation_condition`
6. `scanner`
7. `site_workflow`
8. `preparation_batch`
9. `scan_batch`
10. `acquisition_order`
11. `technical_replicate`
12. `biological_replicate`
13. `notes`

`repeat_acquisition_id` is an optional physical-repeat column. When present, it
must be populated on every row; use `R1` for the first or only acquisition and a
distinct canonical value for each intentional repeat of the same physical
acquisition condition. If the column is absent, the matrix declares no
intentional same-condition repeats.

Other recognized optional prospective-control columns are `operator_id`,
`preparation_order`, `scanner_order`, `temporal_window`, `section_order`,
`section_distance`, `fold_id`, and `registration_quality`. Unknown columns fail
closed so metadata are not silently ignored. Workflow definitions may instead
reside in a versioned manifest referenced by the study; the required workflow
attributes are listed below. The report distinguishes absent metadata from
recorded but operationally imbalanced execution.

## Canonical values

Required values other than `notes` must be nonempty. Identifier fields use
stable ASCII letters, digits, periods, underscores, and hyphens. Leading or
trailing whitespace, control characters, noncanonical integers, and
normalization/case/punctuation aliases fail closed rather than being silently
normalized.

`acquisition_order` must be a positive canonical integer, comparable across all
workflow levels in a `scan_batch`, and unique within that scan batch. This
shared order domain is required to audit scanner and workflow counterbalancing.

## Identity invariants

- `observation_id` is unique.
- The base physical acquisition identity is the same `section_id`,
  `preparation_condition`, `scanner`, and `site_workflow`. `scan_batch` and
  `acquisition_order` are acquisition attributes, not part of that identity.
- A base physical acquisition identity cannot be duplicated under another
  observation ID unless a present `repeat_acquisition_id` explicitly
  distinguishes intentional repeats.
- Within a base physical identity, `repeat_acquisition_id` values are unique.
  Reusing the same repeat ID with a new observation ID fails closed, even when
  the rows have different scan batches or acquisition orders. Changing only
  observation ID, scan batch, or acquisition order does not distinguish a
  repeat.
- One `block_id` maps to one `biological_unit`.
- One `section_id` maps to one biological unit, block, preparation condition,
  preparation batch, and technical replicate.
- One `technical_replicate` maps to one section.
- One biological-replicate identifier maps to one biological unit and vice
  versa.
- Repeated scanner or workflow rows for one section preserve its physical and
  preparation identities.

These rules prevent duplicate acquisition events from being relabeled as
independent physical observations and prevent technical repeats from being
represented as independent biological observations.

Intentional repeats remain separate acquisition rows, but they are technical
replication of one prepared section. They may increase row-level observation
count and row-level residual degrees of freedom; they do not increase the
number of biological units, blocks, sections, or independent contrast
supporters.

This rule validates declared matrix identities; it is not immutable source
provenance. If a source event is relabeled with different section, preparation,
scanner, or workflow values, the CSV alone cannot establish that the rows came
from one acquisition. An executed study must retain an immutable source-event
ID, source-file checksum, or equivalent lineage record outside this schema.

## Requested effects

Preparation, scanner, and site/workflow are requested by default. Every
requested factor and the biological blocking factor must contain at least two
levels. A custom audit may narrow the requested main effects explicitly with
`--requested-effects`; the audit never infers that an effect is unrequested just
because its factor has one level.

## Structural support

For each requested contrast, the matrix must provide at least two independent
biological units. The audit separately records blocks, preparation batches,
scan batches, sections, and controlled bridge strata; technical replicates do
not increase biological support. If optional `fold_id` metadata are supplied,
fewer than two folds that each span both contrast levels is flagged. Block,
batch, and fold counts are contrast-carrier counts, not unions of row labels.
Support summaries use only rows in contrast-carrying strata, not unrelated rows
from the same biological units.

Matched serial-section pair counts are conservative. A held
biology/block/preparation-batch/scan-batch stratum contributes one explicit pair
only when exactly one section from each contrasted preparation is present.
Multiple candidate sections are not combined as Cartesian pseudo-pairs;
prospective section-order or section-distance metadata are required to resolve
their pairing.

Support is contrast-specific. A biological unit that does not span one
preparation contrast is reported as a non-supporter for that contrast; it does
not invalidate a contrast already supported by at least two independent
bridging units. Likewise, a missing preparation/scanner cell may leave connected
additive main effects estimable with partial crossing while making the
preparation-by-scanner interaction unavailable. The global design summary must
not overwrite those contrast-specific conclusions.

Preparation/scanner coverage is assessed against each requested contrast rather
than by an unconditional degree rule. Multiple scanner neighbors in both
directions provide stronger global support, but a connected incomplete graph can
still support additive preparation and scanner contrasts when their exact
row-space tests and multi-unit physical bridges pass. Missing cells preclude an
unrestricted interaction when its four-cell rectangles are absent. Workflow
levels similarly require cross-factor coverage and exact matched bridges for a
direct workflow contrast; a limited-degree workflow is qualified rather than
automatically rejected when connected multi-unit matched bridges still identify
the requested contrast.

## Pairwise and higher-order crossing

All six core-factor pairs are reported, but pairwise completeness is not full
factorial completeness. The audit also reports at least:

- biological unit x preparation x scanner;
- preparation x scanner x post-preparation workflow; and
- biological unit x preparation x scanner x post-preparation workflow.

Each table includes observed and possible combinations, an exact coverage
fraction, minimum row replication per observed combination, and the exact
missing combinations. `Fully crossed` is allowed only when the requested
full-factor product is complete. Complete pairs with a missing higher-order cell
are classified `pairwise complete, higher-order incomplete`; connected
incomplete pairwise support is `partially crossed`.

## Randomization and blocking metadata

A prospective study should record:

- randomized scan order and scanner order;
- randomized preparation and batch assignment where physically feasible;
- balanced temporal acquisition windows;
- preparation and scan batch IDs;
- operator and workflow IDs;
- technical and biological replicate IDs;
- block, section, section-order, and section-distance metadata; and
- registration quality for matched serial-section analyses.

Every `site_workflow` level in this package must denote a declared
`post_preparation_workflow` and prospectively specify, in controlled metadata or
a versioned workflow manifest:

- operator or operator pool;
- transfer/storage condition;
- post-preparation handling;
- post-processing pipeline;
- exposure order;
- timing window;
- destructive versus non-destructive operation; and
- carryover risk.

The workflow factor is operationally under-specified when these attributes are
absent. It cannot silently stand for an upstream preparation site, and it must
not be interpreted as one causal mechanism.

The audit hard-fails missing required batch/order/replicate fields and flags
missing recognized optional controls. It audits recorded ordering for perfect
separation, deterministic within-bridge direction, monotonic biological-unit,
block, and factor ordering, and absent counterbalancing. Metadata presence is not evidence that randomized
execution occurred, and a structural crossing result is reported separately
from operational order, batch, and workflow qualifications.

Preparation and scan batches are audited jointly. Exact one-to-one aliasing
means they are bookkeeping axes that cannot be adjusted as separate nuisance
effects; partial association and sufficiently independent crossing are reported
separately. The same rule applies to block and biological unit: a one-to-one
mapping supplies one replication layer, not two, and block/preparation nesting
receives a specific diagnosis.

## Checked example

`example_design_matrix.csv` is an illustrative design, not observed data. It
contains 32 acquisition rows from a `2 x 2 x 2 x 2 x 2` structure:

- two biological units;
- two preparations;
- two scanners;
- two site/workflow levels; and
- two serial-section replicates per biological-unit/preparation cell.

Each prepared section is represented across both scanners and both workflows.
Here, `site_workflow` means a repeatable `post_preparation_workflow` applied to
the already prepared section. The example does not cross upstream site-specific
preparation workflows. Distinct preparations use matched serial sections from
the same biological unit and block; they are not the same section or pixels.
The example records `repeat_acquisition_id=R1` and contains no intentional
same-condition repeats.
The example demonstrates deterministic audit behavior only and is not a power
or optimal-sample-size recommendation.
