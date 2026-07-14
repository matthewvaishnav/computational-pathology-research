# Estimability rules

## Separate algebra from support

The audit separates three questions:

1. **Algebraic estimability**: a requested contrast lies in the row space of an
   explicitly reference-coded fixed-effect design matrix.
2. **Matched structural support**: the contrast is carried by the required
   holding-constant strata and at least two independent biological units.
3. **Operational validity**: recorded order, batch, workflow-definition, and
   counterbalancing metadata do not hide a known operational limitation.

A full-rank headline is insufficient when one biological unit carries the
contrast. Conversely, an incomplete but connected design may support selected
main-effect contrasts without supporting every interaction.

Structural estimability is contrast-specific. A global design-quality label
summarizes coverage and confounding but does not override a supported contrast.
An extra non-bridging biological unit is a reported non-supporter, not a reason
to erase support from two or more independent bridging units.

## Fixed-effect design matrix

The main matrix contains an intercept and reference-coded terms for:

- biological unit;
- preparation condition;
- scanner; and
- site/workflow.

Factors with one level contribute no dummy columns only when their effect was
explicitly not requested. Rank is computed exactly over rational numbers.
Columns are added in a deterministic order, and columns that do not increase
rank are reported as aliased.

Rank and residual degrees of freedom are row-level quantities. Repeated scans,
workflow exposures, or intentional acquisitions of the same prepared section
can increase row count and row-level residual degrees of freedom without adding
an independent biological unit. Row-level residual degrees of freedom are not
independent biological degrees of freedom and cannot compensate for a
two-biological-unit design.

Repeat-event identity uses section, preparation, scanner, and workflow. Scan
batch and acquisition order are execution attributes, so moving a repeat to a
different batch does not make it a new base physical condition. This declared
identity cannot detect source events relabeled across all identity fields;
immutable source-event provenance or checksums remain an external requirement.

A contrast vector `c` is algebraically estimable exactly when appending `c` as a
row does not increase matrix rank. This test is applied to every pair of levels
for each requested factor.

## Pairwise and higher-order crossing

For all six factor pairs, the audit reports observed and possible level
combinations, an exact coverage fraction, minimum observation and biological
replication per observed cell, factor-level degrees, and bipartite connected
components.

The audit separately reports biological-unit x preparation x scanner,
preparation x scanner x post-preparation-workflow, and the requested four-factor
product. Each higher-order result contains observed and possible combinations,
exact coverage, minimum row replication in observed cells, and exact missing
combinations. Complete pairwise tables do not imply complete higher-order
coverage.

Full factorial balance is not required. A connected partial crossing can
support selected additive contrasts, but conclusions are restricted to the
observed connected support.

The global crossing labels are:

- `fully crossed`: the requested full-factor product is complete;
- `pairwise complete, higher-order incomplete`: all relevant pairs are complete
  but at least one required higher-order cell is missing;
- `partially crossed`: the support is connected but pairwise incomplete; and
- `nested/confounded`: nesting or aliasing destroys a requested contrast.

These labels describe the global matrix. For example, a connected three-of-four
preparation/scanner table can retain estimable additive preparation and scanner
contrasts while its preparation-by-scanner interaction remains not estimable.

## Nesting and aliasing

Factor A is exactly nested in factor B when every A level occurs with exactly
one B level. If the reverse is also true, the factors are one-to-one aliased.
If only some A levels have one B neighbor, the audit reports partial nesting and
the exact singleton-level count rather than applying an arbitrary threshold.

Requested effects fail when preparation and scanner are exactly nested in
either direction, site/workflow is nested with preparation or scanner, biology
provides fewer than two bridging units for a requested preparation contrast, or
the requested fixed-effect matrix is deficient.
Exact nesting inside a one-level unrequested factor is diagnostic, not a
competing-parameter failure. Rank deficiency confined to unrequested contrasts
is reported without blocking an algebraically estimable requested contrast.

Degree-one or non-bridging levels are evaluated at the affected contrast. They
do not automatically invalidate the entire matrix when that contrast retains
exact rank, connected support, and at least two independent bridging biological
units. A block assigned to only one preparation receives a specific
block/preparation diagnosis. A one-to-one block/biological-unit mapping is also
reported and does not count as two independent replication layers.

Preparation and scan batches are checked for exact one-to-one aliasing, partial
association, and support sufficient for separate nuisance adjustment. Exact
aliasing is an operational qualification: two batch labels that encode the same
partition are not two independent nuisance axes.

## Connectedness

The global incidence graph contains namespaced factor-level nodes and observed
biology-preparation, preparation-scanner, and scanner-workflow edges. All levels
needed for a cross-component contrast must belong to one component.
Disconnected components block cross-component estimation even if each component
is internally balanced.

## Main-effect support

- **Preparation**: each level contrast requires matched serial sections from the
  same biological unit and block under multiple preparations, with at least two
  independent biological units. Exact scanner/workflow holding strata within
  the same block, preparation batch, and scan batch distinguish direct
  structural support from assumption-dependent support. These are different
  sections, not the same tissue instance or pixels; interpretation is conditional
  on the declared matched-serial-section design and prospective section
  assignment.
- **Scanner**: each contrast requires the same biological unit and preparation
  across scanners. Same-section multi-scanner anchors within the same workflow
  and scan batch distinguish direct physical-material support from
  section/batch-exchangeability assumptions.
- **Post-preparation workflow**: each contrast requires matching biological-unit,
  preparation, and scanner strata across workflows and at least two biological
  units. Direct support holds the prepared section, scanner, and scan batch
  constant. The verdict applies only to the prospectively declared process
  stage. A post-preparation acquisition-workflow bridge does not identify an
  upstream site preparation effect.

For every main-effect contrast, support accounting includes independent
biological units, blocks, participating sections, bridge observations or bridge
strata, preparation batches, scan batches, and non-supporting biological units.
For preparation, participating sections and matched serial-section pairs are
reported separately: no individual section spans mutually exclusive
preparations. An explicit pair is counted only when a held
biology/block/preparation-batch/scan-batch stratum contains exactly one section
from each contrasted preparation. Strata with multiple candidates require
prospective section-order/distance pairing metadata and are reported as
ambiguous rather than expanded into Cartesian pseudo-pairs.

## Interactions

The audit constructs separate reference-coded blocks for:

- preparation x scanner;
- scanner x site/workflow; and
- preparation x site/workflow.

It tests exact difference-in-differences contrasts, expanded-matrix rank,
row-level residual degrees of freedom, factor-pair coverage, and distinct
biological units per cell. A direct interaction verdict additionally requires at least two
biological units to span every four-cell difference-in-differences rectangle in
the relevant held block, workflow, preparation-batch, and scan-batch strata;
for preparation interactions, each prepared section must bridge the paired
downstream scanner or workflow levels. Marginal cell replication without these
within-unit rectangles is assumption-dependent. An interaction is not called
estimable merely because its main effects are estimable. Interactions are
diagnostic unless explicitly requested with `--request-interaction`.

For requested interactions, block, preparation-batch, scan-batch, and optional
fold support are counted only when that stratum carries the complete direct
rectangle. Fewer than two carrier strata are reported as qualifications even
when two biological units make the structural interaction verdict direct.

Each interaction report also exposes biological-unit, block, section, bridge,
batch, and complete-rectangle counts. Preparation interactions use rectangles
across matched serial sections within the same block; scanner-by-workflow
rectangles can be same-section rectangles. Rectangle counts do not create
additional independent biological units.

## Operational validity

Structural estimability and operational validity are parallel outputs. The
operational layer uses the following bounded terms:

- `no identified operational blocker`;
- `order-confounded` when a contrast direction is perfectly separated or
  deterministically earlier/later within every matched bridge;
- `order-imbalanced` when ordering is asymmetric but not perfectly separated;
- `batch-confounded` when a target or nuisance axis is inseparable from recorded
  batch structure;
- `workflow under-specified` when the declared post-preparation operation lacks
  its required prospective definition; and
- `randomization/counterbalancing unverified` or `order metadata insufficient`
  when the recorded metadata cannot establish execution balance.

The acquisition-order audit checks biological unit, block, scanner, workflow,
preparation, scan batch, and preparation batch for perfect separation,
deterministic within-bridge ordering, monotonic block ordering, and absent counterbalancing. An order
finding does not prove causal bias; it identifies an unresolved operational
alternative explanation. Structural support may remain direct while its
operational validity is qualified.

## Verdicts

- `directly estimable`: algebraically estimable, matched holding-constant
  support exists, and every contrast has at least two biological supporters.
- `estimable with partial crossing`: the connected incomplete design supplies
  multi-unit bridges and exact row-space support, but not complete crossing.
- `estimable only under modeling assumptions`: algebraic support depends on
  additivity, between-unit adjustment, section exchangeability, or another
  stated assumption rather than the required direct bridge.
- `not estimable`: nesting, aliasing, disconnection, rank failure, missing
  within-unit support, or single-unit support blocks the contrast.

The declared minimum of two direct biological rectangle supporters is a bare
structural floor used by this audit, not a sample-size recommendation, power
target, or basis for external generalization.

The future scanner-suppressed residual targets take the worse of the relevant
preparation/workflow verdict and the scanner verdict. They remain future
attribution tests and are not findings of this design audit.

## Fail-closed boundary

Malformed schemas and identities exit as input errors. Well-formed but
nonidentifiable designs return a deterministic `not_identifiable` result and a
nonzero status. Rank-engine failure is fatal. No model-quality assumption can
recover an effect that the sampling design does not identify.
