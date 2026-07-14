# Minimum viable design illustrations

These examples define structural support, not universal sample sizes or power
targets. Real designs require domain-specific power, attrition, quality-control,
and feasibility planning after identifiability is established.

## A. Invalid nested design

Two biological units are assigned to `PREP_A` on `SCN_1` at `WF_1`, while two
different units are assigned to `PREP_B` on `SCN_2` at `WF_2`.

Preparation, scanner, and workflow are aliased; biology does not bridge
preparation; and the factor graph may be disconnected. Only a composite
workflow-package difference is describable. No additive model can identify
which named factor generated that difference.

## B. Minimal partially crossed biology/preparation/scanner design

For a design that requests preparation and scanner but holds workflow fixed,
use two biological units and a three-preparation by three-scanner degree-two
cycle:

```text
PREP_A: SCN_1, SCN_2
PREP_B: SCN_2, SCN_3
PREP_C: SCN_3, SCN_1
```

Each observed preparation/scanner edge is supported by both biological units,
and every biological unit contributes all three preparations. Twelve
acquisition rows are the bare structural illustration. The connected partial
crossing can identify supported additive main effects, but missing cells block
an unrestricted preparation-by-scanner interaction.

Invoke a custom audit for this intentionally narrower question with
`--requested-effects preparation,scanner`. This is not a power recommendation.

## C. Site-extended bridge design

Extend a biology/preparation/scanner design with exact shared
`(biological_unit, preparation_condition, scanner)` strata under at least two
workflows. At least two biological units must support each workflow contrast.
The workflow exposure and its process stage must be defined prospectively.

Sharing only preparation and scanner labels while using different biological
units is assumption-dependent. If physical scanners are permanently fixed to
sites, site and device remain nested unless a device moves or a separately
defined repeatable acquisition condition supplies a defensible bridge.

A sparse set of site bridges may identify a site main effect under additivity
while leaving scanner-by-site and preparation-by-site interactions
unidentified. Moving prepared slides across workflows supplies a
post-preparation acquisition bridge, not an upstream site-preparation contrast.

## D. Strong replication design

An illustrative stronger layout could use:

- four or more biological units;
- at least two preparation conditions;
- at least two scanner conditions;
- at least two site/workflow conditions;
- multiple preparation and scan batches; and
- multiple randomized section assignments per biological-unit/preparation cell.

Each prepared section is scanned across scanners, and matched
biology/preparation/scanner strata bridge workflows. Factor-pair cells and
interaction rectangles are supported by multiple biological units and batches.

Increasing these counts improves redundancy but does not by itself prove causal
attribution, eliminate serial-section heterogeneity, verify randomization, or
provide a universal power guarantee.
