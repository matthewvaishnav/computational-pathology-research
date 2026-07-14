# Factor definitions

## Physical-material hierarchy

The audit keeps the following identities distinct:

```text
biological_unit
  -> block_id
    -> section_id
      -> preparation_condition
        -> scanner x post_preparation_workflow acquisition condition
          -> repeat_acquisition_id
```

`site_workflow`, preparation batch, scan batch, acquisition order, and
replicate identifiers are recorded alongside this hierarchy. Acquisition rows,
tiles, regions, repeated scans, and model folds do not become independent
biological replicates.

## Core factors

### biological_unit

The biological anchor used for within-unit contrasts. Preferred examples are a
tissue block, a mapped physical tissue region, or explicitly matched serial
sections from one block. `block_id` and `section_id` retain the physical
substructure within that anchor.

Each block must map to one biological unit. Each section must map to one block,
one biological unit, one preparation condition, one preparation batch, and one
technical-replicate identity. A prepared section may then be observed across
multiple scanners and workflows.

Blocks and biological units are separate hierarchy fields, but their labels do
not automatically provide separate replication. If each biological unit has
exactly one block and each block belongs to exactly one biological unit, the two
fields are one-to-one aliases for this matrix: block effects cannot be separated
from biological-unit effects, and the block count cannot be added to the
biological-unit count. If blocks are assigned to only one preparation, that
block/preparation structure is reported explicitly rather than hidden behind a
generic support warning.

### preparation_condition

A prospectively named intervention or condition upstream of acquisition, such
as fixation protocol, staining protocol or batch policy, section thickness,
coverslip protocol, or slide-processing condition. Preparation is not a scanner
attribute and must not be inferred from scanner identity.

`preparation_batch` is retained separately. If preparation batches are nested
within preparation, their separation is assumption-dependent even when the
four-factor matrix is otherwise full rank. A contrast carried by fewer than two
recorded batches is also flagged.

Only interventions physically assignable at or after sectioning can use the
same-block matched-serial-section bridge directly. Fixation or block-processing
interventions require prospectively split material and separate blocks; without
another valid anchor, their separation is between-block and assumption-dependent.

### scanner

The acquisition device or acquisition condition. For cross-site estimation, a
scanner level must be meaningfully repeatable across `site_workflow` levels.
If `scanner` is a unique physical device fixed at one site, scanner and site are
nested unless a device is moved or another explicit bridge is introduced. If
`scanner` is only a platform/model label, unrecorded device-instance variation
remains a limitation.

The same prepared section scanned across multiple scanners is stronger support
than different sections sharing only biological and preparation labels.

### site_workflow (`post_preparation_workflow`)

In this package, `site_workflow` is a schema-compatible column name for a
prospectively declared **post-preparation workflow**. It is not a generic site,
laboratory, staining, fixation, or preparation-site label, and it is not
equivalent to scanner. A direct structural workflow contrast requires the same
prepared section, scanner, and scan batch under multiple workflow levels. A
broader biological-unit/preparation/scanner bridge is assumption-dependent
because it exchanges sections or batches.

Moving an already prepared slide across sites can identify acquisition-workflow
differences, but it cannot identify upstream site preparation effects.
Consequently, every workflow level must prospectively specify:

- operator or operator pool;
- transfer/storage condition;
- post-preparation handling;
- post-processing pipeline;
- exposure order;
- timing window;
- whether the operation is destructive or non-destructive; and
- carryover risk.

These definitions may be supplied as controlled columns or a versioned workflow
manifest referenced by the matrix. Until they are supplied, workflow is
operationally under-specified even if its structural bridge is complete. The
checked example uses two repeatable post-preparation acquisition-workflow
labels on already prepared sections; its verdict must not be generalized to a
lab, fixation, staining, or other upstream site effect. A workflow label may
aggregate multiple operational causes and is not a single causal mechanism.

## Blocking and replicate fields

- `preparation_batch`: prospective preparation batch identifier.
- `scan_batch`: acquisition session or scan-batch identifier.
- `acquisition_order`: positive canonical integer that is comparable and unique
  across workflow levels within one scan batch, so scanner/workflow ordering can
  be audited rather than partitioned into incomparable workflow-specific
  sequences.
- `repeat_acquisition_id`: optional identifier for an intentional repeat under
  the same section, preparation, scanner, and workflow condition. If present,
  it is nonempty on every row; `R1` denotes the first or only acquisition.
  Distinct repeats use distinct values such as `R1` and `R2`, including when
  the repeats occur in different scan batches.
- `technical_replicate`: globally unique physical-section technical-unit
  identity; it maps to one `section_id`. Acquisition repeats are distinguished
  by `repeat_acquisition_id`, and neither field is counted as an independent
  biological unit.
- `biological_replicate`: stable replicate label mapping one-to-one to
  `biological_unit` in the submitted matrix.

The presence of order and batch identifiers makes randomization auditable; it
does not prove that randomized execution occurred.
If scan batches are nested within scanner or workflow, the affected separation
is likewise assumption-dependent.

`scan_batch` and `acquisition_order` are execution attributes, not part of base
physical-event identity. Changing only the observation ID, scan batch, or
acquisition order does not remove the need to identify a repeat. An intentional
rescan needs a distinct `repeat_acquisition_id`; it remains technical
replication and may increase row-level residual degrees of freedom without
increasing biological support.

These checks depend on the submitted canonical identities. They cannot detect a
source acquisition that has been relabeled under different section, preparation,
scanner, or workflow values. Executed studies therefore require immutable
source-event provenance, such as a source acquisition ID or file checksum, in
addition to this matrix-level audit.

Preparation and scan batches are also separate bookkeeping fields, not presumed
independent nuisance axes. A one-to-one preparation-batch/scan-batch mapping is
reported as exact aliasing and cannot support separate adjustment of the two
batch effects. Partial association is reported separately from batch structure
that is sufficiently crossed for distinct nuisance adjustment. A constant
batch axis is likewise not separately adjustable from the intercept.
