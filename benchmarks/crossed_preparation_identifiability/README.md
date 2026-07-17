# Crossed-preparation identifiability audit

This package audits whether a proposed paired-acquisition sampling matrix can
separately support fixed-effect contrasts for biological identity, preparation
condition, scanner/acquisition condition, and a declared post-preparation
workflow condition before any model training or representation analysis is run.

The motivating question is:

> After controlling tissue identity, does scanner-suppressed representation
> structure track preparation or workflow identity across scanners?

The package evaluates design support for that future question. It does not
measure representation signal, establish scanner invariance, discover a
preparation or workflow effect, or provide causal attribution.

## Files

- `research_question.md`: estimands and scientific boundaries.
- `factor_definitions.md`: factor meanings and physical-material hierarchy.
- `sampling_matrix_spec.md`: CSV schema and fail-closed input rules.
- `estimability_rules.md`: crossing, nesting, graph, rank, and contrast rules.
- `minimum_designs.md`: bounded structural examples, not power calculations.
- `example_design_matrix.csv`: an explicitly illustrative design whose pairwise,
  selected three-way, and requested four-factor coverage are audited separately.
- `identifiability_audit.py`: deterministic, standard-library-only audit.
- `identifiability_report.md`: report generated from the example matrix.

## Commands

Generate the checked example report:

```powershell
python benchmarks\crossed_preparation_identifiability\identifiability_audit.py
```

Audit a custom CSV without overwriting the checked report:

```powershell
python benchmarks\crossed_preparation_identifiability\identifiability_audit.py --input <csv>
```

Emit deterministic JSON without writing repository files:

```powershell
python benchmarks\crossed_preparation_identifiability\identifiability_audit.py --format json
```

Check the report bytes and run the temporary-fixture test suite:

```powershell
python benchmarks\crossed_preparation_identifiability\identifiability_audit.py --check-report
```

Run only the temporary-fixture tests:

```powershell
python benchmarks\crossed_preparation_identifiability\identifiability_audit.py --self-test
```

Preparation, scanner, and site/workflow main effects are requested by default.
For a deliberately narrower custom design, use `--requested-effects`, for
example `--requested-effects preparation,scanner`. Interactions are always
audited diagnostically and become blocking requirements only when named with
`--request-interaction`.

Rank, nesting, and aliasing remain visible for all recorded factors, but only
defects that compromise a requested contrast are blocking. A constant or
aliased unrequested factor cannot create a competing requested-effect
parameter; it remains a diagnostic qualification.

Default execution may replace only `identifiability_report.md` in this package.
JSON, custom-input, report-check, and self-test modes do not write repository
files. Self-tests use disposable operating-system temporary directories and
verify their removal.

## Interpretation

The global design summary distinguishes `fully crossed`, `pairwise complete,
higher-order incomplete`, `partially crossed`, `nested/confounded`,
and `disconnected` designs. `Fully crossed` is
reserved for a complete requested full-factor product; complete pairwise tables
alone are not sufficient. The global summary describes design quality and does
not replace contrast-specific verdicts. A non-bridging unit or unsupported
interaction cannot invalidate a main-effect contrast that retains its required
rank, connectedness, physical bridge, and independent biological supporters.

Each requested contrast has two reported layers. Structural estimability is
classified as:

- directly estimable;
- estimable with partial crossing;
- estimable only under modeling assumptions; or
- not estimable.

Operational validity is reported separately as no identified operational
blocker, order-confounded, order-imbalanced, workflow under-specified,
batch-confounded, or randomization/counterbalancing unverified. Structural
support must not be read as proof that an operationally clean effect exists.

Structural identifiability is not statistical power. A full-rank matrix does
not by itself supply independent biological replication, a physical
counterfactual, randomized execution, or causal interpretation.

Reported residual degrees of freedom are **row-level residual degrees of
freedom**. They are not independent biological degrees of freedom. Repeated
scans or workflows on one prepared section can increase row-level sample size
without adding a biological unit, and row-level residual degrees of freedom
cannot compensate for only two independent biological units.

The example contains 32 illustrative acquisition rows: two biological units,
two preparation conditions, two scanner conditions, two site/workflow levels,
and two section replicates per biological-unit/preparation cell. These counts
demonstrate audit behavior only and are not universal sample-size guidance.
Its workflow labels denote repeatable post-preparation acquisition workflows
applied to already prepared sections. They do not represent, or identify, an
upstream site-specific preparation effect.

In this package, the CSV field `site_workflow` is restricted to a declared
`post_preparation_workflow`. Each level must be prospectively defined by its
operator or operator pool, transfer/storage condition, post-preparation
handling, post-processing pipeline, exposure order, timing window,
destructive/non-destructive status, and carryover risk. A workflow label may
aggregate operational causes; it is not a single causal mechanism.

The illustrative matrix also keeps physical repeats separate from biological
replication. `repeat_acquisition_id` distinguishes an intentional rescan within
the same section, preparation, scanner, and workflow condition, including when
the repeat occurs in a different scan batch. Scan batch and acquisition order
are acquisition attributes, not base-event identity. Changing only
`observation_id`, `scan_batch`, or `acquisition_order` does not remove the need
for a distinct repeat ID.
Block/biological-unit and preparation-batch/scan-batch relationships are
reported explicitly so duplicated labels or one-to-one nuisance axes are not
mistaken for independent replication.

This declared-identity audit does not replace immutable source provenance. If
the same source acquisition is relabeled with different section, preparation,
scanner, or workflow identifiers, the matrix alone cannot prove that the rows
share one origin. A future executed study needs an immutable source-event ID,
file checksum, or equivalent provenance record to detect that failure mode.
