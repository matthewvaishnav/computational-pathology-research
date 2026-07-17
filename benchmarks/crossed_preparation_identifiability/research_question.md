# Research question and estimands

## Primary question

After controlling tissue identity, does scanner-suppressed representation
structure track preparation or workflow identity across scanners?

This package addresses only the prerequisite design question:

> Does the proposed sampling matrix make preparation, scanner, and
> post-preparation workflow contrasts structurally identifiable, and how many
> independent biological units and physical bridge strata support each one?

No representation is loaded and no effect is estimated here.

## Biological anchor

`biological_unit` is the unit whose identity is held constant for matched
contrasts. A tissue block, a mapped physical tissue region, or serial sections
from one block can provide an anchor, but they are not interchangeable:

- rescanning the same prepared section is a same-physical-material scanner
  counterfactual;
- different serial sections are matched block material, not identical cells,
  pixels, or microanatomy; and
- a block-level match cannot remove within-block biological heterogeneity.

Preparation contrasts therefore use matched serial sections within the same
biological unit and block. The preparation conditions occur on different
sections: they are not the same tissue instance, the same pixels, or true
counterfactual copies. Structural support for a preparation contrast is thus
conditional on the declared matched-serial-section design and its prospective
section assignment. Only a valid repeated intervention on the same physical
material, with order and carryover controlled, would provide a same-material
preparation counterfactual.

## Requested contrasts

1. **Biology-and-block-controlled preparation contrast**: the same biological
   unit and block contribute different, matched serial sections under multiple
   preparation conditions. At least two independent biological units must
   support each claimed contrast. The number of participating sections is
   reported separately and does not add independent biological replication.
2. **Biology-and-preparation-controlled scanner contrast**: the same biological
   unit and preparation are observed across scanners. Same-section rescanning
   within the same workflow and scan batch is required for a direct verdict.
3. **Biology/preparation/scanner-controlled post-preparation-workflow
   contrast**: matched biological-unit, preparation, and scanner strata occur
   under multiple `post_preparation_workflow` conditions. Interpretation is
   limited to the declared process stage; repeating an already prepared slide
   under multiple acquisition
   workflows cannot identify an upstream site preparation effect. A direct
   verdict requires the same prepared section, scanner, and scan batch across
   workflows. It also requires the workflow levels to denote prospectively
   specified, physically repeatable exposures rather than labels that silently
   absorb scanner, preparation, operator, storage, or post-processing changes.
4. **Future scanner-suppressed residual association with preparation**: inherits
   the worse of the preparation and scanner design prerequisites and
   additionally requires an independently justified scanner-suppression
   analysis.
5. **Future scanner-suppressed residual association with workflow**: inherits
   the worse of the workflow and scanner design prerequisites and the same
   future-analysis boundary.

## Interaction questions

Preparation-by-scanner, scanner-by-workflow, and preparation-by-workflow
interactions require replicated factor-pair cells and estimable
difference-in-differences contrasts. Connected additive main effects do not
automatically identify interactions.

## Structural and operational conclusions

The audit reports contrast-specific structural estimability separately from
global design quality. An extra biological unit that does not bridge a given
contrast is reported as a non-supporter; it does not erase support supplied by
two or more independent bridging units. Likewise, an unavailable interaction
does not make an otherwise supported additive main effect unavailable.

Operational validity is a separate layer. Fixed acquisition order, batch
aliasing, an under-specified workflow, or unverified counterbalancing can
qualify a structurally supported contrast without changing the algebraic rank.
Neither layer is an effect estimate or a causal conclusion.

## Claim boundary

An identifiable design can support a future statement such as "representation
outcomes differ across measured preparation conditions using matched serial
sections from the same biological units and blocks under the tested design."
It cannot by itself establish causal
proof, isolate biological truth, prove a preparation artifact, prove a site
effect, solve scanner bias, or confer clinical or diagnostic relevance.

The declared two-biological-unit supporter threshold is a bare structural floor,
not a power calculation or evidence of stable generalization. Row-level
residual degrees of freedom from repeated acquisitions are not independent
biological degrees of freedom and cannot replace additional biological units.

Physical-repeat checks operate on the declared section, preparation, scanner,
and workflow identity. They cannot determine that two differently relabeled
rows came from one source acquisition. Immutable source-event identifiers or
checksums remain necessary provenance for an executed study.

A future null result would mean no detectable association under the audited
estimand and analysis, not proof that preparation- or workflow-related signal is
absent.
