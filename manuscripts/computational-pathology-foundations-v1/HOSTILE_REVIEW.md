# Hostile Review — Accountable Neural Aggregation in Computational Pathology

Each criticism records severity, affected section, whether it is fatal, the exact
correction, whether new experiments are required, and whether the manuscript can
remain internally reviewable without those experiments. Criticism is never
answered with marketing language.

## Computational-pathology reviewer

- **C1.1** Severity: high. Affected: Section 4. "Why should I use PA-NF when
  centroid/QR reaches chance scanner removal?" Correction: the manuscript states
  the strong-baseline boundary and positions PA-NF as structured separation with
  an explicit, inspectable, swappable acquisition branch, not raw erasure.
  Fatal: no (the boundary is explicit). New experiments: validated Layer-2
  swapping would strengthen the swap claim. Internally reviewable without them:
  yes.
- **C1.2** Severity: high. Affected: Section 4. "Only one labeled category
  dataset." Correction: stated in Limitations; SCORPION has no category labels.
  Fatal: no. New experiments: additional category datasets for external release.
  Internally reviewable: yes.

## MIL architecture reviewer

- **C2.1** Severity: high. Affected: Section 5. "TransnnMIL superiority is
  unproven and historical numbers were withdrawn." Correction: the manuscript
  labels TransnnMIL as implemented architecture with pending matched controlled
  reruns and never reuses historical QWK. Fatal: no. New experiments: repaired
  matched reruns. Internally reviewable: yes (as architecture).
- **C2.2** Severity: medium. Affected: Section 5. "Adaptive pruning is disabled;
  hierarchical pooling lacks real-data results." Correction: stated in the
  component matrix. Fatal: no. New experiments: proper pruning implementation and
  real-data pooling benchmark.

## Federated-learning reviewer

- **C3.1** Severity: high. Affected: Section 6. "No real multi-center deployment;
  e2e tests have dangling imports." Correction: stated; e2e tests flagged as not
  runnable as written. Fatal: no (framework is infrastructure). New experiments:
  repair e2e tests; real multi-center pilot. Internally reviewable: yes.
- **C3.2** Severity: high. Affected: Section 9. "CAMELYON17 studies are not FL."
  Correction: explicitly labeled centralized frozen-feature proxies. Fatal: no.
  New experiments: full FL validation. Internally reviewable: yes.

## Fairness reviewer

- **C4.1** Severity: high. Affected: Section 7. "FAIR-WEIGHTS-H claims fairness
  by name." Correction: the manuscript states "FAIR in the name does not mean
  fairness has been proven" and reports no superiority. Fatal: no. New
  experiments: prospective and multi-center fairness validation. Internally
  reviewable: yes.
- **C4.2** Severity: medium. Affected: Section 7. "Several protocol elements are
  specification-only." Correction: stated in the signal table. Fatal: no. New
  experiments: implementing the elements.

## Statistics reviewer

- **C5.1** Severity: medium. Affected: Sections 4 and 8. "Fold counts are
  limited; bootstrap intervals are fold-then-unit." Correction: stated; no
  p-values overclaimed. Fatal: no. New experiments: more folds/datasets.
- **C5.2** Severity: medium. Affected: Section 9. "One held-out CAMELYON17
  center." Correction: stated as a proxy with one center. Fatal: no.

## Reproducibility reviewer

- **C6.1** Severity: high. Affected: Section 12. "PCam raw artifacts are
  gitignored; the 0.9394 AUC is documented but not re-verifiable in-repo."
  Correction: the manuscript states the observed run's artifacts are not tracked
  and bounds the claim to the documented run. Fatal: no. New experiments: re-run
  PCam to a tracked artifact. Internally reviewable: yes (documented).
- **C6.2** Severity: low. Affected: Section 12. "Provenance system is described
  approvingly." Correction: framed as a methodological contribution with test
  bindings, not self-congratulation.

## Hostile novelty reviewer

- **C7.1** Severity: high. Affected: throughout. "Priority over prior work is
  unverified." Correction: the manuscript uses "we introduce / we develop" and
  never "first ever" or "state of the art"; the prior-art-review-required doc
  lists every open novelty question. Fatal: no. New experiments: a current
  external literature review before external submission. Internally reviewable:
  yes.
- **C7.2** Severity: medium. Affected: Section 4. "PA-NF is close to known
  domain-separation methods." Correction: the manuscript claims the specific
  architecture as the contribution, not general domain separation.

## Verdict

No criticism is currently fatal to internal review. All high-severity items are
addressed by explicit boundaries or by pending-validation labels. The manuscript
may remain internally reviewable, provided every pending and prohibited status is
explicit, as it is. Public release is blocked until the external literature
review, repaired TransnnMIL reruns, and any stated missing artifacts are
completed.

## Journal editor (coherence for submission)

- **C8.1** Severity: high. Affected: overall. "This is a foundations/position
  manuscript covering 13 lines in 12 pages; is it a coherent paper or a
  repository catalogue?" Correction: the three-level framework is the unifying
  argument; each level section carries technical mechanism depth (branch-token
  fusion, round lifecycle, weight-computation equations), and implementation
  inventories are pushed to the supplement. The manuscript is positioned as a
  foundations/position-and-methods paper, not a results paper. Fatal: no.
  New experiments: none for coherence; focused papers carry the empirical
  depth. Internally reviewable: yes.
- **C8.2** Severity: medium. Affected: Section 12. "The provenance section
  risks reading as self-congratulation." Correction: framed as a
  methodological contribution that makes the negative and pending statuses
  trustworthy; the hostile-review register and validator are themselves the
  evidence. Fatal: no. New experiments: none.
- **C8.3** Severity: high. Affected: status/claims. "Evidence maturity varies
  wildly across lines; the paper must not imply equality." Correction: the
  abstract, evidence-status matrix (Figure 7), and claim ledger distinguish
  corrected evidence, implemented architecture, negative results, proposed
  protocols, and pending validation; the conclusion does not claim empirical
  equality. Fatal: no. New experiments: matched TransnnMIL reruns, FAIR-WEIGHTS-H
  multi-center validation, PathologyFL deployment before focused empirical
  papers. Internally reviewable: yes.

## Editor verdict

The manuscript is internally reviewable as a foundations/position-and-methods
paper. No criticism is currently fatal. Public submission is gated on: a targeted
Scopus/PubMed citation-chasing pass (recommended, not blocking internal
circulation); repaired matched TransnnMIL reruns; FAIR-WEIGHTS-H prospective
validation; and real multi-center PathologyFL deployment — all required before
the focused empirical papers, not before internal review of the foundations
paper.
