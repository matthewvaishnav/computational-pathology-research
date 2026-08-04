# Full-Program Prior-Art Review — Required Questions

**Date:** 2026-08-04
**Status:** this document lists every novelty or priority question that requires
external literature verification before any absolute priority wording may be
used. No citations are fabricated; the manuscript uses "we introduce / we
develop / our protocol" wording until each question is resolved.

## General novelty questions

1. Has any prior work presented a three-level (representation / whole-slide /
   institutional) account of computational-pathology aggregation as a connected
   system, rather than separate papers per level?
2. Is "accountable neural aggregation" terminology already established in a
   specific sense that must be cited or distinguished?

## Level I — representation formation

3. Are exact same-region multi-scanner paired training designs established in
   prior computational-pathology literature (e.g., stain/style-transfer
   datasets, multi-stain registries)? What is the closest prior paired design?
4. Is the specific PA-NF factorizer topology (tissue-oriented branch +
   acquisition branch + decoder + crossed same-region reconstruction +
   consistency + prototype regularization + variance floor) novel in its exact
   combination?
5. Is an explicit, inspectable, decoder-swappable acquisition branch claimed by
   prior work (vs. linear subspace removal / adversarial removal / domain
   adaptation)?
6. Is the oldstyle centroid/QR scanner-subspace removal baseline novel, or is it
   a rediscovery of a known linear technique? (Its strength is an empirical
   finding, not a novelty claim.)
7. Do prior scanner-leakage audits use the same corrected fixed-estimand,
   sample-blocked, same-region/same-sample-exclusion protocol?

## Level II — whole-slide aggregation

8. Is TransMIL (prior global-correlation MIL) novelty correctly attributed, and
   does TransnnMIL's dual-branch + corrected branch-token self-attention fusion
   differ in a citable way from TransMIL, nnMIL, and AttentionMIL?
9. Do prior MIL papers introduce hierarchical spatial pooling over pathology
   patch grids with learnable cluster centers and region attention/mean/max
   pooling? Which are the closest?
10. Is a topology/GNN branch over pathology patch graphs (k-NN graphs, GATv2 /
    GraphSAGE / GIN) novel in the whole-slide setting? Which are the closest
    graph-MIL works?
11. Is adaptive pruning of MIL attention heads, and coordinate-aware graph
    caching for whole-slide graphs, claimed by prior work?
12. Has the specific branch-token fusion (self-attention over stacked projected
    branch tokens) appeared in prior multi-branch MIL?

## Level III — institutional aggregation

13. Which prior pathology-specific federated-learning frameworks exist, and how
    does PathologyFL differ in its aggregation, monitoring, and
    privacy/secure-aggregation support?
14. Is the separation of training / validation / monitoring institution weights
    novel, or does it generalize known client-selection schemes?
15. Do prior works use uncertainty penalties and integrity gates for
    institution-weight computation?
16. Is the "useful uniqueness" signal (update-global cosine × quality ×
    (1 − risk)) a known weighting scheme under another name?
17. Is Owen/Shapley-style counterfactual contribution attribution for federated
    institutions established prior art that must be cited (FAIR-WEIGHTS-H does
    not implement a Shapley estimator)?
18. Has dominant-site label-noise stress / "when more data is less trustworthy"
    been studied in federated learning for pathology? Which are the closest
    fairness-under-noise works?

## Cross-cutting

19. Are fail-closed provenance validators with immutable release manifests,
    source-tree binding, canonical result hashing, and exact deterministic
    replay used by prior ML-reproducibility initiatives (e.g., repro-hack,
    MLR); how does this system compare?
20. Is the "living claim boundary vs. immutable publication snapshot" mechanism
    used by any prior scientific-audit system?

## Priority-wording rule

Until each question is resolved by a current external literature review, the
manuscript uses "we introduce", "we develop", "our architecture", "our
protocol", and "the distinguishing design combines…" and never "the first ever"
or "state of the art". Focused papers A–D must each run their own scoped
literature review before external submission.
