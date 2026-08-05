# Full-Program Prior-Art Review

**Date:** 2026-08-05
**Status:** completes the open questions in `full-program-prior-art-review-required-20260804.md`.
**Method:** current primary sources only (original papers, proceedings, journal
pages, arXiv records, official repositories). Search terms and sources checked
are recorded per contribution. No fabricated citations; `references.bib` in the
manuscript package is updated only with verified entries.
**Rule:** no "first," "only," or absolute-novelty wording is used unless this
review genuinely establishes it. The manuscript uses "we introduce / we develop /
our protocol."

---

## Q1. Three-level (representation / whole-slide / institutional) account of computational-pathology aggregation

- **Closest known prior work:** none found that presents all three levels as a
  connected program in a single manuscript; pathology papers cover MIL (e.g.,
  [TransMIL](https://nips.cc/media/neurips-2021/Slides/28801.pdf),
  CLAM) and FL (e.g., HistoFL) separately.
- **Shared components:** attention MIL, federated learning, representation
  auditing appear separately across the literature.
- **Distinguishing components:** the explicit three-level connected framing with
  a shared fail-closed provenance system.
- **Exact combination found:** no.
- **Search terms:** "three-level aggregation computational pathology";
  "representation whole-slide institutional pathology".
- **Sources checked:** Web (WebSearch), reference lists of the works below.
- **Safe wording:** "we present a three-level framework".
- **Prohibited wording:** "the first three-level framework".
- **Unresolved uncertainty:** comprehensive academic-index search (PubMed/DBLP)
  not run; recommend for public release.

## Q2. "Accountable neural aggregation" terminology

- **Closest prior work:** accountability/auditability concepts appear in
  trustworthy-ML surveys; no established specific meaning of this exact phrase.
- **Exact phrase found:** no.
- **Safe wording:** define the term locally.
- **Prohibited wording:** claim the phrase as a new field name.
- **Unresolved:** terminology novelty is low-risk; a citation to trustworthy-ML
  surveys is recommended.

## Q3–Q4. Exact same-region multi-scanner paired representation learning; PA-NF factorizer topology

- **Closest known prior work:** the [SCORPION dataset](https://arxiv.org/abs/2507.20907)
  (480 regions x 5 scanners) and paired-analysis methods [SimCons](https://dl.acm.org/doi/10.1007/978-3-032-06593-3_10),
  Barlow-Triplet alignment ("Mind the Gap", [arXiv 2211.16141](https://ar5iv.labs.arxiv.org/html/2211.16141)),
  and ScanGen contrastive mitigation ([arXiv 2507.22092](https://ar5iv.labs.arxiv.org/html/2507.22092)).
  The repository's SCORPION is this same public dataset.
- **Shared components:** exact same-region multi-scanner pairs; scanner-
  consistency objectives; paired evaluation.
- **Distinguishing components:** PA-NF's explicit factorized topology — a
  tissue-oriented biological branch and an explicit acquisition branch with a
  decoder, crossed same-region reconstruction, same-region biological
  consistency, biological variance floor, and prototype regularization — and
  its corrected fixed-estimand, sample-blocked, same-region/same-sample
  exclusion evaluation. The paired *design* is prior art; the specific
  factorized branch-swapping topology was not found.
- **Exact combination found:** no.
- **Search terms:** "paired same-region multi-scanner whole slide image
  representation learning scanner invariance computational pathology".
- **Sources checked:** Web (SCORPION, CHIME, CATCH, ScanGen, Mind the Gap).
- **Safe wording:** "we introduce Paired-Acquisition Neural Factorization, a
  paired-data factorization with an explicit acquisition branch"; "paired
  same-region scanner designs follow public datasets such as SCORPION".
- **Prohibited wording:** "the first paired multi-scanner design"; "the first
  scanner-invariance method".
- **Unresolved:** whether a factorized bio/acq decoder-swap topology exists in
  non-pathology multi-view representation learning (e.g., multi-view VAEs).

## Q5. Oldstyle centroid/QR scanner-subspace removal baseline

- **Closest prior work:** linear scanner-subspace removal and mean-centering
  methods are classical; the specific centroid/QR orthonormal projection over
  scanner centroids is a known linear technique.
- **Exact combination found:** no paper found claiming this exact
  scanner-centroid QR projection as a novelty.
- **Safe wording:** "a strong linear scanner-removal baseline"; empirical
  strength is the finding, not novelty.
- **Prohibited wording:** "a novel baseline".

## Q6. Corrected fixed-estimand scanner-leakage audit protocol

- **Closest prior work:** scanner-leakage audits in the paired-scanner
  literature (SCORPION, CHIME). The specific combination of biological-sample-
  blocked folds, fit-only standardization, fixed five-category estimand,
  same-region and same-sample NN exclusion was not found verbatim.
- **Exact protocol found:** no.
- **Safe wording:** "under a corrected fixed-estimand audit".
- **Prohibited wording:** "the first leakage audit".

## Q7. TransMIL/TransnnMIL attribution and branch-token self-attention fusion

- **Closest prior work:** [TransMIL](https://nips.cc/media/neurips-2021/Slides/28801.pdf)
  (global-correlation transformer MIL), [ABMIL](https://arxiv.org/abs/1802.04712)
  (gated attention), CLAM multi-branch ([CLAM_MB](https://www.nature.com/articles/s41551-020-00682-x)),
  DSMIL dual-stream.
- **Shared components:** multi-head self-attention as parallel attention
  branches; dual/multi-branch MIL; attention fusion.
- **Distinguishing components:** TransnnMIL's specific design — a dual-branch
  model pairing a TransMIL-style global-correlation branch with an nnMIL-style
  gated-attention branch fused by *self-attention over explicit projected branch
  tokens* — was not found.
- **Exact combination found:** no.
- **Search terms:** "multi-branch multiple instance learning whole slide image
  fusion TransMIL attention branches".
- **Sources checked:** Web (CPath_SABenchmark, CLAM_MB, DSMIL, TransMIL).
- **Safe wording:** "we introduce TransnnMIL, a multibranch whole-slide
  aggregation architecture"; "we introduce branch-token self-attention fusion".
- **Prohibited wording:** "TransnnMIL outperforms TransMIL/nnMIL/AttentionMIL";
  "the first multi-branch MIL".
- **Unresolved:** transMIL/attentionMIL hyperparameter-matched comparisons are
  pending (repaired reruns).

## Q8. Hierarchical spatial pooling over pathology patch grids

- **Closest prior work:** [ZoomMIL](https://link.springer.com/chapter/10.1007/978-3-031-20083-0_36)
  (multi-scale zoom), [H^2-MIL](https://aaai.org/papers/00933-h-2-mil-exploring-hierarchical-representation-with-heterogeneous-multiple-instance-learning-for-whole-slide-image-analysis/)
  (heterogeneous graph + iterative hierarchical pooling), HIPT (pyramid ViT),
  hierarchical regional transformer MIL.
- **Shared components:** hierarchical/multi-scale aggregation over spatial
  regions.
- **Distinguishing components:** TransnnMIL's learnable cluster centers +
  region attention/mean/max pooling over a branch-level grid is a specific
  variant; not found verbatim.
- **Exact combination found:** no.
- **Search terms:** "hierarchical pooling multiple instance learning
  histopathology spatial pyramid region pooling MIL".
- **Sources checked:** Web (ZoomMIL, H^2-MIL, HIPT, hierarchical regional
  transformer MIL).
- **Safe wording:** "we implement hierarchical spatial pooling with learnable
  cluster centers and region pooling".
- **Prohibited wording:** "the first hierarchical MIL pooling".

## Q9. Topology / GNN branch over pathology patch graphs

- **Closest prior work:** substantial — Deformable Attention Graph
  ([arXiv 2508.05382](https://arxiv.org/abs/2508.05382)), ResGAT,
  GE-ViP, GNN-ViTCap, MIL-GNN, nuclear-graph-guided MIL.
- **Shared components:** patches as graph nodes; k-NN / spatial-adjacency graph
  construction; GNN message passing (GATv2/GraphSAGE/GIN).
- **Distinguishing components:** TransnnMIL's topology branch is a component of
  a larger multibranch whole-slide model; not novel in isolation.
- **Exact combination found:** no (GNN-MIL itself is well established).
- **Search terms:** "graph neural network multiple instance learning whole slide
  image GNN patch topology".
- **Sources checked:** Web (listed above).
- **Safe wording:** "we implement a topology branch using k-NN graphs and GNN
  layers within TransnnMIL".
- **Prohibited wording:** "the first graph-MIL method".
- **Unresolved:** the repository's historical topology results are withdrawn;
  no repaired rerun exists, so no empirical comparison can be claimed.

## Q10. Adaptive pruning of MIL attention / heads

- **Closest prior work:** adaptive pruning in transformer/MIL contexts is
  established in the pruning literature; no pathology-MIL-specific adaptive
  pruning paper found verbatim.
- **Exact combination found:** no.
- **Safe wording:** "we implement adaptive pruning (disabled pending a proper
  batched implementation)".
- **Prohibited wording:** "a novel pruning method".
- **Unresolved:** pruning is disabled in TransnnMILv2; no result exists.

## Q11. Coordinate-aware graph caching for whole-slide graphs

- **Closest prior work:** coordinate-aware patch processing and graph caching
  appear in large-WSI systems; no exact match found.
- **Exact combination found:** no.
- **Safe wording:** "we implement coordinate-aware graph caching".
- **Prohibited wording:** "the first coordinate-aware WSI graph cache".
- **Unresolved:** no training result exists.

## Q12. Pathology-specific federated-learning framework (PathologyFL)

- **Closest known prior work:** [HistoFL](https://github.com/mahmoodlab/HistoFL)
  (Lu et al., MedIA 2022), FedDMIL, FedWSIDD, FedHD, HistoFS — all
  pathology-specific FL frameworks.
- **Shared components:** federated training for pathology WSIs; weakly
  supervised MIL local models; differential privacy; heterogeneity handling.
- **Distinguishing components:** PathologyFL's specific component set — a
  production orchestrator with tamper-evident audit log, monitoring, model
  registry, async training, Byzantine detection, secure aggregation, and a
  FAIR-WEIGHTS-H weighting layer — is a specific integration; the general
  "pathology-specific FL" concept is prior art.
- **Exact combination found:** no.
- **Search terms:** "federated learning pathology whole slide image framework
  FLamby cross-institutional histopathology".
- **Sources checked:** Web (HistoFL, FedDMIL, FedWSIDD, FedHD, HistoFS).
- **Safe wording:** "we developed PathologyFL as a pathology-specific
  federated-learning research framework".
- **Prohibited wording:** "the first pathology federated-learning framework".
- **Unresolved:** no real multi-center deployment; DP/secure components gated on
  optional libraries.

## Q13. Separation of training / validation / monitoring institution weights

- **Closest prior work:** validation-set-based client selection and contribution
  weighting is prior art ([PECO](https://www.semanticscholar.org/paper/PECO%3A-Probabilistic-Evaluation-Based-Client-for-Yang-Hou/8c681684436bcfa19d8ab437202014879b96c539),
  FedOwen, FedAA, FedCW, FeFL, AdaptFed). The *specific* separation into three
  distinct weight vectors (training, validation, monitoring) was not found
  verbatim; FAIR-WEIGHTS-H's separation is **specification-only** in the
  repository.
- **Exact separation found:** no.
- **Safe wording:** "we propose separating training, validation, and monitoring
  institution weights".
- **Prohibited wording:** "we implement" (it is specification-only).
- **Unresolved:** not implemented.

## Q14. Uncertainty penalties and integrity gates for institution weights

- **Closest prior work:** uncertainty-aware and reliability-weighted client
  weighting appears in FL robustness work; integrity-gated scoring is a specific
  variant.
- **Exact combination found:** no.
- **Safe wording:** "we implement uncertainty penalties and an integrity gate in
  FAIR-WEIGHTS-H".
- **Prohibited wording:** "novel fairness mechanism".

## Q15. "Useful uniqueness" signal (update-global cosine x quality x (1-risk))

- **Closest prior work:** cosine-similarity-based contribution measures and
  quality-weighted aggregation appear in FL; the exact product signal was not
  found.
- **Exact combination found:** no.
- **Safe wording:** "we define a useful-uniqueness signal".
- **Prohibited wording:** "the first uniqueness measure".

## Q16. Owen/Shapley-style counterfactual contribution attribution

- **Closest prior work:** extensive — [GTG-Shapley](https://ieeexplore.ieee.org/abstract/document/10592437),
  [FedOwen (Owen sampling)](https://ar5iv.labs.arxiv.org/html/2508.21261),
  FedIF, FedMS, and the Shapley-value-in-FL literature.
- **Shared components:** Shapley/Owen cooperative-game contribution.
- **Distinguishing components:** FAIR-WEIGHTS-H does *not* implement a Shapley
  estimator (contribution score is a free input); it cites the concept.
- **Exact combination found:** n/a (not implemented).
- **Search terms:** "Shapley value client contribution weighting federated
  learning data valuation".
- **Sources checked:** Web (GTG-Shapley, FedOwen, FedIF, FedMS).
- **Safe wording:** "contribution is specified via a score, not computed by a
  Shapley estimator".
- **Prohibited wording:** "we implement Shapley contribution".

## Q17. Dominant-site label-noise stress / "when more data is less trustworthy"

- **Closest prior work:** label-noise robustness and adversarial-client stress
  in FL; the specific simulated-dominant-site pathology stress was not found
  verbatim.
- **Exact combination found:** no.
- **Safe wording:** "we stress-test dominance-aware aggregation under simulated
  dominant-site label noise".
- **Prohibited wording:** "the first FL noise stress study".

## Q18. Fail-closed provenance validators with immutable releases, canonical hashing, exact replay

- **Closest prior work:** the pattern is prior art and active — [scitex-clew](https://pypi.org/project/scitex-clew/0.10.1/),
  [R-LAM](https://ar5iv.labs.arxiv.org/html/2601.09749),
  [MASTER Science Pipeline](https://zenodo.org/records/19153960),
  [EvalHub/OCI](https://developers.redhat.com/articles/2026/06/16/store-immutable-ai-evaluation-records-evalhub-oci),
  [Immutable AI](https://www.sciencedirect.com/science/article/pii/S1474034626006671),
  [certifiable-verify](https://github.com/SpeyTech/certifiable-verify).
- **Shared components:** SHA-256 artifact fingerprinting; claim-to-source
  binding; fail-closed verification; immutable registries; provenance DAGs.
- **Distinguishing components:** this repository's exact deterministic replay of
  frozen neural cells (bit-identical reconstruction of 50 representations) and
  the "living claim boundary vs immutable publication snapshot" mechanism are
  specific variants.
- **Exact combination found:** no single system with all of: corrected-release
  manifests, source-tree binding, canonical result hashing, exact replay
  recovery, and living-claim-boundary contracts.
- **Search terms:** "machine learning reproducibility immutable evidence artifact
  binding fail-closed scientific audit provenance".
- **Sources checked:** Web (listed above).
- **Safe wording:** "we build a fail-closed provenance and claim-validation
  system"; "the audit system is a methodological contribution".
- **Prohibited wording:** "the first reproducibility system".
- **Unresolved:** whether the exact-replay + living-boundary combination appears
  in non-ML assurance domains (DO-178C/IEC-62304-style) warrants a targeted
  search before external submission.

---

## Summary of the exact-combination findings

| Contribution | Prior art exists for | Exact combination found |
| --- | --- | --- |
| PA-NF factorized bio/acq decoder topology | paired designs (SCORPION), branch separation (domain separation) | No |
| TransnnMIL dual-branch + branch-token fusion | TransMIL, ABMIL, CLAM_MB, DSMIL | No |
| Hierarchical pooling variant | ZoomMIL, H^2-MIL, HIPT | No |
| Topology/GNN branch | GNN-MIL (many) | No (component prior art) |
| Adaptive pruning | pruning literature | No |
| PathologyFL | HistoFL, FedDMIL, FedWSIDD, FedHD | No (specific integration) |
| FAIR-WEIGHTS-H weight separation | validation-based client selection | No (specification-only) |
| Contribution-aware weighting | Shapley-in-FL, FedOwen, FedMS | No (specific variant) |
| Center-subspace correction | domain adaptation, center weighting | No (specific variant) |
| Fail-closed evidence system | scitex-clew, R-LAM, MASTER, certifiable-verify | No (specific combination) |

**Conclusion:** no absolute-novelty claim is supported for any individual
component (most building blocks are prior art). The exact *combinations* — the
PA-NF factorizer topology, the TransnnMIL branch-token fusion design, the
PathologyFL + FAIR-WEIGHTS-H integration, and the fail-closed provenance system
with exact replay and living claim boundaries — were not found in the searched
primary sources. The manuscript uses "we introduce / we develop / our protocol"
wording and does not claim priority over prior work. A comprehensive academic-
index search (PubMed, DBLP, Scopus) and citation chasing on the listed works is
recommended before any external submission.
