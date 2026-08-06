# Full-Program Prior-Art Review — Final (2026-08-05)

**Supersedes / extends:** `full-program-prior-art-review-20260804.md`.
**Search log:** `prior-art-search-log-20260805.csv`.
**Search window:** publications through 2026-08-05.
**Databases used:** OpenAlex API, Crossref API, Semantic Scholar result metadata,
arXiv, and general web primary-source search (original papers, proceedings,
journal pages, official repositories). Academic-index confirmation via the
OpenAlex public API is recorded per query in the search log.
**Rule:** no "first / only / unprecedented / entirely novel / unique in all
literature" wording is used unless the evidence genuinely supports it. Allowed:
"we introduce", "we develop", "our implementation combines", "we did not
identify prior work with this exact combination in the searched literature",
"the distinguishing design is".

---

## Academic-index confirmations (new in this final review)

| Question | Academic-index finding | Verdict |
| --- | --- | --- |
| PA-NF paired-scanner factorization | No OpenAlex result matched a neural factorization with explicit biological/acquisition branches on paired same-region scans. | Exact combination not found. |
| Paired multi-scanner design | Confirmed prior art: SCORPION dataset (Ryu et al., MICCAI UNSURE 2023 / arXiv:2507.20907), CHIME, CATCH, SimCons, Barlow-triplet (Mind-the-Gap), ScanGen. | Paired design is prior art; PA-NF topology not found. |
| TransnnMIL multibranch fusion | Closest partial: SlideMamba (Khan et al., 2026, Scientific Reports) fuses a GNN topology branch with a Mamba global-context branch via entropy-weighted fusion; a dual-path graph-driven MIL (AAAI 2026) is architecturally adjacent but details unverified. | No exact TransMIL+gated-attention+branch-token-self-attention combination found. |
| PathologyFL | Prior art confirmed: HistoFL (Lu et al., MedIA 2022) and Swarm learning (Saldanha et al., Nature Medicine 2022) are pathology-specific decentralized-learning systems. No institutional-weight-separation scheme found. | Pathology-specific FL is prior art; PathologyFL is a specific integration. |
| FAIR-WEIGHTS-H contribution weighting | Shapley value client weighting is established prior art (ShapleyFL, KDD 2023; FedOwen; GTG-Shapley; FedMS; PECO; FedCW; FedAA). No train/val/monitor weight-separation protocol found. | Contribution-aware weighting is prior art; FAIR-WEIGHTS-H is a specific variant; the weight-separation element is specification-only. |
| Provenance system | Claim-to-artifact binding, immutable hashing, and fail-closed verification are active prior art (scitex-clew, R-LAM, MASTER Science Pipeline, EvalHub/OCI, certifiable-verify, Immutable AI). | The pattern is prior art; the repository's exact deterministic replay of frozen cells plus the living-claim-boundary mechanism is a specific variant. |

## Consolidated conclusions (all contribution records)

### PA-NF
- **Closest prior work:** SCORPION (paired same-region dataset), SimCons, Barlow-triplet alignment, ScanGen contrastive mitigation; domain-separation and multi-view factorization generally.
- **Shared components:** exact same-region multi-scanner pairs; scanner-consistency objectives; paired evaluation.
- **Distinguishing components:** the specific factorized topology — tissue-oriented biological branch + explicit acquisition branch + decoder + crossed same-region reconstruction + same-region biological consistency + biological variance floor + prototype regularization — and the corrected fixed-estimand, sample-blocked, same-region/same-sample-exclusion audit.
- **Exact combination found:** no (in the searched academic indexes).
- **Safe wording:** "we introduce Paired-Acquisition Neural Factorization"; "paired same-region scanner designs follow public datasets such as SCORPION".
- **Prohibited wording:** "the first paired multi-scanner design"; "scanner-free biology"; "best scanner-removal".
- **Unresolved:** comprehensive PubMed/Scopus index search for multi-view factorization in non-pathology literature is recommended before external submission; Layer-2 swapping remains unverified.

### TransnnMIL
- **Closest prior work:** TransMIL, ABMIL, CLAM (single/multi-branch), DSMIL, H^2-MIL, HIPT, hierarchical-regional-transformer MIL, graph-MIL (Deformable Attention Graph, ResGAT, GE-ViP, GNN-ViTCap, MIL-GNN), SlideMamba (2026).
- **Shared components:** transformer MIL, attention MIL, dual/multi-branch MIL, hierarchical pooling, GNN/topology MIL, adaptive/pruned attention.
- **Distinguishing components:** the specific dual-branch pairing of TransMIL-style global correlation with nnMIL-style gated attention fused by self-attention over explicit projected branch tokens, in combination with hierarchical pooling, topology branch, adaptive pruning, and coordinate-aware graph caching as one architecture family.
- **Exact combination found:** no.
- **Safe wording:** "we introduce TransnnMIL, a multibranch whole-slide aggregation architecture"; "we introduce branch-token self-attention fusion".
- **Prohibited wording:** "outperforms TransMIL/nnMIL/AttentionMIL/CLAM/SlideMamba"; "the first multi-branch MIL"; "first graph MIL".
- **Unresolved:** repaired matched controlled reruns (including SlideMamba-style hybrid comparisons) are pending; historical QWK values are withdrawn.

### PathologyFL
- **Closest prior work:** HistoFL, Swarm learning, FedDMIL, FedWSIDD, FedHD, HistoFS.
- **Shared components:** federated training for pathology WSIs; weakly supervised MIL local models; DP; heterogeneity handling.
- **Distinguishing components:** the specific component set (production orchestrator with tamper-evident audit log, monitoring, model registry, async training, Byzantine detection, secure aggregation, FAIR-WEIGHTS-H weighting layer).
- **Exact combination found:** no.
- **Safe wording:** "we developed PathologyFL as a pathology-specific federated-learning research framework".
- **Prohibited wording:** "the first pathology federated-learning framework"; clinical outcomes; full multi-center validation.
- **Unresolved:** no real multi-center deployment; DP/secure components gated on optional libraries (not validated in this environment).

### FAIR-WEIGHTS-H
- **Closest prior work:** Shapley/Owen contribution weighting (ShapleyFL, FedOwen, GTG-Shapley, FedMS), validation-based client selection (PECO, FedCW, FedAA, AdaptFed).
- **Shared components:** contribution-aware weighting; validation-signal-driven aggregation; uncertainty/fairness-aware aggregation.
- **Distinguishing components:** the specific signal set (integrity gate, uncertainty penalty, useful uniqueness, bounded weights) and the PathoAlign audit bridge with representation-risk and reason codes; the train/validation/monitor weight separation is specification-only.
- **Exact combination found:** no.
- **Safe wording:** "we propose and partially implement FAIR-WEIGHTS-H".
- **Prohibited wording:** "proves fairness"; "better than equal or volume weighting"; "we implement" for specification-only components.
- **Unresolved:** prospective and multi-center validation; no established performance or fairness superiority.

### Provenance system
- **Closest prior work:** scitex-clew, R-LAM, MASTER Science Pipeline, EvalHub/OCI, certifiable-verify, Immutable AI, artifact evaluation (reproducibility badges).
- **Shared components:** SHA-256 artifact fingerprinting; claim-to-source binding; fail-closed verification; immutable registries; provenance DAGs.
- **Distinguishing components:** exact deterministic replay of frozen neural cells (bit-identical reconstruction of 50 representations) and the living-claim-boundary vs immutable-publication-snapshot contract.
- **Exact combination found:** no single system with all components.
- **Safe wording:** "we build a fail-closed provenance and claim-validation system".
- **Prohibited wording:** "the first reproducibility system".
- **Unresolved:** targeted assurance-domain (DO-178C/IEC-62304-style) search recommended before external submission.

---

## Overall conclusion

No individual building block in the program is claimed as novel in isolation
(paired scanner designs, transformer/graph/hierarchical MIL, pathology FL,
contribution weighting, and fail-closed evidence systems are all prior art). The
exact *combinations* — the PA-NF factorizer topology, the TransnnMIL
branch-token-fusion architecture family, the PathologyFL + FAIR-WEIGHTS-H
integration, and the provenance system with exact replay and living claim
boundaries — were not found in the searched academic indexes or primary sources.
The manuscript therefore uses "we introduce / we develop / our implementation
combines" wording and does not claim absolute priority. A targeted Scopus and
PubMed citation-chasing pass on the listed closest works is recommended before
journal submission.
