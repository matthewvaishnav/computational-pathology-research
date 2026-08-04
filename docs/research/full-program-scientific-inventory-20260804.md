# Full Computational-Pathology Scientific Inventory

**Date:** 2026-08-04
**Branch:** `research/full-computational-pathology-foundations-manuscript-20260804`
**Scope:** repository-wide inventory of the complete research program.
**Authority:** `CLAIM_BOUNDARY.md` overrides older manuscripts, reports, and
tables whenever they conflict.

This inventory classifies every research line separately by its *authored
contribution* (implementation, protocol, theory), its *empirical evidence*
(corrected/negative/pending/withdrawn), and its *prohibited interpretations*.
Statuses never conflate architecture with empirical validation.

## Status vocabulary

- `active_corrected_empirical_evidence`
- `implemented_architecture_pending_controlled_validation`
- `implemented_research_infrastructure`
- `proposed_protocol_with_execution_validation`
- `synthetic_mechanism_evidence`
- `negative_or_mixed_empirical_result`
- `historical_withdrawn_evidence`
- `future_protocol_only`
- `prohibited_by_evidence_scope`

---

## Line 1 — PCam patch benchmark foundation

- **Identifier:** `pcam_patch_benchmark`
- **Scientific level:** Level I (representation formation) / benchmark foundation
- **Problem addressed:** patch-level tumor-detection evaluation and engineering
  foundation for the program.
- **Authored contribution:** PCam dataset API, binary classification head,
  feature extractors, full-config training, full-test evaluation with ROC AUC /
  accuracy / F1 and bootstrap CIs, cross-validation, threshold analysis,
  failure-asymmetry analysis, federated smoke and heterogeneity benchmarks.
- **Implementation paths:** `src/data/datasets/pcam_dataset.py`,
  `src/models/components/heads.py`, `experiments/train_pcam.py`,
  `experiments/evaluate_pcam.py`, `experiments/compare_pcam_baselines.py`,
  `scripts/cross_validate_pcam.py`, `scripts/optimize_threshold.py`,
  `scripts/federated/run_pcam_federated_smoke.py`.
- **Tests:** `tests/test_pcam_dataset.py`, `test_pcam_evaluation_ci.py`,
  `test_evaluate_pcam.py`, `test_compare_pcam_baselines.py`,
  `test_pcam_experiment_configs.py`, `test_pcam_nan_cascade_*`.
- **Datasets:** official PCam train/val/test splits.
- **Result artifacts:** `results/pcam_comparison` (evaluation status failed for
  the compared variants); the documented observed run (ROC AUC 0.9394, accuracy
  0.8526, F1 0.8507 on the official 32,768-patch test split) has artifacts
  gitignored; `docs/results/pcam-results.md`, `docs/PCAM_REAL_RESULTS.md`.
- **Source commits:** PCam training/eval/config history.
- **Current status:** `active_corrected_empirical_evidence` (patch benchmark);
  cross-validation `future_protocol_only` (paused); threshold study
  `historical_withdrawn_evidence`.
- **Strongest supported claim:** a working patch-level pipeline with a
  documented observed test-split performance (ROC AUC 0.9394) and bootstrap
  infrastructure.
- **Pending validation:** completed cross-validation; a calibration study at a
  clinically meaningful operating point.
- **Withdrawn claims:** threshold optimization as confirmatory/deployment
  evidence; any clinical or superiority language.
- **Prohibited interpretations:** clinical validation, diagnosis-saving
  language, state-of-the-art or superiority over unrelated published models,
  deployment thresholds.
- **Manuscript role:** benchmark and engineering foundation (Section 3).
- **Focused-paper role:** none standalone; foundation for federated/MIL work.

## Line 2 — Paired-scanner and paired-acquisition design

- **Identifier:** `paired_scanner_design`
- **Scientific level:** Level I
- **Problem addressed:** scanner/acquisition confounding in representation
  formation; how exact same-region paired scans can isolate acquisition signal.
- **Authored contribution:** the paired-design protocol (exact same-region
  multi-scanner pairs), pairing ladder, broken-pair and scanner-balanced random
  controls, paired permutation nulls, and biological-sample-blocked fold
  protocol.
- **Implementation paths:** `experiments/paired_acquisition/` runners
  (`run_pair_structure_boundary_test.py`,
  `run_real_paired_scanner_bottleneck_allocation_validation.py`,
  `run_fixed_estimand_real_feature_space_adjudication*.py`).
- **Tests:** paired-scanner validation tests,
  `tests/test_fixed_estimand_real_feature_space_adjudication*.py`.
- **Datasets:** SCORPION; multi-scanner canine SCC.
- **Result artifacts:** corrected-20260726 release; real paired-scanner
  validation (`complete_mixed_real_paired_scanner_allocation_effects`);
  fixed-estimand adjudication v2
  (`complete_no_neural_feature_space_increment_supported`).
- **Source commits:** pair-structure boundary commits, real-validation commit
  `e95d8526`, adjudication commits.
- **Current status:** `active_corrected_empirical_evidence` (paired supervision,
  partial structured separation); `negative_or_mixed_empirical_result` (no
  neural feature-space increment).
- **Strongest supported claim:** true-pair superiority over broken-pair controls
  demonstrates paired supervision; partial structured separation under tested
  conditions.
- **Pending validation:** validated Layer-2 scanner swapping (requires verified
  swap metadata); pixel-space validation.
- **Withdrawn claims:** historical slide-independent sign-flip p-values;
  cosine-as-biology.
- **Prohibited interpretations:** perfect disentanglement, scanner-free biology,
  clinical utility.
- **Manuscript role:** Level I (Section 4).
- **Focused-paper role:** core of Focused Paper A.

## Line 3 — Paired-Acquisition Neural Factorization (PA-NF)

- **Identifier:** `paired_acquisition_neural_factorization`
- **Scientific level:** Level I
- **Problem addressed:** explicit, inspectable, bottleneckable, decoder-swappable
  separation of biological and acquisition representation with paired
  supervision.
- **Authored contribution:** factorizer architecture with tissue-oriented
  biological branch, acquisition branch, decoder, crossed reconstruction,
  same-region biological consistency, biological variance floor, prototype
  regularization; parameter-matched B32/B64 families.
- **Implementation paths:** `src/features/paired_acquisition/` (or the factorial
  modules), `experiments/paired_acquisition/` runners,
  `src/paired_acquisition_provenance.py`,
  `src/paired_acquisition_factorial.py`.
- **Tests:** PA-NF audit and validation test suites; corrected-evidence tests.
- **Datasets:** SCORPION; canine SCC.
- **Result artifacts:** corrected-20260726 evidence; SCORPION capacity-matched
  evidence; dimensionality-xcov-factorial evidence; real paired-scanner
  validation; 50-cell exact recovery; fixed-estimand adjudication v2.
- **Source commits:** corrected evidence commit `4239d4f7`; remediation commit
  `32f357e1`; real-validation `e95d8526`; recovery and adjudication commits.
- **Current status:** `implemented_architecture_pending_controlled_validation`
  (architecture); `negative_or_mixed_empirical_result` (corrected neural
  increment); `active_corrected_empirical_evidence` (paired supervision,
  partial separation).
- **Strongest supported claim:** paired supervision and partial structured
  separation under corrected evaluation; the acquisition branch is an explicit,
  inspectable structure.
- **Pending validation:** corrected category gain over simple baselines;
  Layer-2 swapping.
- **Withdrawn claims:** any historical superiority over oldstyle centroid/QR.
- **Prohibited interpretations:** best scanner-removal method; learns
  scanner-free biology; clinical utility; perfect factorization.
- **Manuscript role:** Level I (Section 4) with the negative result stated
  directly.
- **Focused-paper role:** Focused Paper A.

## Line 4 — SCORPION

- **Identifier:** `scorpion`
- **Scientific level:** Level I
- **Problem addressed:** scanner confounding on aligned multi-scanner H&E
  regions.
- **Authored contribution:** paired-scanner study, capacity-matched campaign,
  fold-aware analysis, cross-backbone transfer.
- **Implementation paths:** `experiments/paired_acquisition/run_*scorpion*`,
  `src/models/scorpion_pathoalign.py`.
- **Tests:** SCORPION manifest and capacity-matched tests; corrected-evidence
  tests.
- **Datasets:** SCORPION (48 slides, 480 regions, 5 scanners, 2400 patches).
- **Result artifacts:** corrected-20260726/scorpion; scorpion-capacity-matched
  evidence; fold-aware contrasts.
- **Source commits:** capacity-matched evidence commit `0adea50f`/merge
  `30778499`.
- **Current status:** `active_corrected_empirical_evidence` (scanner, retrieval,
  geometry); **no category labels**, so no biological-accessibility conclusion.
- **Strongest supported claim:** scanner recoverability and region-retrieval
  findings under fold-aware evaluation.
- **Pending validation:** validated category labels (absent).
- **Prohibited interpretations:** any category-accessibility claim.
- **Manuscript role:** Level I dataset; SCORPION conclusion
  `feature_only_no_biological_claim`.
- **Focused-paper role:** Focused Paper A dataset.

## Line 5 — Multi-scanner canine SCC

- **Identifier:** `canine_scc`
- **Scientific level:** Level I
- **Problem addressed:** category-labeled external paired-scanner validation.
- **Authored contribution:** fixed five-category estimand
  (Dermis, Epidermis, Inflamm/Necrosis, SCC, Subcutis; Bone and Cartilage
  excluded), biological-sample-blocked folds, fit-only standardization,
  same-region/same-sample NN exclusion, corrected probes.
- **Implementation paths:** `experiments/paired_acquisition/run_biological_label_preservation_fixed_estimand.py`,
  canine data handling.
- **Tests:** corrected fixed-estimand tests; adjudication v1/v2 tests.
- **Datasets:** external multi-scanner canine SCC.
- **Result artifacts:** corrected-20260726/canine; fixed-estimand adjudication
  v2 (`complete_no_neural_feature_space_increment_supported`).
- **Current status:** `active_corrected_empirical_evidence` (category endpoints);
  `negative_or_mixed_empirical_result` (no neural category increment).
- **Strongest supported claim:** corrected five-category descriptive endpoints;
  B64 does not improve category accessibility but increases retrieval and
  scanner recoverability.
- **Pending validation:** additional category datasets.
- **Prohibited interpretations:** clinical endpoints (categories are descriptive
  labels).
- **Manuscript role:** Level I dataset.
- **Focused-paper role:** Focused Paper A.

## Line 6 — Scanner and center subspaces

- **Identifier:** `scanner_center_subspaces`
- **Scientific level:** Level I
- **Problem addressed:** how scanner and center identity occupy low-dimensional
  subspaces of representation space, and whether they can be removed or
  separated.
- **Authored contribution:** centroid/QR, PCA, paired-linear, adversarial, and
  center-projection subspace methods; center-subspace projection diagnostics.
- **Implementation paths:** `experiments/paired_acquisition/run_*subspace*`,
  `scripts/camelyon17/run_pathoalign_v6c_center_projection_rank_sweep.py`,
  `run_pathoalign_v7_center_projection_baseline.py`.
- **Tests:** baseline and subspace tests.
- **Datasets:** SCORPION, canine SCC, CAMELYON17.
- **Result artifacts:** oldstyle residual branch separation audit;
  CAMELYON17 v6c/v7 center-projection summaries.
- **Current status:** `active_corrected_empirical_evidence` (centroid/QR is the
  strongest raw scanner-removal baseline; linear center subspace partially
  removable); `negative_or_mixed_empirical_result` (adversarial center removal
  did not reduce leakage).
- **Strongest supported claim:** oldstyle centroid/QR reaches chance scanner
  probe with preserved category accuracy; a supervised linear center subspace is
  partially removable with preserved tumor AUC.
- **Pending validation:** nonlinear scanner-removal baselines.
- **Prohibited interpretations:** center invariance; universal center-removal
  law.
- **Manuscript role:** Level I baselines; CAMELYON17 center studies.
- **Focused-paper role:** Focused Papers A and C.

## Line 7 — Synthetic identifiability and mechanism studies

- **Identifier:** `synthetic_identifiability_mechanism`
- **Scientific level:** Level I (mechanism)
- **Problem addressed:** when latent biological and acquisition factors are
  recoverable under controlled generators; capacity allocation.
- **Authored contribution:** synthetic generators, identifiability diagnostics,
  capacity-allocation factorials, task-sufficiency benchmarks, whitening audits,
  paired-consensus and routed-bottleneck variants.
- **Implementation paths:** `experiments/paired_acquisition/run_synthetic_*`,
  `run_crossed_target_*`, `run_*capacity*`, `run_task_*`.
- **Tests:** synthetic factorial and mechanism test suites.
- **Datasets:** controlled synthetic generators (no human data).
- **Result artifacts:** synthetic factorial
  (`complete_capacity_gain_with_scanner_tradeoff`); dimensionality-xcov
  factorial evidence; mechanism result dirs.
- **Current status:** `synthetic_mechanism_evidence`.
- **Strongest supported claim:** mechanism, identifiability, and capacity
  allocation under controlled generators.
- **Pending validation:** transport to real corrected category accessibility
  (failed under the corrected estimand).
- **Prohibited interpretations:** pathology claims; clinical utility.
- **Manuscript role:** Section 10 (synthetic mechanism studies).
- **Focused-paper role:** Focused Paper A (identifiability).

## Line 8 — TransnnMIL whole-slide aggregation

- **Identifier:** `transnnmil_whole_slide`
- **Scientific level:** Level II
- **Problem addressed:** multibranch whole-slide neural aggregation.
- **Authored contribution:** canonical dual-branch TransnnMIL (TransMIL-style
  global correlation + nnMIL-style gated attention), corrected branch-token
  self-attention fusion, TransnnMILv2 three-branch variant, concat/gate/learned
  branch-attention fusion controls, hierarchical spatial pooling, topology/GNN
  branch, adaptive pruning, graph caching and coordinate-aware processing, PANDA
  training infrastructure, Phikon feature bags.
- **Implementation paths:** `src/models/transnnmil/` (all files),
  `src/models/mil/{transmil,nnmil,attention_mil,clam}.py`,
  `scripts/training/train_panda_transnnmil_baseline.py`.
- **Tests:** `tests/models/test_transnnmil_*`,
  `test_topology_branch.py`, `test_hierarchical_pooling.py`,
  `test_adaptive_pruning.py`, `test_transnnmil_fusion_salvage.py`.
- **Datasets:** PANDA (Phikon feature bags), synthetic MIL.
- **Result artifacts:** `results/panda_transnnmil_baseline`,
  `results/panda_transnnmil_ablation`, `results/transnnmil_stability_summary`,
  `results/panda_transnnmil_threshold_ready`.
- **Source commits:** repair commit `38144d8a`; salvage `932c7435`; regression
  `55ece77f`; controls alignment `e976b1ce`; remediation `32f357e1`.
- **Current status:** `implemented_architecture_pending_controlled_validation`;
  historical fusion/topology numbers `historical_withdrawn_evidence`; post-repair
  PANDA runs are preliminary/pending.
- **Strongest supported claim:** TransnnMIL is an implemented, tested multibranch
  architecture; post-repair PANDA stability and ablation runs are competitive but
  not superiority evidence.
- **Pending validation:** repaired matched controlled reruns against TransMIL,
  nnMIL, AttentionMIL, concat, gate, and learned branch-attention controls.
- **Withdrawn claims:** historical fusion and topology QWK values; any
  superiority over AttentionMIL/TransMIL/nnMIL.
- **Prohibited interpretations:** outperforms TransMIL/nnMIL; improves PANDA
  grading; clinical readiness; state of the art.
- **Manuscript role:** Level II (Section 5).
- **Focused-paper role:** Focused Paper B.

## Line 9 — PathologyFL

- **Identifier:** `pathology_fl`
- **Scientific level:** Level III
- **Problem addressed:** pathology-specific federated-learning research
  infrastructure.
- **Authored contribution:** coordinator, client, aggregator (FedAvg/FedProx/
  FedAdam/weighted/Byzantine-robust/secure/pathology-aware/FAIR-WEIGHTS-H),
  privacy engines (DP-SGD, budget tracking), secure communication (TLS gRPC) and
  secure aggregation (TenSEAL), monitoring and Byzantine detection, async
  training, fault tolerance, compression, model registry.
- **Implementation paths:** `src/features/federated/pathology_fl/` (all files).
- **Tests:** `tests/federated/test_fl_integration.py`,
  `test_pathology_fl_aggregation.py`, `test_pathology_fl_production_integration.py`,
  `test_pathology_fl_privacy_regressions.py`, `test_secure_aggregation*.py`,
  `test_async_training.py`, `test_fault_tolerance.py`, `test_model_registry.py`,
  `test_federated_monitoring.py`, `test_hospital_client.py`, `test_local_trainer.py`.
- **Datasets:** PCam patches (simulated sites), PANDA-derived features
  (simulated federations).
- **Result artifacts:** PCam federated smoke; PANDA simulated federations;
  communication/privacy/infrastructure probes.
- **Source commits:** PathologyFL history.
- **Current status:** `implemented_research_infrastructure` (execution
  validated); **no real multi-center deployment**.
- **Strongest supported claim:** executable pathology-specific federated
  research framework with validated integration and smoke behavior.
- **Pending validation:** real multi-center deployment; the e2e federated tests
  have dangling imports (not runnable as written); DP/secure guarantees gated on
  optional libraries and not independently audited.
- **Prohibited interpretations:** clinical outcomes; full multi-center
  validation; clinical readiness.
- **Manuscript role:** Level III (Section 6).
- **Focused-paper role:** Focused Paper C.

## Line 10 — FAIR-WEIGHTS-H

- **Identifier:** `fair_weights_h`
- **Scientific level:** Level III
- **Problem addressed:** auditable institutional contribution, safety, and
  representation weighting.
- **Authored contribution:** hybrid institutional-weighting protocol separating
  training/validation/monitoring signals (spec), integrity gates, uncertainty
  penalties, useful uniqueness, bounded weights, PathoAlign audit bridge with
  representation-risk and reason codes, conservative mode, aggregator
  integration.
- **Implementation paths:** `src/features/federated/pathology_fl/weighting/fair_weights_h.py`,
  `weighting/pathoalign_bridge.py`, `aggregator/pathoalign_fair.py`,
  `aggregator/weighted.py`.
- **Tests:** `tests/federated/test_fair_weights_h.py`,
  `test_weighting_perturbations.py`, `test_weighting_benchmark.py`,
  `test_synthetic_federation.py`.
- **Datasets:** synthetic federations; PCam patches; PANDA-derived simulated
  federations.
- **Result artifacts:** `results/fair_weights_h_panda_feature_stress_*`,
  `results/fair_weights_h_stress_aggregate` (not forward-valid manifests).
- **Current status:** `proposed_protocol_with_execution_validation` (engine
  implemented + tested; several protocol elements specification-only); no
  established performance/fairness superiority.
- **Strongest supported claim:** execution stability and aggregation behavior
  are tested; conditional simulated-site stress results show contribution-aware
  blends can help at 25% dominant-site label noise.
- **Pending validation:** consistent performance/fairness superiority; the
  FAIR-WEIGHTS-H aggregator integration has no dedicated test; subgroup
  constraints and weight-change limits unimplemented.
- **Prohibited interpretations:** FAIR-WEIGHTS-H proves fairness; better than
  equal/volume weighting; clinical readiness.
- **Manuscript role:** Section 7.
- **Focused-paper role:** Focused Paper C.

## Line 11 — PANDA institutional-shift and ordinal-learning studies

- **Identifier:** `panda_studies`
- **Scientific level:** Level II/III
- **Problem addressed:** whole-slide ordinal grading and simulated institutional
  shift.
- **Authored contribution:** Phikon feature bags, MIL baselines, ordinal ISUP
  grading, dominant-site stress, corruption and label-shift experiments,
  ordinal-harm-aware analysis, dominance-detector tuning, centralized-vs-
  federated comparison.
- **Implementation paths:** `scripts/training/train_panda_*`,
  `scripts/experiments/run_panda_*`, `run_fair_weights_h_panda_*`,
  `analyze_panda_fedavg_failure_modes.py`, `analyze_dominance_*`.
- **Tests:** PANDA dataset and MIL tests.
- **Datasets:** PANDA (ISUP 0-5, 10,611 readable Phikon slide features).
- **Result artifacts:** `results/panda_*`, `results/cross_site_fedavg_panda_*`,
  `results/dominance_detector_switch_*`, `results/ordinal_harm_*`.
- **Current status:** `active_corrected_empirical_evidence` (simulated-site
  results); TransnnMIL superiority `implemented_architecture_pending_controlled_validation`.
- **Strongest supported claim:** simulated-site FedAvg-vs-centralized ordering;
  tuned dominance detector reduces clean false-switch; conditional
  FAIR-WEIGHTS-H stress results.
- **Pending validation:** matched TransnnMIL reruns; real hospital federation.
- **Prohibited interpretations:** clinical validity; real hospital mapping;
  superiority over AttentionMIL/TransMIL/nnMIL.
- **Manuscript role:** Sections 5 and 8.
- **Focused-paper role:** Focused Papers B and C.

## Line 12 — CAMELYON17 center-subspace and held-out-center studies

- **Identifier:** `camelyon17_center`
- **Scientific level:** Level III
- **Problem addressed:** source-center weighting, center leakage, and
  held-out-center generalization.
- **Authored contribution:** WILDS-structured center weighting studies, center
  subspace projection, center-leakage mechanism diagnostics, four-pillars
  federated failure analysis.
- **Implementation paths:** `experiments/camelyon17_*`,
  `scripts/camelyon17/` (weighting, detector, projection runners).
- **Tests:** `tests/test_camelyon_*`, `tests/camelyon17/`,
  `tests/test_pathoalign_v6_analysis.py`.
- **Datasets:** CAMELYON17/WILDS (5 centers, 455,954 patches).
- **Result artifacts:** `results/camelyon17/`,
  `results/camelyon17_supervised_resnet18/`,
  `results/camelyon17_pathoalign_v6c/v7`.
- **Current status:** `active_corrected_empirical_evidence` (centralized
  source-weighting proxies and mechanism diagnostics);
  `negative_or_mixed_empirical_result` (adversarial center removal).
- **Strongest supported claim:** equal-client weighting improves held-out-center
  accuracy over sample-proportional weighting (centralized frozen-feature proxy);
  a supervised linear center subspace is partially removable.
- **Pending validation:** full federated-learning CAMELYON17 validation
  (Track A planned); real FAIR-WEIGHTS-H CAMELYON17 validation.
- **Prohibited interpretations:** full FL validation; general hospital law; one
  held-out center establishes a cross-hospital law.
- **Manuscript role:** Section 9.
- **Focused-paper role:** Focused Paper C (boundary).

## Line 13 — Reproducibility, provenance, claim auditing, fail-closed validation

- **Identifier:** `scientific_audit_provenance`
- **Scientific level:** cross-cutting methodology
- **Problem addressed:** binding every active claim to an immutable, re-verifiable
  artifact; separating code correction from empirical validation.
- **Authored contribution:** provenance validators, immutable release packages,
  canonical hashing, source binding, fail-closed statuses, living claim-boundary
  contract, historical withdrawal records, deterministic exact replay and
  artifact recovery, claim ledgers.
- **Implementation paths:** `scripts/provenance/`,
  `src/paired_acquisition_provenance.py`, `src/paired_acquisition_factorial*.py`,
  `experiments/paired_acquisition/run_real_bottleneck_representation_recovery.py`,
  `run_fixed_estimand_real_feature_space_adjudication*.py`.
- **Tests:** `tests/test_corrected_paired_acquisition_evidence.py`,
  `tests/test_real_bottleneck_representation_recovery.py`,
  `tests/test_fixed_estimand_real_feature_space_adjudication*.py`,
  `tests/test_biological_bottleneck_capacity_allocation_factorial.py`.
- **Datasets:** all program artifacts (provenance applies cross-cuttingly).
- **Result artifacts:** `evidence/paired_acquisition/` three immutable releases;
  recovery result (`complete_exact_real_bottleneck_representation_recovery`);
  adjudication v2 (`complete_no_neural_feature_space_increment_supported`);
  v1 (`fixed_estimand_adjudication_not_ready`, preserved).
- **Source commits:** remediation `32f357e1`; evidence builders/validators;
  recovery/adjudication commits.
- **Current status:** `active_corrected_empirical_evidence` (the audit system is
  itself validated by its tests); `implemented_research_infrastructure`.
- **Strongest supported claim:** fail-closed validation infrastructure binds
  claims to artifacts; exact 50-cell replay reproduced frozen metrics bit-exactly
  (0 deltas); the corrected evidence release validates against its immutable
  snapshot.
- **Pending validation:** none internal (the system is the validation).
- **Prohibited interpretations:** passing tests do not imply clinical
  performance; the audit system is not clinical validation.
- **Manuscript role:** Section 12 (cross-cutting).
- **Focused-paper role:** candidate Focused Paper D (reproducibility), pending
  portfolio decision.

---

## Cross-cutting notes

- No research line is classified by a single status when its architecture,
  experiments, and historical claims have different statuses.
- Historical withdrawn results appear only as motivation for remediation, never
  in active result tables.
- `complete_no_neural_feature_space_increment_supported` is preserved verbatim;
  it is a scientific status, not an architectural invalidation.
