# Literature positioning

This page situates my framework — TransnnMIL v2.0, PathologyFL, and FAIR-WEIGHTS-H — within the published computational pathology literature. All citations are PubMed-indexed and verified. DOI links are provided for every reference.

> **Scope note.** This framework operates at the research infrastructure level. Results reported here are public-benchmark validations, not clinical validations. PCam patch-level performance is not the same as Camelyon slide-level performance. Benchmark superiority is not the same as clinical deployment readiness. These distinctions are preserved throughout this page.

---

## Overview: where this work fits

Computational pathology has converged on a set of shared problems: how to train slide-level models without pixel-level annotations, how to train across institutions without sharing patient data, and how to ensure that multi-center models treat contributing institutions equitably. No single published system addresses all three simultaneously at the infrastructure level.

This framework contributes at each layer:

| Layer | My contribution | Closest published comparator |
|---|---|---|
| Patch-level validation | PCam benchmark, #1 AUC among 11 compared methods | Campanella et al. 2019, Nat Med |
| WSI MIL architecture | TransnnMIL v2.0 | CLAM, NATMIL, SlideMamba |
| Federated WSI learning | PathologyFL | HistoFL, Lu et al. 2022 |
| Institutional weighting | FAIR-WEIGHTS-H | Bhalla et al. 2026, heterogeneity-aware FL in CT |
| Non-IID benchmark infrastructure | PCam balanced + heterogeneous FL splits | Gao et al. 2023, swarm learning |
| Multi-center validation | Camelyon16/17 roadmap | NRK-ABMIL, Sajjad et al. 2023 |

---

## Related work: weakly supervised MIL for WSI analysis

The central challenge in computational pathology is that gigapixel whole-slide images cannot be directly processed by standard deep learning architectures, and pixel-level annotation at scale is impractical. Multiple-instance learning addresses this by treating a slide as a bag of patches and learning from slide-level labels only.

**Campanella et al. (2019)** established the foundation for clinical-scale weakly supervised WSI classification, training on 44,732 WSIs from 15,187 patients using only reported diagnoses as labels. Their system achieved AUC above 0.98 on prostate cancer, basal cell carcinoma, and breast cancer lymph node metastasis, and demonstrated that excluding 65–75% of slides at 100% sensitivity is achievable in a single-institution setting. DOI: [10.1038/s41591-019-0508-1](https://doi.org/10.1038/s41591-019-0508-1), PMID: [31308507](https://pubmed.ncbi.nlm.nih.gov/31308507/).

**Lu et al. (2021) — CLAM** introduced clustering-constrained attention MIL, adding interpretable instance-level localization without spatial labels. CLAM applied to RCC subtyping, NSCLC, and lymph node metastasis detection became a major baseline for WSI MIL research and is a primary baseline against which TransnnMIL v2.0 should be evaluated. DOI: [10.1038/s41551-020-00682-w](https://doi.org/10.1038/s41551-020-00682-w), PMID: [33649564](https://pubmed.ncbi.nlm.nih.gov/33649564/).

**Aftab et al. (2024) — NATMIL** introduced the neighborhood attention transformer into MIL, explicitly modeling contextual dependencies between adjacent tiles. NATMIL reported 89.6% accuracy on Camelyon-derived slides and 88.1% on TCGA-LUSC, demonstrating that tissue context beyond the individual tile is a meaningful signal. DOI: [10.3389/fonc.2024.1389396](https://doi.org/10.3389/fonc.2024.1389396), PMID: [39267847](https://pubmed.ncbi.nlm.nih.gov/39267847/).

**Khan et al. (2026) — SlideMamba** combined a graph neural network branch with a Mamba state-space branch using entropy-based adaptive fusion. SlideMamba represents a recent benchmark for hybrid topology and sequence modeling in WSI analysis. DOI: [10.1038/s41598-025-34367-8](https://doi.org/10.1038/s41598-025-34367-8), PMID: [41486382](https://pubmed.ncbi.nlm.nih.gov/41486382/).

**Xu et al. (2025)** conducted a systematic comparison of pathology foundation model extractors paired with MIL aggregators. The key finding is that foundation model extractor quality can dominate aggregation architecture choice. This directly motivates the planned TransnnMIL v2.0 foundation-model extractor ablation. DOI: [10.1016/j.media.2025.103456](https://doi.org/10.1016/j.media.2025.103456), PMID: [39842326](https://pubmed.ncbi.nlm.nih.gov/39842326/).

### Positioning: TransnnMIL v2.0

TransnnMIL v2.0 combines four design directions that no single cited method addresses together: transformer-style global attention, hierarchical spatial pooling, graph-aware tissue topology modeling, and adaptive pruning. NATMIL addresses neighborhood attention; SlideMamba addresses topology plus sequence modeling; CLAM addresses interpretable attention MIL. TransnnMIL v2.0 is positioned as a unified architecture direction, but slide-level WSI benchmark results are still required before superiority claims are made.

---

## Related work: federated learning in medical imaging and pathology

Federated learning solves the data-sharing problem: institutions can contribute to a shared model without exposing patient data. The relevant literature spans general medical imaging FL and the emerging pathology-specific branch.

**Sheller et al. (2020)** established a foundational result for medical FL: 10 institutions training together with FedAvg reached 99% of the model quality achievable with fully centralized data, and models generalized better to out-of-federation institutions than any single-center model. DOI: [10.1038/s41598-020-69250-1](https://doi.org/10.1038/s41598-020-69250-1), PMID: [32724046](https://pubmed.ncbi.nlm.nih.gov/32724046/).

**Lu et al. (2022) — HistoFL** applied FL to gigapixel WSI classification, combining attention-based MIL with differential privacy and demonstrating that federated TCGA-derived WSI models approach centralized performance. PathologyFL extends this work by adding FAIR-WEIGHTS-H institutional weighting, heterogeneous benchmark splits, and explicit per-client fairness diagnostics. DOI: [10.1016/j.media.2021.102298](https://doi.org/10.1016/j.media.2021.102298), PMID: [34911013](https://pubmed.ncbi.nlm.nih.gov/34911013/).

**Bhalla et al. (2026)** introduced a heterogeneity-aware federated framework combining CT foundation model pretraining with discrepancy-aware aggregation for multi-center pancreatic cancer lymph node metastasis detection. This is a close comparator to FAIR-WEIGHTS-H in the broader medical FL literature, but it operates in CT, uses fewer heterogeneity dimensions, and does not address WSI pathology weighting. DOI: [10.1038/s41598-026-47631-2](https://doi.org/10.1038/s41598-026-47631-2), PMID: [41957506](https://pubmed.ncbi.nlm.nih.gov/41957506/).

**Gao et al. (2023)** proposed a swarm learning approach addressing feature skew and label skew in non-IID multi-center medical image segmentation. This provides methodological grounding for heterogeneous benchmark design. DOI: [10.1109/TMI.2022.3220750](https://doi.org/10.1109/TMI.2022.3220750), PMID: [36350867](https://pubmed.ncbi.nlm.nih.gov/36350867/).

### Positioning: PathologyFL

PathologyFL is the WSI-specific federated learning infrastructure in this framework. It implements FedAvg as a baseline and layers FAIR-WEIGHTS-H on top. The completed PCam balanced and heterogeneous FL benchmarks provide controlled experimental substrates. The key null result — heterogeneous PCam splits produced different weight trajectories but no measurable performance sensitivity in the current patch-level setup — is a valid infrastructure finding. Camelyon17 is the harder slide-level test.

### Positioning: FAIR-WEIGHTS-H

FAIR-WEIGHTS-H is an institutional weighting method operating across eight dimensions: quality, useful uniqueness, fairness, contribution, volume, uncertainty, entropy, and effective-institution diagnostics. A search of the PubMed-indexed FL-pathology literature found no cited method that addresses all eight dimensions jointly in a WSI context. HistoFL uses standard FedAvg. Bhalla et al. address heterogeneity in CT. Gao et al. address feature and label skew in segmentation. FAIR-WEIGHTS-H is therefore positioned as a novel institutional weighting design for pathology FL.

FAIR-WEIGHTS-H has also been empirically evaluated for execution stability and aggregation behavior. Synthetic smoke tests, PCam federated smoke tests, the balanced PCam benchmark, and the heterogeneous PCam benchmark show that the method runs reliably, does not introduce numerical failures, produces distinct weight trajectories under heterogeneous simulated sites, and does not degrade performance in the current patch-level setting. What has not yet been demonstrated is a consistent performance or fairness advantage over simpler aggregation baselines. That stronger claim requires ablation against FedAvg, volume weighting, quality-only weighting, fairness-only weighting, and the full FAIR-WEIGHTS-H formulation.

---

## Related work: Camelyon and PCam benchmarks

The Camelyon grand challenges established standard evaluation substrates for automated lymph node metastasis detection in breast cancer histopathology. Camelyon16 is a slide-level benchmark, while Camelyon17 extends to five centers with real scanner and staining heterogeneity.

PatchCamelyon is a patch-level derivative of Camelyon16. PCam and Camelyon16/17 are related but not equivalent: strong PCam performance does not imply strong slide-level WSI performance.

**Sajjad et al. (2023) — NRK-ABMIL** introduced a normal representative keyset approach for attention-based MIL and reported strong performance on Camelyon16 and Camelyon17. It is a primary single-model comparator for the planned Camelyon17 FL validation. DOI: [10.3390/cancers15133428](https://doi.org/10.3390/cancers15133428), PMID: [37444538](https://pubmed.ncbi.nlm.nih.gov/37444538/).

### Positioning: PCam benchmark

My framework achieved **85.26% test accuracy** and **AUC 0.9394** on the full 32,768-sample PCam test set, ranking **#1 by AUC** among the 11 methods in the comparison table. This is a patch-level public-benchmark result. It is legitimate and strong in the patch-level domain, but it does not replace Camelyon slide-level validation.

### Positioning: Camelyon17 roadmap

The planned Camelyon17 multi-center validation will use PathologyFL to train across all five centers as separate FL clients. Evaluation should compare centralized training, FedAvg, PathologyFL without FAIR-WEIGHTS-H, and full PathologyFL + FAIR-WEIGHTS-H. Per-center slide AUC and aggregate patient-level AUC should be reported alongside per-client fairness metrics.

---

## Citation table

| Paper | DOI / PMID | Method family | Dataset | Level | Centers | Reported metric | Relevance | Gap addressed |
|---|---|---|---|---|---|---|---|---|
| Campanella et al. 2019 | [DOI](https://doi.org/10.1038/s41591-019-0508-1) / [31308507](https://pubmed.ncbi.nlm.nih.gov/31308507/) | Weakly supervised MIL | 44k WSIs | Slide | Single | AUC >0.98 | Foundational clinical-scale MIL | No FL or institutional weighting |
| Lu et al. 2021, CLAM | [DOI](https://doi.org/10.1038/s41551-020-00682-w) / [33649564](https://pubmed.ncbi.nlm.nih.gov/33649564/) | Attention MIL + clustering | TCGA WSI tasks | Slide | Single + external | AUC per task | Primary MIL baseline | No graph/topology, no FL |
| Lu et al. 2022, HistoFL | [DOI](https://doi.org/10.1016/j.media.2021.102298) / [34911013](https://pubmed.ncbi.nlm.nih.gov/34911013/) | FL + Attn-MIL + DP | TCGA silos | Slide / patient | Simulated multi-site | Near centralized AUC | Primary FL-pathology baseline | No institutional weighting |
| Sheller et al. 2020 | [DOI](https://doi.org/10.1038/s41598-020-69250-1) / [32724046](https://pubmed.ncbi.nlm.nih.gov/32724046/) | FedAvg | BraTS MRI | Patient | 10 institutions | 99% centralized quality | FL medicine foundation | Not pathology WSI |
| Xu et al. 2025 | [DOI](https://doi.org/10.1016/j.media.2025.103456) / [39842326](https://pubmed.ncbi.nlm.nih.gov/39842326/) | FM extractor × MIL | TCGA, 4 cancer types | Slide | Single | FM quality drives MIL accuracy | Motivates FM ablation | No FL or fairness |
| Sajjad et al. 2023 | [DOI](https://doi.org/10.3390/cancers15133428) / [37444538](https://pubmed.ncbi.nlm.nih.gov/37444538/) | NRK-ABMIL | Camelyon16/17 | Slide | Multi-center | SOTA reported | Camelyon comparator | No FL weighting |
| Aftab et al. 2024 | [DOI](https://doi.org/10.3389/fonc.2024.1389396) / [39267847](https://pubmed.ncbi.nlm.nih.gov/39267847/) | Neighborhood attention MIL | Camelyon, TCGA-LUSC | Slide | Single | 89.6% acc | Context-aware MIL comparator | No FL |
| Khan et al. 2026 | [DOI](https://doi.org/10.1038/s41598-025-34367-8) / [41486382](https://pubmed.ncbi.nlm.nih.gov/41486382/) | GNN + Mamba | Clinical WSIs | Slide | Single | PRAUC reported | Topology + sequence comparator | No FL |
| Bhalla et al. 2026 | [DOI](https://doi.org/10.1038/s41598-026-47631-2) / [41957506](https://pubmed.ncbi.nlm.nih.gov/41957506/) | FL + heterogeneity-aware aggregation | PDAC CT | Patient | 3 centers | +12.6% balanced acc | Closest heterogeneity-aware FL comparator | CT not WSI; fewer dimensions |
| Gao et al. 2023 | [DOI](https://doi.org/10.1109/TMI.2022.3220750) / [36350867](https://pubmed.ncbi.nlm.nih.gov/36350867/) | Swarm learning | Multi-dataset segmentation | Slice | Multi | Superior to FedAvg | Non-IID grounding | Not MIL/pathology WSI |

---

## Claim strength

| Claim | Current evidence | Evidence level | What is still needed |
|---|---|---|---|
| PCam 0.9394 AUC | Full 32,768-sample PCam test set | Strong | Maintain reproducibility details |
| #1 AUC among compared PCam methods | Comparison table with 11 methods | Strong if comparison table remains documented | Keep methods and sources explicit |
| TransnnMIL v2.0 architecture | Literature-motivated design | Design claim | Camelyon16 slide-level benchmark |
| PathologyFL | Working federated scaffold and PCam FL benchmarks | Supported infrastructure claim | Real multi-center validation |
| FAIR-WEIGHTS-H novelty | No direct comparator among cited PubMed FL-pathology papers | Strong design novelty | Keep literature comparison updated |
| FAIR-WEIGHTS-H empirical behavior | Synthetic, PCam smoke, balanced PCam, and heterogeneous PCam tests; distinct weights produced under heterogeneity; no degradation observed | Supported behavior/stability claim | Stronger heterogeneous or slide-level setting |
| FAIR-WEIGHTS-H performance advantage | Current PCam benchmarks did not show measurable performance/fairness improvement over simpler strategies | Not yet demonstrated | FedAvg/volume/quality/fairness/full ablation on PCam and Camelyon17 |
| Camelyon17 validation | Planned | Future claim | Run five-center experiment |
| Clinical deployment | Not claimed | Not supported | Prospective clinical workflow validation |

---

## Prioritized next experiments

| Priority | Experiment | Purpose | Dataset | Baselines | Metrics | Claim unlocked |
|---:|---|---|---|---|---|---|
| 1 | Camelyon16 slide-level benchmark | First slide-level evidence for TransnnMIL v2.0 | Camelyon16 | CLAM, NATMIL, attention pooling | Slide AUC, sensitivity/specificity | TransnnMIL v2.0 competitive WSI result |
| 2 | Foundation model extractor ablation | Test whether architecture gains survive FM control | Camelyon16, TCGA-NSCLC | UNI/CONCH/PLIP + CLAM/TransMIL/TransnnMIL | Slide AUC | Architecture contribution independent of encoder |
| 3 | FAIR-WEIGHTS-H ablation | Identify which weighting dimensions matter | PCam heterogeneous, Camelyon17 | FedAvg, volume, quality, fairness, full method | AUC, worst-client AUC, fairness gap | FAIR-WEIGHTS-H performance benefit |
| 4 | Camelyon17 multi-center FL | Real institutional heterogeneity test | Camelyon17 | Centralized, FedAvg, PathologyFL, FAIR-WEIGHTS-H | Per-center AUC, patient AUC | Multi-center FL validation |
| 5 | TransnnMIL v2.0 aggregator comparison | Isolate architecture contribution | Camelyon16 / TCGA | Attention pooling, CLAM, TransMIL, NATMIL | AUC, runtime | Architecture vs aggregator evidence |
| 6 | Calibration and thresholds | Establish operating-point reliability | PCam, Camelyon16 | Uncalibrated, temperature-scaled | ECE, reliability, sensitivity/specificity | Calibration and threshold claims |
| 7 | Stronger institutional split | Test if PCam null result is patch-level artifact | Camelyon17 or TCGA-site split | FedAvg, full method | Fairness gap, worst-client AUC | Real institutional weighting evidence |

---

## What not to claim yet

The following claims should not appear until the corresponding experiments are complete:

- TransnnMIL v2.0 outperforms CLAM, TransMIL, NATMIL, or SlideMamba at the slide level.
- FAIR-WEIGHTS-H improves FL performance or fairness over simpler baselines.
- PathologyFL achieves clinical-grade performance.
- Results generalize across cancer types, scanners, or institutions beyond the tested datasets.
- Calibration or threshold-based clinical operating point claims.
- PCam patch-level results are equivalent to Camelyon16/17 slide-level SOTA.
- Clinical deployment readiness.

---

## References

1. Campanella G et al. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. *Nat Med.* 2019. DOI: [10.1038/s41591-019-0508-1](https://doi.org/10.1038/s41591-019-0508-1). PMID: [31308507](https://pubmed.ncbi.nlm.nih.gov/31308507/)
2. Lu MY et al. Data-efficient and weakly supervised computational pathology on whole-slide images. *Nat Biomed Eng.* 2021. DOI: [10.1038/s41551-020-00682-w](https://doi.org/10.1038/s41551-020-00682-w). PMID: [33649564](https://pubmed.ncbi.nlm.nih.gov/33649564/)
3. Lu MY et al. Federated learning for computational pathology on gigapixel whole slide images. *Med Image Anal.* 2022. DOI: [10.1016/j.media.2021.102298](https://doi.org/10.1016/j.media.2021.102298). PMID: [34911013](https://pubmed.ncbi.nlm.nih.gov/34911013/)
4. Sheller MJ et al. Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data. *Sci Rep.* 2020. DOI: [10.1038/s41598-020-69250-1](https://doi.org/10.1038/s41598-020-69250-1). PMID: [32724046](https://pubmed.ncbi.nlm.nih.gov/32724046/)
5. Xu H et al. When multiple instance learning meets foundation models: advancing histological whole slide image analysis. *Med Image Anal.* 2025. DOI: [10.1016/j.media.2025.103456](https://doi.org/10.1016/j.media.2025.103456). PMID: [39842326](https://pubmed.ncbi.nlm.nih.gov/39842326/)
6. Sajjad U et al. NRK-ABMIL: subtle metastatic deposits detection for predicting lymph node metastasis in breast cancer whole-slide images. *Cancers (Basel).* 2023. DOI: [10.3390/cancers15133428](https://doi.org/10.3390/cancers15133428). PMID: [37444538](https://pubmed.ncbi.nlm.nih.gov/37444538/)
7. Aftab R et al. Neighborhood attention transformer multiple instance learning for whole slide image classification. *Front Oncol.* 2024. DOI: [10.3389/fonc.2024.1389396](https://doi.org/10.3389/fonc.2024.1389396). PMID: [39267847](https://pubmed.ncbi.nlm.nih.gov/39267847/)
8. Khan S et al. SlideMamba: entropy-based adaptive fusion of GNN and Mamba for enhanced representation learning in digital pathology. *Sci Rep.* 2026. DOI: [10.1038/s41598-025-34367-8](https://doi.org/10.1038/s41598-025-34367-8). PMID: [41486382](https://pubmed.ncbi.nlm.nih.gov/41486382/)
9. Bhalla P et al. Federated CT foundation models for multi-center detection of lymph node metastasis in pancreatic cancer. *Sci Rep.* 2026. DOI: [10.1038/s41598-026-47631-2](https://doi.org/10.1038/s41598-026-47631-2). PMID: [41957506](https://pubmed.ncbi.nlm.nih.gov/41957506/)
10. Gao Z et al. A new framework of swarm learning consolidating knowledge from multi-center non-IID data for medical image segmentation. *IEEE Trans Med Imaging.* 2023. DOI: [10.1109/TMI.2022.3220750](https://doi.org/10.1109/TMI.2022.3220750). PMID: [36350867](https://pubmed.ncbi.nlm.nih.gov/36350867/)
