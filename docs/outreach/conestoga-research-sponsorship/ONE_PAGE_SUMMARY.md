# One-page project summary

## Paired-Acquisition Neural Factorization

Computational pathology models can encode scanner optics, colour response, compression, staining, and institutional workflow alongside tissue information. This creates a representation-identifiability problem: a model may appear to recognize biology while relying partly on acquisition provenance.

Paired-Acquisition Neural Factorization uses multiple scans of the same underlying tissue region to separate frozen pathology embeddings into:

- a scanner-suppressed tissue factor;
- an acquisition-specific factor.

The method is evaluated by asking whether the tissue factor preserves same-region identity and cross-scanner agreement while reducing linearly recoverable scanner identity, and whether the acquisition factor retains scanner information rather than merely destroying the original representation.

## Current evidence

### Human H&E paired-scanner benchmark

The primary SCORPION study uses 2,400 real-tissue patches from 480 aligned regions across 48 original H&E slides scanned on five devices. A locked objective was evaluated across DINOv2-Base, Phikon, and ImageNet ResNet50 frozen features using five rotating original-slide test folds.

| Frozen representation | Scanner probe: paired baseline | Scanner probe: factorized tissue branch | Mean retrieval: paired baseline | Mean retrieval: factorized tissue branch |
|---|---:|---:|---:|---:|
| DINOv2-Base | 0.7825 | **0.3989** | 0.9999 | 0.9998 |
| Phikon | 0.9543 | **0.5200** | 0.9991 | **0.9997** |
| ResNet50 | 0.6828 | **0.3145** | 0.9645 | **0.9726** |

Across all three backbones, scanner-probe confidence intervals were below zero for the factorized-versus-baseline contrast, cross-scanner cosine agreement improved, retrieval met the predefined noninferiority criteria, and the compact acquisition branch retained strong scanner information.

### Independent external paired-scanner validation

The external study uses a geometry-qualified subset of the public Multi-Scanner Canine Cutaneous Squamous Cell Carcinoma dataset: 44 biological samples, five scanners, 805 complete matched regions, and 4,025 scanner views. Hyperparameters were locked from SCORPION before the five-fold external test.

The biological scanner probe decreased from 0.7529 to **0.3614**, mean cross-scanner cosine increased from 0.6960 to **0.7300**, and mean same-region retrieval changed from 0.9306 to **0.9334**. All predefined external-validation criteria passed.

### Additional controls

A matched-budget pair-allocation study indicates that increasing unique biological-pair diversity improves biological consistency and factor separation more than repeatedly presenting fewer anchors. The repository also contains mechanism and downstream branches for center-leakage attenuation, whole-slide multiple-instance learning, and external-center generalization.

## Reproducibility and research controls

The repository includes preregistered protocols, sample- or slide-blocked evaluation, repeated seeds, uncertainty estimates, explicit negative claim boundaries, immutable run identifiers, SHA-256 artifact bindings, corruption tests, and fail-closed provenance validation. Historical artifacts without complete lineage remain excluded from new claim evidence.

## Current boundary

The evidence supports a **representation-identifiability and scanner-suppression claim on paired-acquisition benchmarks**. It does not establish clinical validity, diagnostic equivalence, causal disease biology, complete disentanglement, or improved patient outcomes.

## Requested support

The project is seeking faculty review and potential supervision, Digital Research Alliance sponsorship, assessment for a paid student research role, guidance on Mitacs or related applied-research funding, and connections to pathology or medical-technology partners for stronger human-tissue external validation.

## Public materials

- [Review-package landing page](README.md)
- [SCORPION study package](https://github.com/matthewvaishnav/paired-acquisition-factorization-scorpion)
- [External canine SCC study package](https://github.com/matthewvaishnav/paired-acquisition-factorization-caninescc)
- [Main research repository](https://github.com/matthewvaishnav/computational-pathology-research)
