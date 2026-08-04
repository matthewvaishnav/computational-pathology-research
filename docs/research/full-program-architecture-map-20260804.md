# Full Program Architecture Map

**Date:** 2026-08-04
**Purpose:** the data-flow and component map connecting the three aggregation
levels and the benchmark/provenance infrastructure.

```
                    ┌─────────────────────────────────────────────────────┐
                    │          Benchmark foundation (Level I)             │
                    │   PCam patches ──► PCam training/eval infrastructure│
                    └───────────────────────┬─────────────────────────────┘
                                            │ frozen patch / foundation features
                                            ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                 LEVEL I — REPRESENTATION FORMATION                      │
   │                                                                          │
   │  frozen DINOv2 / Phikon / ResNet features ──► scanner & center auditing  │
   │        │                                                                │
   │        ├──► Paired-Acquisition Neural Factorization (PA-NF)             │
   │        │       biological branch  /  acquisition branch  /  decoder     │
   │        │       crossed reconstruction · consistency · prototype reg.    │
   │        │       simple baselines: centroid/QR · PCA · paired-linear      │
   │        │       broken-pair & scanner-balanced random controls           │
   │        │       synthetic identifiability · capacity allocation          │
   │        └──► scanner/center subspaces (oldstyle QR · center projection)  │
   └───────────────────────┬──────────────────────────────────────────────────┘
                           │ patch representations / feature bags
                           ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │             LEVEL II — WHOLE-SLIDE AGGREGATION (TransnnMIL)              │
   │   patch feature bags (Phikon) ──► branch A: TransMIL global correlation  │
   │                             ──► branch B: nnMIL gated attention          │
   │   branch-token fusion (corrected self-attention) · hierarchical pooling │
   │   topology/GNN branch · adaptive pruning · graph caching · PANDA eval    │
   └───────────────────────┬──────────────────────────────────────────────────┘
                           │ slide-level predictions / local models
                           ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │            LEVEL III — INSTITUTIONAL AGGREGATION                         │
   │   local institutional models ──► PathologyFL coordinator/client/agg      │
   │        FedAvg · FedProx · FedAdam · weighted · Byzantine-robust          │
   │        secure aggregation · DP · monitoring · async · fault tolerance    │
   │   FAIR-WEIGHTS-H weights: integrity · uncertainty · useful-uniqueness    │
   │        PathoAlign audit bridge · bounded weights · conservative mode     │
   │   stress: dominant-site · corruption · ordinal shift · heterogeneity     │
   └───────────────────────┬──────────────────────────────────────────────────┘
                           │ global model / global evaluation
                           ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │   CROSS-CUTTING: benchmark (PCam/PANDA/CAMELYON17) + provenance +        │
   │   claim validation (immutable releases · source binding · fail-closed     │
   │   statuses · exact replay recovery · living claim boundary)              │
   └──────────────────────────────────────────────────────────────────────────┘
```

## Component table

| Component | Level | Implementation path | Evidence status |
| --- | --- | --- | --- |
| PCam patch benchmark | Foundation | `experiments/`, `src/data/datasets/pcam_dataset.py` | active (patch benchmark; not clinical) |
| PA-NF factorizer | I | `experiments/paired_acquisition/` runners, `src/paired_acquisition_*.py` | corrected paired supervision; negative neural increment |
| Scanner/center subspace baselines | I | `experiments/paired_acquisition/run_*subspace*`, CAMELYON17 v6c/v7 | active (centroid/QR strongest raw removal; center subspace partial) |
| Synthetic generators | I | `experiments/paired_acquisition/run_synthetic_*` | synthetic mechanism evidence |
| TransnnMIL | II | `src/models/transnnmil/`, `src/models/mil/` | implemented architecture; superiority pending; historical withdrawn |
| PANDA MIL baselines | II | `src/models/mil/{attention_mil,nnmil,transmil,clam}.py` | active (QWK baselines) |
| PathologyFL | III | `src/features/federated/pathology_fl/` | implemented infrastructure; no real multi-center |
| FAIR-WEIGHTS-H | III | `.../weighting/fair_weights_h.py`, `.../aggregator/pathoalign_fair.py` | protocol + execution validation; no superiority |
| CAMELYON17 center studies | III | `experiments/camelyon17_*`, `scripts/camelyon17/` | centralized proxies + mechanism diagnostics |
| Provenance / audit | Cross-cutting | `scripts/provenance/`, `src/paired_acquisition_provenance.py` | validated by its own tests; exact replay recovery |

## Data-flow invariants

1. Frozen patch/foundation features are the shared input to Level I auditing and
   Level II feature bags; the same arrays are audited before entering MIL/FL.
2. No frozen evidence, numerical threshold, category set, or result artifact is
   modified by the program organization.
3. Every active empirical claim binds to an immutable result artifact; every
   architectural claim binds to source and tests.
