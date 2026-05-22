# TransnnMIL v2.0 Architecture

TransnnMIL v2.0 is the custom whole-slide multiple-instance learning architecture in this project. It extends TransMIL-style attention with explicit spatial and topology-aware branches.

## Core idea

TransnnMIL v2.0 combines three complementary branches:

1. **TransMIL branch** for global attention over patch features.
2. **Hierarchical pooling branch** for spatial region-level aggregation.
3. **Topology branch** for k-NN graph construction and graph neural processing.

Input:

```text
Patch features: [B, N, D]
Patch coordinates: [B, N, 2]
```

Output:

```text
Class logits: [B, C]
```

## Architecture

```text
Patch features + coordinates
        |
        |---- Branch A: TransMIL / attention MIL
        |---- Branch B: hierarchical spatial pooling
        |---- Branch C: topology / graph branch
        |
   Feature fusion: concat + MLP
        |
   Slide-level logits
```

## Branch A: TransMIL-style global attention

Purpose: learn global interactions among WSI patch embeddings.

Typical flow:

```text
[B, N, D]
  -> optional adaptive pruning
  -> transformer encoder
  -> CLS/global token
  -> [B, hidden_dim]
```

## Branch B: hierarchical pooling

Purpose: preserve spatial locality and reduce token count by clustering patches into regions.

Typical flow:

```text
features + coordinates
  -> spatial clustering
  -> region attention pooling
  -> region-level features
  -> global region summary
```

Benefits:

- reduces compute from patch-level to region-level processing
- supports interpretable region assignments
- captures coarse spatial tissue organization

## Branch C: topology branch

Purpose: model local tissue structure as a graph.

Typical flow:

```text
features + coordinates
  -> k-NN graph construction
  -> GNN layers
  -> graph/global pooling
  -> topology feature vector
```

Supported graph concepts include:

- k-nearest-neighbor patch graphs
- distance-aware or similarity-aware edges
- GAT/GraphSAGE/GIN-style processing depending on implementation

## Fusion

Branch outputs are concatenated and passed through an MLP classifier:

```text
fused = concat([global_attention, region_features, graph_features])
logits = MLP(fused)
```

## Status

This page summarizes the custom TransnnMIL v2.0 architecture. For implementation details, see the source code and the legacy architecture documentation in `docs/TRANSNNMIL_V2_ARCHITECTURE.md`.
