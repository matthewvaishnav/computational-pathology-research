# Unified separation evidence inventory — historical snapshot

**Status:** descriptive inventory only  
**Ranking status:** withdrawn  
**Current claim use:** restricted

This file previously presented results from different datasets, protocols,
representations, dimensionalities, seed counts, and evaluation procedures as a
single scoreboard. That presentation implied controlled head-to-head ranking
where none existed.

The inventory is retained for provenance, but it must not be used to answer
questions such as "what wins" across experiments.

## Current restrictions

1. Cross-dataset values are not directly comparable.
2. Metrics produced under different preprocessing and probe protocols are not
   directly comparable.
3. Missing values prevent complete ranking.
4. The historical canine biological-label audit has been superseded because of
   test-set preprocessing leakage and same-region nearest-neighbour leakage.
5. SCORPION retrieval metrics measure same-region identity, not category-label
   preservation.
6. Raw scanner removal, structured branch separation, downstream label
   preservation, and acquisition-branch inspectability are different estimands
   and must not be collapsed into one score.

## Still-supported qualitative boundary

Under their own protocols, the completed studies support the following narrow
statements:

- oldstyle centroid/QR projection is the stronger raw scanner-removal baseline;
- paired-acquisition models provide an explicit scanner-bearing acquisition
  branch and a biological branch with reduced linearly recoverable scanner
  identity;
- residual scanner and category information remains in both branches;
- SCORPION evidence concerns tissue-region retrieval and cross-scanner
  agreement, not biological-category labels;
- canine category-preservation conclusions require the v2 audit rerun before
  numerical promotion.

## Replacement format

Future summaries must be split into separate within-protocol tables:

1. SCORPION DINOv2 paired baseline versus factorization;
2. frozen cross-backbone transfer, reported by backbone;
3. canine paired-acquisition scanner and retrieval metrics;
4. canine biological-label audit v2, after completion;
5. linear-removal baselines under exactly matched preprocessing;
6. bottleneck comparisons within one frozen protocol.

Only contrasts generated inside the same experiment with the same folds,
features, preprocessing, probes, and uncertainty procedure may be described as
head-to-head comparisons.

The historical CSV files in this directory remain available to reconstruct the
old snapshot. They are not a current scientific leaderboard.
