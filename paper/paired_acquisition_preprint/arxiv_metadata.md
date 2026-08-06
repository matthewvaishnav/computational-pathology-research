# arXiv submission metadata

## Title

Paired-Acquisition Neural Factorization for Computational Pathology

## Authors

Matthew Vaishnav

## Primary category

`cs.CV`

## Secondary categories

`cs.LG`, `q-bio.QM`

## Comments

5 pages, 1 figure, 2 tables. Corrected focused preprint with capacity-matched SCORPION evidence and an independent fixed-estimand multi-scanner canine SCC audit.

## Abstract

Frozen histopathology embeddings can encode scanner identity together with tissue structure. Removing every scanner-predictive direction is unsafe when acquisition and tissue are correlated, yet leaving scanner information uncontrolled can compromise transport across devices. We study Paired-Acquisition Neural Factorization (PA-NF), a two-branch model trained from matched acquisitions of the same tissue region. A tissue-oriented branch is optimized for same-region agreement and reduced scanner recoverability; an acquisition branch is optimized to retain scanner information; a joint decoder and variance controls discourage collapse.

The primary corrected analysis uses SCORPION: 48 human H&E slides, 480 aligned regions, five scanners, and 2,400 patches. In a separately versioned 175-fit, capacity-matched campaign, PA-NF reduced held-out linear scanner balanced accuracy by 0.3108 relative to an equal-parameter two-branch control without scanner objectives (95% fold/slide bootstrap interval [-0.3346, -0.2858]). Mean top-1 same-region retrieval changed by only +0.00004 ([-0.00009, 0.00026]), within a preregistered 0.02 noninferiority margin. The acquisition branch remained strongly scanner informative.

On an independent five-scanner canine cutaneous squamous-cell-carcinoma dataset, corrected five-category evaluation yielded a materially more conservative result. Neural B32/B64 representations did not establish a feature-space increment over simple centroid/QR or paired-linear scanner-removal baselines. Increasing the tissue bottleneck from 32 to 64 dimensions increased scanner recoverability by 0.0489 and retrieval by 0.0515, but did not improve corrected category balanced accuracy (-0.0076, interval [-0.0194, 0.0071]). The evidence supports partial structured separation under the tested protocols, not pure biological factors, complete scanner invariance, diagnostic improvement, or clinical utility.
