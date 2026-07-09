# Figure 4: Bottleneck Comparison / Capacity-Constrained Separation

**Intended manuscript location:** Section 7 (Result 4: Bottleneck Comparison)
**Claim:** CLAIM_4_BOTTLENECK_FRONTIER
**Type:** Multi-panel main-text figure

## Caption

**Figure 4. Bottlenecking the acquisition branch reduces biological leakage while preserving scanner capture — a directional separation-frontier improvement.** **(a)** Canine SCC DINOv2 bottleneck comparison. Acquisition-branch category leakage (x-axis, lower = less biological information in scanner branch) vs acquisition-branch scanner capture (y-axis, higher = better scanner encoding). True_pair (64D, no bottleneck) at upper right: high scanner capture (0.865) but high category leakage (0.346). acq_dim8_default (8D, default cross-covariance) and acq_dim16_stronger_xcov (16D, stronger cross-covariance penalty 0.20) move left (reduced leakage: 0.160 and 0.169 respectively) at the same scanner capture level (0.864). Arrow indicates directional improvement — less leakage, preserved capture. Labeled "directional separation-frontier improvement" (not "Pareto front" — the comparison is sparse with 2 dimensions × 2 regularization strengths at full scale; 6 smoke variants at 1-fold support the direction). **(b–d)** SCORPION cross-backbone tissue/pair-retrieval leakage in the acquisition branch. Grouped bars for true_pair (64D) and acq_dim8_default (8D) across three backbones. DINOv2: 0.0944 → 0.0231 (δ = −0.071). Phikon: 0.0739 → 0.0204 (δ = −0.054). ResNet50: 0.1705 → 0.0505 (δ = −0.120). All three backbones show reduced tissue-retrieval leakage under bottlenecking. SCORPION metrics are labeled "tissue/pair-retrieval leakage" (not "category leakage" — SCORPION lacks biological category labels). Error bars: ±1 standard deviation across 25 runs (5-fold × 5-seed). Sources: acquisition bottleneck separation frontier (commit a89bfb32), frontier-selected downstream validation (c29a038d), frontier-selected cross-backbone validation (0e2af247).

## Data Source

- **Primary bottleneck:** a89bfb32 — acquisition bottleneck separation frontier
- **Downstream validation:** c29a038d — frontier-selected downstream validation
- **Cross-backbone:** 0e2af247 — frontier-selected cross-backbone validation
- **Consolidated:** 1c527697 — unified separation scoreboard

## Key Visual Signals

1. Panel A: Bottlenecked variants move left (less leakage) without moving down (same scanner capture)
2. Panel A: Labeled "Directional separation-frontier improvement" (NOT "Pareto front")
3. Panel A: Caption notes "Sparse comparison (4 full-scale variants); supports directional improvement, not dense frontier mapping"
4. Panels B-D: Cross-backbone consistency — all three SCORPION backbones show reduced leakage
5. SCORPION panels explicitly labeled "Tissue/pair-retrieval leakage" (not "category leakage")
6. Biological-branch category accuracy and scanner suppression preserved within narrow bands

## Forbidden Language

- "Frontier sweep" (unqualified) → use "bottleneck comparison" or "capacity-constrained separation audit"
- "Pareto front" / "Pareto optimal" → use "directional separation-frontier improvement"
- "Eliminates biological leakage" → acq_dim8 still has category probe 0.160
- "Dimension 8 is optimal" → only 8 and 16 tested at full scale
- "SCORPION category leakage reduced" → use "tissue/pair-retrieval leakage"

## Appendix Trail

Appendix F: Smoke + full variant metrics, per-scanner downstream, cross-backbone raw metrics.
