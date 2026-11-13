# MAGNET v4: empirically calibrated scanner-response energy — canine SCC retrospective development

**Status: numerical development improvement; not a physical scanner theory, independent confirmation, or biologically pure factorization.** Original PA-NF and MAGNET v0/v3 failures remain preserved.

## Mechanism

Earlier MAGNET v3 assumed a scanner-wide strictly convex quadratic potential, fitting only a rank-16 operator. That model lost region identity and had source-to-target MSE 0.461526. MAGNET v4 instead uses actual 5-view, same-region scanner observations to define a **strictly convex conditional displacement energy**:

`E(d|x,s,t)=λ/2 ||d−μ_{s,t}||² + 1/2 Σ_j w_j(x,s)||d−(x_{j,t}−x_{j,s})||²`.

Its unique equilibrium is the scanner mean displacement shrunk toward a local weighted mean of matched training-region differences. `A` is exclusively the recorded scanner identity, not a per-region acquisition vector. The correction varies with source tissue geometry as real scanner response may. This energy is only guaranteed strictly convex in **d**, not in the biological state or a global image-formation potential.

For every original outer fold: 44 biological sample blocks total, 805 matched regions, 4,025 five-scanner 768D DINOv2 observations. Each outer sample-heldout fold has four nested inner-fold validations for `k∈{8,16,32,64,128}`, `γ∈{.25,.5,.75,1}`. All five selected `k=32,γ=1`. Original uploaded ZIP SHA256: `78a736bd8c672d9fedb2719db89cc9045567efaff726d67af5029c6d72511e25`.

## Real canine results

| Five-fold score | Scanner mean | MAGNET v3 | MAGNET v4 |
|---|---:|---:|---:|
| Source-only heldout target feature MSE ↓ | 0.432823 | 0.461526 | **0.403703** |
| Cross-scanner region retrieval ↑ | 0.892294 | 0.831530 | **0.910068** (gauged) |
| Six-class tissue balanced accuracy ↑ | 0.452351 | 0.452547 | 0.450351 (gauged) |
| B scanner linear balanced accuracy ↓ | 0.217189 | 0.215782 | 0.212835 (gauged) |
| B scanner ExtraTrees balanced accuracy ↓ | 0.322118 | 0.382669 | 0.224511 (gauged) |
| B scanner random-RBF feature probe balanced accuracy ↓ | 0.376725 | not scored | **0.253772** (gauged; still FAIL vs chance 0.20) |

The sample-block mean MSE improvement over scanner mean is +0.027660 with **descriptive** bootstrap interval [+0.024351,+0.031213], positive for all **44/44 previously-exposed samples**. This is **not** a prospective confidence interval.

Broken true-region training correspondence within tissue category: heldout MSE **0.507769**, versus 0.403703 genuine pairs. Direct CS2→NZ20 and CS2→GT450→NZ20 compositions disagree by feature MSE **0.010746**: the local field is **not** exactly invertible/closed under composition.

## Gates and boundaries

- **PASS:** source-only target prediction; 16,100 ordered matched-pair predictions/method; analytic unique conditional displacement minimum; no free `A(x)` channel; improved reconstruction and region retrieval; category retention within 0.01 balanced accuracy; broken-pair control.
- **FAIL:** full scanner independence (stronger RBF probe still 25.4% vs 20% chance); exact inverse/composition; any proof that the shared B excludes scanner-constant slide-preparation provenance.
- **BLOCKED:** independent biological-reference/preparation intervention, no untouched canine test folds, independently adjudicated diagnosis and native cellular-scale resolution.
- **DO NOT PROMOTE** this model to canonical PA-NF or publication claims. The new method is a form of local kernel response regression and should not be claimed to be uniquely novel without prior-art comparison.

The GitHub branch contains a compact algorithmic core, not the full evaluation runner. Full SHA-checked loader, exact original real-data fold replay, tests (8/8), per-pair errors, broken-pair and nonlinear controls, proof and log are in the conversation package `PA_NF_MAGNET_v4_Canine_20261103.zip`, SHA256 `f02e53f89ad527429b79e9a24c0b75aa4eb30c779076775e242c8854826cbb33`. No original DINOv2 feature arrays, source microscopy, or checkpoints are pushed to GitHub.

**Scientific conclusion:** local empirically measured scanner interactions mitigate the v3 operator's predictive misspecification, but biological/acquisition **purity has not been achieved**. Next mechanism must add independent preparation/biology anchors and an exactly composable invertible scanner representation, and pass external untouched heldout confirmation.
