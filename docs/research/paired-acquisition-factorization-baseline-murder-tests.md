# Historical simple-baseline audit — superseded interpretation

**Original run date:** 2026-07-03  
**Current status:** descriptive historical results only

The numerical tables produced by the original baseline runner are retained for
auditability. The former conclusion that Paired-Acquisition Neural Factorization
achieved the best scanner-suppression/tissue-preservation trade-off is withdrawn.

## Why the interpretation was withdrawn

### Representation geometry was not a biological endpoint

The historical report interpreted lower cross-scanner cosine after PCA removal
as "tissue damage." That conclusion is not supported. Cosine values can change
under centering, standardization, dimensionality reduction, and learned nonlinear
projections without establishing loss of biological information.

The problem is visible in the k=0 control: applying no scanner-direction removal
still changed cosine and retrieval relative to the original frozen features
because the baseline pipeline changed the feature geometry through
standardization. Consequently, cosine values from those differently transformed
representations cannot serve as a direct measure of retained biology.

### The baseline set did not include the strongest known linear removal result

The historical runner used a standardized logistic/scanner-direction baseline.
A separate oldstyle centroid/QR audit later reached substantially lower raw
scanner-probe accuracy. Therefore, this report cannot support the statement that
no simple baseline approached the neural method on scanner removal.

### The paired baseline and factorization were not capacity matched

The paired-consistency reference used one latent branch, while the factorization
used biological and acquisition branches and reconstructed from both. The old
comparison did not isolate whether scanner suppression came from paired
identification, additional branch capacity, reconstruction routing, scanner
supervision, or individual regularizers.

A frozen capacity-matched ablation runner now exists at:

`experiments/scorpion/run_pathoalign_capacity_matched_ablations.py`

New conclusions require those 175 fits and their preregistered analysis.

## What the historical tables still show

Under the exact original protocols:

- the factorized biological branch had lower linear scanner-probe accuracy than
  the paired-consistency reference;
- PCA and the tested standardized scanner-direction removals changed scanner
  recoverability and representation geometry;
- same-region top-1 retrieval was nearly saturated for many SCORPION
  representations;
- the tested metrics did not establish a biological-preservation ranking.

These are descriptive observations, not proof that one representation preserved
more biological information than another.

## Current permitted comparison language

Safe:

> Under the historical protocol, Paired-Acquisition Neural Factorization reduced
> linearly recoverable scanner identity more than the paired-consistency reference
> and the particular standardized post-hoc baselines tested in that runner.

Safe:

> The oldstyle centroid/QR baseline is the strongest historical raw
> scanner-removal result, but its category-preservation comparison must be
> regenerated under the corrected biological-label audit v2.

Not safe:

> PCA caused tissue damage.

Not safe:

> Neural factorization achieved the best tissue-preserving trade-off.

Not safe:

> No simple baseline came close to the neural method.

Not safe:

> Cosine agreement proves biological preservation.

## Required replacement evidence

1. Run the capacity-matched SCORPION objective ablations.
2. Run the canine biological-label audit v2 with the oldstyle centroid/QR
   baseline included under fit-only, sample-blocked evaluation.
3. Evaluate downstream labels or hard retrieval outcomes that exclude same-region
   and same-sample shortcuts.
4. Report scanner suppression, representation agreement, downstream retention,
   rank, and acquisition capture as separate outcomes rather than one informal
   trade-off score.
5. Publish new outputs through the forward-valid provenance contract.

## Historical artifacts

The original result directories remain available:

- `results/paired_acquisition_factorization_baseline_murder_test/`
- `results/paired_acquisition_factorization_baseline_murder_test_caninescc/`

They are not current authority for biological preservation or baseline supremacy.
The repository-wide authority is `CLAIM_BOUNDARY.md`.
