# Synthetic crossed-factor identifiability diagnostic

## Status

This is a **post-confirmatory exploratory diagnostic**. It does not alter, rerun, or reinterpret the frozen private crossed-target campaign. Its purpose is to determine why the confirmed PA-NF objective produced stable reconstruction-contributing branches without the intended crossed-target semantics.

## Question

Does the unchanged PA-NF objective identify biological and acquisition factors when those factors are known exactly, or does it learn an arbitrary distributed code that only works jointly?

## Design

The experiment generates complete grids of 256 biological identities × 5 scanners under two frozen renderers:

1. Linear: `x = M_b b + 0.75 M_a a + noise`
2. Nonlinear: a frozen random MLP plus a weak linear residual

Each identity has exactly one scanner combination withheld. Every identity and every scanner remains represented during training, but the withheld pair is never shown to the model.

Three model families are trained:

- `pa_nf`: the existing `ScorpionProjection("pathoalign")` and existing `projection_loss`, unchanged.
- `joint_autoencoder`: a negative control with one unconstrained code split arbitrarily into two branches.
- `oracle_supervised`: a positive control whose branches are explicitly supervised against the known biological and acquisition latents.

## Primary endpoint

For identity `b`, unseen scanner `a2`, an observed source `x[b,a1]`, and a different-identity acquisition donor `x[b',a2]`, construct:

```text
D(C(x[b,a1]), A(x[b',a2]))
```

Then calculate:

```text
counterfactual_delta = MSE(swapped, x[b',a2]) - MSE(swapped, x[b,a2])
```

Positive values mean the output is closer to the correct biological identity under the donor scanner than to the donor's biological identity. The script reports a bootstrap interval across identities for every renderer/model/seed run.

## Secondary diagnostics

- `R²(z_b -> b)` and `R²(z_b -> a)`
- `R²(z_a -> a)` and `R²(z_a -> b)`
- combined-code recovery of `[b,a]`
- held-out scanner balanced accuracy from each branch
- biological identity top-1 retrieval
- content- and acquisition-branch ablation penalties

The central failure signature is:

1. combined code recovers the joint factors;
2. both branches contribute to reconstruction;
3. individual branch allocation is impure;
4. held-out counterfactual recombination fails.

That would demonstrate that the objective identifies joint information without identifying its allocation.

## Execution

Run a smoke test first:

```powershell
py experiments/paired_acquisition/run_synthetic_crossed_factor_identifiability.py `
  --mode smoke `
  --device cuda `
  --output-root results/synthetic_crossed_factor_identifiability_smoke_20260731
```

The smoke configuration uses 40 identities, 2 seeds, 20 epochs, both renderers, and all three models.

After the smoke artifact and tests pass, run the full exploratory grid:

```powershell
py experiments/paired_acquisition/run_synthetic_crossed_factor_identifiability.py `
  --mode full `
  --device cuda `
  --output-root results/synthetic_crossed_factor_identifiability_full_20260731
```

The full configuration uses 256 identities, 10 seeds, 250 epochs, 5,000 identity-bootstrap replicates, both renderers, and all three models: 60 total fits.

The output root is fail-closed and cannot already exist. It contains per-run JSON, a dataset manifest, an aggregate JSON result, and a compact CSV summary.

## Interpretation boundaries

This experiment is diagnostic rather than confirmatory. Passing synthetic tests would not establish pathology-domain factorization. Failure on the linear renderer would show that the current objective is structurally non-identifying even under ideal data. Passing linear but failing nonlinear would indicate dependence on restrictive factor geometry. Passing both would redirect diagnosis toward pathology feature sufficiency, imperfect pairing, or tissue-dependent scanner effects.
