# PA-NF v2 crossed-intervention development

## Status

Development-only. This line does **not** reinterpret or tune against the frozen 29 September 2026 SCORPION shortcut/conflict outcomes. PA-NF v1 and those outcomes remain unchanged.

## Why v2 exists

The 29 September falsification showed that PA-NF v1 biological features resisted a deliberately injected scanner shortcut relative to raw DINOv2 and strongly separated biological from acquisition behavior, but the registered equal-capacity two-branch control tied PA-NF on the primary downstream superiority endpoints.

The v2 question is therefore stronger and more structural:

> Can biological and acquisition codes be made independently reusable under acquisition intervention, rather than merely showing lower scanner probe accuracy?

## Architecture

For every observation `x`:

- `z_b = E_b(x)` is the image-inferred biological representation.
- `z_a = E_a(x)` is the image-inferred acquisition representation.
- `D(z_b, z_a)` reconstructs an observation.

Scanner identity is **not** an inference input. Known scanner labels are used only during development training to anchor image-inferred acquisition embeddings to learned scanner prototypes.

## Candidate objective

For paired observations of the same tissue identity under scanners `s` and `t`:

1. Self reconstruction: `D(E_b(x_s), E_a(x_s)) -> x_s`.
2. Crossed reconstruction: `D(E_b(x_s), E_a(x_t)) -> x_t`.
3. Factor cycle: re-encode the crossed output and recover the source biological code plus target acquisition code.
4. Same-identity biological consistency across observed scanner views.
5. Acquisition prototype anchoring for known scanners during training only.
6. Biological variance floor and prototype regularization to resist collapse.

## Primary causal control

`v2_control_no_cross_cycle` uses the **exact same architecture and parameter count**. It receives the same self-reconstruction, biological-consistency, acquisition-organization and anti-collapse terms. Only crossed reconstruction and re-encoding cycle weights are set to zero.

This makes the development contrast specifically test whether compositional crossed/cycle training adds value beyond an equally large two-branch model.

## Development data

The runner uses the existing synthetic known-factor crossed grids with linear and nonlinear renderers. Each biological identity has five scanner states and one identity/scanner combination is withheld for evaluation.

The registered 29 September SCORPION outcome files are not read by this runner.

## Development promotion gate

Promotion requires all of the following on the synthetic development grid:

- exact candidate/control parameter equality;
- candidate crossed-factorization success for every development seed;
- mean acquisition-transfer delta greater than control;
- mean biology-retention delta greater than control;
- mean cycle error lower than control.

This gate only decides whether v2 is worth freezing. Passing it is **not** confirmatory evidence for pathology robustness.

## Smoke run

```powershell
python experiments\paired_acquisition\run_pa_nf_v2_crossed_intervention_development.py `
    --mode smoke `
    --device cuda
```

Smoke mode uses 64 identities, 80 epochs, three model seeds, both renderers and both capacity-matched model families.

The default output directory is:

`results/pa_nf_v2_crossed_intervention_development_smoke`

The runner is fail-closed and refuses to overwrite an existing output directory.

## Full development grid

Use a fresh output root:

```powershell
python experiments\paired_acquisition\run_pa_nf_v2_crossed_intervention_development.py `
    --mode full `
    --device cuda `
    --output-root results\pa_nf_v2_crossed_intervention_development_full_20260929
```

Full mode uses 256 identities, 250 epochs and ten model seeds.

## Prospective confirmation boundary

If v2 earns promotion, architecture, loss weights and confirmation gates must be frozen before any new confirmatory outcomes are read.

Preferred confirmation: five-rotation leave-one-scanner-out transport with frozen PA-NF v2 versus an identical-capacity v2 control lacking crossed/cycle losses. PA-NF v1 and raw DINOv2 remain secondary comparators.
