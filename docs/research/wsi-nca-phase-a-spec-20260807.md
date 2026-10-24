# WSI-NCA Phase A — Frozen Falsification Specification

**Frozen:** 2026-08-07  
**Status:** proposed architecture / unvalidated research line  
**Branch:** `research/wsi-nca-phase-a-20260807`

## Claim boundary

This experiment does **not** claim a new biological mechanism, clinical utility,
state-of-the-art performance, self-repair, scanner invariance, or superiority to
MIL/GNN baselines. Phase A asks one bounded question:

> Can repeated application of one shared local neural update law over true WSI
> patch topology produce a more useful slide representation than a static readout
> over the same frozen patch features?

A positive result is architecture evidence only. A negative result is retained.

## Why this experiment is isolated from the paired-acquisition core

PA-NF is the umbrella computational-pathology research system, and its
paired-acquisition core already studies explicit tissue/acquisition structure.
Phase A intentionally does **not** add tissue/acquisition factorization. The
cellular dynamics must first earn their existence independently. Only a
forward-valid positive result warrants the later Factorized Tissue Dynamics
experiment.

## Data contract

Each slide is represented by an HDF5 file containing:

- `features`: float array `[N, D]` from a frozen feature extractor;
- `coordinates`: integer/float array `[N, 2]` in a common slide coordinate frame.

The existing OpenSlide PANDA extractor already writes both datasets. Training
must fail closed when coordinates are absent or misaligned with features.

## Model contract

For slide patch `i`:

1. Initialize state from the frozen feature:

   `h_i^0 = phi(x_i)`

2. Construct a spatial k-nearest-neighbour graph from WSI coordinates.

3. Reuse **one and the same** local transition module at every cell and every
   developmental step:

   `h_i^(t+1) = U_theta(h_i^t, {h_j^t, p_j-p_i : j in N(i)})`

4. Read out the final states with the exact same attention readout used by the
   `T=0` control.

The defining Phase A property is parameter sharing across developmental time.
A stack of independently parameterized GNN layers is a baseline, not WSI-NCA.

## Primary controlled comparison

Run matched training/evaluation with:

- `T = 0` — static control;
- `T = 1`;
- `T = 2`;
- `T = 4`;
- `T = 8`;
- `T = 16`.

All runs must use the same:

- frozen feature files;
- slide split;
- seeds;
- initializer;
- readout;
- classifier;
- tuning budget;
- optimizer family;
- evaluation metrics.

The primary question is not whether one hand-picked `T` wins. It is whether a
repeatable developmental-depth effect exists under matched controls.

## Spatial falsification controls

For every credible positive result, compare:

1. **True spatial topology** — kNN over WSI coordinates.
2. **Shuffled coordinate assignment** — same coordinate set, randomly reassigned
   to patch embeddings within each slide.
3. **Embedding topology** — kNN over initial patch states rather than tissue
   coordinates.
4. **Fixed-depth GNN control** — comparable parameter/compute budget without
   recurrent temporal weight sharing.

A result that survives coordinate shuffling does not support a claim that tissue
organization caused the gain.

## Required implementation invariants

- Padded cells cannot send messages or enter slide readout.
- Self-neighbours are excluded.
- Relative position, not absolute slide origin, enters local messages.
- `T=0` uses the same initialization, readout, and classifier as `T>0`.
- Neighbor topology is inspectable in model outputs.
- Cell states are returned for trajectory/perturbation analysis.
- No random projection may be created inside `forward`.
- All trainable modules must be registered and reproducible.

## Phase A metrics

For PANDA ordinal grading feasibility:

- quadratic weighted kappa (QWK);
- macro F1;
- accuracy;
- validation loss.

Performance metrics alone do not establish self-organization. Dynamics analysis
is required before that wording is promoted.

## Secondary dynamics analyses

Only after the primary comparison is complete:

- state-change norm by time step: `||H^(t+1)-H^t||`;
- sensitivity to additional inference steps beyond the training horizon;
- local/contiguous cell-state ablation;
- recovery after state damage;
- trajectory similarity across slides.

"Self-repair", "attractor", and "regeneration" remain prohibited claims until
those behaviours are directly measured against appropriate controls.

## Promotion gate to Factorized Tissue Dynamics

Do not merge PA-NF paired-acquisition factorization into this architecture merely
because the model trains.

Proceed only if at least one forward-valid result shows that:

- recurrent local dynamics add signal beyond `T=0`; and
- true spatial topology matters relative to a shuffled-topology control; and
- the result is not explained by an obvious parameter/compute mismatch.

At that point the next hypothesis is:

> Tissue-oriented and acquisition-oriented latent fields may obey different
> spatial dynamics, with paired acquisitions providing direct constraints on
> which cross-scanner structure should converge and which acquisition structure
> should remain represented.

That is a separate experiment and inherits the PA-NF claim boundary: partial
structured separation only unless stronger evidence is obtained.
