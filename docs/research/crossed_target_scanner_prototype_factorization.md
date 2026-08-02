# Crossed-Target Scanner-Prototype Factorization

## Status

Post-confirmatory exploratory architecture experiment.

This work does not reinterpret the failed unchanged PA-NF objective. It begins a
new method line whose claims must be earned on known-factor synthetic systems
before any return to pathology features.

## Research question

Under what architectural and supervisory constraints are paired acquisitions
sufficient to identify intervention-consistent biological and acquisition
representations?

The immediate test is deliberately narrow:

> Does replacing unrestricted per-observation acquisition codes with shared
> scanner prototypes, together with explicit crossed-target supervision, produce
> correct recombination on unseen biological-identity × scanner cells?

## Motivation

The preceding two-axis diagnostic established that excellent reconstruction and
high factor recoverability do not imply factorization. The unchanged PA-NF
objective permitted an acquisition branch that carried both biology and scanner
state and dominated the decoder. Crossed decoding therefore followed donor
biology instead of preserving source biology.

The next experiment changes the objective and architecture rather than tuning
the failed loss.

## Proposed model

For observation `x[b,s]`, biological identity `b`, and scanner `s`:

```text
z_b = C_theta(x[b,s])
p_s = learned scanner prototype
x_hat[b,s'] = D_phi(z_b, p_s')
```

There is no per-observation acquisition encoder. Every sample from scanner `s`
receives the same prototype `p_s`, so the acquisition path cannot carry
sample-specific donor biology.

The decoder uses prototype-conditioned feature-wise modulation rather than an
unrestricted concatenation-only decoder.

## Training objectives

### Self reconstruction

```text
D(C(x[b,s]), p_s) -> x[b,s]
```

### Crossed-target reconstruction

For two observed scanner views of the same identity:

```text
D(C(x[b,s1]), p_s2) -> x[b,s2]
```

This directly specifies which factor must be retained and which factor must be
changed.

### Biological consistency

```text
C(x[b,s1]) ~= C(x[b,s2])
```

A variance floor prevents content collapse. Scanner prototypes are centered and
encouraged to remain distinct.

## Model families

1. **PA-NF** — unchanged failed objective, rerun as a direct same-grid reference.
2. **Prototype reconstruction** — scanner prototypes and structured decoder, but
   no crossed-target loss. This isolates architectural sufficiency.
3. **Crossed-target prototype** — the proposed architecture plus explicit
   crossed-target supervision.
4. **Supervised oracle** — known-factor positive control.

## Primary evaluation

Every biological identity has one held-out scanner cell. The model receives a
biological code from an observed scanner and the prototype of the held-out
scanner.

Two independent contrasts are required:

1. **Biology retention:** output must be closer to the correct identity under the
   target scanner than to a different identity under that scanner.
2. **Acquisition transfer:** output must be closer to the correct target scanner
   than to the source identity under its original scanner.

Both bootstrap intervals must be above zero, and a majority of identities must
succeed on both axes.

## Factor-allocation gates

The biological branch must:

- recover biological factors;
- exclude acquisition factors.

The scanner-prototype branch must:

- recover acquisition factors;
- exclude biological factors.

Combined factors must remain recoverable.

The proposed model succeeds only when both two-axis recombination and factor
allocation pass.

## Interpretation matrix

| Prototype reconstruction | Crossed-target prototype | Interpretation |
|---|---|---|
| fail | pass | Crossed supervision supplies the missing identification signal |
| pass | pass | Shared prototypes and structured decoding are sufficient; crossed loss is redundant under this synthetic regime |
| fail | fail | Current architecture/objective remains insufficient |
| pass | fail | Implementation or optimization defect; crossed training should not degrade a valid factorization without explanation |

The supervised oracle must pass and PA-NF must remain rejected. Otherwise the
experiment is not interpretable.

## Execution gate

Smoke mode uses 40 identities, two seeds, and 100 requested epochs. The oracle
receives at least 250 epochs through the validated v2 control logic.

The full experiment is not allowed unless:

- every oracle seed passes;
- every PA-NF seed is rejected;
- every crossed-target prototype seed passes.

The architecture-only control determines whether the crossed objective adds
incremental value but does not independently close the path if the proposed
model succeeds.

## Relation to the broader architecture path

The long-term objective is a general computational-pathology neural backbone,
not a one-off scanner harmonizer. Paired acquisition is the current
identifiability path because it provides unusually strong controlled variation:
the tissue is fixed while acquisition changes.

This stage remains intentionally small. A method that cannot identify known
factors on linear and nonlinear crossed grids has no justification for being
scaled to whole-slide pathology.

After synthetic success, the next gates are:

1. frozen pathology-feature crossed reconstruction;
2. donor-invariance and scanner-intervention stress tests;
3. comparison with additive, affine, low-rank, and unrestricted decoders;
4. tissue-preservation evaluation under held-out slides and scanner pairs;
5. only then integration into the broader computational-pathology backbone.
