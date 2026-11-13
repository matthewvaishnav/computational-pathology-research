# WSI-NCA receptive-field-depth falsification

**Executed:** 2026-08-10

**Status:** synthetic mechanism evidence only; not pathology evidence

**Machine-readable record:**
`results/wsi_nca_phase_a/synthetic_receptive_field_depth.json`

## Scientific question

Can repeated application of one shared local update rule extract a signal that
one spatial pass cannot?

## Causal construction

Both labels use the same eight equally spaced coordinates and the same four-zero,
four-one feature multiset. The class templates are `00101101` and `00110101`.
Their exact multisets of colored, signed-edge rooted neighborhoods are identical
at depths zero and one and first differ at depth two. Therefore:

- T=0 cannot separate the labels from the unordered features;
- T=1 cannot separate them even with the global attention readout, because its
  complete multiset of cell states is matched by construction;
- T>=2 is representationally capable because two-hop signatures differ;
- coordinate reassignment destroys the stable feature/topology correspondence.

Each generated class pair shares a random global feature complement and a random
coordinate translation. The model uses only relative positions inside messages.
Across three random initializations, the maximum random-weight T=1 class-logit
difference was `1.19e-7`, numerically confirming that the attention readout did
not bypass the one-hop boundary.

## Locked 60-epoch observations

Validation accuracy for seeds `[7, 19, 43]`:

| Control | Per-seed accuracy | Mean | Parameters |
|---|---:|---:|---:|
| T=0, tied, real topology | 0.50 / 0.50 / 0.50 | 0.500 | 6,891 |
| T=1, tied, real topology | 0.50 / 0.50 / 0.50 | 0.500 | 6,891 |
| T=2, tied, real topology | 1.00 / 0.50 / 1.00 | 0.833 | 6,891 |
| T=4, tied, real topology | 1.00 / 0.50 / 1.00 | 0.833 | 6,891 |
| T=4 real-trained, shuffled evaluation | 0.539 / 0.500 / 0.477 | 0.505 | 6,891 |
| T=4, tied, shuffled train/evaluation | 0.469 / 0.488 / 0.496 | 0.484 | 6,891 |
| T=4, untied, real topology | 1.00 / 0.50 / 1.00 | 0.833 | 25,179 |

The two successful tied T4 seeds fell to mean accuracy `0.508` when evaluated
with reassigned coordinates. The untied T4 control used 18,288 more parameters
(`3.654x` total) but had the same per-seed locked-horizon accuracy as tied T4.
This is parameter efficiency on this task, not evidence that tied dynamics are
generally superior.

## Negative finding and diagnosis

Seed 19 remained at chance on both training and validation for T=2 tied, T=4
tied, and T=4 untied at the locked 60-epoch horizon. This failure was retained in
the primary aggregate.

It was not caused by readout leakage, label-bearing geometry, or a hard
architectural collision: seed 19 had nonzero initial class-sensitive slide states,
logits, and gradients. In a labeled post-hoc diagnostic that changed only the
training horizon from 60 to 240 epochs, all three failed runs reached 1.00 train
and validation accuracy. The bounded diagnosis is optimization sensitivity at
the locked horizon.

## Supported and unsupported statements

Supported:

> On this constructed matched-multiset task, repeated use of the shared local
> rule propagated information beyond one hop and enabled separation that T=0 and
> T=1 could not express. The useful signal depended on the real feature/topology
> correspondence.

Not supported:

- reliable optimization across seeds at the 60-epoch budget;
- an advantage of tied over untied depth in predictive accuracy;
- generalization to unseen graph families or pathology data;
- self-repair, attractors, regeneration, morphogenesis, or clinical utility.

The task intentionally uses two fixed motif families with translation and global
feature-complement nuisance variation. It establishes a receptive-field-depth
mechanism, not broad synthetic generalization.

## Next promotion gate

Execute the PANDA causal ladder on one frozen coordinate-bearing split with
multiple seeds: T0 -> T1 -> T4, T4 real -> shuffled, and tied T4 -> untied T4.
No recurrent-dynamics claim is promoted unless T4 improves on T1 and the gain
depends on real tissue topology.
