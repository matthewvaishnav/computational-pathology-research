# WSI-NCA PANDA-300 frozen scientific ladder

Status: **completed, bounded negative result**.

## Protocol

The post-calibration protocol was frozen before execution:

- cohort: frozen 300-slide PANDA coordinate-feature bundle;
- split: stratified 180 train / 60 validation / 60 test with `split_seed=42`;
- model seeds: `7, 19, 42, 43, 67`;
- 20 epochs, batch size 4, maximum 512 patches/slide;
- hidden width 256, 8 spatial neighbours;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- checkpoint selection: maximum validation QWK only;
- held-out test metrics were not used for tuning.

Each seed evaluated:

1. T0 static real→real;
2. T1 tied real→real;
3. T4 tied real→real and, from the same selected checkpoint, real→shuffled test topology;
4. T4 tied shuffled-train→shuffled-test;
5. T4 untied real→real.

## Held-out test result

| Control | Mean QWK ± SD | Mean accuracy | Mean macro-F1 |
|---|---:|---:|---:|
| T0 static real→real | 0.6777 ± 0.0328 | 0.4867 | 0.4039 |
| T1 tied real→real | 0.6665 ± 0.0938 | 0.4400 | 0.3770 |
| T4 tied real→real | 0.6469 ± 0.0425 | 0.4833 | 0.4306 |
| T4 tied real→shuffled test | 0.6546 ± 0.0311 | 0.4867 | 0.4206 |
| T4 tied shuffled→shuffled | 0.6401 ± 0.0518 | 0.4733 | 0.4150 |
| T4 untied real→real | 0.6572 ± 0.0589 | 0.4867 | 0.4010 |

## Paired seed diagnostics

Using QWK and pairing controls by model seed:

- T4 tied real minus T1: mean `-0.0195`, median `-0.0135`; T4 higher in 2/5 seeds.
- T4 tied real minus T0: mean `-0.0308`, median `-0.0372`; T4 higher in 1/5 seeds.
- T4 tied real minus the same checkpoint under shuffled test coordinates: mean `-0.0077`, median `+0.0023`; real topology higher in 3/5 seeds.
- T4 tied real minus shuffled-train/shuffled-test: mean `+0.0068`, median `-0.0104`; real-trained T4 higher in 2/5 seeds.
- T4 tied minus T4 untied: mean `-0.0103`, median `-0.0454`; tied higher in 2/5 seeds.

With only five seeds, these are descriptive paired diagnostics, not a claim of statistical significance.

## Interpretation

The frozen PANDA-300 experiment **does not support a recurrence advantage**. T4 tied did not outperform T1 or the static T0 control on mean held-out QWK.

The experiment also **does not support dependence of the predictive result on the real spatial feature/topology correspondence**. Shuffling held-out coordinates for the same selected T4 checkpoint did not cause a consistent QWK decrease; the mean QWK was slightly higher under shuffled evaluation. Training and evaluating T4 on shuffled topology also remained close to the real-topology T4 result.

The untied T4 control had a slightly higher mean QWK than tied T4, so this experiment does not establish tied predictive superiority. Any parameter-efficiency statement must remain architectural/descriptive rather than predictive.

## Claim boundary

Supported:

> On the frozen five-seed PANDA-300 protocol, repeated tied spatial updates did not produce a held-out QWK advantage over either T1 or the static T0 control, and the T4 result did not consistently deteriorate when the same selected checkpoint was evaluated with shuffled coordinates. The tested real-data setting therefore does not support a recurrence-dependent or topology-dependent predictive advantage.

Not supported:

- PANDA benefit from WSI-NCA recurrence;
- predictive superiority of tied over untied dynamics;
- a pathology-utility or clinical-value claim;
- self-repair, attractors, regeneration, morphogenesis, or acquisition-invariance claims.

The earlier synthetic receptive-field-depth experiment remains a separate mechanism result: it demonstrates representational capability of repeated local propagation on the constructed matched-multiset task, not PANDA utility.

Machine-readable aggregate: `results/wsi_nca_phase_a/panda_300_scientific_ladder_summary.json`.
