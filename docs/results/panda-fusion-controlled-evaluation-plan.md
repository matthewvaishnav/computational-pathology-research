# Controlled PANDA fusion evaluation plan

Status: draft implementation plan for issue #42.

## Scientific question

Does any non-degenerate TransnnMIL fusion variant improve over both standalone
branches and simple fusion controls when every model is trained and evaluated
under the same locked protocol?

The models are:

1. standalone nnMIL;
2. standalone TransMIL;
3. historical TransnnMIL;
4. concatenation fusion;
5. gated fusion;
6. branch-attention fusion.

No model should be called improved before the full comparison is complete.

## Claim boundary

The public PANDA release is a development resource. Results produced by this
plan are internal development-set evidence, even when a confirmation partition
is kept untouched until the final comparison. They are not a replacement for
blinded external validation on an independent cohort.

## Locked data protocol

Generate one split assignment with
`scripts/data/build_panda_locked_split_manifest.py` and reuse the resulting CSV
unchanged for every architecture and seed.

Recommended initial partition:

- train: 70%;
- selection: 15%;
- confirmation: 15%.

Stratify by provider and ISUP grade when provider metadata is available. The
script records the source-manifest SHA-256, output SHA-256, seed, fractions, and
partition counts. The split assignment and metadata must be archived with the
experiment outputs before training begins.

The selection partition is used for checkpoint selection and early stopping.
The confirmation partition is evaluated once per finalized seed/model run and
must not influence architecture, hyperparameter, or epoch selection.

## Locked optimization protocol

All six models must share:

- the same feature files and patch sampling policy;
- the same split manifest;
- the same repeated seed list;
- the same optimizer, learning-rate schedule, weight decay, epoch budget,
  gradient clipping, early stopping rule, and class weighting;
- the same batch size unless a documented memory constraint requires a change;
- the same metric implementation.

Any architecture-specific exception must be declared before results are viewed.

## Required outputs per run

Each model/seed run must save:

- configuration and git commit SHA;
- split-manifest and source-manifest hashes;
- checkpoint selected only on the selection partition;
- selection and confirmation predictions keyed by image_id;
- QWK, accuracy, macro-F1, confusion matrix, parameter count, and wall-clock
  training/inference time;
- branch-specific predictions for TransnnMIL variants;
- branch ablation results;
- branch gradient norms during training;
- learned branch weights and collapse fractions for gate and attention variants.

## Repeated-seed analysis

Use at least five prespecified seeds. Report every seed plus mean, standard
deviation, minimum, and maximum. Do not report only the best run.

For each fusion model, compare confirmation predictions with both standalone
branches and with concat/gate controls using paired bootstrap confidence
intervals for the QWK difference. Bootstrap resampling must operate on the same
confirmation cases for both models.

## Decision rule

A fusion variant may be described as improved only when it:

1. exceeds standalone nnMIL and standalone TransMIL on the untouched
   confirmation partition;
2. exceeds historical TransnnMIL;
3. exceeds or clearly differentiates itself from concat and gate controls;
4. shows a stable paired effect across seeds;
5. does not exhibit practical branch collapse.

Beating only the known-degenerate historical fusion is insufficient.

## Implementation sequence

1. Generate and archive the locked split manifest.
2. Refactor the trainer to consume explicit train/selection/confirmation rows.
3. Add standalone nnMIL and TransMIL to the identical runner.
4. Add runtime, parameter, prediction, branch-ablation, and collapse exports.
5. Add a paired repeated-seed aggregation script.
6. Run CPU smoke tests before any full PANDA training.
7. Run the prespecified seed/model matrix and produce publication-ready tables.

## Merge boundary

This branch should remain a draft until the runner can execute all six models
on one immutable split assignment and tests verify that confirmation rows are
never used for checkpoint selection.
