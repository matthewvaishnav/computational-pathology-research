# Data provenance and claim boundaries

## Canonical statement

The pathology benchmarks used for scientific evaluation in this repository are derived from **real human histopathology data**.

- **CAMELYON16/17** contain digitized human breast-cancer lymph-node tissue.
- **PatchCamelyon (PCam)** contains pathology patches derived from CAMELYON lymph-node slides.
- **PANDA** contains digitized human prostate-biopsy tissue.

The repository must not describe these datasets, their image tensors, their extracted features, or results computed from them as synthetic.

## CAMELYON17 evidence in this repository

The active CAMELYON17/WILDS work uses the real multi-center pathology dataset. The audited dataset contains 455,954 labeled image examples across five centers, with source-domain, out-of-distribution validation, and held-out out-of-distribution test centers. The image pipeline has been exercised with frozen ImageNet features and CAMELYON17-trained supervised ResNet18 features.

The main evidence note is:

```text
docs/research/camelyon17-external-center-validation-note.md
```

Associated scripts and result artifacts are listed in that note.

## Test fixtures are not research datasets

A small number of legacy scripts create generated tensors or mock slide manifests for unit tests and smoke tests. Those fixtures exist only to test software behavior such as batching, masking, serialization, and failure handling. They are not the source of the repository's reported CAMELYON, PCam, or PANDA scientific results.

Any document that mentions a generated fixture must label it explicitly as **test-only** or **software-only** and must not generalize that fixture into a statement about the project's research data.

## Required wording

Use:

> Evaluated on real human histopathology data from public pathology benchmarks; generated fixtures are used only for isolated software tests.

Do not use:

> The repository is synthetic-only.

> CAMELYON results are synthetic.

> Real CAMELYON data has not been integrated.

## Clinical boundary

Real human research data is not the same as clinical deployment or prospective clinical validation. The project remains research-only and is not diagnostic software or a patient-care system.
