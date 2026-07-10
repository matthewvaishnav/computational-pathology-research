# Paired Scanner Counterfactual Benchmark

This benchmark defines paired multi-scanner histopathology as observed
counterfactual data: the same tissue region is observed under different
scanner or acquisition conditions. The core object is:

```text
x(region, scanner)
```

The benchmark asks whether representations, decoders, and future pixel-level
models can preserve the region identity while changing the acquisition
condition.

## Benchmark layers

### Layer 1: Feature-space counterfactual audit

Question: can frozen/pathology representation features separate biological
structure from acquisition variation?

Currently supported by repo artifacts when a dataset has feature arrays,
scanner identifiers, pair or region identifiers, and optional category labels.
Existing artifacts include frozen-feature paired-region audits, branch
separation outputs, bottlenecked leakage-control runs, and the oldstyle
centroid/QR baseline family.

### Layer 2: Decoder-space counterfactual audit

Question: can biological and acquisition branches be recombined so decoded
features behave like the target acquisition while preserving biological
identity?

Currently supported when projected biological/acquisition branch archives and
decoder or composition weights are present. Existing artifacts include
projected feature archives with acquisition branch arrays, checkpoints with
biological/acquisition/decoder weights, pair-assignment metadata, and acquisition
swap audit evidence.

### Layer 3: Pixel-space paired scanner counterfactual reconstruction

Question: given an image patch from scanner A and target scanner B, can a model
predict the real paired scanner-B image patch of the same tissue region?

This layer is future work unless the repo contains actual paired patch images or
WSI paths with registration-ready coordinates and registration confidence or QC
metadata. The benchmark must not synthesize or infer pixel-level supervision
when those files are absent.

## Supported now

The repository can support a benchmark v0 for feature-space and decoder-space
audits where the feasibility audit confirms:

- scanner labels are available;
- paired region/sample identifiers are available;
- frozen feature arrays are available;
- biological and acquisition branch embeddings are available;
- decoder/composition weights are available;
- pair-assignment or swap metadata is available.

Category labels are not required for pair retrieval, but they are required for
category preservation and leakage metrics.

## Future pixel requirements

Pixel-space reconstruction requires explicit evidence for paired image patches
or WSI paths, local region correspondence, patch coordinates, scanner-specific
paired acquisitions, and registration confidence or QC rules. If any of these
are missing, Layer 3 remains a design target rather than a runnable benchmark.

## Claim boundaries

Allowed language:

- observed scanner counterfactuals;
- paired acquisition as supervision;
- feature-space counterfactual audit;
- decoder-space factor-like evidence;
- pixel-space future work;
- conditional cross-dataset evidence.

Unsupported language:

- universal disentanglement proven;
- pixel-level acquisition modeling proven;
- clinical validation;
- diagnostic performance;
- deployment or patient-care readiness;
- FDA or HIPAA readiness;
- scanner bias solved;
- scanner-free representation;
- perfect causal factorization;
- breakthrough or solves scanner bias claims.
