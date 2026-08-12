---
license: mit
pretty_name: Paired-Acquisition Factorization Evidence
language:
  - en
tags:
  - computational-pathology
  - representation-learning
  - reproducibility
  - paired-acquisition
  - negative-results
---

# Paired-Acquisition Factorization Evidence

This dataset repository is a compact, machine-readable evidence release for two
distinct Paired-Acquisition Neural Factorization (PA-NF) comparisons. It is an
evidence dataset, not a trained model, frozen feature archive, or image dataset.

## Data origin

Files are copied without reinterpretation from the authoritative program hub:

- `evidence/paired_acquisition/corrected-20260726`;
- `evidence/paired_acquisition/scorpion-capacity-matched-20260726`.

The source repository is
`matthewvaishnav/computational-pathology-research` at immutable release-source
commit `edf8b2b96fbdb8b21fbecc03b8a19ac0351e1dce`. The SCORPION campaign itself
was executed at `0adea50f1ef22865969109f1834a3c175e3f8b43`.

## Unit of observation

Units differ by file and are explicitly named in the manifests: registered
fit/cell, fold-level contrast, seed-averaged slide or biological-sample metric,
and aggregate evidence record. Rows must not be treated as independent patients
or clinical outcomes.

## Schema

| Path family | Format | Meaning |
|---|---|---|
| `corrected-20260726/` | JSON, CSV, Markdown | Corrected SCORPION and canine fixed-estimand evidence, environment, manifests, and bounded interpretation. |
| `scorpion-capacity-matched-20260726/campaign/` | JSON, JSONL, CSV | Registered campaign design, complete append-only run ledger, inventory, and run metrics. |
| `scorpion-capacity-matched-20260726/analysis/` | JSON, CSV | Fold-aware analysis specification, summaries, and contrasts. |
| `release-provenance.json` | JSON | HF preparation time, exact Git source, source paths, and release-spec hash. |
| `checksums.sha256` | SHA256 text | Complete integrity inventory for the HF release folder. |

Each evidence family includes its own schema version and release manifest. Those
machine-readable manifests are authoritative over this overview.

## Sample counts

- SCORPION capacity-matched campaign: 7 variants × 5 folds × 5 seeds =
  **175/175 valid registered fits**, 36 aggregate contrast rows, and no failed,
  invalid, or mixed-configuration cells.
- Corrected canine dimensionality × cross-covariance campaign:
  **450/450 registered cells** in the promoted repository record.
- SCORPION source design: 2,400 patches from 480 aligned regions on 48 original
  H&E slides scanned by five devices.

These counts describe the evidence designs; raw images and external feature
arrays are not included.

## Preprocessing

This release does not re-run, normalize, or transform scientific results. The
pre-registered analysis outputs, environment records, command records, and
hashes are copied from Git. Producer and analysis code paths are recorded in each
release manifest. Reproduction must use those manifests and immutable commits.

## Coordinates

No pixel or WSI coordinates are distributed. Same-region correspondence and
slide/sample blocking are represented through the tracked study manifests and
analysis design, not through image data in this HF repository.

## Exclusions

The compact release intentionally excludes:

- raw SCORPION or canine images;
- frozen foundation-model feature archives;
- projected feature arrays;
- trained checkpoints;
- durable terminal logs and local run directories;
- superseded canine analyses;
- historical slide-independent SCORPION inference;
- unified cross-protocol leaderboard interpretations.

External artifacts referenced by hash in a release manifest are not silently
reconstructed or represented as included files.

## Checksums

`checksums.sha256` is generated after the release folder is assembled. The
publishing tool verifies every local file, uploads in one commit, downloads every
released file at the returned immutable HF revision, recomputes SHA256, and only
then records that revision in the GitHub release registry.

## Licensing

The researcher-authored aggregate evidence package is distributed under the MIT
license inherited from the source repository. Underlying images, external
features, third-party model weights, and their licenses are not redistributed or
relicensed by this release.

## Intended use

- reproduce the promoted fold-aware aggregate analyses;
- audit registered campaign completeness and provenance;
- inspect the distinct positive SCORPION and negative canine boundaries;
- build evidence-aware comparisons without downloading raw pathology images.

## Limitations

This release is not evidence of pure biological factors, complete scanner
invariance, information-theoretic independence, universal harmonization
superiority, diagnostic improvement, patient benefit, or clinical readiness.
Probe and retrieval metrics are representation diagnostics. The canine public
source is bounded by its documented coarse 4 µm/pixel release resolution.

## Provenance

- Program hub: <https://github.com/matthewvaishnav/computational-pathology-research>
- Release-source commit: `edf8b2b96fbdb8b21fbecc03b8a19ac0351e1dce`
- SCORPION execution commit: `0adea50f1ef22865969109f1834a3c175e3f8b43`
- Corrected evidence manifest:
  `evidence/paired_acquisition/corrected-20260726/release_manifest.json`
- Capacity-matched evidence manifest:
  `evidence/paired_acquisition/scorpion-capacity-matched-20260726/release_manifest.json`
- Current claim boundary:
  <https://github.com/matthewvaishnav/computational-pathology-research/blob/main/CLAIM_BOUNDARY.md>

## Citation

Use the repository `CITATION.cff` and cite the primary foundations manuscript:

> Matthew Vaishnav. *Accountable Neural Aggregation in Computational Pathology:
> From Paired-Acquisition Representations to Whole-Slide and Institutional
> Learning* (2026).

Also cite the original SCORPION and canine dataset publications when using the
corresponding evidence family.

## Claim boundary

Supported SCORPION statement:

> On the registered SCORPION structured-separation objective, PA-NF has a
> controlled comparative advantage over the equal-capacity two-branch neural
> control: tissue-branch scanner balanced accuracy is reduced by `0.3108` with a
> fold-aware 95% interval of `[-0.3346, -0.2858]`, registered same-region
> retrieval noninferiority is preserved, and acquisition-branch scanner
> information remains strong.

Separate supported canine statement:

> Under the corrected five-category canine fixed-estimand comparison, no
> additional neural feature-space improvement over the strongest simple
> scanner-removal baselines was established.

The canine result does not negate the SCORPION controlled advantage, and the
SCORPION result does not establish superiority to every simple scanner-removal
method. Both remain endpoint-, comparator-, dataset-, and protocol-specific.
