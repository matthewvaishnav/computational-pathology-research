# PLISM feasibility package

This package converts the public PLISM deposits into deterministic, provenance-bounded metadata products before any image payload is downloaded or any model is trained.

## Scientific purpose

PLISM is the strongest public feasibility target found for separating preparation and scanner effects because it combines:

- 13 H&E staining conditions applied to serial sections;
- seven whole-slide scanner domains;
- 91 original whole-slide images;
- registered tile correspondences with shared coordinates;
- 46 tissue categories;
- public Figshare Plus deposits under CC BY 4.0.

The package does **not** treat serial sections as identical tissue, scanner model names as verified physical device identities, or registered tiles as proof of causal preparation effects.

## Current status

The metadata-only Figshare manifest stage is runnable. The crossing stage is deliberately **fail-closed** for the original public five-column PLISM image list because that table does not expose a verified slide, section, WSI, archive-group, or equivalent provenance parent.

A coordinate is local to a source image and cannot by itself identify an independent registered field across the archive. The crossing analysis must not run until a provenance-enriched input or independently verified mapping supplies a parent identifier that is stable across rescans of the same stained section and distinguishes different serial sections.

## Commands

Fetch only public deposit metadata and write a normalized manifest:

```powershell
python benchmarks\plism_feasibility\fetch_figshare_manifest.py
```

Run deterministic offline self-tests:

```powershell
python benchmarks\plism_feasibility\fetch_figshare_manifest.py --self-test
python benchmarks\plism_feasibility\build_crossing_matrix.py --self-test
```

Check a previously generated manifest without network access:

```powershell
python benchmarks\plism_feasibility\fetch_figshare_manifest.py --check-manifest
```

After recovering and validating a provenance-bearing parent field, run the crossing audit on a provenance-enriched CSV:

```powershell
python benchmarks\plism_feasibility\build_crossing_matrix.py --input path\to\plism_provenance_enriched.csv
```

Accepted parent headers include `slide_id`, `section_id`, `wsi_id`, `archive_group_id`, or an equivalent documented parent identifier. The unmodified public five-column image list is expected to fail with `missing_or_ambiguous_header:parent_id`.

## Outputs

Manifest stage:

- `figshare_manifest.json`: normalized article and file metadata, file sizes, checksums supplied by Figshare, download URLs, and a deterministic manifest fingerprint.
- `storage_plan.md`: metadata-only storage accounting and a staged acquisition recommendation.

Crossing stage, only after the provenance gate passes:

- `normalized_observations.csv`: canonical parent, tissue, stain, scanner-domain, coordinate, registered-group, and image-path records.
- `crossing_matrix.csv`: observed counts for every stain × scanner cell.
- `crossing_summary.json`: provenance-parent counts, complete-group counts, missing cells, dimensions, and an audit fingerprint.
- `crossing_report.md`: human-readable feasibility decision and claim boundaries.

The crossing audit rejects missing or ambiguous parent provenance, duplicate paths, duplicate group–stain–scanner identities, contradictory path labels, malformed coordinates, and ambiguous input headers. Registered groups are defined as `parent_id|coordinate`; the highest available biological or provenance parent must remain intact across train/test splits.

No image files are downloaded by this package.

## Hard claim boundaries

1. A shared registered coordinate is an image correspondence, not a globally unique physical-slide or specimen identifier.
2. A provenance parent must be documented or independently verified; it must not be inferred from tissue category, stain, scanner, or coordinate coincidence.
3. A shared registered coordinate is not a guarantee of pixel-identical biological material across serial sections.
4. A scanner model/domain label is not a physical scanner serial number or immutable acquisition event.
5. Scanner contrasts are same-section comparisons only within a fixed stain and provenance-bounded registered group; stain contrasts are serial-section correspondences.
6. Missing preparation batch, scan batch, acquisition order, section distance, and source-event provenance remain missing.
7. The first analysis is exploratory feasibility. It cannot be promoted to confirmatory evidence without recovering the missing provenance.
