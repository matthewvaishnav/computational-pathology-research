# PLISM feasibility package

This package converts the public PLISM deposits into deterministic, provenance-bounded metadata products before any image payload is downloaded or any model is trained.

## Scientific purpose

PLISM is the strongest public feasibility target found for separating preparation and scanner effects because it combines:

- 13 H&E staining conditions applied to serial sections;
- seven whole-slide scanner domains;
- 91 original whole-slide images;
- registered tile groups with shared coordinates;
- 46 tissue categories;
- public Figshare Plus deposits under CC BY 4.0.

The package does **not** treat serial sections as identical tissue, scanner model names as verified physical device identities, or registered tiles as proof of causal preparation effects.

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

After obtaining the public PLISM image-list CSV, normalize it and audit the actual stain × scanner crossing:

```powershell
python benchmarks\plism_feasibility\build_crossing_matrix.py --input path\to\plism_image_list.csv
```

## Outputs

Manifest stage:

- `figshare_manifest.json`: normalized article and file metadata, file sizes, checksums supplied by Figshare, download URLs, and a deterministic manifest fingerprint.
- `storage_plan.md`: metadata-only storage accounting and a staged acquisition recommendation.

Crossing stage:

- `normalized_observations.csv`: canonical tissue, stain, scanner-domain, coordinate, registered-group, and image-path records.
- `crossing_matrix.csv`: observed counts for every stain × scanner cell.
- `crossing_summary.json`: complete-group counts, missing cells, dimensions, and an audit fingerprint.
- `crossing_report.md`: human-readable feasibility decision and claim boundaries.

The crossing audit rejects duplicate paths, duplicate group–stain–scanner identities, contradictory path labels, malformed coordinates, and ambiguous input headers. Registered groups are defined from tissue and supplied coordinates and must remain intact across train/test splits.

No image files are downloaded by this package.

## Hard claim boundaries

1. A shared registered coordinate is an image correspondence, not a guarantee of pixel-identical biological material across serial sections.
2. A scanner model/domain label is not a physical scanner serial number or immutable acquisition event.
3. Scanner contrasts are same-section comparisons only within a fixed stain and registered group; stain contrasts are serial-section correspondences.
4. Missing preparation batch, scan batch, acquisition order, section distance, and source-event provenance remain missing.
5. The first analysis is exploratory feasibility. It cannot be promoted to confirmatory evidence without recovering the missing provenance.
