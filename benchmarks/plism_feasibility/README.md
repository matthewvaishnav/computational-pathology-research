# PLISM feasibility package

This package converts the public PLISM deposits into a deterministic, provenance-bounded manifest before any image payload is downloaded or any model is trained.

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
```

Check a previously generated manifest without network access:

```powershell
python benchmarks\plism_feasibility\fetch_figshare_manifest.py --check-manifest
```

## Outputs

- `figshare_manifest.json`: normalized article and file metadata, file sizes, checksums supplied by Figshare, download URLs, and a deterministic manifest fingerprint.
- `storage_plan.md`: metadata-only storage accounting and a staged acquisition recommendation.

No image files are downloaded by this package.

## Hard claim boundaries

1. A shared registered coordinate is an image correspondence, not a guarantee of pixel-identical biological material across serial sections.
2. A scanner model/domain label is not a physical scanner serial number or immutable acquisition event.
3. Missing preparation batch, scan batch, acquisition order, section distance, and source-event provenance remain missing.
4. The first analysis is exploratory feasibility. It cannot be promoted to confirmatory evidence without recovering the missing provenance.
