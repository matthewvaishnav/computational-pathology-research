# ScanComp first-IFD BigTIFF adapter (read-only)

## Purpose

This change integrates ScanComp decoding into the repository's established `WSIReader` at `src/data/wsi_pipeline/reader.py`. Ordinary SVS/TIFF/NDPI OpenSlide behavior remains unchanged. Only source filenames matching the official four-scanner public ScanComp pattern (`aaa123_hms360.big.tiff`, etc.) are eligible for fallback after an OpenSlide failure or its absence. This avoids another standalone image-processing stack.

Our previously observed `hmi825` files show a readable first image directory and a malformed following TIFF directory. The adapter does not infer that the files are intact, and does not traverse the malformed directory. It reads bounded compressed source tiles from the *first* image IFD; it cannot reconstruct a true pyramid, validate registration, or obtain clinical labels.

## Local setup

In an existing Python environment that supports the repository, add optional TIFF codecs:

```powershell
python -m pip install tifffile imagecodecs
```

Fetch the branch or pull request into your local repository, then run:

```powershell
cd "$env:USERPROFILE\computational-pathology-research"
python -m scripts.external_multiscanner.scancomp_validate_wsi --data-dir "$env:USERPROFILE\Downloads\hmi" --out-dir "$env:USERPROFILE\Documents\PA_NF_ScanComp_Intake\hmi825_repo" --slide-id hmi825
```

Output: `SCANCOMP_PIXEL_DECODE_RECEIPT.json`, `README_RESULTS.txt`, and at most four `*_UNREGISTERED_sparse_contact_sheet.png` files. The source TIFFs are never modified. If the 25 source-region probes for each scanner cannot all be decoded, the command exits with status 2 and the JSON documents the errors. SHA-256 streaming hashes are recorded by default for all four source files. The older `--sha256` option remains supported as a no-op for compatibility. Prior receipts lacking hashes require a new intake run for byte-level provenance.

Run synthetic tests:

```powershell
python -m pytest tests/test_scancomp_tiff_reader.py -q
```

**Scientific boundaries:** Output is a pixel-decoding receipt only. The contact sheets are sparse samples at similar *relative* coordinates, not registered tissue or full WSI thumbnails. The PA-NF frozen checkpoint scanner heads and decoder labels were trained for SCORPION scanners; do not reinterpret them as ScanComp scanner identities. Clinical G0 remains blocked by missing independent diagnosis and patient/case linkage, irrespective of decode success. No feature extraction, training or protected test evaluation is performed by this change.
