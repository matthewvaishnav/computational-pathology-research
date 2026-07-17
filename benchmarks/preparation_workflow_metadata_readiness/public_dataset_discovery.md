# Public dataset discovery for crossed preparation and scanner studies

Audit date: 2026-07-17

## Question

Does a public pathology dataset currently expose enough explicit provenance to support the next crossed-preparation experiment without inferring missing identities from filenames, sites, or publication prose?

## Search scope

The discovery pass searched public dataset repositories and primary dataset papers for combinations of:

- the same glass section scanned on multiple scanner systems;
- matched serial sections prepared under multiple staining or preparation conditions;
- raw whole-slide images or registered coordinates;
- stable biological, block, section, scanner, batch, order, and source-event identifiers;
- explicit access and license terms.

The registry records only what is explicit in the inspected primary sources. Missing fields remain `unknown` or `partial` rather than being upgraded by inference.

## Ranked result

### 1. PLISM original whole-slide images

**Status: strongest public preparation/scanner candidate, but not confirmatory-ready.**

Primary sources:

- Scientific Data paper: https://doi.org/10.1038/s41597-024-03122-5
- Original WSI deposit: https://doi.org/10.25452/figshare.plus.24988074
- Registered tile deposit: https://doi.org/10.25452/figshare.plus.23614422

Explicitly supported:

- 46 human tissue types represented in tissue microarrays;
- 13 H&E staining conditions;
- serial sections used because mutually exclusive staining conditions cannot be applied to one section;
- seven slide scanners, plus mobile-phone imaging in the broader dataset;
- the same stained sections imaged across scanner domains;
- 91 original WSIs;
- registered image groups and common patch coordinates;
- CC BY 4.0 access.

Still missing or incomplete for the repository's confirmatory gate:

- stable block identifiers and full biological hierarchy;
- preparation-batch identifiers;
- physical scanner device identifiers rather than model/domain labels alone;
- scan-batch identifiers;
- acquisition order and counterbalancing records;
- exact serial-section order and distance for every comparison;
- immutable source-event bindings connecting every WSI to a prospective acquisition record;
- a prospectively declared post-preparation workflow factor.

Decision: use PLISM as the first feasibility target. It is the only candidate found that directly combines serial-section preparation variation with repeated scanner domains and public registered correspondences. Results must remain exploratory unless the missing provenance can be recovered from deposit metadata or the authors.

### 2. Multi-Scanner Canine Cutaneous Squamous Cell Carcinoma

**Status: strong confirmatory-style scanner benchmark; not a preparation benchmark.**

Primary sources:

- Dataset: https://doi.org/10.5281/zenodo.7418555
- Paper: https://arxiv.org/abs/2301.04423

Explicitly supported:

- 44 samples;
- 220 images;
- the same glass slides digitized on five scanner systems;
- local correspondences and transferred annotations;
- raw pyramidal images and file-level MD5 records;
- public access.

Limitations:

- only one stated H&E preparation per section;
- no matched serial preparation contrast;
- no explicit scan batches or acquisition order;
- scanner system names do not establish physical device serial identity;
- incomplete block and prospective source-event hierarchy.

Decision: retain as the scanner-only positive control and provenance comparison dataset. Do not use it to claim preparation effects.

### 3. PLISM registered tiles

**Status: low-cost feasibility surface.**

The registered tile release contains 3,417 aligned groups across the PLISM staining/device combinations. It is suitable for testing loaders, grouping logic, paired sampling, and exploratory representation analyses before downloading and processing the full original WSI archive.

Decision: begin implementation against the registered tiles, then verify conclusions against original WSIs where the required source records can be reconstructed.

### 4. ANHIR serial sections

**Status: useful serial-section registration resource; structurally unsuitable as the main crossed design.**

Primary source:

- https://anhir.grand-challenge.org/Data/

The challenge provides serial histological sections with multiple stains and several tissue subsets. However, section order is not guaranteed in some series, scanner reporting differs by subset, and the resource does not provide a clean same-section multi-scanner crossing.

Decision: use only for registration and serial-section robustness tests.

### 5. EPFL mouse liver serial sections

**Status: small serial-section engineering fixture.**

Primary source:

- https://doi.org/10.5281/zenodo.12072433

The deposit contains 15 H&E serial sections scanned on one VS200 scanner. It is useful for validating section-sequence ingestion and registration tooling, but it cannot estimate scanner or preparation contrasts.

## Discovery conclusion

No inspected public dataset currently satisfies every field required by the merged metadata-readiness audit for a confirmatory crossed preparation/scanner analysis.

PLISM is nevertheless a materially stronger candidate than the current repository artifacts because it provides explicit preparation variation, serial sections, multiple scanners, public original WSIs, and registered coordinates in one resource. The next experiment should therefore be a bounded PLISM feasibility study with three hard restrictions:

1. do not represent scanner model labels as verified physical device identity;
2. do not represent serial sections as identical tissue or pixel counterfactuals;
3. do not promote exploratory preparation/scanner separation to confirmatory evidence unless batch, order, section, and immutable acquisition provenance are recovered.

## Next implementation package

Create a dedicated PLISM feasibility package containing:

- a metadata downloader that retrieves deposit manifests before image payloads;
- a manifest normalizer mapping tissue, staining condition, imaging device, section/group, coordinates, filenames, checksums, and archive IDs;
- an automated readiness-registry generator;
- contradiction checks for section/device/preparation grouping;
- a small registered-tile smoke test;
- an original-WSI acquisition plan with storage estimates and checksum verification;
- an executed crossed-design matrix passed through the existing identifiability audit;
- an exploratory representation protocol whose claims are bounded by the missing provenance.

No model training should begin until the normalized PLISM manifest has been audited and the exact supported contrasts are printed deterministically.
