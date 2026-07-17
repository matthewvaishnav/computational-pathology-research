# Preparation/workflow metadata readiness report

Audit ID: `preparation_workflow_metadata_readiness_v1`
Input SHA-256: `802f8716bd1fd390d54ff13c8946ef91f18c787fa8d6720e43629006d2c49148`
Audit fingerprint: `b880cb0ae22e7ee62a9ba8117e119f1d8e11c16e9500c88ecd4b819af9b7b471`

## Boundaries

- This is a metadata feasibility result, not an experimental result.
- No candidate is confirmatory-ready unless every required field is explicit and verified.
- Scanner suppression is not evidence of biological validity.
- Absence of metadata is not evidence that the underlying factor was absent.
- Inferred site or scanner labels are not process provenance.

## Candidate results

### ANHIR challenge serial-section histology dataset (`anhir_serial_sections`)

- Overall tier: **candidate_discovery**
- Preparation contrast: **candidate_discovery**
- Scanner contrast: **candidate_discovery**
- Workflow contrast: **candidate_discovery**
- Evidence: `verified_primary_source` from `https://anhir.grand-challenge.org/Data/`
- Blocking reasons: `access_or_license_unresolved`, `missing_acquisition_order`, `missing_biological_anchor`, `missing_block_identity`, `missing_immutable_source_provenance`, `missing_preparation_batch`, `missing_same_section_scanner_pairing`, `missing_scan_batch`, `missing_scanner_identity`, `missing_workflow_definition`
- Recommended next action: Resolve the listed provenance and access gaps; do not treat inferred labels as process provenance.

### Current repository paired-acquisition artifacts (`current_repo_paired_acquisition_artifacts`)

- Overall tier: **candidate_discovery**
- Preparation contrast: **candidate_discovery**
- Scanner contrast: **candidate_discovery**
- Workflow contrast: **candidate_discovery**
- Evidence: `verified_repository_metadata` from `benchmarks/paired_acquisition_provenance_manifest/ and benchmarks/crossed_preparation_identifiability/`
- Blocking reasons: `access_or_license_unresolved`, `missing_acquisition_order`, `missing_biological_anchor`, `missing_block_identity`, `missing_immutable_source_provenance`, `missing_preparation_batch`, `missing_preparation_condition`, `missing_same_section_scanner_pairing`, `missing_scan_batch`, `missing_scanner_identity`, `missing_section_identity`, `missing_serial_section_relationship`, `missing_workflow_definition`
- Recommended next action: Resolve the listed provenance and access gaps; do not treat inferred labels as process provenance.

### Whole slide images of mouse liver serial sections (`mouse_liver_serial_sections_zenodo`)

- Overall tier: **candidate_discovery**
- Preparation contrast: **candidate_discovery**
- Scanner contrast: **candidate_discovery**
- Workflow contrast: **candidate_discovery**
- Evidence: `verified_primary_source` from `https://doi.org/10.5281/zenodo.12072433`
- Blocking reasons: `access_or_license_unresolved`, `missing_acquisition_order`, `missing_biological_anchor`, `missing_block_identity`, `missing_immutable_source_provenance`, `missing_preparation_batch`, `missing_same_section_scanner_pairing`, `missing_scan_batch`, `missing_scanner_identity`, `missing_workflow_definition`, `site_not_equivalent_to_workflow`
- Recommended next action: Resolve the listed provenance and access gaps; do not treat inferred labels as process provenance.

### Multi-Scanner Canine Cutaneous Squamous Cell Carcinoma Histopathology Dataset (`multiscanner_canine_scc`)

- Overall tier: **candidate_discovery**
- Preparation contrast: **candidate_discovery**
- Scanner contrast: **candidate_discovery**
- Workflow contrast: **candidate_discovery**
- Evidence: `verified_primary_source` from `https://doi.org/10.5281/zenodo.7418555 and https://arxiv.org/abs/2301.04423`
- Blocking reasons: `missing_acquisition_order`, `missing_block_identity`, `missing_preparation_batch`, `missing_scan_batch`, `missing_scanner_identity`, `missing_serial_section_relationship`, `missing_workflow_definition`
- Recommended next action: Resolve the listed provenance and access gaps; do not treat inferred labels as process provenance.

### PLISM original whole-slide images (`plism_original_wsi`)

- Overall tier: **candidate_discovery**
- Preparation contrast: **candidate_discovery**
- Scanner contrast: **candidate_discovery**
- Workflow contrast: **candidate_discovery**
- Evidence: `verified_primary_source` from `https://doi.org/10.1038/s41597-024-03122-5 and https://doi.org/10.25452/figshare.plus.24988074`
- Blocking reasons: `factor_nesting_unresolved`, `missing_acquisition_order`, `missing_biological_anchor`, `missing_block_identity`, `missing_immutable_source_provenance`, `missing_preparation_batch`, `missing_scan_batch`, `missing_scanner_identity`, `missing_section_identity`, `missing_workflow_definition`, `site_not_equivalent_to_workflow`
- Recommended next action: Resolve the listed provenance and access gaps; do not treat inferred labels as process provenance.

### PLISM registered multi-device and staining tiles (`plism_registered_tiles`)

- Overall tier: **candidate_discovery**
- Preparation contrast: **candidate_discovery**
- Scanner contrast: **candidate_discovery**
- Workflow contrast: **candidate_discovery**
- Evidence: `verified_primary_source` from `https://doi.org/10.1038/s41597-024-03122-5 and https://doi.org/10.25452/figshare.plus.23614422`
- Blocking reasons: `factor_nesting_unresolved`, `missing_acquisition_order`, `missing_biological_anchor`, `missing_block_identity`, `missing_immutable_source_provenance`, `missing_preparation_batch`, `missing_scan_batch`, `missing_scanner_identity`, `missing_section_identity`, `missing_workflow_definition`, `site_not_equivalent_to_workflow`
- Recommended next action: Resolve the listed provenance and access gaps; do not treat inferred labels as process provenance.
