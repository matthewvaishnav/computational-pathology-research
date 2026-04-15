# Bugfix Requirements Document

## Introduction

This document addresses four critical reproducibility and trust issues in the computational pathology research repository that undermine the framework's reliability for research use. These bugs affect manifest generation, experiment reproducibility, documentation accuracy, and project metadata consistency - all essential for a production-grade research framework.

## Bug Analysis

### Current Behavior (Defect)

#### Bug 1: BenchmarkManifest Path Handling

1.1 WHEN manifest_path is a relative filename without directory components (e.g., "manifest.jsonl") THEN the system crashes with FileNotFoundError because `os.path.dirname("manifest.jsonl")` returns empty string "" and `os.makedirs("")` raises FileNotFoundError

1.2 WHEN users attempt to use simple relative filenames for manifests THEN they are forced to use directory-prefixed paths or absolute paths, reducing usability

#### Bug 2: compare_pcam_baselines Command Recording

1.3 WHEN compare_pcam_baselines.py records a manifest entry THEN it reconstructs the train/eval command by inventing a `*.yaml` glob pattern from the first config's directory, not the actual command that was executed

1.4 WHEN the actual command used explicit config file lists or different glob patterns THEN the recorded command in the manifest can reproduce a different set of experiments than what produced the results

#### Bug 3: README Evaluation Command

1.5 WHEN users follow the CAMELYON quick-start in README.md line 142 THEN they execute `experiments/evaluate_camelyon.py --generate-attention-heatmaps` which fails because the current evaluator exposes different CLI flags

1.6 WHEN users attempt to use the documented command THEN they encounter argument errors because `--generate-attention-heatmaps` is not recognized, while the actual flags are `--tile-scores-dir`, `--heatmaps-dir`, and `--save-predictions-csv`

#### Bug 4: CITATION.cff Metadata Mismatch

1.7 WHEN citation tooling or package consumers read CITATION.cff line 4 and pyproject.toml line 6 THEN they encounter conflicting metadata where title/description/author identity differ between the two files

1.8 WHEN researchers cite the project or package managers parse metadata THEN they report inconsistent project ownership and descriptions, creating a trust issue for a research repository

### Expected Behavior (Correct)

#### Bug 1: BenchmarkManifest Path Handling

2.1 WHEN manifest_path is a relative filename without directory components (e.g., "manifest.jsonl") THEN the system SHALL handle it gracefully by either skipping directory creation when dirname is empty, or creating the file in the current directory without error

2.2 WHEN users provide simple relative filenames for manifests THEN the system SHALL accept them and create the manifest file in the current working directory

#### Bug 2: compare_pcam_baselines Command Recording

2.3 WHEN compare_pcam_baselines.py records a manifest entry THEN it SHALL record the actual command-line arguments that were used to run the comparison, preserving the exact config paths or patterns provided

2.4 WHEN users read the recorded command from the manifest THEN they SHALL be able to reproduce the exact same experiment set that produced the results

#### Bug 3: README Evaluation Command

2.5 WHEN users follow the CAMELYON quick-start in README.md THEN they SHALL execute a command that works with the current evaluator's CLI interface

2.6 WHEN users run the documented evaluation command THEN it SHALL successfully execute using the correct flags: `--tile-scores-dir`, `--heatmaps-dir`, and `--save-predictions-csv`

#### Bug 4: CITATION.cff Metadata Mismatch

2.7 WHEN citation tooling or package consumers read CITATION.cff and pyproject.toml THEN they SHALL encounter consistent metadata where title, description, and author identity match between the two files

2.8 WHEN researchers cite the project or package managers parse metadata THEN they SHALL report consistent project ownership and descriptions, establishing trust for the research repository

### Unchanged Behavior (Regression Prevention)

#### Bug 1: BenchmarkManifest Path Handling

3.1 WHEN manifest_path is a directory-prefixed relative path (e.g., "benchmarks/manifest.jsonl") THEN the system SHALL CONTINUE TO create the parent directory and manifest file as it currently does

3.2 WHEN manifest_path is an absolute path THEN the system SHALL CONTINUE TO handle it correctly with directory creation

3.3 WHEN the default manifest path is used (None parameter) THEN the system SHALL CONTINUE TO use "benchmarks/manifest.jsonl" as the default

#### Bug 2: compare_pcam_baselines Command Recording

3.4 WHEN compare_pcam_baselines.py creates manifest entries with other fields (metrics, artifact_paths, caveats, notes) THEN the system SHALL CONTINUE TO populate these fields correctly

3.5 WHEN the manifest recording is disabled with --no-manifest flag THEN the system SHALL CONTINUE TO skip manifest recording

3.6 WHEN multiple variants are compared THEN the system SHALL CONTINUE TO aggregate results and generate comparison reports correctly

#### Bug 3: README Evaluation Command

3.7 WHEN users follow other README commands for PCam training and evaluation THEN these commands SHALL CONTINUE TO work as documented

3.8 WHEN users use the evaluate_camelyon.py script with the correct current flags THEN it SHALL CONTINUE TO function properly

3.9 WHEN users generate attention heatmaps using the correct current interface THEN the functionality SHALL CONTINUE TO work

#### Bug 4: CITATION.cff Metadata Mismatch

3.10 WHEN citation tooling reads CITATION.cff for bibliographic information THEN it SHALL CONTINUE TO parse the file correctly in CFF format

3.11 WHEN package managers read pyproject.toml for package metadata THEN they SHALL CONTINUE TO parse the file correctly in TOML format

3.12 WHEN the repository references (CAMELYON dataset citations) are read from CITATION.cff THEN they SHALL CONTINUE TO be preserved and correctly formatted
