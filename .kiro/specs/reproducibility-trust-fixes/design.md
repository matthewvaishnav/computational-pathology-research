# Reproducibility and Trust Fixes Design

## Overview

This design addresses four critical bugs that undermine the computational pathology research framework's reliability and trustworthiness. These bugs affect:

1. **BenchmarkManifest Path Handling**: Crashes when using simple relative filenames
2. **Command Recording Accuracy**: Records incorrect commands that cannot reproduce experiments
3. **Documentation Accuracy**: README contains outdated CLI commands that fail
4. **Metadata Consistency**: Conflicting project identity between CITATION.cff and pyproject.toml

The fix strategy is surgical and minimal: handle edge cases in path creation, record actual command-line arguments, update documentation to match current CLI, and align metadata across files. All fixes preserve existing functionality for non-buggy inputs.

## Glossary

- **Bug_Condition (C)**: The condition that triggers each bug - varies per bug (simple filename, command reconstruction, outdated docs, metadata mismatch)
- **Property (P)**: The desired behavior when bug conditions are met - graceful handling, accurate recording, correct documentation, consistent metadata
- **Preservation**: Existing behavior that must remain unchanged - directory-prefixed paths, other manifest fields, working commands, valid metadata formats
- **BenchmarkManifest**: The class in `src/utils/benchmark_manifest.py` that manages experiment tracking via JSON Lines manifest files
- **compare_pcam_baselines.py**: The script in `experiments/compare_pcam_baselines.py` that runs multiple model variants and records results
- **_record_comparison_to_manifest**: The function that creates manifest entries with reconstructed commands (Bug 2 location)
- **evaluate_camelyon.py**: The evaluation script with CLI flags `--tile-scores-dir`, `--heatmaps-dir`, `--save-predictions-csv`
- **CITATION.cff**: Citation File Format metadata for bibliographic tools
- **pyproject.toml**: Python package metadata for build tools and package managers

## Bug Details

### Bug 1: BenchmarkManifest Path Handling

The bug manifests when a user provides a simple relative filename without directory components (e.g., "manifest.jsonl") to BenchmarkManifest. The `__init__` method calls `os.path.dirname(manifest_path)` which returns an empty string "", then `os.makedirs("")` raises FileNotFoundError because it cannot create a directory with an empty name.

**Formal Specification:**
```
FUNCTION isBugCondition_Bug1(input)
  INPUT: input of type string (manifest_path)
  OUTPUT: boolean
  
  RETURN (os.path.dirname(input) == "")
         AND (input is not None)
         AND (input is a relative filename without "/" or "\\")
END FUNCTION
```

### Bug 2: compare_pcam_baselines Command Recording

The bug manifests when compare_pcam_baselines.py records a manifest entry. The `_record_comparison_to_manifest` function reconstructs the command by inventing a `*.yaml` glob pattern from the first config's parent directory, not the actual command-line arguments that were executed. This means if the user provided explicit config file lists or different glob patterns, the recorded command will reproduce a different set of experiments.

**Formal Specification:**
```
FUNCTION isBugCondition_Bug2(input)
  INPUT: input of type CommandLineArgs (actual args used to run script)
  OUTPUT: boolean
  
  RETURN (input.configs contains explicit file list OR custom glob pattern)
         AND (recorded_command uses invented pattern from first_config.parent)
         AND (recorded_command != actual_command)
END FUNCTION
```

### Bug 3: README Evaluation Command

The bug manifests when users follow the CAMELYON quick-start documentation in README.md line 143-149. The documented command uses `--generate-attention-heatmaps` flag which does not exist in the current evaluate_camelyon.py CLI. The actual flags are `--tile-scores-dir`, `--heatmaps-dir`, and `--save-predictions-csv`.

**Formal Specification:**
```
FUNCTION isBugCondition_Bug3(input)
  INPUT: input of type CommandString (command from README)
  OUTPUT: boolean
  
  RETURN (input contains "--generate-attention-heatmaps")
         AND (evaluate_camelyon.py does not accept this flag)
         AND (argparse will raise error)
END FUNCTION
```

### Bug 4: CITATION.cff Metadata Mismatch

The bug manifests when citation tooling or package consumers read both CITATION.cff and pyproject.toml. The files contain conflicting metadata:
- CITATION.cff line 4: title = "Computational Pathology Research Framework"
- pyproject.toml line 6: description = "Novel multimodal fusion architectures for computational pathology"
- CITATION.cff line 7-8: author = "Matthew Vaishnav"
- pyproject.toml line 12: authors = "Research Team"

**Formal Specification:**
```
FUNCTION isBugCondition_Bug4(input)
  INPUT: input of type MetadataFiles (CITATION.cff, pyproject.toml)
  OUTPUT: boolean
  
  RETURN (input.citation_cff.title != input.pyproject_toml.description)
         OR (input.citation_cff.authors != input.pyproject_toml.authors)
         OR (semantic_identity_differs(input.citation_cff, input.pyproject_toml))
END FUNCTION
```

### Examples

**Bug 1 Examples:**
- `BenchmarkManifest("manifest.jsonl")` → FileNotFoundError (current behavior)
- `BenchmarkManifest("results.jsonl")` → FileNotFoundError (current behavior)
- `BenchmarkManifest("benchmarks/manifest.jsonl")` → Works correctly (preserved behavior)
- `BenchmarkManifest("/absolute/path/manifest.jsonl")` → Works correctly (preserved behavior)

**Bug 2 Examples:**
- Actual command: `python experiments/compare_pcam_baselines.py --configs config1.yaml config2.yaml`
- Recorded command: `python experiments/compare_pcam_baselines.py --configs "experiments/configs/pcam_comparison/*.yaml"`
- Result: Recorded command may run different configs than what produced the results

**Bug 3 Examples:**
- README command: `python experiments/evaluate_camelyon.py --checkpoint ... --generate-attention-heatmaps`
- Error: `error: unrecognized arguments: --generate-attention-heatmaps`
- Correct command: `python experiments/evaluate_camelyon.py --checkpoint ... --heatmaps-dir results/camelyon/heatmaps`

**Bug 4 Examples:**
- CITATION.cff: "Computational Pathology Research Framework" by "Matthew Vaishnav"
- pyproject.toml: "Novel multimodal fusion architectures" by "Research Team"
- Result: Inconsistent project identity across metadata files

## Expected Behavior

### Preservation Requirements

**Bug 1 - Unchanged Behaviors:**
- Directory-prefixed relative paths (e.g., "benchmarks/manifest.jsonl") must continue to create parent directories
- Absolute paths must continue to be handled correctly
- Default manifest path (None parameter) must continue to use "benchmarks/manifest.jsonl"
- All other BenchmarkManifest methods (add_entry, read_all, etc.) must continue to work

**Bug 2 - Unchanged Behaviors:**
- Manifest entries with other fields (metrics, artifact_paths, caveats, notes) must continue to be populated correctly
- Manifest recording disabled with --no-manifest flag must continue to skip recording
- Multiple variant comparison and aggregation must continue to work correctly
- All other comparison functionality must remain unchanged

**Bug 3 - Unchanged Behaviors:**
- Other README commands for PCam training and evaluation must continue to work as documented
- evaluate_camelyon.py with correct current flags must continue to function properly
- Heatmap generation functionality must continue to work with current interface
- All other evaluation features must remain unchanged

**Bug 4 - Unchanged Behaviors:**
- CITATION.cff must continue to be valid CFF format for bibliographic tools
- pyproject.toml must continue to be valid TOML format for package managers
- CAMELYON dataset citations in CITATION.cff must be preserved and correctly formatted
- All other metadata fields must remain valid

**Scope:**
All inputs that do NOT involve the specific bug conditions should be completely unaffected by these fixes. This includes:
- Bug 1: Paths with directory components, absolute paths, default paths
- Bug 2: Manifest entries created by other scripts, other manifest operations
- Bug 3: Other README commands, other evaluation scripts
- Bug 4: Other metadata fields, file format validity

## Hypothesized Root Cause

### Bug 1: BenchmarkManifest Path Handling

Based on the code analysis, the root cause is:

1. **Missing Edge Case Handling**: The `__init__` method in `src/utils/benchmark_manifest.py` line 51 calls `os.makedirs(os.path.dirname(manifest_path), exist_ok=True)` without checking if dirname returns an empty string
   - When manifest_path = "manifest.jsonl", `os.path.dirname("manifest.jsonl")` returns ""
   - `os.makedirs("")` raises FileNotFoundError: "Cannot create a file when that file already exists"

2. **Assumption of Directory Components**: The code assumes all manifest paths will have directory components, which is reasonable for the default case but not enforced

### Bug 2: compare_pcam_baselines Command Recording

Based on the code analysis in `experiments/compare_pcam_baselines.py` lines 268-279, the root cause is:

1. **Command Reconstruction Instead of Recording**: The `_record_comparison_to_manifest` function reconstructs the command from available data rather than recording the actual sys.argv
   - Line 268-276: Invents a glob pattern from the first config's parent directory
   - Line 277-279: Constructs a command string with the invented pattern
   - The actual command-line arguments (sys.argv) are never captured

2. **Pattern Inference Logic**: The code tries to infer a pattern, but this is fundamentally flawed because:
   - Users may provide explicit file lists: `--configs config1.yaml config2.yaml`
   - Users may provide custom glob patterns: `--configs "custom_dir/*.yaml"`
   - The inferred pattern may match different files than what was actually used

### Bug 3: README Evaluation Command

Based on the code analysis, the root cause is:

1. **Documentation Drift**: README.md line 143-149 documents `--generate-attention-heatmaps` flag which does not exist in current evaluate_camelyon.py
   - The CLI was refactored to use separate flags: `--tile-scores-dir`, `--heatmaps-dir`, `--save-predictions-csv`
   - The README was not updated to reflect the new CLI interface

2. **No Automated Documentation Validation**: There is no mechanism to ensure README commands stay in sync with actual CLI interfaces

### Bug 4: CITATION.cff Metadata Mismatch

Based on the file analysis, the root cause is:

1. **Inconsistent Project Identity**: The two files describe different aspects of the project:
   - CITATION.cff (line 4): "Computational Pathology Research Framework" - describes the framework itself
   - pyproject.toml (line 6): "Novel multimodal fusion architectures" - describes research contributions
   - These are not semantically equivalent descriptions

2. **Inconsistent Authorship**: The two files attribute the project to different entities:
   - CITATION.cff (line 7-8): "Matthew Vaishnav" - individual author
   - pyproject.toml (line 12): "Research Team" - generic team attribution
   - This creates ambiguity about project ownership

3. **No Metadata Synchronization**: The two files are maintained independently with no validation that they represent the same project consistently

## Correctness Properties

Property 1: Bug Condition 1 - Simple Filename Handling

_For any_ manifest_path where the bug condition holds (simple relative filename without directory components), the fixed BenchmarkManifest.__init__ SHALL handle it gracefully by either skipping directory creation when dirname is empty, or creating the file in the current directory without raising FileNotFoundError.

**Validates: Requirements 2.1, 2.2**

Property 2: Bug Condition 2 - Accurate Command Recording

_For any_ command-line invocation of compare_pcam_baselines.py, the fixed _record_comparison_to_manifest function SHALL record the actual command-line arguments that were used (from sys.argv), preserving the exact config paths or patterns provided, enabling exact reproduction of the experiment set.

**Validates: Requirements 2.3, 2.4**

Property 3: Bug Condition 3 - Correct Documentation

_For any_ user following the CAMELYON quick-start in README.md, the fixed documentation SHALL provide a command that works with the current evaluate_camelyon.py CLI interface, using the correct flags (--tile-scores-dir, --heatmaps-dir, --save-predictions-csv) instead of the non-existent --generate-attention-heatmaps flag.

**Validates: Requirements 2.5, 2.6**

Property 4: Bug Condition 4 - Consistent Metadata

_For any_ citation tooling or package consumer reading CITATION.cff and pyproject.toml, the fixed metadata files SHALL provide consistent project identity where title/description and author information align semantically, establishing trust for the research repository.

**Validates: Requirements 2.7, 2.8**

Property 5: Preservation 1 - Directory-Prefixed Paths

_For any_ manifest_path where the bug condition does NOT hold (directory-prefixed relative paths, absolute paths, None/default), the fixed BenchmarkManifest.__init__ SHALL produce the same result as the original function, preserving directory creation behavior.

**Validates: Requirements 3.1, 3.2, 3.3**

Property 6: Preservation 2 - Other Manifest Operations

_For any_ manifest operation that is NOT command recording in compare_pcam_baselines.py (other manifest fields, --no-manifest flag, variant aggregation), the fixed code SHALL produce the same result as the original code, preserving all other manifest functionality.

**Validates: Requirements 3.4, 3.5, 3.6**

Property 7: Preservation 3 - Other README Commands

_For any_ README command that is NOT the CAMELYON evaluation command (PCam commands, other evaluation scripts), the fixed documentation SHALL continue to work as currently documented, preserving all other command accuracy.

**Validates: Requirements 3.7, 3.8, 3.9**

Property 8: Preservation 4 - Metadata Format Validity

_For any_ metadata field that is NOT title/description/author (dataset citations, license, version, keywords), the fixed metadata files SHALL continue to be valid in their respective formats (CFF, TOML) and preserve all other metadata correctly.

**Validates: Requirements 3.10, 3.11, 3.12**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**Bug 1: File**: `src/utils/benchmark_manifest.py`

**Function**: `BenchmarkManifest.__init__`

**Specific Changes**:
1. **Add Empty String Check**: Before calling `os.makedirs`, check if `os.path.dirname(manifest_path)` returns an empty string
   - If empty, skip directory creation (file will be created in current directory)
   - If not empty, proceed with `os.makedirs` as before

2. **Implementation Approach**:
   ```python
   dir_path = os.path.dirname(manifest_path)
   if dir_path:  # Only create directory if dirname is not empty
       os.makedirs(dir_path, exist_ok=True)
   ```

**Bug 2: File**: `experiments/compare_pcam_baselines.py`

**Function**: `_record_comparison_to_manifest`

**Specific Changes**:
1. **Capture Actual Command**: Instead of reconstructing the command, capture the actual sys.argv at script entry point
   - Store sys.argv in a module-level variable or pass it to the function
   - Use the actual command-line arguments in the manifest entry

2. **Remove Pattern Inference Logic**: Delete lines 268-276 that invent glob patterns from config paths

3. **Implementation Approach**:
   - At script entry (main function): `actual_command = " ".join(sys.argv)`
   - Pass actual_command to `_record_comparison_to_manifest`
   - Use actual_command directly in the manifest entry instead of reconstructed comparison_command

**Bug 3: File**: `README.md`

**Specific Changes**:
1. **Update Command Syntax**: Replace the outdated command at lines 143-149 with the correct current CLI flags
   - Remove: `--generate-attention-heatmaps`
   - Add: `--heatmaps-dir results/camelyon/heatmaps`
   - Keep: `--save-predictions-csv` (already correct)

2. **Implementation Approach**:
   ```bash
   python experiments/evaluate_camelyon.py \
     --checkpoint checkpoints/camelyon/best_model.pth \
     --data-root data/camelyon \
     --output-dir results/camelyon \
     --save-predictions-csv \
     --heatmaps-dir results/camelyon/heatmaps
   ```

**Bug 4: Files**: `CITATION.cff` and `pyproject.toml`

**Specific Changes**:
1. **Align Project Description**: Choose one consistent description that represents the project accurately
   - Option A: Use "Computational Pathology Research Framework" in both files (framework-focused)
   - Option B: Use "Production-grade PyTorch framework for computational pathology research" (more descriptive)
   - Recommendation: Option B provides more context while remaining accurate

2. **Align Authorship**: Choose one consistent author attribution
   - Option A: Use "Matthew Vaishnav" in both files (individual attribution)
   - Option B: Use "Research Team" in both files (team attribution)
   - Recommendation: Use the actual author/maintainer name for clarity and accountability

3. **Implementation Approach**:
   - Update CITATION.cff title and abstract to match pyproject.toml description (or vice versa)
   - Update pyproject.toml authors to match CITATION.cff authors (or vice versa)
   - Ensure semantic consistency across both files

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate each bug on unfixed code, then verify the fixes work correctly and preserve existing behavior. Since these are four independent bugs, each will have its own exploratory and validation tests.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate all four bugs BEFORE implementing the fixes. Confirm or refute the root cause analysis for each bug. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that trigger each bug condition and observe the failures on UNFIXED code to understand the root causes.

**Test Cases**:

1. **Bug 1 - Simple Filename Test**: Create BenchmarkManifest with "manifest.jsonl" (will fail on unfixed code with FileNotFoundError)
2. **Bug 1 - Another Simple Filename**: Create BenchmarkManifest with "results.jsonl" (will fail on unfixed code)
3. **Bug 2 - Explicit Config List**: Run compare_pcam_baselines.py with explicit config files, check recorded command (will show incorrect glob pattern on unfixed code)
4. **Bug 2 - Custom Glob Pattern**: Run compare_pcam_baselines.py with custom glob, check recorded command (will show different pattern on unfixed code)
5. **Bug 3 - README Command Execution**: Execute the exact command from README.md line 143-149 (will fail with unrecognized argument on unfixed code)
6. **Bug 4 - Metadata Consistency Check**: Parse both CITATION.cff and pyproject.toml, compare title/description and authors (will show mismatches on unfixed code)

**Expected Counterexamples**:
- Bug 1: FileNotFoundError when dirname is empty string
- Bug 2: Recorded command uses invented glob pattern instead of actual arguments
- Bug 3: argparse error for unrecognized --generate-attention-heatmaps flag
- Bug 4: Inconsistent project identity and authorship across metadata files

### Fix Checking

**Goal**: Verify that for all inputs where each bug condition holds, the fixed functions produce the expected behavior.

**Pseudocode:**

**Bug 1:**
```
FOR ALL manifest_path WHERE isBugCondition_Bug1(manifest_path) DO
  manifest := BenchmarkManifest_fixed(manifest_path)
  ASSERT manifest is created successfully
  ASSERT manifest.manifest_path == manifest_path
  ASSERT no FileNotFoundError is raised
END FOR
```

**Bug 2:**
```
FOR ALL command_args WHERE isBugCondition_Bug2(command_args) DO
  recorded_command := run_compare_pcam_baselines_fixed(command_args)
  ASSERT recorded_command == actual_command_from_sys_argv
  ASSERT recorded_command can reproduce exact same experiment set
END FOR
```

**Bug 3:**
```
FOR ALL readme_commands WHERE isBugCondition_Bug3(readme_commands) DO
  result := execute_readme_command_fixed(readme_commands)
  ASSERT result executes successfully
  ASSERT no argparse errors
  ASSERT uses correct CLI flags
END FOR
```

**Bug 4:**
```
FOR ALL metadata_files WHERE isBugCondition_Bug4(metadata_files) DO
  citation_metadata := parse_citation_cff_fixed()
  pyproject_metadata := parse_pyproject_toml_fixed()
  ASSERT citation_metadata.title semantically_matches pyproject_metadata.description
  ASSERT citation_metadata.authors == pyproject_metadata.authors
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where each bug condition does NOT hold, the fixed functions produce the same result as the original functions.

**Pseudocode:**

**Bug 1:**
```
FOR ALL manifest_path WHERE NOT isBugCondition_Bug1(manifest_path) DO
  ASSERT BenchmarkManifest_original(manifest_path) == BenchmarkManifest_fixed(manifest_path)
END FOR
```

**Bug 2:**
```
FOR ALL manifest_operations WHERE NOT isBugCondition_Bug2(manifest_operations) DO
  ASSERT original_behavior(manifest_operations) == fixed_behavior(manifest_operations)
END FOR
```

**Bug 3:**
```
FOR ALL readme_commands WHERE NOT isBugCondition_Bug3(readme_commands) DO
  ASSERT original_command_works(readme_commands) == fixed_command_works(readme_commands)
END FOR
```

**Bug 4:**
```
FOR ALL metadata_fields WHERE NOT isBugCondition_Bug4(metadata_fields) DO
  ASSERT original_metadata(metadata_fields) == fixed_metadata(metadata_fields)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for non-bug inputs, then write property-based tests capturing that behavior.

**Test Cases**:

1. **Bug 1 Preservation**: Verify directory-prefixed paths ("benchmarks/manifest.jsonl"), absolute paths, and default paths continue to work
2. **Bug 2 Preservation**: Verify other manifest fields (metrics, artifact_paths), --no-manifest flag, and variant aggregation continue to work
3. **Bug 3 Preservation**: Verify other README commands (PCam training, other evaluation scripts) continue to work as documented
4. **Bug 4 Preservation**: Verify other metadata fields (license, version, keywords, dataset citations) remain valid and unchanged

### Unit Tests

- Test BenchmarkManifest with simple filenames, directory-prefixed paths, absolute paths, and None/default
- Test compare_pcam_baselines.py command recording with explicit file lists, glob patterns, and --no-manifest flag
- Test README command execution with current CLI flags
- Test metadata parsing and consistency validation for CITATION.cff and pyproject.toml
- Test edge cases: empty strings, special characters in paths, missing files, invalid metadata

### Property-Based Tests

- Generate random manifest paths (simple filenames, directory paths, absolute paths) and verify correct handling
- Generate random config file lists and glob patterns, verify recorded commands match actual commands
- Generate random CLI flag combinations for evaluate_camelyon.py, verify all valid combinations work
- Generate random metadata field values, verify consistency validation works correctly

### Integration Tests

- Test full workflow: create manifest with simple filename, add entries, read entries, generate markdown
- Test full workflow: run compare_pcam_baselines.py with various config patterns, verify manifest entries are accurate
- Test full workflow: follow README quick-start commands end-to-end, verify all commands execute successfully
- Test full workflow: parse metadata from both files, verify citation tools and package managers can consume consistently
