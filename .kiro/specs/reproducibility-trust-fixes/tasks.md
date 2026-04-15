# Implementation Plan

## Bug 1: BenchmarkManifest Path Handling

- [x] 1. Write bug condition exploration test for Bug 1
  - **Property 1: Bug Condition** - Simple Filename Crashes
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: Scope the property to concrete failing cases: simple relative filenames without directory components (e.g., "manifest.jsonl", "results.jsonl")
  - Test that BenchmarkManifest(simple_filename) where os.path.dirname(simple_filename) == "" does NOT raise FileNotFoundError
  - Test that the manifest file is created successfully in the current directory
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS with FileNotFoundError (this is correct - it proves the bug exists)
  - Document counterexamples found (e.g., "BenchmarkManifest('manifest.jsonl') raises FileNotFoundError: Cannot create a file when that file already exists")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2_

- [x] 2. Write preservation property tests for Bug 1 (BEFORE implementing fix)
  - **Property 2: Preservation** - Directory-Prefixed and Absolute Paths
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs:
    - Directory-prefixed relative paths (e.g., "benchmarks/manifest.jsonl")
    - Absolute paths (e.g., "/tmp/manifest.jsonl")
    - Default path (None parameter)
  - Write property-based tests capturing observed behavior:
    - For directory-prefixed paths: parent directory is created, manifest file is created
    - For absolute paths: full directory path is created, manifest file is created
    - For None parameter: default "benchmarks/manifest.jsonl" is used
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 3. Fix for BenchmarkManifest simple filename handling

  - [x] 3.1 Implement the fix in src/utils/benchmark_manifest.py
    - Add empty string check before os.makedirs call in __init__ method
    - If os.path.dirname(manifest_path) returns empty string, skip directory creation
    - If not empty, proceed with os.makedirs as before
    - Implementation: `dir_path = os.path.dirname(manifest_path); if dir_path: os.makedirs(dir_path, exist_ok=True)`
    - _Bug_Condition: isBugCondition_Bug1(manifest_path) where os.path.dirname(manifest_path) == ""_
    - _Expected_Behavior: BenchmarkManifest handles simple filenames gracefully without FileNotFoundError_
    - _Preservation: Directory-prefixed paths, absolute paths, and default paths continue to work_
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 3.3_

  - [x] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Simple Filename Handling
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2_

  - [x] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Directory-Prefixed and Absolute Paths
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no regressions)

- [x] 4. Checkpoint - Ensure Bug 1 tests pass
  - Ensure all Bug 1 tests pass, ask the user if questions arise

## Bug 2: compare_pcam_baselines Command Recording

- [x] 5. Write bug condition exploration test for Bug 2
  - **Property 1: Bug Condition** - Inaccurate Command Recording
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: Scope the property to concrete failing cases: explicit config file lists or custom glob patterns
  - Test that compare_pcam_baselines.py records the ACTUAL command-line arguments (sys.argv) in the manifest
  - Test with explicit config list: `--configs config1.yaml config2.yaml`
  - Test with custom glob pattern: `--configs "custom_dir/*.yaml"`
  - Verify recorded command matches actual command, not invented glob pattern
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (recorded command uses invented pattern instead of actual arguments)
  - Document counterexamples found (e.g., "Actual: --configs config1.yaml config2.yaml, Recorded: --configs 'experiments/configs/pcam_comparison/*.yaml'")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.3, 1.4_

- [x] 6. Write preservation property tests for Bug 2 (BEFORE implementing fix)
  - **Property 2: Preservation** - Other Manifest Operations
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy manifest operations:
    - Other manifest fields (metrics, artifact_paths, caveats, notes) are populated correctly
    - --no-manifest flag skips manifest recording
    - Multiple variant comparison and aggregation work correctly
  - Write property-based tests capturing observed behavior patterns
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.4, 3.5, 3.6_

- [x] 7. Fix for compare_pcam_baselines command recording

  - [x] 7.1 Implement the fix in experiments/compare_pcam_baselines.py
    - Capture actual sys.argv at script entry point in main function
    - Store in module-level variable or pass to _record_comparison_to_manifest
    - Remove pattern inference logic (lines 268-276 that invent glob patterns)
    - Use actual command-line arguments directly in manifest entry
    - Implementation: `actual_command = " ".join(sys.argv)` at script entry, pass to recording function
    - _Bug_Condition: isBugCondition_Bug2(input) where recorded_command != actual_command_
    - _Expected_Behavior: Recorded command matches actual sys.argv, enabling exact reproduction_
    - _Preservation: Other manifest fields, --no-manifest flag, variant aggregation continue to work_
    - _Requirements: 1.3, 1.4, 2.3, 2.4, 3.4, 3.5, 3.6_

  - [x] 7.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Accurate Command Recording
    - **IMPORTANT**: Re-run the SAME test from task 5 - do NOT write a new test
    - The test from task 5 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 5
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.3, 2.4_

  - [x] 7.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Other Manifest Operations
    - **IMPORTANT**: Re-run the SAME tests from task 6 - do NOT write new tests
    - Run preservation property tests from step 6
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no regressions)

- [x] 8. Checkpoint - Ensure Bug 2 tests pass
  - Ensure all Bug 2 tests pass, ask the user if questions arise

## Bug 3: README Evaluation Command

- [x] 9. Write bug condition exploration test for Bug 3
  - **Property 1: Bug Condition** - Outdated README Command
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: Scope the property to the concrete failing case: the exact command from README.md lines 143-149
  - Test that the README command uses correct CLI flags that evaluate_camelyon.py accepts
  - Test that --generate-attention-heatmaps is NOT used (does not exist)
  - Test that correct flags are used: --tile-scores-dir, --heatmaps-dir, --save-predictions-csv
  - Parse README.md, extract evaluation command, verify it uses correct flags
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (README uses --generate-attention-heatmaps which doesn't exist)
  - Document counterexamples found (e.g., "README line 143-149 uses --generate-attention-heatmaps, but evaluate_camelyon.py only accepts --heatmaps-dir")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.5, 1.6_

- [x] 10. Write preservation property tests for Bug 3 (BEFORE implementing fix)
  - **Property 2: Preservation** - Other README Commands
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy README commands:
    - PCam training commands work as documented
    - Other evaluation scripts work as documented
    - Heatmap generation with current interface works
  - Write property-based tests capturing observed behavior patterns
  - Parse README.md, extract all commands except the buggy evaluation command
  - Verify each command uses valid CLI flags for its respective script
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.7, 3.8, 3.9_

- [x] 11. Fix for README evaluation command

  - [x] 11.1 Implement the fix in README.md
    - Update CAMELYON quick-start command at lines 143-149
    - Remove: --generate-attention-heatmaps flag
    - Add: --heatmaps-dir results/camelyon/heatmaps flag
    - Keep: --save-predictions-csv flag (already correct)
    - Ensure command syntax matches current evaluate_camelyon.py CLI interface
    - _Bug_Condition: isBugCondition_Bug3(command) where command contains --generate-attention-heatmaps_
    - _Expected_Behavior: README command uses correct CLI flags that work with current evaluator_
    - _Preservation: Other README commands continue to work as documented_
    - _Requirements: 1.5, 1.6, 2.5, 2.6, 3.7, 3.8, 3.9_

  - [x] 11.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Correct README Command
    - **IMPORTANT**: Re-run the SAME test from task 9 - do NOT write a new test
    - The test from task 9 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 9
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.5, 2.6_

  - [x] 11.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Other README Commands
    - **IMPORTANT**: Re-run the SAME tests from task 10 - do NOT write new tests
    - Run preservation property tests from step 10
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no regressions)

- [x] 12. Checkpoint - Ensure Bug 3 tests pass
  - Ensure all Bug 3 tests pass, ask the user if questions arise

## Bug 4: CITATION.cff Metadata Mismatch

- [x] 13. Write bug condition exploration test for Bug 4
  - **Property 1: Bug Condition** - Inconsistent Metadata
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: Scope the property to the concrete failing case: title/description and author mismatches
  - Test that CITATION.cff title/abstract semantically matches pyproject.toml description
  - Test that CITATION.cff authors match pyproject.toml authors
  - Parse both files, compare metadata fields for consistency
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (metadata fields are inconsistent)
  - Document counterexamples found (e.g., "CITATION.cff: 'Computational Pathology Research Framework' by 'Matthew Vaishnav', pyproject.toml: 'Novel multimodal fusion architectures' by 'Research Team'")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.7, 1.8_

- [x] 14. Write preservation property tests for Bug 4 (BEFORE implementing fix)
  - **Property 2: Preservation** - Other Metadata Fields
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy metadata fields:
    - CITATION.cff is valid CFF format for bibliographic tools
    - pyproject.toml is valid TOML format for package managers
    - CAMELYON dataset citations are preserved and correctly formatted
    - Other metadata fields (license, version, keywords) are valid
  - Write property-based tests capturing observed behavior patterns
  - Parse both files, validate format compliance and non-buggy field preservation
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.10, 3.11, 3.12_

- [x] 15. Fix for CITATION.cff and pyproject.toml metadata consistency

  - [x] 15.1 Implement the fix in CITATION.cff and pyproject.toml
    - Align project description: Use consistent description in both files
    - Recommendation: "Production-grade PyTorch framework for computational pathology research"
    - Update CITATION.cff title and abstract to match pyproject.toml description
    - Align authorship: Use consistent author attribution in both files
    - Recommendation: Use actual author/maintainer name for clarity
    - Update pyproject.toml authors to match CITATION.cff authors
    - Ensure semantic consistency across both files
    - _Bug_Condition: isBugCondition_Bug4(metadata) where title/authors differ between files_
    - _Expected_Behavior: Consistent project identity and authorship across metadata files_
    - _Preservation: Format validity, dataset citations, other metadata fields preserved_
    - _Requirements: 1.7, 1.8, 2.7, 2.8, 3.10, 3.11, 3.12_

  - [x] 15.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Consistent Metadata
    - **IMPORTANT**: Re-run the SAME test from task 13 - do NOT write a new test
    - The test from task 13 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 13
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.7, 2.8_

  - [x] 15.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Other Metadata Fields
    - **IMPORTANT**: Re-run the SAME tests from task 14 - do NOT write new tests
    - Run preservation property tests from step 14
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no regressions)

- [x] 16. Checkpoint - Ensure Bug 4 tests pass
  - Ensure all Bug 4 tests pass, ask the user if questions arise

## Final Checkpoint

- [x] 17. Final verification - Ensure all tests pass
  - Run all exploration tests (tasks 1, 5, 9, 13) - all should PASS after fixes
  - Run all preservation tests (tasks 2, 6, 10, 14) - all should still PASS
  - Verify no regressions across all four bug fixes
  - Ask the user if questions arise
