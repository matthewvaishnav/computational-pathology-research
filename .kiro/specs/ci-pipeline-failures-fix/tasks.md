# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - TOML Parsing Failure on Malformed Include Line
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the TOML parsing error exists
  - **Scoped PBT Approach**: Scope the property to the concrete failing case - parsing pyproject.toml with the malformed `include = '\.pyi?` line
  - Test that parsing pyproject.toml with tomli/tomllib raises TOMLDecodeError for the unclosed string literal
  - Test that pip install simulation fails during TOML parsing phase
  - Test that the error points to the [tool.black] section around line 60
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)
  - Document counterexamples found (e.g., "TOMLDecodeError: Unterminated string at line 60")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3_

- [~] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Other Configuration Sections Remain Valid
  - **IMPORTANT**: Follow observation-first methodology
  - Create a reference pyproject.toml with [tool.black] section removed (this will be parseable)
  - Parse the reference file and observe all configuration values for [build-system], [project], [tool.pytest.ini_options], [tool.mypy], and [tool.setuptools.packages.find]
  - Write property-based tests that verify these sections would parse to the same values after the fix
  - Test that [tool.black] line-length and target-version settings are preserved
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on reference file (without malformed line)
  - **EXPECTED OUTCOME**: Tests PASS on reference file (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on reference file
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [~] 3. Fix for corrupted pyproject.toml [tool.black] include line

  - [~] 3.1 Implement the fix
    - Open pyproject.toml file
    - Locate line 60 in the [tool.black] section with the malformed `include = '\.pyi?` line
    - Replace the incomplete line with the complete valid TOML: `include = '\.pyi?$'`
    - Verify the closing single quote is added
    - Verify the regex pattern is completed with the `$` anchor
    - Do not modify any other lines in pyproject.toml
    - _Bug_Condition: isBugCondition(input) where input.file == "pyproject.toml" AND fileContains(input.file, "include = '\\.pyi?") AND NOT fileContains(input.file, "include = '\\.pyi?$'")_
    - _Expected_Behavior: TOML parser successfully parses pyproject.toml without TOMLDecodeError, pip install completes successfully_
    - _Preservation: All other pyproject.toml sections ([build-system], [project], [tool.pytest.ini_options], [tool.mypy]) parse to the same values; Black's line-length and target-version settings remain unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [~] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - TOML Parsing Success
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - Verify TOML parsing succeeds without TOMLDecodeError
    - Verify pip install simulation completes successfully
    - Verify the parsed include value is exactly `'\.pyi?$'`
    - _Requirements: 2.1, 2.2, 2.3_

  - [~] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Configuration Sections Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Verify [build-system], [project], [tool.pytest.ini_options], [tool.mypy] sections parse to same values
    - Verify [tool.black] line-length and target-version are unchanged
    - Confirm all tests still pass after fix (no regressions)

- [~] 4. Checkpoint - Ensure all tests pass
  - Run all tests to verify the fix is complete
  - Verify pip install -e . succeeds in a clean environment
  - Verify CI workflows can proceed past dependency installation
  - Ask the user if questions arise
