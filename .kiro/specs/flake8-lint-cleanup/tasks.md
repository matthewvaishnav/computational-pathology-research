# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Flake8 Lint Violations Across 11 Categories
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate lint violations exist across all 11 categories
  - **Scoped PBT Approach**: Run flake8 on the entire codebase to capture all violations by category
  - Test that flake8 reports violations for F841, F541, E231, E225, E712, E713, E722, E731, F811, F821, F402
  - Run `flake8 src/ tests/ experiments/ --max-line-length=100 --max-complexity=15 --statistics` on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS with 60+ violations across 11 categories (this is correct - it proves the bug exists)
  - Document counterexamples found:
    - F841: Unused variables in multiple files (scatter, bars, result, metrics)
    - F541: F-strings without placeholders
    - E231: Missing whitespace after comma in lists/function calls
    - E225: Missing whitespace around operators
    - E712: Boolean comparisons in tests/test_camelyon_config.py lines 85, 119
    - E713: Membership tests using 'not X in Y'
    - E722: Bare except in src/clinical/batch_inference.py line 187, experiments/train_camelyon.py line 545
    - E731: Lambda assignments in tests/test_statistical.py, monitor_training.py lines 62, 84
    - F811: Redefinition of unused names
    - F821: Undefined 'run_id' in experiments/train_pcam.py
    - F402: Import shadowed by loop variable
  - Mark task complete when test is written, run, and failures are documented
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Runtime Behavior and Test Results Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code:
    - Run complete test suite: `pytest tests/` and record all pass/fail results
    - Run specific test files that will be modified: tests/test_statistical.py, tests/test_camelyon_config.py
    - Verify exception handling works in src/clinical/batch_inference.py and experiments/train_camelyon.py
    - Verify statistical functions produce correct numerical outputs
    - Verify boolean logic produces correct results
  - Write property-based tests capturing observed behavior patterns:
    - Test suite preservation: All tests that pass on unfixed code must pass on fixed code
    - Function output preservation: Functions with removed unused variables return identical results
    - Exception handling preservation: Specified exception types catch same errors as bare except
    - Boolean logic preservation: 'is True' produces same results as '== True'
    - Lambda conversion preservation: def functions produce identical results to lambda equivalents
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 3. Fix flake8 lint violations across all 11 categories

  - [x] 3.1 Fix F841 - Unused Variables
    - Remove unused variable assignments across src/, tests/, experiments/
    - Remove matplotlib object captures: change `scatter = ax.scatter(...)` to `ax.scatter(...)`
    - Remove unused return values: change `result = func()` to `func()` when not used
    - Prefix intentionally unused variables with underscore: `_unused = value`
    - _Bug_Condition: violations.count(F841) > 0_
    - _Expected_Behavior: violations.count(F841) == 0_
    - _Preservation: All tests pass, function outputs unchanged_
    - _Requirements: 1.1, 2.1, 3.1, 3.4_

  - [x] 3.2 Fix F541 - F-string Without Placeholders
    - Convert f-strings without placeholders to regular strings
    - Change `f"static text"` to `"static text"`
    - Verify no hidden placeholders that should be escaped
    - _Bug_Condition: violations.count(F541) > 0_
    - _Expected_Behavior: violations.count(F541) == 0_
    - _Preservation: String values remain identical_
    - _Requirements: 1.2, 2.2, 3.2_

  - [x] 3.3 Fix E231 - Missing Whitespace After Comma
    - Add space after commas in lists, tuples, function calls
    - Change `[1,2,3]` to `[1, 2, 3]`
    - Change `func(a,b,c)` to `func(a, b, c)`
    - Change `{a:1,b:2}` to `{a: 1, b: 2}`
    - _Bug_Condition: violations.count(E231) > 0_
    - _Expected_Behavior: violations.count(E231) == 0_
    - _Preservation: Code logic unchanged, only formatting_
    - _Requirements: 1.3, 2.3, 3.2_

  - [x] 3.4 Fix E225 - Missing Whitespace Around Operator
    - Add spaces around assignment operators: `x=5` to `x = 5`
    - Add spaces around arithmetic operators: `a+b` to `a + b`
    - Add spaces around comparison operators: `x<5` to `x < 5`
    - _Bug_Condition: violations.count(E225) > 0_
    - _Expected_Behavior: violations.count(E225) == 0_
    - _Preservation: Code logic unchanged, only formatting_
    - _Requirements: 1.4, 2.4, 3.2_

  - [x] 3.5 Fix E712 - Boolean Comparison Style
    - Fix tests/test_camelyon_config.py line 85: `assert data["download"] == False` to `assert data["download"] is False`
    - Fix tests/test_camelyon_config.py line 119: `assert fe["pretrained"] == True` to `assert fe["pretrained"] is True`
    - _Bug_Condition: violations.count(E712) > 0_
    - _Expected_Behavior: violations.count(E712) == 0_
    - _Preservation: Boolean logic produces identical results_
    - _Requirements: 1.5, 2.5, 3.7_

  - [x] 3.6 Fix E713 - Membership Test Style
    - Reorder membership tests: `if not x in list:` to `if x not in list:`
    - Maintain identical boolean logic
    - _Bug_Condition: violations.count(E713) > 0_
    - _Expected_Behavior: violations.count(E713) == 0_
    - _Preservation: Membership test results unchanged_
    - _Requirements: 1.6, 2.6, 3.8_

  - [x] 3.7 Fix E722 - Bare Except Clause
    - Fix src/clinical/batch_inference.py line 187: change `except:` to `except queue.Full:`
    - Fix experiments/train_camelyon.py line 545: change `except:` to `except (ValueError, RuntimeError):`
    - Specify appropriate exception types based on context
    - _Bug_Condition: violations.count(E722) > 0_
    - _Expected_Behavior: violations.count(E722) == 0_
    - _Preservation: Exception handling catches same errors_
    - _Requirements: 1.7, 2.7, 3.5_

  - [x] 3.8 Fix E731 - Lambda Assignment
    - Convert lambda assignments to def functions in tests/test_statistical.py
    - Convert lambda in monitor_training.py lines 62, 84: `key=lambda path: path.stat().st_mtime`
    - Convert lambda in experiments/run_statistical_analysis.py line 248: `model_factory=lambda: AblationWrapper(...)`
    - Example: `metric_fn = lambda yt, yp, yprob: accuracy_score(yt, yp)` becomes proper def function
    - _Bug_Condition: violations.count(E731) > 0_
    - _Expected_Behavior: violations.count(E731) == 0_
    - _Preservation: Functions produce identical results_
    - _Requirements: 1.8, 2.8, 3.6_

  - [x] 3.9 Fix F811 - Redefinition of Unused Name
    - Remove first definition if never used
    - Rename if both definitions serve different purposes
    - _Bug_Condition: violations.count(F811) > 0_
    - _Expected_Behavior: violations.count(F811) == 0_
    - _Preservation: Code logic unchanged_
    - _Requirements: 1.9, 2.9, 3.2_

  - [x] 3.10 Fix F821 - Undefined Name
    - Fix experiments/train_pcam.py: define run_id variable before use, or add as parameter, or remove reference
    - Ensure all referenced names are defined in scope
    - _Bug_Condition: violations.count(F821) > 0_
    - _Expected_Behavior: violations.count(F821) == 0_
    - _Preservation: Code logic unchanged_
    - _Requirements: 1.10, 2.10, 3.2_

  - [x] 3.11 Fix F402 - Import Shadowed by Loop Variable
    - Rename loop variables that shadow imports
    - Example: `for json in items:` becomes `for json_item in items:` if json is imported
    - _Bug_Condition: violations.count(F402) > 0_
    - _Expected_Behavior: violations.count(F402) == 0_
    - _Preservation: Loop logic unchanged_
    - _Requirements: 1.11, 2.11, 3.2_

  - [ ] 3.12 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Zero Flake8 Violations
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run `flake8 src/ tests/ experiments/ --max-line-length=100 --max-complexity=15 --statistics`
    - **EXPECTED OUTCOME**: Test PASSES with 0 violations across all 11 categories (confirms bug is fixed)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11_

  - [x] 3.13 Verify preservation tests still pass
    - **Property 2: Preservation** - Runtime Behavior Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run complete test suite: `pytest tests/`
    - Run specific modified test files: tests/test_statistical.py, tests/test_camelyon_config.py
    - Verify exception handling still works correctly
    - Verify statistical functions produce identical outputs
    - Verify boolean logic produces identical results
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no regressions)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 4. Checkpoint - Ensure all tests pass and flake8 reports zero violations
  - Run full test suite: `pytest tests/`
  - Run flake8: `flake8 src/ tests/ experiments/ --max-line-length=100 --max-complexity=15`
  - Verify CI pipeline passes all checks (black, isort, mypy, pytest, flake8)
  - Ensure all tests pass, ask the user if questions arise
