# macOS CI Timeout Fix Bugfix Design

## Overview

The macOS CI pipeline fails with exit code 137 (SIGKILL) during property-based tests due to memory exhaustion. The `test_patch_extraction_coordinate_consistency` test generates slide dimensions up to 50,000x50,000 pixels with max_examples=100, creating excessive memory usage that exceeds CI runner limits. The fix involves implementing CI-aware test configuration that reduces resource usage on CI environments while maintaining comprehensive test coverage in local development.

## Glossary

- **Bug_Condition (C)**: The condition that triggers memory exhaustion - when property-based tests generate large slide dimensions (>10,000 pixels) with high example counts (>50) on resource-constrained CI environments
- **Property (P)**: The desired behavior when tests run on CI - tests should complete within available memory and time limits without being killed
- **Preservation**: Existing comprehensive test coverage in local development environments that must remain unchanged
- **test_patch_extraction_coordinate_consistency**: The property-based test in `tests/dataset_testing/property_based/test_openslide_properties.py` that validates coordinate consistency during patch extraction
- **CI Environment**: GitHub Actions runners with limited memory (7GB for macOS) and time constraints
- **max_examples**: Hypothesis setting that controls the number of test cases generated per property test
- **slide_dimensions**: The width and height parameters for synthetic slide generation in property tests

## Bug Details

### Bug Condition

The bug manifests when property-based tests generate slide dimensions exceeding CI memory capacity combined with high example counts. The `test_patch_extraction_coordinate_consistency` function generates slides up to 50,000x50,000 pixels with max_examples=100, causing memory usage to exceed the ~7GB limit on macOS CI runners.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type PropertyTestExecution
  OUTPUT: boolean
  
  RETURN input.slide_width > 10000
         AND input.slide_height > 10000
         AND input.max_examples > 50
         AND input.environment == "CI"
         AND input.available_memory < required_memory(input.slide_width, input.slide_height, input.max_examples)
END FUNCTION
```

### Examples

- **Large slide with high examples**: slide_width=50000, slide_height=50000, max_examples=100 on macOS CI → Process killed with SIGKILL after ~16 minutes
- **Medium slide with high examples**: slide_width=25000, slide_height=25000, max_examples=100 on macOS CI → High memory usage, potential timeout
- **Large slide with low examples**: slide_width=50000, slide_height=50000, max_examples=10 on macOS CI → Should complete successfully
- **Small slide with high examples**: slide_width=5000, slide_height=5000, max_examples=100 on macOS CI → Should complete successfully

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Local development environments must continue to run comprehensive tests with full parameter ranges
- Test coverage for coordinate consistency and alignment preservation must remain complete
- Property validation logic and assertions must remain identical
- Other CI platforms (Ubuntu, Windows) should continue to execute successfully if they have sufficient resources

**Scope:**
All test executions that do NOT involve resource-constrained CI environments should be completely unaffected by this fix. This includes:
- Local development testing with full parameter ranges
- Manual test execution with custom configurations
- Non-property-based tests (unit tests, integration tests)

## Hypothesized Root Cause

Based on the bug description and CI logs, the most likely issues are:

1. **Excessive Memory Allocation**: Large slide dimensions (50,000x50,000) combined with patch extraction operations create memory usage that exceeds CI runner limits
   - Each mock slide requires substantial memory for coordinate tracking
   - Multiple test examples compound the memory usage
   - CI runners have limited memory (~7GB on macOS)

2. **Inappropriate Test Configuration for CI**: The current test uses the same parameters for all environments
   - max_examples=100 is excessive for resource-constrained environments
   - slide dimension ranges (up to 50,000) are too large for CI memory limits

3. **Lack of Environment Detection**: Tests don't adapt their behavior based on execution environment
   - No differentiation between local development and CI execution
   - No memory-aware parameter adjustment

4. **Cumulative Resource Usage**: Property-based tests may not properly clean up resources between examples
   - Mock objects and temporary data may accumulate
   - Memory fragmentation from repeated large allocations

## Correctness Properties

Property 1: Bug Condition - CI Resource Constraints

_For any_ property-based test execution where the bug condition holds (large slide dimensions with high example counts on CI), the fixed test configuration SHALL use reduced parameters that complete within available memory and time limits without being killed.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Preservation - Local Development Coverage

_For any_ test execution that is NOT on a resource-constrained CI environment (local development, sufficient CI resources), the fixed test configuration SHALL produce the same comprehensive test coverage as the original configuration, preserving all parameter ranges and example counts.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `tests/dataset_testing/property_based/test_openslide_properties.py`

**Function**: `test_patch_extraction_coordinate_consistency`

**Specific Changes**:
1. **Environment Detection**: Add CI environment detection logic
   - Check for CI environment variables (CI=true, GITHUB_ACTIONS=true)
   - Implement helper function to determine execution context

2. **Adaptive Test Configuration**: Implement CI-aware parameter adjustment
   - Reduce max_examples from 100 to 20 on CI environments
   - Limit slide dimensions to 10,000x10,000 on CI environments
   - Maintain full parameters for local development

3. **Memory-Aware Parameter Selection**: Adjust Hypothesis strategies based on environment
   - Create CI-specific strategy variants with reduced ranges
   - Use conditional strategy selection based on environment detection

4. **Resource Cleanup Enhancement**: Improve cleanup between test examples
   - Ensure proper disposal of mock objects and temporary data
   - Add explicit garbage collection hints for large allocations

5. **Configuration Centralization**: Create shared configuration for CI test limits
   - Define CI-specific limits in a central location
   - Allow easy adjustment of CI parameters without code changes

### Implementation Details

**Environment Detection Helper**:
```python
def is_ci_environment() -> bool:
    """Detect if running in CI environment."""
    return os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'

def get_test_config():
    """Get test configuration based on environment."""
    if is_ci_environment():
        return {
            'max_examples': 20,
            'max_slide_dimension': 10000,
            'deadline': 30000  # 30 seconds
        }
    else:
        return {
            'max_examples': 100,
            'max_slide_dimension': 50000,
            'deadline': 60000  # 60 seconds
        }
```

**Adaptive Strategy Implementation**:
```python
@given(
    slide_width=st.integers(
        min_value=1000, 
        max_value=get_test_config()['max_slide_dimension']
    ),
    slide_height=st.integers(
        min_value=1000, 
        max_value=get_test_config()['max_slide_dimension']
    ),
    patch_size=patch_size_strategy(),
    num_levels=st.integers(min_value=1, max_value=5),
)
@settings(
    max_examples=get_test_config()['max_examples'], 
    deadline=get_test_config()['deadline']
)
```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that simulate CI environment conditions with large slide dimensions and high example counts. Run these tests on the UNFIXED code to observe failures and understand the root cause.

**Test Cases**:
1. **CI Memory Exhaustion Test**: Simulate CI environment with slide_width=50000, slide_height=50000, max_examples=100 (will fail on unfixed code)
2. **CI Timeout Test**: Monitor execution time and memory usage during large parameter tests (will timeout on unfixed code)
3. **Local Environment Test**: Run same parameters in local environment to verify it works with sufficient resources (may succeed on unfixed code)
4. **Boundary Test**: Test with slide dimensions around 10,000-15,000 to find memory threshold (may fail on unfixed code)

**Expected Counterexamples**:
- Process killed with SIGKILL (exit code 137) due to memory exhaustion
- Possible causes: excessive memory allocation, lack of resource cleanup, inappropriate parameter ranges for CI

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := test_patch_extraction_coordinate_consistency_fixed(input)
  ASSERT expectedBehavior(result)
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT test_original(input) = test_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-CI environments

**Test Plan**: Observe behavior on UNFIXED code first for local development environments, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Local Development Preservation**: Verify that local environments continue to use full parameter ranges and example counts
2. **Test Coverage Preservation**: Verify that the same coordinate consistency properties are validated
3. **Other Platform Preservation**: Verify that Ubuntu and Windows CI continue to work if they have sufficient resources
4. **Non-Property Test Preservation**: Verify that unit tests and integration tests continue to work unchanged

### Unit Tests

- Test environment detection logic with various CI environment variables
- Test configuration selection based on environment detection
- Test parameter adjustment for CI vs local environments
- Test resource cleanup and memory management improvements

### Property-Based Tests

- Generate random CI environment scenarios and verify tests complete within resource limits
- Generate random local environment scenarios and verify full test coverage is maintained
- Test boundary conditions around memory and time limits across many scenarios

### Integration Tests

- Test full CI pipeline execution with fixed configuration
- Test local development workflow with preserved comprehensive testing
- Test that visual feedback and error reporting continue to work correctly