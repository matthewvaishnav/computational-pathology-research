# PCam NaN Cascade Fix Bugfix Design

## Overview

The PCam training pipeline experiences cascading NaN losses when model parameters become corrupted during mixed precision training. The current implementation only skips individual batches with NaN losses but fails to detect when the underlying model parameters themselves are corrupted, leading to consecutive NaN losses that effectively halt training progress. This design implements automatic recovery from model parameter corruption while maintaining training stability through checkpoint restoration and gradient scaler reinitialization.

## Glossary

- **Bug_Condition (C)**: The condition that triggers cascading NaN losses - when model parameters become corrupted with NaN values during training
- **Property (P)**: The desired behavior when cascading NaN losses are detected - automatic recovery through checkpoint restoration and training continuation
- **Preservation**: Existing single-batch NaN handling and normal training behavior that must remain unchanged by the fix
- **handleKeyPress**: The training loop function in `experiments/train_pcam.py` that processes batches and handles NaN detection
- **cascadingNaNCount**: A counter tracking consecutive batches with NaN losses to detect parameter corruption
- **lastValidCheckpoint**: The most recent checkpoint saved before parameter corruption occurred

## Bug Details

### Bug Condition

The bug manifests when model parameters become corrupted with NaN values during mixed precision training, causing the training loop to encounter consecutive NaN losses. The current `train_one_epoch` function in `experiments/train_pcam.py` only skips individual batches but does not detect when the underlying model parameters are corrupted, leading to indefinite batch skipping without recovery.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type TrainingBatch
  OUTPUT: boolean
  
  RETURN consecutiveNaNCount >= 3
         AND modelParametersContainNaN(feature_extractor, encoder, head)
         AND NOT recoveryAttempted
END FUNCTION
```

### Examples

- **Batch 150**: NaN loss detected, batch skipped (normal behavior)
- **Batch 151**: NaN loss detected, batch skipped (normal behavior) 
- **Batch 152**: NaN loss detected, batch skipped (cascading detected - should trigger recovery)
- **Batch 153-200**: All NaN losses, all skipped without recovery (bug manifestation)
- **Edge case**: Mixed precision scaler state corruption causing persistent NaN generation

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Isolated NaN losses (1-2 non-consecutive batches) must continue to be skipped without triggering recovery
- Normal training without NaN issues must continue without any additional overhead or interference
- Validation NaN handling must continue to skip validation batches without affecting training state
- Checkpointing and logging functionality must continue to save checkpoints and log metrics as before

**Scope:**
All inputs that do NOT involve cascading NaN losses (3+ consecutive batches) should be completely unaffected by this fix. This includes:
- Single isolated NaN batches
- Normal training batches without numerical issues
- Validation and testing phases
- Model saving and loading operations

## Hypothesized Root Cause

Based on the bug description and code analysis, the most likely issues are:

1. **Model Parameter Corruption**: Mixed precision training can corrupt model parameters with NaN values
   - Feature extractor parameters become NaN during gradient updates
   - Encoder or classification head parameters accumulate NaN values
   - Corrupted parameters propagate NaN through all subsequent forward passes

2. **Gradient Scaler State Issues**: The AMP gradient scaler state becomes inconsistent
   - Scaler internal state corrupted by repeated NaN handling
   - Scale factor becomes invalid (NaN or extreme values)
   - Optimizer state becomes inconsistent with corrupted gradients

3. **Insufficient Parameter Validation**: No validation of model parameters after updates
   - Current code only checks loss and gradients, not final parameter states
   - Parameter corruption can occur during optimizer.step() without detection

4. **Missing Recovery Mechanism**: No automatic recovery from parameter corruption
   - System continues training with corrupted parameters indefinitely
   - No checkpoint restoration when corruption is detected

## Correctness Properties

Property 1: Bug Condition - Cascading NaN Recovery

_For any_ training state where cascading NaN losses occur (3+ consecutive batches with NaN), the fixed training loop SHALL detect parameter corruption, restore from the last valid checkpoint, reinitialize the gradient scaler and optimizer states, and continue training successfully.

**Validates: Requirements 2.1, 2.2, 2.4**

Property 2: Preservation - Single NaN Batch Handling

_For any_ training batch where isolated NaN losses occur (1-2 non-consecutive batches), the fixed training loop SHALL produce exactly the same behavior as the original code, skipping those batches without triggering recovery mechanisms.

**Validates: Requirements 3.1, 3.2, 3.3**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `experiments/train_pcam.py`

**Function**: `train_one_epoch`

**Specific Changes**:
1. **Add Cascading NaN Detection**: Track consecutive NaN batches with a counter
   - Initialize `consecutive_nan_count = 0` at epoch start
   - Increment counter when NaN detected, reset to 0 on successful batch
   - Trigger recovery when counter reaches threshold (3)

2. **Add Model Parameter Validation**: Check for NaN in model parameters after each batch
   - Validate feature_extractor, encoder, and head parameters
   - Detect parameter corruption immediately after optimizer.step()

3. **Implement Checkpoint-Based Recovery**: Restore from last valid checkpoint when corruption detected
   - Load model states from most recent valid checkpoint
   - Reinitialize optimizer and scheduler states
   - Reset gradient scaler to clean state

4. **Add Recovery State Management**: Track recovery attempts and prevent infinite loops
   - Limit recovery attempts per epoch (max 3)
   - Log recovery actions for debugging
   - Fail gracefully if recovery unsuccessful

5. **Enhance Checkpoint Strategy**: Save checkpoints more frequently during unstable periods
   - Save checkpoint every N batches when NaN detected
   - Maintain rolling window of recent valid checkpoints
   - Validate checkpoint integrity before saving

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that inject NaN values into model parameters during training and verify that consecutive NaN losses occur without recovery. Run these tests on the UNFIXED code to observe failures and understand the root cause.

**Test Cases**:
1. **Parameter Corruption Test**: Inject NaN into feature extractor parameters mid-training (will fail on unfixed code)
2. **Scaler State Corruption Test**: Corrupt gradient scaler state and verify cascading NaN (will fail on unfixed code)
3. **Mixed Precision Instability Test**: Use extreme learning rates to trigger natural parameter corruption (will fail on unfixed code)
4. **Recovery Limit Test**: Trigger multiple cascading events to test recovery limits (may fail on unfixed code)

**Expected Counterexamples**:
- Training continues indefinitely with consecutive NaN losses without recovery
- Possible causes: no parameter validation, no checkpoint restoration, corrupted scaler state

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := train_one_epoch_fixed(input)
  ASSERT recoverySuccessful(result) AND trainingContinues(result)
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT train_one_epoch_original(input) = train_one_epoch_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for normal training and isolated NaN batches, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Normal Training Preservation**: Verify normal batches continue to train correctly after fix
2. **Single NaN Preservation**: Verify isolated NaN batches are still skipped without recovery
3. **Validation Preservation**: Verify validation phase behavior unchanged
4. **Checkpointing Preservation**: Verify normal checkpoint saving continues to work

### Unit Tests

- Test cascading NaN detection logic with various consecutive counts
- Test model parameter validation functions for different corruption patterns
- Test checkpoint restoration with various corruption scenarios
- Test recovery state management and attempt limiting

### Property-Based Tests

- Generate random training states and verify cascading NaN detection works correctly
- Generate random parameter corruption patterns and verify recovery behavior
- Test that all non-cascading scenarios continue to work across many configurations
- Verify checkpoint integrity across various corruption and recovery cycles

### Integration Tests

- Test full training pipeline with injected parameter corruption
- Test recovery behavior across epoch boundaries
- Test interaction between recovery mechanism and early stopping
- Test that visual feedback and logging work correctly during recovery events