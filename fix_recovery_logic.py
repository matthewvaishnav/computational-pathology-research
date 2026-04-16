#!/usr/bin/env python3
"""
Fix the cascading NaN recovery logic to trigger recovery based on consecutive NaN count alone,
not requiring parameter corruption.
"""

import re

# Read the file
with open("experiments/train_pcam.py", "r", encoding="utf-8") as f:
    content = f.read()

# Pattern to match the buggy recovery logic
old_pattern = r"""                # Check for cascading NaN condition
                if consecutive_nan_count >= cascading_nan_threshold:
                    # Check if model parameters contain NaN
                    if check_model_parameters_for_nan\(feature_extractor, encoder, head\):
                        logger\.error\(
                            f"Cascading NaN detected: \{consecutive_nan_count\} consecutive NaN batches "
                            f"with corrupted model parameters at batch \{batch_idx\}\. "
                            f"Model parameter corruption detected - attempting recovery\."
                        \)

                        # Attempt recovery from checkpoint
                        recovery_successful, recovery_attempts = perform_recovery\(
                            feature_extractor,
                            encoder,
                            head,
                            optimizer,
                            scheduler,
                            scaler,
                            config,
                            run_id,
                            epoch,
                            batch_idx,
                            recovery_attempts,
                            max_recovery_attempts,
                        \)

                        if recovery_successful:
                            logger\.info\(
                                "Recovery successful\. Resetting consecutive NaN count and continuing training\."
                            \)
                            consecutive_nan_count = 0
                            recovery_attempted = True
                            continue
                        else:
                            logger\.error\("Recovery failed\. Training cannot continue\."\)
                            raise RuntimeError\(
                                f"Cascading NaN losses detected with model parameter corruption\. "
                                f"Recovery failed after \{recovery_attempts\} attempts\. "
                                f"Consecutive NaN count: \{consecutive_nan_count\}, "
                                f"Batch: \{batch_idx\}, Epoch: \{epoch\}"
                            \)
                    else:
                        logger\.warning\(
                            f"Cascading NaN detected: \{consecutive_nan_count\} consecutive NaN batches "
                            f"at batch \{batch_idx\}, but model parameters are valid\. Continuing training\."
                        \)"""

# Replacement pattern
new_pattern = """                # Check for cascading NaN condition
                if consecutive_nan_count >= cascading_nan_threshold:
                    # Trigger recovery regardless of parameter state
                    # Cascading NaN indicates fundamental training instability
                    params_have_nan = check_model_parameters_for_nan(feature_extractor, encoder, head)
                    logger.error(
                        f"Cascading NaN detected: {consecutive_nan_count} consecutive NaN batches "
                        f"at batch {batch_idx}. Model parameters {'contain' if params_have_nan else 'do not contain'} NaN. "
                        f"Attempting recovery from checkpoint."
                    )

                    # Attempt recovery from checkpoint
                    recovery_successful, recovery_attempts = perform_recovery(
                        feature_extractor,
                        encoder,
                        head,
                        optimizer,
                        scheduler,
                        scaler,
                        config,
                        run_id,
                        epoch,
                        batch_idx,
                        recovery_attempts,
                        max_recovery_attempts,
                    )

                    if recovery_successful:
                        logger.info(
                            "Recovery successful. Resetting consecutive NaN count and continuing training."
                        )
                        consecutive_nan_count = 0
                        recovery_attempted = True
                        continue
                    else:
                        logger.error("Recovery failed. Training cannot continue.")
                        raise RuntimeError(
                            f"Cascading NaN losses detected. "
                            f"Recovery failed after {recovery_attempts} attempts. "
                            f"Consecutive NaN count: {consecutive_nan_count}, "
                            f"Batch: {batch_idx}, Epoch: {epoch}"
                        )"""

# Replace all occurrences
content_fixed = re.sub(old_pattern, new_pattern, content, flags=re.MULTILINE)

# Count replacements
num_replacements = content.count("but model parameters are valid. Continuing training.")
print(f"Found {num_replacements} occurrences to fix")

# Write back
with open("experiments/train_pcam.py", "w", encoding="utf-8") as f:
    f.write(content_fixed)

print(f"Fixed recovery logic in experiments/train_pcam.py")
print("Recovery will now trigger after 3 consecutive NaN batches regardless of parameter state")
