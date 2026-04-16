#!/usr/bin/env python3
"""Fix all remaining occurrences of the buggy recovery logic."""

# Read the file
with open("experiments/train_pcam.py", "r", encoding="utf-8") as f:
    content = f.read()

# Simple string replacement for the warning message
old_warning = "but model parameters are valid. Continuing training."
new_warning = "Model parameters do not contain NaN. Attempting recovery anyway."

# Count occurrences before
count_before = content.count(old_warning)
print(f"Found {count_before} occurrences of the old warning message")

# Replace
content = content.replace(old_warning, new_warning)

# Count after
count_after = content.count(old_warning)
print(f"Remaining occurrences: {count_after}")

# Now fix the logic - replace the else block that continues training
# with code that triggers recovery

old_else_block = """                    else:
                        logger.warning(
                            f"Cascading NaN detected: {consecutive_nan_count} consecutive NaN batches "
                            f"at batch {batch_idx}, Model parameters do not contain NaN. Attempting recovery anyway."
                        )"""

new_else_block = """                    else:
                        logger.warning(
                            f"Cascading NaN detected: {consecutive_nan_count} consecutive NaN batches "
                            f"at batch {batch_idx}. Model parameters do not contain NaN. Attempting recovery anyway."
                        )
                        
                        # Attempt recovery even without parameter corruption
                        # Cascading NaN indicates fundamental training instability
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

# Count else blocks
count_else = content.count("Model parameters do not contain NaN. Attempting recovery anyway.")
print(f"Found {count_else} else blocks to fix")

# Replace
content = content.replace(old_else_block, new_else_block)

# Write back
with open("experiments/train_pcam.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed all recovery logic!")
print("Recovery will now trigger after 3 consecutive NaN batches regardless of parameter state")
