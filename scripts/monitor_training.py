#!/usr/bin/env python3
"""Monitor PCam training progress."""

import json
import time
from pathlib import Path


def monitor_training():
    """Monitor training progress from status file."""
    status_file = Path("logs/pcam/training_status.json")

    print("Monitoring PCam training progress...")
    print("Press Ctrl+C to stop monitoring (training will continue)\n")

    last_epoch = 0

    try:
        while True:
            if status_file.exists():
                try:
                    with open(status_file, "r") as f:
                        status = json.load(f)

                    state = status.get("state", "unknown")
                    epoch = status.get("epoch", 0)

                    if epoch != last_epoch:
                        print(f"\n{'='*60}")
                        print(f"Epoch {epoch}/20 - State: {state}")

                        if "train_metrics" in status:
                            metrics = status["train_metrics"]
                            print(f"  Train Loss: {metrics.get('loss', 0):.4f}")
                            print(f"  Train Acc:  {metrics.get('accuracy', 0):.4f}")
                            print(f"  Train AUC:  {metrics.get('auc', 0):.4f}")

                        last_epoch = epoch

                except json.JSONDecodeError:
                    pass

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Training continues in background.")
        print("Check logs/pcam/training_status.json for current status.")


if __name__ == "__main__":
    monitor_training()
