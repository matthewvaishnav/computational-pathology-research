"""Quick training monitor to check progress."""

import json
from datetime import datetime
from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints/pcam_real")
LOG_DIR = Path("logs/pcam_real")
STATUS_FILE = LOG_DIR / "training_status.json"
TOTAL_EPOCHS = 20


def print_status() -> None:
    """Print current training status."""
    if not STATUS_FILE.exists():
        print("\n[WARN] Status file not found")
        return

    with open(STATUS_FILE, encoding="utf-8") as handle:
        status = json.load(handle)

    print("\n[STATUS] Training Status:")
    print(f"  State: {status.get('state', 'unknown')}")
    print(f"  Epoch: {status.get('epoch', '?')}/{TOTAL_EPOCHS}")

    batch_idx = status.get("batch_idx", "?")
    total_batches = status.get("total_batches", "?")
    print(f"  Batch: {batch_idx}/{total_batches}")

    loss = status.get("loss")
    if loss is None:
        print("  Current Loss: N/A")
    else:
        print(f"  Current Loss: {loss:.4f}")

    if {"epoch", "batch_idx", "total_batches"} <= status.keys():
        epoch_progress = status["batch_idx"] / status["total_batches"] * 100
        total_progress = (
            (status["epoch"] - 1) / TOTAL_EPOCHS
            + (status["batch_idx"] / status["total_batches"]) / TOTAL_EPOCHS
        ) * 100
        print(f"  Epoch Progress: {epoch_progress:.1f}%")
        print(f"  Total Progress: {total_progress:.1f}%")

    if "timestamp" in status:
        last_update = datetime.fromtimestamp(status["timestamp"])
        time_since = datetime.now() - last_update
        minutes = time_since.seconds // 60
        seconds = time_since.seconds % 60
        print(f"  Last Update: {minutes}m {seconds}s ago")


def print_checkpoints() -> None:
    """Print latest checkpoints."""
    print("\n[DISK] Checkpoints:")
    if not CHECKPOINT_DIR.exists():
        print("  Checkpoint directory not found")
        return

    checkpoints = sorted(
        CHECKPOINT_DIR.glob("*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        print("  No checkpoints found")
        return

    for checkpoint in checkpoints[:3]:
        size_mb = checkpoint.stat().st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(checkpoint.stat().st_mtime)
        modified_str = modified.strftime("%I:%M %p")
        summary = f"  {checkpoint.name}: {size_mb:.1f} MB " f"(modified {modified_str})"
        print(summary)


def print_logs() -> None:
    """Print latest TensorBoard log file."""
    if not LOG_DIR.exists():
        return

    event_files = sorted(
        LOG_DIR.glob("events.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not event_files:
        return

    latest = event_files[0]
    size_kb = latest.stat().st_size / 1024
    modified = datetime.fromtimestamp(latest.stat().st_mtime)
    print("\n[LOG] Latest Log:")
    print(f"  {latest.name[:50]}...")
    print(f"  Size: {size_kb:.2f} KB")
    print(f"  Modified: {modified.strftime('%I:%M:%S %p')}")


def main() -> None:
    """Run the monitor."""
    print("=" * 60)
    print("PCam Training Monitor")
    print("=" * 60)
    print_status()
    print_checkpoints()
    print_logs()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
