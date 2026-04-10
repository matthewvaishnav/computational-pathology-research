"""Quick training monitor to check progress."""

import json
import time
from pathlib import Path
from datetime import datetime

status_file = Path("logs/pcam_real/training_status.json")
checkpoint_dir = Path("checkpoints/pcam_real")

print("=" * 60)
print("PCam Training Monitor")
print("=" * 60)

# Check status file
if status_file.exists():
    with open(status_file) as f:
        status = json.load(f)

    print(f"\n📊 Training Status:")
    print(f"  State: {status.get('state', 'unknown')}")
    print(f"  Epoch: {status.get('epoch', '?')}/20")
    print(f"  Batch: {status.get('batch_idx', '?')}/{status.get('total_batches', '?')}")
    print(
        f"  Current Loss: {status.get('loss', 'N/A'):.4f}"
        if "loss" in status
        else "  Current Loss: N/A"
    )

    # Calculate progress
    if "epoch" in status and "batch_idx" in status and "total_batches" in status:
        epoch_progress = (status["batch_idx"] / status["total_batches"]) * 100
        total_progress = (
            (status["epoch"] - 1) / 20 + (status["batch_idx"] / status["total_batches"]) / 20
        ) * 100
        print(f"  Epoch Progress: {epoch_progress:.1f}%")
        print(f"  Total Progress: {total_progress:.1f}%")

    # Time since last update
    if "timestamp" in status:
        last_update = datetime.fromtimestamp(status["timestamp"])
        time_since = datetime.now() - last_update
        print(f"  Last Update: {time_since.seconds // 60}m {time_since.seconds % 60}s ago")
else:
    print("\n⚠️  Status file not found")

# Check checkpoints
print(f"\n💾 Checkpoints:")
if checkpoint_dir.exists():
    checkpoints = sorted(
        checkpoint_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if checkpoints:
        for i, ckpt in enumerate(checkpoints[:3]):
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
            print(f"  {ckpt.name}: {size_mb:.1f} MB (modified {mtime.strftime('%I:%M %p')})")
    else:
        print("  No checkpoints found")
else:
    print("  Checkpoint directory not found")

# Check log files
log_dir = Path("logs/pcam_real")
if log_dir.exists():
    event_files = sorted(log_dir.glob("events.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if event_files:
        latest = event_files[0]
        size_kb = latest.stat().st_size / 1024
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        print(f"\n📝 Latest Log:")
        print(f"  {latest.name[:50]}...")
        print(f"  Size: {size_kb:.2f} KB")
        print(f"  Modified: {mtime.strftime('%I:%M:%S %p')}")

print("\n" + "=" * 60)
