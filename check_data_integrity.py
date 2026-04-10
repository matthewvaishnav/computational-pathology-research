"""Check PCam data integrity."""

import h5py
import numpy as np
from pathlib import Path

data_dir = Path("data/pcam_real")

print("=" * 60)
print("PCam Data Integrity Check")
print("=" * 60)

splits = ["train", "valid", "test"]

for split in splits:
    print(f"\n📁 Checking {split} split...")

    # Check x (images)
    x_file = data_dir / f"camelyonpatch_level_2_split_{split}_x.h5"
    y_file = data_dir / f"camelyonpatch_level_2_split_{split}_y.h5"

    if not x_file.exists():
        print(f"  ❌ Missing: {x_file.name}")
        continue
    if not y_file.exists():
        print(f"  ❌ Missing: {y_file.name}")
        continue

    try:
        with h5py.File(x_file, "r") as f:
            x_data = f["x"]
            print(f"  ✅ Images: {x_data.shape} ({x_data.dtype})")
            print(f"     Size: {x_file.stat().st_size / (1024**3):.2f} GB")

            # Try to read a few samples
            print(f"     Testing sample reads...")
            test_indices = [0, 100, 1000, 10000]
            failed = 0
            for idx in test_indices:
                if idx < x_data.shape[0]:
                    try:
                        sample = x_data[idx]
                        if np.all(sample == 0):
                            print(f"       ⚠️  Sample {idx}: All zeros")
                            failed += 1
                    except Exception as e:
                        print(f"       ❌ Sample {idx}: {str(e)[:50]}")
                        failed += 1

            if failed == 0:
                print(f"     ✅ All test samples readable")
            else:
                print(f"     ⚠️  {failed}/{len(test_indices)} samples failed")

        with h5py.File(y_file, "r") as f:
            y_data = f["y"]
            print(f"  ✅ Labels: {y_data.shape} ({y_data.dtype})")
            print(f"     Size: {y_file.stat().st_size / (1024**2):.2f} MB")

            # Check label distribution
            labels = y_data[: min(10000, y_data.shape[0])]
            unique, counts = np.unique(labels, return_counts=True)
            print(f"     Label distribution (first 10k):")
            for label, count in zip(unique, counts):
                print(f"       Class {label[0]}: {count} ({count/len(labels)*100:.1f}%)")

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")

print("\n" + "=" * 60)
print("Summary:")
print("  If you see many 'All zeros' or read errors, the data may be corrupted.")
print("  Consider re-downloading with: python scripts/download_pcam.py")
print("=" * 60)
