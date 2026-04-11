"""Check PCam data integrity."""

from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path("data/pcam_real")
SPLITS = ["train", "valid", "test"]
TEST_INDICES = [0, 100, 1000, 10000]


def check_split(split: str) -> None:
    """Validate one PCam split."""
    print(f"\n[DIR] Checking {split} split...")

    x_file = DATA_DIR / f"camelyonpatch_level_2_split_{split}_x.h5"
    y_file = DATA_DIR / f"camelyonpatch_level_2_split_{split}_y.h5"

    if not x_file.exists():
        print(f"  [MISSING] {x_file.name}")
        return
    if not y_file.exists():
        print(f"  [MISSING] {y_file.name}")
        return

    try:
        with h5py.File(x_file, "r") as x_handle:
            x_data = x_handle["x"]
            print(f"  [OK] Images: {x_data.shape} ({x_data.dtype})")
            size_gb = x_file.stat().st_size / (1024**3)
            print(f"     Size: {size_gb:.2f} GB")
            print("     Testing sample reads...")

            failed = 0
            for idx in TEST_INDICES:
                if idx >= x_data.shape[0]:
                    continue
                try:
                    sample = x_data[idx]
                    if np.all(sample == 0):
                        print(f"       [WARN] Sample {idx}: all zeros")
                        failed += 1
                except Exception as exc:  # pragma: no cover
                    message = f"       [ERROR] Sample {idx}: {str(exc)[:50]}"
                    print(message)
                    failed += 1

            if failed == 0:
                print("     [OK] All test samples readable")
            else:
                total = len(TEST_INDICES)
                print(f"     [WARN] {failed}/{total} samples failed")

        with h5py.File(y_file, "r") as y_handle:
            y_data = y_handle["y"]
            print(f"  [OK] Labels: {y_data.shape} ({y_data.dtype})")
            size_mb = y_file.stat().st_size / (1024**2)
            print(f"     Size: {size_mb:.2f} MB")

            labels = y_data[: min(10_000, y_data.shape[0])]
            unique, counts = np.unique(labels, return_counts=True)
            print("     Label distribution (first 10k):")
            for label, count in zip(unique, counts):
                pct = count / len(labels) * 100
                print(f"       Class {label[0]}: {count} ({pct:.1f}%)")
    except Exception as exc:  # pragma: no cover
        print(f"  [ERROR] {exc}")


def main() -> None:
    """Run the integrity check."""
    print("=" * 60)
    print("PCam Data Integrity Check")
    print("=" * 60)

    for split in SPLITS:
        check_split(split)

    summary = "  If you see many 'all zeros' or read errors, the data may be " "corrupted."
    redownload = "  Consider re-downloading with: " "python scripts/download_pcam.py"

    print("\n" + "=" * 60)
    print("Summary:")
    print(summary)
    print(redownload)
    print("=" * 60)


if __name__ == "__main__":
    main()
