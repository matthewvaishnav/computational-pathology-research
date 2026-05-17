"""
Quick setup script to run after PANDA dataset finishes downloading.

This script will:
1. Verify the download
2. Create slide index
3. Extract features
4. Start training

Usage:
    python setup_panda_after_download.py --data_dir data/panda
"""

import argparse
import subprocess
import sys
from pathlib import Path

def check_download(data_dir):
    """Check if PANDA dataset is downloaded."""
    data_dir = Path(data_dir)
    
    print("\n" + "=" * 70)
    print("1. Checking PANDA Download")
    print("=" * 70)
    
    required_files = [
        "train.csv",
        "train_images",
    ]
    
    all_present = True
    for item in required_files:
        path = data_dir / item
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {item} ({size_mb:.2f} MB)")
            else:
                num_files = len(list(path.iterdir()))
                print(f"  ✓ {item}/ ({num_files} files)")
        else:
            print(f"  ✗ {item} NOT FOUND")
            all_present = False
    
    return all_present


def create_slide_index(data_dir):
    """Create slide index from CSV."""
    print("\n" + "=" * 70)
    print("2. Creating Slide Index")
    print("=" * 70)
    
    cmd = f"""
python -c "
from pathlib import Path
from src.data.panda_dataset import PANDASlideIndex

index = PANDASlideIndex.from_csv(
    csv_path='{data_dir}/train.csv',
    image_dir='{data_dir}/train_images',
    split_ratios=(0.7, 0.15, 0.15),
    stratify=True,
    seed=42
)
index.save('{data_dir}/slide_index.json')
print(f'Created index with {{len(index)}} slides')
print(f'Grade distribution: {{index.get_grade_distribution()}}')
"
"""
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("  ✓ Slide index created successfully")
        print(result.stdout)
        return True
    else:
        print("  ✗ Failed to create slide index")
        print(result.stderr)
        return False


def extract_features(data_dir, model="resnet50"):
    """Extract features from slides."""
    print("\n" + "=" * 70)
    print(f"3. Extracting Features (using {model})")
    print("=" * 70)
    print("  This will take 4-6 hours for the full dataset...")
    print("  You can monitor progress in a separate terminal")
    
    features_dir = Path(data_dir) / f"features_{model}"
    
    cmd = [
        "python",
        "scripts/extract_panda_features.py",
        "--data_dir", str(data_dir),
        "--output_dir", str(features_dir),
        "--model", model,
        "--batch_size", "64",
        "--num_workers", "4",
    ]
    
    print(f"\n  Command: {' '.join(cmd)}")
    print("\n  Starting feature extraction in background...")
    
    # Start in background
    subprocess.Popen(cmd)
    
    print(f"\n  ✓ Feature extraction started")
    print(f"  Features will be saved to: {features_dir}")
    print(f"  Check progress with: python check_panda_status.py")
    
    return True


def start_training(data_dir, features_dir):
    """Start PANDA training."""
    print("\n" + "=" * 70)
    print("4. Starting Training")
    print("=" * 70)
    
    cmd = [
        "python",
        "experiments/train_panda.py",
        "--data_dir", str(data_dir),
        "--features_dir", str(features_dir),
        "--index_path", f"{data_dir}/slide_index.json",
        "--ordinal",
        "--epochs", "40",
        "--batch_size", "32",
        "--lr", "5e-4",
    ]
    
    print(f"\n  Command: {' '.join(cmd)}")
    print("\n  Note: Training will start after feature extraction completes")
    print("  You can run this command manually when ready:")
    print(f"\n  {' '.join(cmd)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup PANDA after download")
    parser.add_argument("--data_dir", type=str, default="data/panda", help="PANDA data directory")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet18"], help="Feature extraction model")
    parser.add_argument("--skip_features", action="store_true", help="Skip feature extraction")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PANDA Setup After Download")
    print("=" * 70)
    
    # Step 1: Check download
    if not check_download(args.data_dir):
        print("\n❌ Download incomplete. Please ensure all files are downloaded.")
        return 1
    
    # Step 2: Create index
    if not create_slide_index(args.data_dir):
        print("\n❌ Failed to create slide index")
        return 1
    
    # Step 3: Extract features
    if not args.skip_features:
        features_dir = Path(args.data_dir) / f"features_{args.model}"
        extract_features(args.data_dir, args.model)
    else:
        print("\n⏭️  Skipping feature extraction")
        features_dir = Path(args.data_dir) / f"features_{args.model}"
    
    # Step 4: Training instructions
    start_training(args.data_dir, features_dir)
    
    print("\n" + "=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print("""
Next steps:
1. Wait for feature extraction to complete (~4-6 hours)
2. Check status: python check_panda_status.py
3. Start training when features are ready
4. Monitor training: tail -f logs/panda/train.log (or check log file)
5. Evaluate model: python experiments/evaluate_panda.py --checkpoint checkpoints/panda/best_model.pth
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
