"""
Prepare PANDA Features for Training

1. Normalize coordinates to [0, 1]
2. Add labels from train.csv
3. Create train/val/test splits
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize_coordinates_inplace(features_dir, dry_run=False):
    """Normalize coordinates in HDF5 files (in-place)."""
    features_dir = Path(features_dir)
    h5_files = list(features_dir.glob('*.h5'))
    
    print(f"Found {len(h5_files)} HDF5 files")
    
    if dry_run:
        print("DRY RUN - no files will be modified")
    
    for file_path in tqdm(h5_files, desc="Normalizing coordinates"):
        try:
            with h5py.File(file_path, 'r+' if not dry_run else 'r') as f:
                # Get coordinate key
                coord_key = 'coords' if 'coords' in f else 'coordinates'
                
                if coord_key not in f:
                    print(f"Warning: {file_path.name} has no coordinates")
                    continue
                
                coords = f[coord_key][:]
                
                # Check if already normalized
                if coords.max() <= 1.5:
                    continue  # Already normalized
                
                # Normalize to [0, 1]
                coords_min = coords.min(axis=0)
                coords_max = coords.max(axis=0)
                coords_range = coords_max - coords_min
                
                # Avoid division by zero
                coords_range[coords_range == 0] = 1.0
                
                coords_norm = (coords - coords_min) / coords_range
                
                if not dry_run:
                    # Update in-place
                    del f[coord_key]
                    f.create_dataset(coord_key, data=coords_norm)
        
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    print(f"✓ Coordinate normalization {'would be' if dry_run else 'is'} complete")


def add_labels_from_csv(features_dir, csv_path, dry_run=False):
    """Add labels from PANDA train.csv to HDF5 files."""
    features_dir = Path(features_dir)
    
    # Load labels
    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found - skipping label addition")
        print("You can download it from: https://www.kaggle.com/c/prostate-cancer-grade-assessment/data")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} labels from {csv_path}")
    
    # Create slide_id -> label mapping
    label_map = dict(zip(df['image_id'], df['isup_grade']))
    
    h5_files = list(features_dir.glob('*.h5'))
    matched = 0
    
    for file_path in tqdm(h5_files, desc="Adding labels"):
        slide_id = file_path.stem
        
        if slide_id not in label_map:
            continue
        
        label = label_map[slide_id]
        matched += 1
        
        if not dry_run:
            try:
                with h5py.File(file_path, 'r+') as f:
                    if 'label' in f:
                        del f['label']
                    f.create_dataset('label', data=label)
                    
                    if 'slide_id' not in f:
                        f.create_dataset('slide_id', data=slide_id)
            except Exception as e:
                print(f"Error adding label to {file_path.name}: {e}")
    
    print(f"✓ Added labels to {matched}/{len(h5_files)} files")
    
    if matched < len(h5_files):
        print(f"Warning: {len(h5_files) - matched} files have no labels")


def create_splits(features_dir, output_file, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create train/val/test splits."""
    features_dir = Path(features_dir)
    h5_files = list(features_dir.glob('*.h5'))
    
    # Get slide IDs
    slide_ids = [f.stem for f in h5_files]
    
    # Shuffle
    np.random.seed(seed)
    indices = np.random.permutation(len(slide_ids))
    
    # Split
    n_train = int(len(slide_ids) * train_ratio)
    n_val = int(len(slide_ids) * val_ratio)
    
    train_ids = [slide_ids[i] for i in indices[:n_train]]
    val_ids = [slide_ids[i] for i in indices[n_train:n_train+n_val]]
    test_ids = [slide_ids[i] for i in indices[n_train+n_val:]]
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"✓ Created splits:")
    print(f"  Train: {len(train_ids)} ({100*train_ratio:.0f}%)")
    print(f"  Val: {len(val_ids)} ({100*val_ratio:.0f}%)")
    print(f"  Test: {len(test_ids)} ({100*(1-train_ratio-val_ratio):.0f}%)")
    print(f"  Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare PANDA features for training')
    parser.add_argument('features_dir', type=str, help='Directory containing HDF5 feature files')
    parser.add_argument('--csv_path', type=str, default='panda/train.csv', help='Path to PANDA train.csv')
    parser.add_argument('--splits_file', type=str, default='panda/splits.json', help='Output splits file')
    parser.add_argument('--normalize', action='store_true', help='Normalize coordinates')
    parser.add_argument('--add_labels', action='store_true', help='Add labels from CSV')
    parser.add_argument('--create_splits', action='store_true', help='Create train/val/test splits')
    parser.add_argument('--all', action='store_true', help='Run all preparation steps')
    parser.add_argument('--dry_run', action='store_true', help='Dry run (no modifications)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("PANDA Feature Preparation")
    print("=" * 60)
    
    if args.all:
        args.normalize = True
        args.add_labels = True
        args.create_splits = True
    
    if args.normalize:
        print("\n1. Normalizing coordinates...")
        normalize_coordinates_inplace(args.features_dir, args.dry_run)
    
    if args.add_labels:
        print("\n2. Adding labels...")
        add_labels_from_csv(args.features_dir, args.csv_path, args.dry_run)
    
    if args.create_splits:
        print("\n3. Creating splits...")
        if not args.dry_run:
            create_splits(args.features_dir, args.splits_file)
        else:
            print("DRY RUN - splits not created")
    
    print("\n" + "=" * 60)
    print("✓ Preparation complete!")
    print("=" * 60)
    
    if not args.dry_run:
        print("\nNext steps:")
        print("1. Verify features: python scripts/verify_panda_features.py", args.features_dir)
        print("2. Start training: python scripts/train_v2_0.py --data_dir", args.features_dir)


if __name__ == '__main__':
    main()
